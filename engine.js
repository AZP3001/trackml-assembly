// engine.js — the WebAssembly backend, and the only file that knows the
// simulation isn't JavaScript any more.
//
// Everything above this line (script.js, the editor, tracks.js, image-import.js)
// talks to the same API the JS edition exposed — generateTrackFromPath returns
// the same track object, Engine.createBrain/brainToJSON speak the same on-disk
// format — so saved AIs and published tracks move between the two versions
// untouched.
//
// SHAPE OF THE THING
//   * One wasm instance on the main thread ("master"). It owns track geometry
//     and the authoritative brain array, and it runs the evolve step. It never
//     simulates a car.
//   * One wasm instance per worker. Each owns a contiguous slice of the
//     population and simulates only that slice.
//
// Brains flow master -> workers and never the other way, because running a
// generation does not modify a brain. Workers send back a small flat buffer:
// render rows when the screen needs them, bare fitness when it doesn't.
//
// WHY NOT SHARED MEMORY. Threads inside a single wasm instance would need
// SharedArrayBuffer, which needs COOP/COEP response headers, which GitHub Pages
// cannot send. Separate instances with their own memories need no headers at
// all and scale the same way across cores — the population partitions cleanly,
// since cars never interact.

// The build stamp the deploy wrote into the page. The wasm URLs carry it too,
// so a new deploy is a new URL and the browser fetches it; between deploys the
// module comes straight from cache instead of being downloaded and recompiled
// on every single load.
const ASSET_VERSION = (typeof document !== 'undefined' &&
    document.querySelector('meta[name="asset-version"]')?.content) || 'dev';
const WASM_URL_SCALAR = `./wasm/sim.wasm?v=${ASSET_VERSION}`;
const WASM_URL_SIMD   = `./wasm/sim-simd.wasm?v=${ASSET_VERSION}`;

// The standard one-module probe for the 128-bit SIMD proposal: a function body
// that does nothing but `i32.const 0; i8x16.splat; drop`. An engine without SIMD
// fails to *validate* it, which is the cheap check — actually instantiating the
// real SIMD module on such an engine throws instead, and we would rather find
// out in four bytes than in forty kilobytes.
const SIMD_PROBE = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0,
    1, 4, 1, 96, 0, 0,
    3, 2, 1, 0,
    10, 9, 1, 7, 0, 65, 0, 253, 15, 26, 11
]);
function simdSupported() {
    try { return WebAssembly.validate(SIMD_PROBE); } catch (e) { return false; }
}

// How many workers to spawn: every logical core, on every device.
//
// Phones used to get roughly the fast half and never more than four, because
// with equal slices a slow core set the pace at every round's barrier. Slices
// are now sized by each worker's measured speed (_sliceUp), so a little core
// gets a little slice and finishes with the rest — it adds throughput instead
// of holding everyone back. The CPU Cores slider still goes down to 1.
function workerCountFor(cores) {
    return Math.max(1, cores | 0);
}

const CORE_ICON_SVG = '<svg class="w-3 h-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20v2"></path><path d="M12 2v2"></path><path d="M17 20v2"></path><path d="M17 2v2"></path><path d="M2 12h2"></path><path d="M2 17h2"></path><path d="M2 7h2"></path><path d="M20 12h2"></path><path d="M20 17h2"></path><path d="M20 7h2"></path><path d="M7 20v2"></path><path d="M7 2v2"></path><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="8" y="8" width="8" height="8" rx="1"></rect></svg>';

const Engine = {
    module: null,        // compiled WebAssembly.Module, cloned out to workers
    master: null,        // { exports, memory } on the main thread
    workers: [],
    coreCount: 1,
    slices: [],          // { ids, count, start } per worker: the global car ids it drives
    _ready: null,
    _runState: null,
    _pendingCoreLabel: '',
    _hidden: 5,
    _popSize: 0,
    _hasGlobalBest: false,
    _globalBestFitness: -Infinity,

    // ---- startup ------------------------------------------------------
    // Compile once, instantiate many. WebAssembly.Module is structured-
    // cloneable, so workers get the compiled artifact by postMessage instead of
    // each re-fetching and re-compiling the same bytes.
    usingSimd: false,

    ready: function() {
        if (this._ready) return this._ready;
        this._ready = (async () => {
            this.usingSimd = simdSupported();
            try {
                this.module = await this._compile(this.usingSimd ? WASM_URL_SIMD : WASM_URL_SCALAR);
            } catch (e) {
                // Validation said yes but compilation said no — an engine with a
                // partial implementation, or a missing file. Scalar still runs.
                if (!this.usingSimd) throw e;
                console.warn('SIMD module failed to compile, falling back to scalar:', e.message);
                this.usingSimd = false;
                this.module = await this._compile(WASM_URL_SCALAR);
            }
            this.master = await this._instantiate();

            this.hardwareCores = navigator.hardwareConcurrency || 4;
            this.coreCount = workerCountFor(this.hardwareCores);
            this._target = { mode: 'cpu', cores: this.coreCount };

            this.workers = await this._spawnPool('cpu', this.coreCount, 0);
            this._incoming = [];
            this._pendingCoreLabel = this.coreLabelHTML();
            return this;
        })();
        return this._ready;
    },

    _compile: async function(url) {
        // Cached normally. Freshness comes from the ?v= stamp in the URL, not
        // from refusing to cache — the page itself is always fetched fresh and
        // it is what decides which version this is.
        const res = await fetch(url);
        if (!res.ok) throw new Error(`could not load ${url} (HTTP ${res.status})`);
        return WebAssembly.compile(await res.arrayBuffer());
    },

    _instantiate: async function() {
        const instance = await WebAssembly.instantiate(this.module, {});
        return { ex: instance.exports, mem: instance.exports.memory };
    },

    // ---- the worker pool -----------------------------------------------
    // Two shapes. 'cpu': one sim-worker.js (its own wasm instance, its own
    // slice of the population) per core, `cores` of them. 'gpu': a single
    // gpu-worker.js, which speaks the same protocol but runs every car on the
    // GPU. Both change LIVE, mid-generation, from the Compute panel:
    //
    //   1. whatever new workers the target shape needs are spawned and brought
    //      up to date (track, config) in the background while the current pool
    //      keeps racing — that is the slow part (a GPU has to find an adapter
    //      and compile its shaders), and it costs the running race nothing;
    //   2. then new rounds are held back, the one in flight (at most one) is
    //      allowed to finish, and every current worker hands back its cars
    //      exactly as they are ('export');
    //   3. and the pool is swapped, re-sliced, and every car handed to its new
    //      worker ('import') to carry on from precisely where it was.
    //
    // Only step 2 stops the race, and only for one message round trip, which
    // is why moving the slider or flipping to GPU takes effect within a frame
    // or two instead of at the end of the generation — the population is
    // never regenerated and nobody goes back to the start line.
    //
    // A worker is killed only after it has answered its export, never while
    // a round is waiting on it: run()'s promise counts replies, and one that
    // never arrives would freeze the main loop for good.
    mode: 'cpu',
    gpuLabel: '',
    gpuError: '',
    onPoolChange: null,
    _target: null,
    _incoming: [],
    _reshaping: null,
    _reshapeAgain: false,
    _gate: null,
    _idleWaiters: [],
    _exportSink: null,
    // Bumped by anything that hands the pool a fresh population. A reshape
    // that sees it change while it was collecting cars knows those cars are
    // stale and ships the new population instead.
    _popEpoch: 0,
    _lastConfig: null,

    // Spawns `count` workers of one kind with indices firstIndex.., resolves
    // once every one has finished its handshake and been caught up on the
    // current track and config. They sit in _incoming — reachable by
    // setTrack/pushConfig, invisible to run() — until a reshape adopts them.
    _spawnPool: function(mode, count, firstIndex) {
        const url = mode === 'gpu' ? `./gpu-worker.js?v=${ASSET_VERSION}` : `./sim-worker.js?v=${ASSET_VERSION}`;
        const spawned = [];
        const one = index => new Promise((resolve, reject) => {
            const w = new Worker(url);
            spawned.push(w);
            w.onerror = e => reject(new Error(e.message || 'worker failed to start'));
            w.onmessage = e => {
                const d = e.data;
                if (d.type === 'gpu-error') { reject(new Error(d.message)); return; }
                if (d.type !== 'ready') return;
                w.onmessage = ev => this._handleWorkerMessage(ev.data);
                w.onerror = ev => console.error('Worker error:', ev.message);
                if (d.gpu) this.gpuLabel = d.gpu;
                if (this._lastTrackMsg) w.postMessage({ type: 'track', def: this._lastTrackMsg.def, config: this._lastTrackMsg.config });
                else if (this._lastConfig) w.postMessage({ type: 'config', args: this._lastConfig });
                this._incoming.push(w);
                resolve(w);
            };
            w.postMessage({ type: 'init', module: this.module, index });
        });
        return Promise.all(Array.from({ length: count }, (_, k) => one(firstIndex + k))).catch(err => {
            for (const w of spawned) w.terminate();
            this._incoming = this._incoming.filter(w => !spawned.includes(w));
            throw err;
        });
    },

    // The badge in the Compute panel builds this fresh off the pool as it
    // actually is, so it only changes once a reshape has landed.
    coreLabelHTML: function() {
        if (this.mode === 'gpu') {
            const name = String(this.gpuLabel || '').replace(/[^\w .+-]/g, '');
            return `${CORE_ICON_SVG} GPU · WebGPU${name ? ' · ' + name : ''}`;
        }
        const n = this.workers.length, m = this.hardwareCores || n;
        return `${CORE_ICON_SVG} ${n}${n < m ? '/' + m : ''} Cores · WASM${this.usingSimd ? '+SIMD' : ''}`;
    },

    // Both are wired to controls that exist before ready() has finished; a
    // click that early is simply ignored rather than thrown.
    setCoreCount: function(n) {
        if (!this._target) return Promise.resolve();
        const cap = this.hardwareCores || this.workers.length || 1;
        this._target.cores = Math.max(1, Math.min(cap, n | 0));
        return this._kickReshape();
    },

    // Resolves once the pool has actually changed (or failed to — gpuError
    // then says why, and the pool stays on the CPU).
    setComputeMode: function(mode) {
        if (!this._target) return Promise.resolve();
        this._target.mode = mode === 'gpu' ? 'gpu' : 'cpu';
        if (mode === 'gpu') this.gpuError = '';
        return this._kickReshape();
    },

    _poolMatches: function() {
        const t = this._target;
        return t.mode === this.mode && (t.mode === 'gpu' || this.workers.length === t.cores);
    },

    // One reshape at a time; a request that arrives mid-reshape is picked up
    // the moment the current one lands, so dragging the slider across its
    // whole range converges on wherever it stopped instead of queueing one
    // reshape per notch.
    _kickReshape: function() {
        if (this._reshaping) { this._reshapeAgain = true; return this._reshaping; }
        if (this._poolMatches()) return Promise.resolve();
        const p = this._reshape().catch(err => {
            console.warn('compute pool change failed:', err);
            if (this._target.mode === 'gpu' && this.mode !== 'gpu') this.gpuError = (err && err.message) || String(err);
            // Settle the target on the pool that actually exists, so a failure
            // can't turn into a retry loop.
            this._target = { mode: this.mode, cores: this.mode === 'cpu' ? this.workers.length : this._target.cores };
        }).then(() => {
            this._reshaping = null;
            if (this.onPoolChange) this.onPoolChange();
            if (this._reshapeAgain) { this._reshapeAgain = false; return this._kickReshape(); }
        });
        this._reshaping = p;
        return p;
    },

    _reshape: async function() {
        const target = { mode: this._target.mode, cores: this._target.cores };
        const old = this.workers;
        let keep, incoming;
        if (target.mode === 'cpu' && this.mode === 'cpu') {
            // Same kind, different count: keep the ones that stay, add or drop
            // at the tail, so every kept worker's index still matches its
            // position (the throughput table and the slicer both trust that).
            keep = old.slice(0, Math.min(old.length, target.cores));
            incoming = target.cores > old.length ? await this._spawnPool('cpu', target.cores - old.length, old.length) : [];
        } else {
            keep = [];
            incoming = await this._spawnPool(target.mode, target.mode === 'gpu' ? 1 : target.cores, 0);
        }

        this._closeGate();
        try {
            await this._whenIdle();
            const epoch = this._popEpoch;
            const exported = this._popSize > 0 ? await this._exportAll(old) : null;
            const pool = keep.concat(incoming);
            for (const w of old) if (!pool.includes(w)) w.terminate();
            this._incoming = this._incoming.filter(w => !pool.includes(w));
            this.workers = pool;
            this.mode = target.mode;
            this.coreCount = pool.length;
            // Timings and recycled buffers belong to the workers that took
            // them; the new pool starts from an even split and measures again.
            this._rate = [];
            this._spare = [];
            const fresh = !exported || epoch !== this._popEpoch;
            if (!fresh) {
                for (const e of exported) if (e.gateRatio) this._lastGateRatio = e.gateRatio;
            }
            if (this._popSize > 0) {
                const oldSlices = this.slices;
                this._sliceUp();
                if (fresh) this._shipBrains(); else this._importState(exported, oldSlices);
            }
        } finally {
            this._openGate();
        }
    },

    _closeGate: function() {
        let open;
        const promise = new Promise(r => { open = r; });
        this._gate = { promise, open };
    },
    _openGate: function() {
        const g = this._gate;
        this._gate = null;
        if (g) g.open();
    },
    _whenIdle: function() {
        if (!this._runState) return Promise.resolve();
        return new Promise(r => this._idleWaiters.push(r));
    },

    // Every current worker's cars, as they are right now. A worker that
    // doesn't answer within a few seconds (a crashed thread, a hung driver)
    // isn't allowed to hold the whole app hostage: the reshape gives up on
    // migrating and ships a fresh generation to the new pool instead.
    _exportAll: function(workers) {
        return new Promise(resolve => {
            const out = [];
            let left = workers.length;
            const timer = setTimeout(() => { this._exportSink = null; resolve(null); }, 5000);
            this._exportSink = d => {
                out.push(d);
                if (--left === 0) { clearTimeout(timer); this._exportSink = null; resolve(out); }
            };
            workers.forEach(w => w.postMessage({ type: 'export' }));
        });
    },

    // Hand each worker of the (already re-sliced) pool its cars: the brains
    // and focus window are the master's for this generation, the car state is
    // whatever the old pool exported, reassembled in car-id order through the
    // slicing the old pool was given.
    _importState: function(exported, oldSlices) {
        const ex = this.master.ex;
        const W = ex.state_words();
        const all = new Uint32Array(this._popSize * W);
        for (const e of exported) {
            const ids = oldSlices[e.index] && oldSlices[e.index].ids;
            if (!ids) continue;
            for (let j = 0; j < e.count; j++) all.set(e.state.subarray(j * W, (j + 1) * W), ids[j] * W);
        }
        const stride = ex.brain_stride();
        const brains = this._f32(this.master, ex.brains_ptr(), (ex.max_cars() + 1) * stride);
        this.workers.forEach((w, i) => {
            const s = this.slices[i];
            const b = this._gather(brains, s.ids, stride);
            const state = this._gather(all, s.ids, W);
            const gateRatio = i === 0 && this._lastGateRatio ? this._lastGateRatio.slice() : null;
            const transfer = [b.buffer, state.buffer];
            if (gateRatio) transfer.push(gateRatio.buffer);
            w.postMessage({
                type: 'import', start: s.start, count: s.count, hidden: this._hidden,
                seed: (Math.random() * 0xffffffff) >>> 0, brains: b, state, gateRatio
            }, transfer);
        });
    },

    // The GPU went away under a running race (driver reset, a mobile browser
    // reclaiming it in the background). The GPU worker still holds the last
    // state it read back, so the cars move to the CPU from there.
    _onGpuLost: function(message) {
        this.gpuError = message || 'GPU device lost';
        if (this.mode === 'gpu' || this._target.mode === 'gpu') {
            this._target.mode = 'cpu';
            this._kickReshape();
        }
    },

    // Measured, not estimated: car-steps actually simulated per second of
    // wall time (so in the normal, frame-paced mode it reads the pace the
    // screen allows, and in hyper mode the real ceiling of the backend), and
    // how long one round takes end to end.
    throughput: { stepsPerSec: 0, roundMs: 0 },
    _tpSteps: 0, _tpT0: 0, _tpLast: 0,
    _noteThroughput: function(st) {
        const now = performance.now();
        let steps = 0;
        for (const r of st.rows) steps += r.carSteps || 0;
        const ms = now - st.t0;
        this.throughput.roundMs = this.throughput.roundMs ? this.throughput.roundMs * 0.8 + ms * 0.2 : ms;
        // Wall time since the previous round landed (or this round's own
        // duration after a gap, so a pause is never counted as running time).
        const gap = this._tpLast && now - this._tpLast < 1000 ? now - this._tpLast : ms;
        this._noteBench(steps, gap, st.gen, st.wantRender);
        if (!this._tpT0 || st.t0 - this._tpLast > 1000) { this._tpT0 = st.t0; this._tpSteps = 0; }
        this._tpSteps += steps;
        this._tpLast = now;
        const span = now - this._tpT0;
        if (span >= 500) {
            this.throughput.stepsPerSec = this._tpSteps * 1000 / span;
            this._tpSteps = 0;
            this._tpT0 = now;
        }
    },

    // CPU benchmark of the current run, Hyper mode only (at 1x the rate is just
    // 60 steps a second per car): car-steps/s averaged over the whole run and
    // over generations 1-100, 100-1000 and 1000+, plus one sample per
    // generation for the graph (the last 1000). app.evolve() calls
    // benchGenEnd(); resetBench() starts over when the setup being measured
    // changes (new run, track, CPU/GPU pool).
    bench: null,
    resetBench: function() {
        this.bench = { ms: 0, steps: 0, gens: 0, win: [[0, 0], [0, 0], [0, 0]], samples: [], sampleGen: [], genSteps: 0, genMs: 0, recentMs: 0, recentSteps: 0 };
    },
    _noteBench: function(steps, ms, gen, wantRender) {
        if (wantRender) return;
        const b = this.bench || (this.resetBench(), this.bench);
        const w = b.win[gen <= 100 ? 0 : gen <= 1000 ? 1 : 2];
        w[0] += steps; w[1] += ms;
        b.steps += steps; b.ms += ms;
        b.genSteps += steps; b.genMs += ms;
    },
    benchGenEnd: function(gen) {
        const b = this.bench;
        if (!b || !(b.genMs > 0)) return;
        b.samples.push(b.genSteps * 1000 / b.genMs); b.sampleGen.push(gen);
        // Smoothed cost of one generation lately: tells "the CPU got slower"
        // (car-steps/s falls) apart from "generations got longer" (steps/gen rises).
        const a = b.gens ? 0.1 : 1;
        b.recentMs += (b.genMs - b.recentMs) * a; b.recentSteps += (b.genSteps - b.recentSteps) * a;
        if (b.samples.length > 1000) { b.samples.shift(); b.sampleGen.shift(); }
        b.gens++;
        b.genSteps = 0; b.genMs = 0;
    },

    // A wasm instance's memory can be detached and replaced when it grows, so a
    // cached view can silently go stale. Re-derive on every use; it is a couple
    // of nanoseconds and it is the difference between reading geometry and
    // reading a zero-length buffer.
    _f32: function(inst, ptr, len) { return new Float32Array(inst.mem.buffer, ptr, len); },
    _i32: function(inst, ptr, len) { return new Int32Array(inst.mem.buffer, ptr, len); },

    // ---- track geometry ------------------------------------------------
    // Called from the editor on every frame while a track is being dragged
    // around, so this is a hot path in its own right even though no car is
    // moving. The generator itself is in sim.c; this only marshals.
    buildTrack: function(id, name, pathInput, width, customStartPos, customStartAngle, zones, auto) {
        zones = zones || [];
        auto = auto || {};
        const autoOn = auto.enabled ? 1 : 0;
        const autoBlend = typeof auto.blend === 'number' ? auto.blend : 0;
        const empty = () => makeTrack(id, name, pathInput || [], width, { x: 100, y: 100 }, 0, zones,
            new Float32Array(0), new Float32Array(0), new Int32Array(0), new Float32Array(0),
            new Uint8Array(0), new Float32Array(0), { enabled: !!autoOn, blend: autoBlend });
        if (!this.master || !pathInput || pathInput.length < 3) return empty();

        const ex = this.master.ex;
        const nPts = Math.min(pathInput.length, ex.max_path_pts());
        const nZones = Math.min(zones.length, ex.max_zones());

        const pIn = this._f32(this.master, ex.path_in_ptr(), nPts * 4);
        for (let i = 0; i < nPts; i++) {
            const p = pathInput[i];
            pIn[i * 4]     = p.x;
            pIn[i * 4 + 1] = p.y;
            pIn[i * 4 + 2] = p.type === 'corner' ? 1 : 0;
            pIn[i * 4 + 3] = p.radius !== undefined ? p.radius : 60;
        }
        const zIn = this._f32(this.master, ex.zone_in_ptr(), Math.max(1, nZones * 5));
        for (let i = 0; i < nZones; i++) {
            const z = zones[i];
            zIn[i * 5]     = z.x;
            zIn[i * 5 + 1] = z.y;
            zIn[i * 5 + 2] = z.radius;
            // -1 for anything unrecognised: sim.c matches on exact ids, so an
            // unknown zone has no effect, which is what the JS edition's
            // string comparisons did with one too.
            zIn[i * 5 + 3] = ZONE_TYPE_ID[z.type] !== undefined ? ZONE_TYPE_ID[z.type] : -1;
            zIn[i * 5 + 4] = z.killTimer !== undefined ? z.killTimer : 150;
        }

        const hasStart = customStartPos ? 1 : 0;
        const hasAngle = (customStartAngle !== undefined && customStartAngle !== null) ? 1 : 0;
        const ok = ex.track_build(
            ex.path_in_ptr(), nPts, width,
            hasStart, hasStart ? customStartPos.x : 0, hasStart ? customStartPos.y : 0,
            hasAngle, hasAngle ? customStartAngle : 0,
            ex.zone_in_ptr(), nZones,
            autoOn, autoBlend);
        if (!ok) return empty();

        // Copy the results out. The arena they live in is reset by the next
        // track_build, and the editor calls that once a frame.
        const nc = ex.track_centerline_count();
        const centerF32 = this._f32(this.master, ex.track_centerline_ptr(), nc * 2).slice();
        // One half-width per centreline sample. With auto width off these are
        // all the same number, and the renderer takes a faster path.
        const widthF32 = this._f32(this.master, ex.track_widths_ptr(), nc).slice();

        const nw = ex.track_wall_count();
        const wallsRaw = this._f32(this.master, ex.track_walls_ptr(), nw * 5);
        const wallsSegRaw = this._i32(this.master, ex.track_walls_ptr(), nw * 5);
        const wallsF32 = new Float32Array(nw * 4);
        const wallSeg = new Int32Array(nw);
        for (let i = 0; i < nw; i++) {
            wallsF32[i * 4]     = wallsRaw[i * 5];
            wallsF32[i * 4 + 1] = wallsRaw[i * 5 + 1];
            wallsF32[i * 4 + 2] = wallsRaw[i * 5 + 2];
            wallsF32[i * 4 + 3] = wallsRaw[i * 5 + 3];
            wallSeg[i] = wallsSegRaw[i * 5 + 4];
        }

        // Stride 7: p1x,p1y,p2x,p2y,cx,cy and then an apex flag, read as an
        // int through its own view over the same bytes.
        const ncp = ex.track_cp_count();
        const cpF32 = this._f32(this.master, ex.track_cps_ptr(), ncp * CP_STRIDE).slice();
        const cpApexRaw = this._i32(this.master, ex.track_cps_ptr(), ncp * CP_STRIDE);
        const cpApex = new Uint8Array(ncp);
        for (let i = 0; i < ncp; i++) cpApex[i] = cpApexRaw[i * CP_STRIDE + 6] ? 1 : 0;

        return makeTrack(id, name, pathInput, width,
            { x: ex.track_start_x(), y: ex.track_start_y() }, ex.track_start_angle(),
            zones, centerF32, wallsF32, wallSeg, cpF32, cpApex,
            widthF32, { enabled: !!autoOn, blend: autoBlend }, ex.track_start_cp());
    },

    // ---- config --------------------------------------------------------
    // Sensor reach for the current Max Speed, straight from the module that
    // raycasts with it — so the overlay the UI draws can never disagree with
    // what the cars actually saw.
    sensorLength: function() {
        return this.master ? this.master.ex.sensor_len() : 180;
    },

    // The most recent {def, config} sent to every worker via setTrack(),
    // config kept current by pushConfig() too — replayed at a freshly spawned
    // worker (see _spawnPool) so it starts from the same track and config as
    // everyone else instead of an empty, unbuilt instance.
    _lastTrackMsg: null,

    // Every worker that will be simulating soon: the pool, plus any spawned
    // for a reshape that hasn't landed yet — a track or config change made
    // while one is coming up has to reach it too.
    _allWorkers: function() { return this._incoming.length ? this.workers.concat(this._incoming) : this.workers; },

    // Mutation settings only matter where evolve() runs: the master.
    _pushMutation: function(state) {
        if (state.mutRate !== undefined) this.master.ex.set_mutation(state.mutRate, state.sigma, state.sigmaFloor);
    },

    // sim.c set_config's arguments, in order.
    _configArgs: function(state) {
        const p = state.physics;
        return [p.maxSpeed, p.acceleration, p.turnSpeed, p.brakeStrength,
                state.initialTTL, state.targetLaps, state.focusPct, state.hiddenLayers,
                state.nudgeMode ? 1 : 0, state.paceReward, this._cullFor(state)];
    },

    // Retire cars more than 5% of the way behind the champion's last run (sim.c
    // cfg_cull). Only where it is safe and unseen: the track is solved
    // (Optimize), nothing is drawn (Hyper), the champion really is car 0 (at
    // least one elite), and the step runs in wasm (the GPU step has no such rule).
    // Measured over 25 tracks x 3 seeds: 40% fewer car-steps per Optimize
    // generation, best laps unchanged (geometric mean 0.997 of without).
    CULL_BEHIND: 0.95,
    _cullFor: function(state) {
        return state.hyperMode && state.profile === 'optimize' && state.eliteClones >= 1 && this.mode !== 'gpu' ? this.CULL_BEHIND : 0;
    },
    // The champion's gate times: _champProfile as car 0 reports them during a
    // generation, _refProfile the finished one every worker is judging by.
    // _refDirty: the settings changed mid-generation, so this one is not kept.
    _champProfile: null, _refProfile: null, _refDirty: false,

    // The scoring changed (profile, lap target), so the stored all-time-best
    // score no longer compares with what a generation can earn. Forget the
    // number, keep the brain: it still runs as car 0, and the next generation's
    // best — at least as good as it, under the new scoring — takes over.
    rescore: function() { this._globalBestFitness = -Infinity; },

    pushConfig: function(state) {
        const args = this._configArgs(state);
        // Anything but the cull switch itself changes what a run is worth as a reference.
        const clearRef = !this._lastConfig || args.slice(0, 10).some((v, i) => v !== this._lastConfig[i]);
        if (clearRef) { this._refDirty = true; this._refProfile = null; }
        if (this.master) { this.master.ex.set_config(...args); this._pushMutation(state); }
        this._allWorkers().forEach(w => w.postMessage({ type: 'config', args, clearRef }));
        this._lastConfig = args;
        if (this._lastTrackMsg) this._lastTrackMsg.config = args;
    },

    // Hand every worker the raw track definition and let it rebuild the geometry
    // in its own instance. The generator is deterministic, so all of them land
    // on byte-identical walls — and the message stays a few hundred bytes
    // instead of the whole wall list.
    setTrack: function(track, state) {
        const def = {
            path: track.path,
            width: track.trackWidth,
            startPos: track.startPos,
            startAngle: track.startAngle,
            zones: track.zones || [],
            autoWidth: !!track.autoWidth,
            autoWidthBlend: track.autoWidthBlend || 0
        };
        const config = this._configArgs(state);
        if (this.master) { this.master.ex.set_config(...config); this._pushMutation(state); }
        this._allWorkers().forEach(w => w.postMessage({ type: 'track', def, config }));
        this._lastTrackMsg = { def, config };
        this._lastConfig = config;
    },

    // ---- population ----------------------------------------------------
    // Partition the population across workers and give each its slice. Slices
    // are contiguous so a render row's car id is just `sliceStart + i`.
    initPopulation: function(popSize, hiddenLayers, loadedBrainJSON) {
        const ex = this.master.ex;
        this._popSize = Math.min(popSize, ex.max_cars());
        this._hidden = hiddenLayers;
        this._hasGlobalBest = false;
        this._globalBestFitness = -Infinity;
        this._champProfile = null; this._refProfile = null; this._refDirty = false;
        this._lastGateRatio = null;
        this._popEpoch++;
        // _rate is deliberately NOT cleared here: it describes the machine,
        // not the population, and a Reset would otherwise throw away the one
        // measurement that takes several generations to settle.

        ex.pop_init(this._popSize, 0, hiddenLayers, (Math.random() * 0xffffffff) >>> 0);

        if (loadedBrainJSON) {
            // Park the imported brain in the stash, then seed the field from it:
            // car 0 verbatim, everyone else a mutated copy.
            this.writeBrainJSON(ex.stash_slot(), loadedBrainJSON);
            ex.seed_from_stash();
            ex.copy_brain(0, ex.stash_slot());
            this._hasGlobalBest = true;
        } else {
            ex.pop_randomize_brains();
        }

        this._sliceUp();
        this._shipBrains();
    },

    // Measured throughput per worker, in car-steps per millisecond, as a
    // rolling average. Null until a worker has reported at least one round.
    _rate: [],
    // Smoothing. A single round is noisy — a slice whose cars all crashed in
    // the first few frames finishes in no time — so each reading only moves
    // the estimate a fifth of the way. Rebalancing therefore follows a
    // sustained difference between cores and ignores one unlucky generation.
    _RATE_ALPHA: 0.2,

    _noteWorkerTiming: function(data) {
        // Too small a sample says more about postMessage than about the core.
        if (!(data.busyMs > 0.5) || !(data.carSteps > 0)) return;
        const r = data.carSteps / data.busyMs;
        const prev = this._rate[data.index];
        this._rate[data.index] = prev > 0 ? prev + (r - prev) * this._RATE_ALPHA : r;
    },

    // Partition the population across workers.
    //
    // Sized by measured speed rather than equally. Every round ends at a
    // barrier — nothing can be drawn or bred until the LAST worker reports —
    // so equal slices on unequal cores means everyone waits for the slowest,
    // and a phone's little cores are a third the speed of its big ones. Giving
    // each worker a share proportional to how fast it has actually been makes
    // them finish together, which is the only thing the barrier cares about.
    // Before any timings exist this is an even split.
    //
    // And dealt, not cut into blocks. evolve() puts the elites — the cars that
    // drive longest — at the front, so contiguous blocks left worker 0 running
    // long after the others had no cars left: 57-74% core use on the harder
    // tracks in tools measurement, 74-81% dealt. Each car goes to whichever
    // worker is furthest below its speed-weighted share, which interleaves the
    // field and still hands faster cores proportionally more of it.
    _sliceUp: function() {
        const n = this.workers.length || 1;
        const pop = this._popSize;
        const weights = new Array(n);
        let total = 0;
        for (let i = 0; i < n; i++) {
            const r = this._rate[i];
            weights[i] = r > 0 ? r : 0;
            total += weights[i];
        }
        // Any worker that has not reported yet, or a pool with no timings at
        // all, falls back to an even share for everyone — mixing measured and
        // assumed weights would hand the unmeasured ones whatever was left.
        if (!(total > 0) || weights.some(w => w === 0)) weights.fill(1);

        const ids = Array.from({ length: n }, () => []);
        for (let k = 0; k < pop; k++) {
            let best = 0, bestV = Infinity;
            for (let i = 0; i < n; i++) {
                const v = (ids[i].length + 1) / weights[i];
                if (v < bestV) { bestV = v; best = i; }
            }
            ids[best].push(k);
        }
        this.slices = ids.map(a => ({ ids: Int32Array.from(a), count: a.length, start: a.length ? a[0] : 0 }));
    },

    // Rows `ids` of a flat array of `width`-sized records, packed in order.
    _gather: function(src, ids, width) {
        const out = new src.constructor(ids.length * width);
        for (let j = 0; j < ids.length; j++) out.set(src.subarray(ids[j] * width, (ids[j] + 1) * width), j * width);
        return out;
    },

    // Copy each worker's slice of the brain array out of master memory and send
    // it. ~600KB at the largest settings, once per generation — next to the
    // per-frame structured clone of 200 brain objects the JS edition did, this
    // barely registers.
    _shipBrains: function() {
        const ex = this.master.ex;
        const stride = ex.brain_stride();
        const all = this._f32(this.master, ex.brains_ptr(), (ex.max_cars() + 1) * stride);
        // Which slots evolve() bred as Focus % mutants (shown on screen).
        const allFocused = this._i32(this.master, ex.car_focused_ptr(), ex.max_cars());
        this.workers.forEach((w, i) => {
            const s = this.slices[i];
            const blob = this._gather(all, s.ids, stride);
            const focused = this._gather(allFocused, s.ids, 1);
            w.postMessage({
                type: 'pop',
                start: s.start, count: s.count,
                hidden: this._hidden,
                seed: (Math.random() * 0xffffffff) >>> 0,
                brains: blob, focused, ref: this._refProfile
            }, [blob.buffer, focused.buffer]);
        });
    },

    // ---- running -------------------------------------------------------
    // `wantRender` is false in hyper mode: nothing is drawn, so the workers skip
    // packing render rows and send back three floats per car instead of
    // eighteen. Combined with the much larger iteration chunk hyper mode uses,
    // that is most of why hyper mode pulls away from the JS edition.
    // Buffers handed back by the workers on the previous round, returned to
    // them here to be filled again. A transfer detaches the buffer from
    // whichever side sent it, so ping-ponging the same allocation between the
    // two is what keeps this from producing a fresh ~36KB of garbage on every
    // step. Safe at this point by construction: the caller consumed the
    // previous results before asking for the next ones.
    _spare: [],

    // `gen` only labels the round for the benchmark.
    run: function(iters, wantRender, gen) {
        // A pool reshape is swapping workers; the round starts on the new ones.
        if (this._gate) return this._gate.promise.then(() => this.run(iters, wantRender, gen));
        return new Promise(resolve => {
            this._runState = {
                completed: 0, total: this.workers.length,
                maxLaps: 0, allCrashed: true, wantRender, resolve,
                rows: [], t0: performance.now(), gen: gen | 0
            };
            for (let i = 0; i < this.workers.length; i++) {
                const spare = this._spare[i];
                this._spare[i] = null;
                const msg = { type: 'run', iters, wantRender, snapshot: this.workers.length > 1 };
                if (spare) { msg.recycle = spare; this.workers[i].postMessage(msg, [spare]); }
                else this.workers[i].postMessage(msg);
            }
        });
    },

    // A generation ends at the step the first car anywhere reaches the target
    // lap count, but each worker only stops for its OWN cars. In hyper mode a
    // round is 2500 steps, so without this every other slice drove on for up
    // to 2499 frames past the real finish, banking score no car in the
    // finisher's slice had time to earn — enough to take the stash with an
    // inflated all-time best that fair runs then could never beat, and a
    // result that depended on how many cores the population was split over.
    // Any slice that ran past the earliest finish is rewound to the round's
    // start (sim-worker.js keeps it) and replayed to exactly that step; the
    // simulation is deterministic, so the replay is exact. Returns true while
    // replays are in flight.
    _alignFinish: function(st) {
        const target = this._lastConfig ? this._lastConfig[5] : Infinity;
        let k = Infinity;
        for (const r of st.rows) if (r.maxLaps >= target && r.iters < k) k = r.iters;
        if (!(k < Infinity)) return false;
        const over = st.rows.filter(r => typeof r.iters === 'number' && r.iters > k);
        if (!over.length) return false;
        st.rows = st.rows.filter(r => !over.includes(r));
        st.completed -= over.length;
        for (const r of over) {
            const buf = r.buffer.buffer;
            this.workers[r.index].postMessage({ type: 'rerun', iters: k, wantRender: st.wantRender, recycle: buf }, [buf]);
        }
        return true;
    },

    // Hand a consumed buffer back to the worker that produced it.
    recycle: function(index, buffer) {
        if (buffer && buffer.byteLength) this._spare[index] = buffer;
    },

    _handleWorkerMessage: function(data) {
        if (data.type === 'exported') { if (this._exportSink) this._exportSink(data); return; }
        if (data.type === 'gpu-lost') { this._onGpuLost(data.message); return; }
        const st = this._runState;
        if (!st || data.type !== 'done') return;
        this._noteWorkerTiming(data);
        if (data.gateRatio) this._lastGateRatio = data.gateRatio;
        if (data.profile) this._champProfile = data.profile;
        // Row i of this worker's results is global car ids[i].
        data.ids = this.slices[data.index] ? this.slices[data.index].ids : null;
        st.rows.push(data);
        if (++st.completed === st.total) {
            if (this._alignFinish(st)) return;
            // Recomputed from the final rows: a replayed slice replaces its
            // overshooting report.
            st.maxLaps = 0; st.allCrashed = true;
            for (const r of st.rows) {
                if (r.maxLaps > st.maxLaps) st.maxLaps = r.maxLaps;
                if (!r.allCrashed) st.allCrashed = false;
            }
            this._runState = null;
            this._noteThroughput(st);
            st.resolve(st);
            const waiters = this._idleWaiters;
            this._idleWaiters = [];
            for (const f of waiters) f();
        }
    },

    // ---- evolution -----------------------------------------------------
    // Selection is global, so it happens here on the master, which is the one
    // instance holding every brain. Workers never breed.
    // Worker 0's per-gate pace telemetry for car 0 (sim.c gate_ratio), carried
    // across a pool change so a migrated race keeps it.
    _lastGateRatio: null,

    // sigmaGen is computed by the caller (app.evolve in script.js): it needs
    // the whole run's history, and this object only sees one generation.
    evolve: function(fitness, eliteClones, sigmaGen) {
        const ex = this.master.ex;
        const ev = this._f32(this.master, ex.ev_fitness_ptr(), ex.max_cars());
        let bestIdx = 0, bestFit = -Infinity;
        for (let i = 0; i < this._popSize; i++) {
            const f = fitness[i] || 0;
            ev[i] = f;
            if (f > bestFit) { bestFit = f; bestIdx = i; }
        }
        // The all-time best is kept in the stash slot and cloned forward, which
        // is what makes elite clones actually prevent regression.
        if (!this._hasGlobalBest || bestFit > this._globalBestFitness) {
            ex.copy_brain(bestIdx, ex.stash_slot());
            this._globalBestFitness = bestFit;
            this._hasGlobalBest = true;
        }
        ex.evolve(eliteClones, this._hasGlobalBest ? 1 : 0, sigmaGen | 0);
        this._refProfile = this._refDirty ? null : this._champProfile;
        this._champProfile = null; this._refDirty = false;
        this._popEpoch++;
        // A generation boundary is the one moment the partition can move for
        // free: every worker is idle and about to be handed a fresh slice
        // anyway, so re-balancing them here costs nothing beyond arithmetic.
        this._sliceUp();
        this._shipBrains();
        return { bestFitness: bestFit, globalBest: this._globalBestFitness };
    },

    resetPopulation: function() {
        this._popEpoch++;
        this.workers.forEach(w => w.postMessage({ type: 'reset' }));
    },

    // ---- brain import / export ----------------------------------------
    // The JSON shape is the nested-array format the original used, so a brain
    // saved from the JS edition loads here and vice versa.
    brainToJSON: function(slot) {
        const ex = this.master.ex;
        const h = this._hidden, iC = BRAIN_INPUTS, oC = 2;
        const stride = ex.brain_stride();
        const b = this._f32(this.master, ex.brains_ptr() + slot * stride * 4, stride);
        const rows = (off, r, c) => {
            const out = new Array(r);
            for (let i = 0; i < r; i++) {
                const row = new Array(c);
                for (let j = 0; j < c; j++) row[j] = b[off + i * c + j];
                out[i] = row;
            }
            return out;
        };
        return {
            weightsIH: rows(0, iC, h),
            weightsHO: rows(iC * h, h, oC),
            biasH: Array.from(b.subarray(iC * h + h * oC, iC * h + h * oC + h)),
            biasO: Array.from(b.subarray(iC * h + h * oC + h, stride))
        };
    },

    writeBrainJSON: function(slot, j) {
        const ex = this.master.ex;
        const stride = ex.brain_stride();
        const h = this._hidden, iC = BRAIN_INPUTS, oC = 2;
        const b = this._f32(this.master, ex.brains_ptr() + slot * stride * 4, stride);
        // A file written before the network gained its two recurrent inputs
        // has 9 rows, not 11. The missing rows are zeroed rather than
        // rejected, which is exactly neutral: a zero weight means the brain
        // ignores the new input and drives as it always did, and evolution is
        // free to find a use for it from there.
        for (let i = 0; i < iC; i++) {
            const row = j.weightsIH[i];
            for (let k = 0; k < h; k++) b[i * h + k] = row ? row[k] : 0;
        }
        const off = iC * h;
        for (let i = 0; i < h; i++) for (let k = 0; k < oC; k++) b[off + i * oC + k] = j.weightsHO[i][k];
        const bh = off + h * oC;
        for (let i = 0; i < h; i++) b[bh + i] = j.biasH[i];
        for (let i = 0; i < oC; i++) b[bh + h + i] = j.biasO[i];
    },

    // Validate a brain file completely before a single number is written.
    // Writes go through a Float32Array view sized to the brain stride, and an
    // out-of-range index on a typed array is silently DISCARDED rather than
    // throwing — so an oversized file would not crash, it would just quietly
    // produce a brain full of zeros. Everything gets checked up front instead.
    validateBrainJSON: function(j) {
        if (!j || !Array.isArray(j.weightsIH) || !Array.isArray(j.weightsHO)
            || !Array.isArray(j.biasH) || !Array.isArray(j.biasO)) return 'not a TrackML brain file';
        if (j.weightsIH.length !== BRAIN_INPUTS && j.weightsIH.length !== LEGACY_BRAIN_INPUTS) {
            return `expected ${BRAIN_INPUTS} input rows (or ${LEGACY_BRAIN_INPUTS} from an older build), found ${j.weightsIH.length}`;
        }
        const h = j.biasH.length;
        if (j.weightsHO.length !== h) return 'hidden layer size is inconsistent within the file';
        if (j.biasO.length !== 2) return `expected 2 outputs, found ${j.biasO.length}`;
        const maxH = this.master ? this.master.ex.max_hidden() : 32;
        if (!(h >= 1 && h <= maxH)) return `hidden layer of ${h} is outside the supported range of 1-${maxH}`;
        for (const row of j.weightsIH) if (!Array.isArray(row) || row.length !== h) return 'ragged weightsIH';
        for (const row of j.weightsHO) if (!Array.isArray(row) || row.length !== 2) return 'ragged weightsHO';
        const finite = v => typeof v === 'number' && Number.isFinite(v);
        for (const row of j.weightsIH) if (!row.every(finite)) return 'weightsIH contains a non-finite value';
        for (const row of j.weightsHO) if (!row.every(finite)) return 'weightsHO contains a non-finite value';
        if (!j.biasH.every(finite)) return 'biasH contains a non-finite value';
        if (!j.biasO.every(finite)) return 'biasO contains a non-finite value';
        return null;
    },

    hiddenOf: function(j) { return j.biasH.length; },

    // ---- whole-population export / import ------------------------------
    // Saving one brain loses the run: reloading it re-seeds everybody from
    // mutated copies of that single network, throwing away all the diversity
    // the population had built up. The brains are already one contiguous
    // Float32Array, so shipping the lot costs nothing but the bytes.
    exportPopulation: function() {
        const ex = this.master.ex;
        const stride = ex.brain_stride();
        const live = this._f32(this.master, ex.brains_ptr(), this._popSize * stride).slice();
        const stash = this._f32(this.master, ex.brains_ptr() + ex.stash_slot() * stride * 4, stride).slice();
        return {
            popSize: this._popSize,
            hidden: this._hidden,
            inputs: BRAIN_INPUTS,
            stride,
            hasGlobalBest: this._hasGlobalBest,
            globalBestFitness: Number.isFinite(this._globalBestFitness) ? this._globalBestFitness : null,
            // Base64 rather than a JSON array of numbers: ~4x smaller and it
            // round-trips exactly, where decimal text does not.
            brains: bytesToBase64(new Uint8Array(live.buffer)),
            stashBrain: bytesToBase64(new Uint8Array(stash.buffer))
        };
    },

    // Returns null on success, or a human-readable reason it was rejected.
    importPopulation: function(p) {
        if (!p || typeof p !== 'object') return 'not a population file';
        const ex = this.master.ex;
        const h = p.hidden | 0;
        if (!(h >= 1 && h <= ex.max_hidden())) return `hidden layer of ${p.hidden} is outside 1-${ex.max_hidden()}`;
        if (p.inputs !== undefined && p.inputs !== BRAIN_INPUTS) {
            return `this file was saved by a build whose networks had ${p.inputs} inputs, not ${BRAIN_INPUTS}`;
        }
        const popSize = Math.min(p.popSize | 0, ex.max_cars());
        if (popSize < 1) return 'population size is missing or zero';

        this._popSize = popSize;
        this._hidden = h;
        ex.pop_init(popSize, 0, h, (Math.random() * 0xffffffff) >>> 0);
        const stride = ex.brain_stride();
        if ((p.stride | 0) !== stride) return `brain layout mismatch (file ${p.stride}, this build ${stride})`;

        let live, stash;
        try {
            live = new Float32Array(base64ToBytes(p.brains).buffer);
            stash = new Float32Array(base64ToBytes(p.stashBrain).buffer);
        } catch (e) { return 'the brain data is corrupt'; }
        if (live.length < popSize * stride) return 'the brain data is shorter than the population it claims';

        this._f32(this.master, ex.brains_ptr(), popSize * stride).set(live.subarray(0, popSize * stride));
        if (stash.length >= stride) {
            this._f32(this.master, ex.brains_ptr() + ex.stash_slot() * stride * 4, stride).set(stash.subarray(0, stride));
        }
        this._hasGlobalBest = !!p.hasGlobalBest;
        this._globalBestFitness = typeof p.globalBestFitness === 'number' ? p.globalBestFitness : -Infinity;
        this._lastGateRatio = null;

        this._sliceUp();
        this._shipBrains();
        return null;
    }
};

// Base64 helpers for the population blob. Done in chunks because spreading a
// megabyte-sized array into String.fromCharCode overflows the call stack.
function bytesToBase64(bytes) {
    let out = '';
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
        out += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(out);
}
function base64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

// Network input count — seven sensors, speed, bearing to the next gate, and
// the previous frame's two outputs. Must track IN_N in sim.c.
const BRAIN_INPUTS = 11;
// What that number was before the recurrent inputs existed. Brain files saved
// by an older build have this many rows and are still loadable.
const LEGACY_BRAIN_INPUTS = 9;

const ZONE_TYPE_ID = { speed: 0, precision: 1, focus: 2, spawnkill: 3 };

// Floats per checkpoint in wasm memory — must track the Checkpoint struct in
// sim.c, which carries an apex flag after the six coordinates.
const CP_STRIDE = 7;

// The track object the rest of the app sees.
//
// Geometry comes back as flat typed arrays — that is what the canvas wants and
// what keeps the editor's per-frame rebuild from allocating thousands of little
// {p1:{x,y}} objects. `walls` and `checkpoints` are still here as lazy getters
// for anything that wants the old object shape; nothing on a hot path does.
function makeTrack(id, name, path, trackWidth, startPos, startAngle, zones,
                   centerF32, wallsF32, wallSeg, cpF32, cpApex, widthF32, auto, startCp) {
    const t = {
        id, name, path, trackWidth, startPos, startAngle, zones,
        centerF32, wallsF32, wallSeg, cpF32, cpApex, widthF32,
        autoWidth: !!(auto && auto.enabled),
        autoWidthBlend: (auto && typeof auto.blend === 'number') ? auto.blend : 0,
        wallCount: wallsF32.length / 4,
        cpCount: cpF32.length / CP_STRIDE,
        // The gate whose crossing actually completes a lap (see sim.c's
        // updateCar) — not necessarily checkpoint 0, on a track whose start
        // has been dragged elsewhere. This is which one to draw as the
        // finish line.
        startCp: startCp || 0,
        segStep: 34
    };
    let _walls = null, _cps = null;
    Object.defineProperty(t, 'walls', {
        enumerable: false,
        get() {
            if (_walls) return _walls;
            _walls = [];
            for (let i = 0; i < t.wallCount; i++) {
                _walls.push({
                    p1: { x: wallsF32[i * 4], y: wallsF32[i * 4 + 1] },
                    p2: { x: wallsF32[i * 4 + 2], y: wallsF32[i * 4 + 3] },
                    segmentIndex: wallSeg[i] < 0 ? undefined : wallSeg[i]
                });
            }
            return _walls;
        }
    });
    Object.defineProperty(t, 'checkpoints', {
        enumerable: false,
        get() {
            if (_cps) return _cps;
            _cps = [];
            for (let i = 0; i < t.cpCount; i++) {
                _cps.push({
                    index: i,
                    p1: { x: cpF32[i * CP_STRIDE], y: cpF32[i * CP_STRIDE + 1] },
                    p2: { x: cpF32[i * CP_STRIDE + 2], y: cpF32[i * CP_STRIDE + 3] },
                    center: { x: cpF32[i * CP_STRIDE + 4], y: cpF32[i * CP_STRIDE + 5] },
                    apex: !!cpApex[i]
                });
            }
            return _cps;
        }
    });
    return t;
}

// The name the rest of the app (and every published track in tracks.js) calls.
// `auto` is optional and defaults to off, so every existing call — including
// every track in tracks.js — keeps its constant width.
function generateTrackFromPath(id, name, pathInput, width, customStartPos, customStartAngle, zones = [], auto = null) {
    return Engine.buildTrack(id, name, pathInput, width, customStartPos, customStartAngle, zones, auto);
}
