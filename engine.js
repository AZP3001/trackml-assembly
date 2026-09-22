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

// How many workers to spawn.
//
// hardwareConcurrency is a count of logical cores, and on a phone that count
// is a lie about what they are worth: an eight-core phone is typically four
// fast cores and four slow ones, and the two kinds can differ by a factor of
// three. Every round here ends at a barrier — the main thread cannot draw the
// frame until the LAST worker reports — so a slice handed to a slow core sets
// the pace for all of them, and adding those cores makes the whole thing
// slower, not faster. On top of that each worker carries its own wasm instance,
// and eight of those on a phone is memory the tab does not have to spare and
// heat it cannot shed.
//
// So on a phone or tablet: roughly the fast half, and never more than four.
// On a desktop, unchanged — every core, as before.
function workerCountFor(cores, isMobile) {
    const n = Math.max(1, cores | 0);
    if (!isMobile) return n;
    return Math.max(1, Math.min(4, Math.min(n, Math.ceil(n / 2))));
}
// script.js decides what counts as a phone; engine.js is loaded first, so it
// asks rather than deciding again, and falls back to "desktop" on its own.
function engineIsMobile() {
    return typeof IS_MOBILE !== 'undefined' ? !!IS_MOBILE : false;
}

const CORE_ICON_SVG = '<svg class="w-3 h-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20v2"></path><path d="M12 2v2"></path><path d="M17 20v2"></path><path d="M17 2v2"></path><path d="M2 12h2"></path><path d="M2 17h2"></path><path d="M2 7h2"></path><path d="M20 12h2"></path><path d="M20 17h2"></path><path d="M20 7h2"></path><path d="M7 20v2"></path><path d="M7 2v2"></path><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="8" y="8" width="8" height="8" rx="1"></rect></svg>';

const Engine = {
    module: null,        // compiled WebAssembly.Module, cloned out to workers
    master: null,        // { exports, memory } on the main thread
    workers: [],
    coreCount: 1,
    slices: [],          // { start, count } per worker
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
            this.coreCount = workerCountFor(this.hardwareCores, engineIsMobile());
            this._target = { mode: 'cpu', cores: this.coreCount };
            this._crashBase = new Int32Array(this.master.ex.max_gates());

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
    // Crash tallies from workers that have since been retired (or had their
    // cars migrated), folded in so evolve() still sees the whole generation.
    _crashBase: null,
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
                for (const e of exported) {
                    for (let g = 0; g < this._crashBase.length; g++) this._crashBase[g] += e.crashCount[g];
                    if (e.gateRatio) this._lastGateRatio = e.gateRatio;
                }
            }
            this._crashCountByWorker = [];
            if (this._popSize > 0) {
                this._sliceUp();
                if (fresh) this._shipBrains(); else this._importState(exported);
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
    // whatever the old pool exported, reassembled in car-id order.
    _importState: function(exported) {
        const ex = this.master.ex;
        const W = ex.state_words();
        const all = new Uint32Array(this._popSize * W);
        for (const e of exported) if (e.count > 0) all.set(e.state.subarray(0, e.count * W), e.start * W);
        const stride = ex.brain_stride();
        const brains = this._f32(this.master, ex.brains_ptr(), (ex.max_cars() + 1) * stride);
        const focusLo = ex.focus_lo(), focusHi = ex.focus_hi();
        this.workers.forEach((w, i) => {
            const s = this.slices[i];
            const b = brains.slice(s.start * stride, (s.start + s.count) * stride);
            const state = all.slice(s.start * W, (s.start + s.count) * W);
            const gateRatio = s.start === 0 && this._lastGateRatio ? this._lastGateRatio.slice() : null;
            const transfer = [b.buffer, state.buffer];
            if (gateRatio) transfer.push(gateRatio.buffer);
            w.postMessage({
                type: 'import', start: s.start, count: s.count, hidden: this._hidden,
                seed: (Math.random() * 0xffffffff) >>> 0, brains: b, focusLo, focusHi, state, gateRatio
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

    pushConfig: function(state) {
        const p = state.physics;
        const args = [p.maxSpeed, p.acceleration, p.turnSpeed, p.brakeStrength,
                      state.initialTTL, state.targetLaps, state.focusPct, state.hiddenLayers,
                      state.nudgeMode ? 1 : 0];
        if (this.master) this.master.ex.set_config(...args);
        this._allWorkers().forEach(w => w.postMessage({ type: 'config', args }));
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
        const p = state.physics;
        const config = [p.maxSpeed, p.acceleration, p.turnSpeed, p.brakeStrength,
                        state.initialTTL, state.targetLaps, state.focusPct, state.hiddenLayers,
                        state.nudgeMode ? 1 : 0];
        if (this.master) this.master.ex.set_config(...config);
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
        this._lastGateRatio = null;
        this._crashCountByWorker = [];
        this._crashBase.fill(0);
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

    // Partition the population across workers. Slices stay contiguous so a
    // render row's car id is just `sliceStart + i`.
    //
    // Sized by measured speed rather than equally. Every round ends at a
    // barrier — nothing can be drawn or bred until the LAST worker reports —
    // so equal slices on unequal cores means everyone waits for the slowest,
    // and a phone's little cores are a third the speed of its big ones. Giving
    // each worker a share proportional to how fast it has actually been makes
    // them finish together, which is the only thing the barrier cares about.
    // Before any timings exist this is exactly the old equal split.
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
        if (!(total > 0) || weights.some(w => w === 0)) {
            for (let i = 0; i < n; i++) weights[i] = 1;
            total = n;
        }

        this.slices = [];
        let start = 0;
        for (let i = 0; i < n; i++) {
            // The last worker takes the remainder, so rounding can never lose
            // or duplicate a car.
            const count = i === n - 1
                ? pop - start
                : Math.min(pop - start, Math.max(0, Math.round((pop * weights[i]) / total)));
            this.slices.push({ start, count: Math.max(0, count) });
            start += count;
        }
    },

    // Copy each worker's slice of the brain array out of master memory and send
    // it. ~600KB at the largest settings, once per generation — next to the
    // per-frame structured clone of 200 brain objects the JS edition did, this
    // barely registers.
    _shipBrains: function() {
        const ex = this.master.ex;
        const stride = ex.brain_stride();
        const all = this._f32(this.master, ex.brains_ptr(), (ex.max_cars() + 1) * stride);
        // Which slots evolve() bred as focused clones, and the track window
        // their reward is boosted inside — both computed on the master (the
        // only instance holding the whole population and the only one that
        // calls evolve()) and shipped out alongside the brains, exactly like
        // config. Workers never breed, so they have no other way to know.
        const allFocused = this._i32(this.master, ex.car_focused_ptr(), ex.max_cars());
        const focusLo = ex.focus_lo(), focusHi = ex.focus_hi();
        this.workers.forEach((w, i) => {
            const s = this.slices[i];
            const blob = all.slice(s.start * stride, (s.start + s.count) * stride);
            const focused = allFocused.slice(s.start, s.start + s.count);
            w.postMessage({
                type: 'pop',
                start: s.start, count: s.count,
                hidden: this._hidden,
                seed: (Math.random() * 0xffffffff) >>> 0,
                brains: blob, focused, focusLo, focusHi
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

    run: function(iters, wantRender) {
        // A pool reshape is swapping workers; the round starts on the new ones.
        if (this._gate) return this._gate.promise.then(() => this.run(iters, wantRender));
        return new Promise(resolve => {
            this._runState = {
                completed: 0, total: this.workers.length,
                maxLaps: 0, allCrashed: true, wantRender, resolve,
                rows: [], t0: performance.now()
            };
            for (let i = 0; i < this.workers.length; i++) {
                const spare = this._spare[i];
                this._spare[i] = null;
                const msg = { type: 'run', iters, wantRender };
                if (spare) { msg.recycle = spare; this.workers[i].postMessage(msg, [spare]); }
                else this.workers[i].postMessage(msg);
            }
        });
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
        if (data.maxLaps > st.maxLaps) st.maxLaps = data.maxLaps;
        if (!data.allCrashed) st.allCrashed = false;
        if (data.gateRatio) this._lastGateRatio = data.gateRatio;
        // Overwritten, not accumulated: each worker's array is already the
        // running total of ITS crashes so far this generation (sim.c only
        // resets it once per generation, at pop_reset), so the latest message
        // from a given worker is already its correct cumulative count. evolve()
        // sums across the per-worker entries once, at generation boundary.
        if (data.crashCount) this._crashCountByWorker[data.index] = data.crashCount;
        st.rows.push(data);
        if (++st.completed === st.total) {
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
    // Set from worker 0's last 'done' message (see _handleWorkerMessage) —
    // that worker owns global car 0, and car 0 mirrors the stash exactly
    // whenever one exists, so its per-gate pace this generation is exactly
    // the telemetry evolve() needs to find the stash's weakest stretch. The
    // master itself never simulates a car, so it has no other way to see it.
    _lastGateRatio: null,
    // One entry per worker, each that worker's own crash_count array (see
    // sim.c) as of its last report. Unlike gate_ratio this is population-wide
    // — every worker's every car, not one worker's car 0 — so evolve() sums
    // ACROSS entries rather than taking the newest one.
    _crashCountByWorker: [],

    // sigmaGen and lapCompletions are computed by the caller (app.evolve in
    // script.js), not here: they both need the whole run's history (has a lap
    // EVER completed, how many separate generations has it happened in), and
    // this object only ever sees one generation at a time. See the comment on
    // MUT_SIGMA_* and FEW_LAPS_THRESHOLD in sim.c for what each one controls.
    evolve: function(fitness, eliteClones, sigmaGen, lapCompletions) {
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
        if (this._lastGateRatio) {
            this._f32(this.master, ex.gate_ratio_ptr(), ex.max_gates()).set(this._lastGateRatio);
        }
        {
            const maxGates = ex.max_gates();
            const sum = this._i32(this.master, ex.crash_count_ptr(), maxGates);
            sum.set(this._crashBase);
            for (const wc of this._crashCountByWorker) {
                if (!wc) continue;
                for (let g = 0; g < maxGates; g++) sum[g] += wc[g];
            }
        }
        ex.evolve(eliteClones, this._hasGlobalBest ? 1 : 0, sigmaGen | 0, lapCompletions | 0);
        this._crashBase.fill(0);
        this._popEpoch++;
        // A generation boundary is the one moment the partition can move for
        // free: every worker is idle and about to be handed a fresh slice
        // anyway, so re-balancing them here costs nothing beyond arithmetic.
        this._sliceUp();
        this._shipBrains();
        return { bestFitness: bestFit, globalBest: this._globalBestFitness };
    },

    resetPopulation: function() {
        this._crashBase.fill(0);
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
        this._crashCountByWorker = [];

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
