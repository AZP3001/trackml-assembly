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

            this.coreCount = navigator.hardwareConcurrency || 4;
            this._pendingCoreLabel = `<svg class="w-3 h-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20v2"></path><path d="M12 2v2"></path><path d="M17 20v2"></path><path d="M17 2v2"></path><path d="M2 12h2"></path><path d="M2 17h2"></path><path d="M2 7h2"></path><path d="M20 12h2"></path><path d="M20 17h2"></path><path d="M20 7h2"></path><path d="M7 20v2"></path><path d="M7 2v2"></path><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="8" y="8" width="8" height="8" rx="1"></rect></svg> ${this.coreCount} Cores · WASM${this.usingSimd ? '+SIMD' : ''}`;

            await this._spawnWorkers();
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

    _spawnWorkers: function() {
        return Promise.all(Array.from({ length: this.coreCount }, (_, i) => new Promise((resolve, reject) => {
            const w = new Worker(`./sim-worker.js?v=${ASSET_VERSION}`);
            w.onerror = e => { console.error('Worker error:', e.message); reject(new Error(e.message)); };
            w.onmessage = e => {
                if (e.data.type === 'ready') { resolve(w); return; }
                this._handleWorkerMessage(e.data);
            };
            w.postMessage({ type: 'init', module: this.module, index: i });
            this.workers.push(w);
        })));
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
            widthF32, { enabled: !!autoOn, blend: autoBlend });
    },

    // ---- config --------------------------------------------------------
    // Sensor reach for the current Max Speed, straight from the module that
    // raycasts with it — so the overlay the UI draws can never disagree with
    // what the cars actually saw.
    sensorLength: function() {
        return this.master ? this.master.ex.sensor_len() : 180;
    },

    pushConfig: function(state) {
        const p = state.physics;
        const args = [p.maxSpeed, p.acceleration, p.turnSpeed, p.brakeStrength,
                      state.initialTTL, state.targetLaps, state.mutationRate, state.hiddenLayers];
        if (this.master) this.master.ex.set_config(...args);
        this.workers.forEach(w => w.postMessage({ type: 'config', args }));
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
                        state.initialTTL, state.targetLaps, state.mutationRate, state.hiddenLayers];
        if (this.master) this.master.ex.set_config(...config);
        this.workers.forEach(w => w.postMessage({ type: 'track', def, config }));
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

    _sliceUp: function() {
        const n = this.workers.length || 1;
        const per = Math.ceil(this._popSize / n);
        this.slices = [];
        for (let i = 0; i < n; i++) {
            const start = Math.min(i * per, this._popSize);
            const count = Math.min(per, this._popSize - start);
            this.slices.push({ start, count });
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
        this.workers.forEach((w, i) => {
            const s = this.slices[i];
            const blob = all.slice(s.start * stride, (s.start + s.count) * stride);
            w.postMessage({
                type: 'pop',
                start: s.start, count: s.count,
                hidden: this._hidden,
                seed: (Math.random() * 0xffffffff) >>> 0,
                brains: blob
            }, [blob.buffer]);
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
        return new Promise(resolve => {
            this._runState = {
                completed: 0, total: this.workers.length,
                maxLaps: 0, allCrashed: true, wantRender, resolve,
                rows: []
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
        const st = this._runState;
        if (!st || data.type !== 'done') return;
        if (data.maxLaps > st.maxLaps) st.maxLaps = data.maxLaps;
        if (!data.allCrashed) st.allCrashed = false;
        st.rows.push(data);
        if (++st.completed === st.total) {
            this._runState = null;
            st.resolve(st);
        }
    },

    // ---- evolution -----------------------------------------------------
    // Selection is global, so it happens here on the master, which is the one
    // instance holding every brain. Workers never breed.
    evolve: function(fitness, eliteClones, generation) {
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
        ex.evolve(eliteClones, this._hasGlobalBest ? 1 : 0, generation | 0);
        this._shipBrains();
        return { bestFitness: bestFit, globalBest: this._globalBestFitness };
    },

    resetPopulation: function() {
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
                   centerF32, wallsF32, wallSeg, cpF32, cpApex, widthF32, auto) {
    const t = {
        id, name, path, trackWidth, startPos, startAngle, zones,
        centerF32, wallsF32, wallSeg, cpF32, cpApex, widthF32,
        autoWidth: !!(auto && auto.enabled),
        autoWidthBlend: (auto && typeof auto.blend === 'number') ? auto.blend : 0,
        wallCount: wallsF32.length / 4,
        cpCount: cpF32.length / CP_STRIDE,
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
