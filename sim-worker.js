// sim-worker.js — hosts one wasm instance and simulates one slice of the
// population. It holds no simulation logic of its own; every message below is
// a thin shim onto an export of sim.c.
//
// The worker is a real file rather than a Blob URL so that it keeps the page's
// origin: a Blob worker gets an opaque one, and that makes relative fetches
// (and debugging) needlessly awkward. It never fetches the wasm anyway — the
// main thread compiles once and posts the WebAssembly.Module across, which is
// structured-cloneable and already compiled, so eight workers cost one compile.

let ex = null;          // wasm exports
let memory = null;
let index = -1;
let popCount = 0;
let popStart = 0;

// A wasm memory can be detached and replaced when it grows, so views are
// derived per use rather than cached.
const f32 = (ptr, len) => new Float32Array(memory.buffer, ptr, len);
const i32 = (ptr, len) => new Int32Array(memory.buffer, ptr, len);

self.onmessage = async (e) => {
    const msg = e.data;

    switch (msg.type) {
        case 'init': {
            index = msg.index;
            const instance = await WebAssembly.instantiate(msg.module, {});
            ex = instance.exports;
            memory = ex.memory;
            self.postMessage({ type: 'ready', index });
            break;
        }

        case 'config':
            ex.set_config(...msg.args);
            break;

        case 'track': {
            // Rebuild the geometry locally from the raw definition. The
            // generator is deterministic, so this lands on exactly the same
            // walls the main thread computed — without shipping any of them.
            const d = msg.def;
            ex.set_config(...msg.config);
            const nPts = Math.min(d.path.length, ex.max_path_pts());
            const pIn = f32(ex.path_in_ptr(), Math.max(1, nPts * 4));
            for (let i = 0; i < nPts; i++) {
                const p = d.path[i];
                pIn[i * 4]     = p.x;
                pIn[i * 4 + 1] = p.y;
                pIn[i * 4 + 2] = p.type === 'corner' ? 1 : 0;
                pIn[i * 4 + 3] = p.radius !== undefined ? p.radius : 60;
            }
            const zones = d.zones || [];
            const nz = Math.min(zones.length, ex.max_zones());
            const zIn = f32(ex.zone_in_ptr(), Math.max(1, nz * 5));
            const ZID = { speed: 0, precision: 1, focus: 2, spawnkill: 3 };
            for (let i = 0; i < nz; i++) {
                const z = zones[i];
                zIn[i * 5]     = z.x;
                zIn[i * 5 + 1] = z.y;
                zIn[i * 5 + 2] = z.radius;
                zIn[i * 5 + 3] = ZID[z.type] !== undefined ? ZID[z.type] : -1;
                zIn[i * 5 + 4] = z.killTimer !== undefined ? z.killTimer : 150;
            }
            const hasStart = d.startPos ? 1 : 0;
            const hasAngle = (d.startAngle !== undefined && d.startAngle !== null) ? 1 : 0;
            ex.track_build(
                ex.path_in_ptr(), nPts, d.width,
                hasStart, hasStart ? d.startPos.x : 0, hasStart ? d.startPos.y : 0,
                hasAngle, hasAngle ? d.startAngle : 0,
                ex.zone_in_ptr(), nz,
                d.autoWidth ? 1 : 0, d.autoWidthBlend || 0);
            break;
        }

        case 'pop': {
            popStart = msg.start;
            popCount = msg.count;
            ex.pop_init(popCount, popStart, msg.hidden, msg.seed);
            const stride = ex.brain_stride();
            f32(ex.brains_ptr(), popCount * stride).set(msg.brains.subarray(0, popCount * stride));
            // Which of this slice's cars evolve() bred as focused clones, and
            // the checkpoint window their reward is boosted inside. Bred and
            // computed on the master; this worker never breeds, so it has no
            // other way to know either one.
            if (msg.focused) i32(ex.car_focused_ptr(), popCount).set(msg.focused);
            ex.set_focus_window(msg.focusLo === undefined ? -1 : msg.focusLo, msg.focusHi === undefined ? -1 : msg.focusHi);
            ex.pop_reset();
            break;
        }

        case 'reset':
            ex.pop_reset();
            break;

        // Pool reshape, outgoing half: hand every car this worker holds back
        // to the master exactly as it is mid-generation (see pack_state in
        // sim.c), plus this slice's crash tally and — for the slice holding
        // global car 0 — its per-gate pace, so neither is lost when the cars
        // move to a different worker.
        case 'export': {
            ex.pack_state();
            const words = ex.state_words();
            const state = new Uint32Array(memory.buffer, ex.state_ptr(), popCount * words).slice();
            const crashCount = i32(ex.crash_count_ptr(), ex.max_gates()).slice();
            const gateRatio = popStart === 0 ? f32(ex.gate_ratio_ptr(), ex.max_gates()).slice() : null;
            const transfer = [state.buffer, crashCount.buffer];
            if (gateRatio) transfer.push(gateRatio.buffer);
            self.postMessage({ type: 'exported', index, start: popStart, count: popCount, state, crashCount, gateRatio }, transfer);
            break;
        }

        // Incoming half: take over a slice mid-generation. Same as 'pop' up to
        // the brains and focus window, then the cars are overwritten with the
        // state they had wherever they were before, instead of being reset to
        // the start line. The crash tally starts from zero here — the master
        // folded every old worker's count into its own running total before
        // handing the cars out.
        case 'import': {
            popStart = msg.start;
            popCount = msg.count;
            ex.pop_init(popCount, popStart, msg.hidden, msg.seed);
            const stride = ex.brain_stride();
            f32(ex.brains_ptr(), popCount * stride).set(msg.brains.subarray(0, popCount * stride));
            ex.set_focus_window(msg.focusLo === undefined ? -1 : msg.focusLo, msg.focusHi === undefined ? -1 : msg.focusHi);
            new Uint32Array(memory.buffer, ex.state_ptr(), popCount * ex.state_words()).set(msg.state);
            ex.unpack_state();
            if (msg.gateRatio) f32(ex.gate_ratio_ptr(), ex.max_gates()).set(msg.gateRatio);
            break;
        }

        case 'run': {
            // Timed, so the master can see how fast THIS worker's core really
            // is. On a phone the pool is spread over cores that differ by a
            // factor of three, and the master divides this by the car-steps
            // below to size the next generation's slices accordingly.
            const t0 = performance.now();
            const maxLaps = ex.run(msg.iters);
            const busyMs = performance.now() - t0;
            const carSteps = ex.car_steps();
            const allCrashed = ex.all_crashed() === 1;

            // Hyper mode puts nothing on screen, so only fitness/laps/lap time
            // cross the boundary instead of the full render row.
            let src, len;
            if (msg.wantRender) {
                ex.write_render();
                len = popCount * ex.render_stride();
                src = f32(ex.render_ptr(), len);
            } else {
                ex.write_fitness();
                len = popCount * ex.fitness_stride();
                src = f32(ex.fitness_ptr(), len);
            }
            // Refill the buffer the main thread handed back rather than
            // allocating a new one per step. It only comes back when it is the
            // right size; anything else (a population or mode change) falls
            // through to a fresh allocation.
            let buffer;
            if (msg.recycle && msg.recycle.byteLength === len * 4) {
                buffer = new Float32Array(msg.recycle);
                buffer.set(src);
            } else {
                buffer = src.slice();
            }
            // Only the worker holding global car 0 has anything worth
            // reporting here — car 0 mirrors the stash exactly whenever one
            // exists, so this is evolve()'s only window onto where the
            // all-time-best brain is currently slowest. A small, fixed-size
            // copy (MAX_GATES floats) regardless of population size.
            const gateRatio = popStart === 0 ? f32(ex.gate_ratio_ptr(), ex.max_gates()).slice() : null;
            // Unlike gateRatio, every worker sends this one — it's where THIS
            // SLICE's cars died this generation, and the master wants the sum
            // over every slice, not just one car's. Still small and fixed-size
            // (MAX_GATES ints) regardless of population.
            const crashCount = i32(ex.crash_count_ptr(), ex.max_gates()).slice();
            // alive_count travels separately from the buffer: in hyper mode the
            // crashed flags never cross at all, so the main thread has no way to
            // count them itself.
            const transfer = [buffer.buffer, crashCount.buffer];
            if (gateRatio) transfer.push(gateRatio.buffer);
            self.postMessage({
                type: 'done', index, start: popStart, count: popCount,
                maxLaps, allCrashed, alive: ex.alive_count(), busyMs, carSteps, crashCount,
                render: !!msg.wantRender, stride: ex.fitness_stride(), buffer, gateRatio
            }, transfer);
            break;
        }
    }
};
