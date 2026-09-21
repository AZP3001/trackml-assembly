// bench.mjs — how much faster the WebAssembly backend actually is.
//
// Runs the identical workload through sim.wasm and through the JavaScript
// edition's worker (tools/reference-worker.mjs, copied verbatim), single
// threaded, same track, same brains, same number of steps. Both versions use
// the same worker-pool-per-core scheme on top of this, so the ratio measured
// here is the ratio you get in the app.
//
//   node tools/bench.mjs [population] [steps]        # scalar build
//   WASM=wasm/sim-simd.wasm node tools/bench.mjs      # SIMD build
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createReferenceWorker } from './reference-worker.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.env.WASM || join(here, '..', 'wasm', 'sim.wasm');
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports;
const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);
const i32 = (p, n) => new Int32Array(w.memory.buffer, p, n);

const POP = Number(process.argv[2] || 200);
const STEPS = Number(process.argv[3] || 3000);
const HIDDEN = 5, TTL = 1e9;      // effectively immortal, so the work is comparable
const IN_N = 9, OUT_N = 2;
const cfg = { maxSpeed: 10, acceleration: 0.05, turnSpeed: 0.04, brakeStrength: 0.2, targetLaps: 1e9 };

const CW = 1200, CH = 900, cx = CW / 2, cy = CH / 2;
const path = Array.from({ length: 48 }, (_, i) => {
    const a = (Math.PI * 2 * i) / 48, r = 340 + Math.sin(a * 3) * 70;
    return { x: Math.round(cx + Math.cos(a) * r), y: Math.round(cy + Math.sin(a) * r), type: 'corner', radius: 60 };
});

// --- build the track once, share the geometry with both engines ---
const pIn = f32(w.path_in_ptr(), path.length * 4);
path.forEach((p, i) => { pIn[i*4]=p.x; pIn[i*4+1]=p.y; pIn[i*4+2]=1; pIn[i*4+3]=60; });
w.track_build(w.path_in_ptr(), path.length, 60, 0, 0, 0, 0, 0, w.zone_in_ptr(), 0, 0, 0);

const nw = w.track_wall_count(), wf = f32(w.track_walls_ptr(), nw*5), wi = i32(w.track_walls_ptr(), nw*5);
const walls = Array.from({ length: nw }, (_, i) => ({
    p1: { x: wf[i*5], y: wf[i*5+1] }, p2: { x: wf[i*5+2], y: wf[i*5+3] },
    segmentIndex: wi[i*5+4] < 0 ? undefined : wi[i*5+4]
}));
const ncp = w.track_cp_count(), cf = f32(w.track_cps_ptr(), ncp*7);  // stride 7: coords + apex flag
const checkpoints = Array.from({ length: ncp }, (_, i) => ({
    index: i, p1: { x: cf[i*7], y: cf[i*7+1] }, p2: { x: cf[i*7+2], y: cf[i*7+3] },
    center: { x: cf[i*7+4], y: cf[i*7+5] }
}));
const track = { walls, checkpoints, zones: [], segStep: 34,
                startPos: { x: w.track_start_x(), y: w.track_start_y() }, startAngle: w.track_start_angle() };

// --- identical brains on both sides ---
w.set_config(cfg.maxSpeed, cfg.acceleration, cfg.turnSpeed, cfg.brakeStrength, TTL, cfg.targetLaps, 0.15, HIDDEN);
w.pop_init(POP, 0, HIDDEN, 999);
w.pop_randomize_brains();
const stride = w.brain_stride();
const brains = f32(w.brains_ptr(), POP * stride).slice();

function refBrain(i) {
    const b = brains.subarray(i * stride, (i + 1) * stride);
    return {
        weightsIH: b.slice(0, IN_N*HIDDEN), weightsHO: b.slice(IN_N*HIDDEN, IN_N*HIDDEN + HIDDEN*OUT_N),
        biasH: b.slice(IN_N*HIDDEN + HIDDEN*OUT_N, IN_N*HIDDEN + HIDDEN*OUT_N + HIDDEN),
        biasO: b.slice(IN_N*HIDDEN + HIDDEN*OUT_N + HIDDEN, stride)
    };
}

function makeRef() {
    const outbox = [];
    const self = { postMessage: m => outbox.push(m) };
    createReferenceWorker(self);
    self.onmessage({ data: { type: 'initTrack', track, config: cfg } });
    self.onmessage({ data: { type: 'initCars', cars: Array.from({ length: POP }, (_, i) => ({
        id: i, x: track.startPos.x, y: track.startPos.y, angle: track.startAngle, vx: 0, vy: 0, speed: 0,
        brain: refBrain(i), fitness: 0, crashed: false, timeToLive: TTL, nextCheckpointIndex: 1,
        framesAlive: 0, lapTimes: [], completedLaps: 0, checkpointsReached: 0
    })) } });
    return { run: iters => self.onmessage({ data: { type: 'run', iters } }), outbox };
}

function time(label, fn) {
    fn();                                  // warm up: let the JIT settle before timing it
    const t0 = process.hrtime.bigint();
    fn();
    const t1 = process.hrtime.bigint();
    const ms = Number(t1 - t0) / 1e6;
    console.log(`  ${label.padEnd(26)} ${ms.toFixed(1).padStart(9)} ms`);
    return ms;
}

console.log(`TrackML backend benchmark — ${POP} cars x ${STEPS} steps, ${walls.length} walls, single thread`);
console.log(`module: ${wasmPath.split('/').pop()}\n`);

const jsMs = time('JavaScript (reference)', () => { const r = makeRef(); r.run(STEPS); });
const wasmMs = time(`WebAssembly (${wasmPath.split('/').pop()})`, () => { w.pop_reset(); w.run(STEPS); });

const carSteps = POP * STEPS;
console.log(`\n  speedup                     ${(jsMs / wasmMs).toFixed(2)}x`);
console.log(`  JS   throughput             ${(carSteps / jsMs / 1000).toFixed(2)} M car-steps/s`);
console.log(`  WASM throughput             ${(carSteps / wasmMs / 1000).toFixed(2)} M car-steps/s`);
console.log(`\n  (single thread; the app runs ${'navigator.hardwareConcurrency'} of these in parallel,`);
console.log('   and in hyper mode skips the per-frame render round trip entirely.)');
