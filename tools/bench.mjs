// bench.mjs — how fast the simulation actually steps, scalar versus SIMD.
//
// This used to time the wasm backend against tools/reference-worker.mjs, a
// hand-maintained JavaScript copy of the same physics. That copy is gone (see
// the note in the README's Testing section): it had to be edited by hand to
// match every change to sim.c, with nothing to catch it drifting, and the
// question it answered — "was the port worth doing" — was settled a long time
// ago. What is still worth measuring on every build is the thing that varies
// per machine: whether the 128-bit SIMD module is actually paying for itself
// on this CPU, since engine.js picks between exactly these two files.
//
//   node tools/bench.mjs [population] [steps]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const POP = Number(process.argv[2] || 200);
const STEPS = Number(process.argv[3] || 3000);
const HIDDEN = 5, TTL = 1e9;   // effectively immortal, so the work is comparable

const CW = 1200, CH = 900, cx = CW / 2, cy = CH / 2;
const path = Array.from({ length: 48 }, (_, i) => {
    const a = (Math.PI * 2 * i) / 48, r = 340 + Math.sin(a * 3) * 70;
    return { x: Math.round(cx + Math.cos(a) * r), y: Math.round(cy + Math.sin(a) * r) };
});

async function timeModule(file) {
    const { instance } = await WebAssembly.instantiate(readFileSync(join(here, '..', file)), {});
    const w = instance.exports;
    const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);

    const pIn = f32(w.path_in_ptr(), path.length * 4);
    path.forEach((p, i) => { pIn[i*4]=p.x; pIn[i*4+1]=p.y; pIn[i*4+2]=1; pIn[i*4+3]=60; });
    if (!w.track_build(w.path_in_ptr(), path.length, 60, 0,0,0, 0,0, w.zone_in_ptr(), 0, 0, 0)) {
        throw new Error(`${file}: track_build failed`);
    }
    // Same seed both sides, so both simulate the identical population.
    w.set_config(10, 0.05, 0.04, 0.2, TTL, 1e9, 0.3, HIDDEN);
    w.pop_init(POP, 0, HIDDEN, 20260921);
    w.pop_randomize_brains();
    w.pop_reset();

    w.run(50);                       // warm up the JIT before timing
    w.pop_reset();
    const t0 = performance.now();
    let frames = 0, carFrames = 0;
    while (frames < STEPS) {
        w.run(1); frames++;
        carFrames += w.alive_count();
        if (w.all_crashed() === 1) { w.pop_reset(); }
    }
    const ms = performance.now() - t0;
    return { ms, carFrames, walls: w.track_wall_count() };
}

const scalar = await timeModule('wasm/sim.wasm');
const simd   = await timeModule('wasm/sim-simd.wasm');

console.log(`TrackML simulation benchmark — ${POP} cars x ${STEPS} steps, ${scalar.walls} walls, single thread\n`);
const row = (name, r) => {
    const rate = r.carFrames / (r.ms / 1000) / 1e6;
    console.log(`  ${name.padEnd(28)} ${r.ms.toFixed(1).padStart(7)} ms   ${rate.toFixed(2)} M car-steps/s`);
};
row('scalar (sim.wasm)', scalar);
row('SIMD   (sim-simd.wasm)', simd);
console.log(`\n  SIMD speedup              ${(scalar.ms / simd.ms).toFixed(2)}x`);
console.log(`
  (single thread; the app runs navigator.hardwareConcurrency of these in
   parallel, and in hyper mode skips the per-frame render round trip entirely.)`);
