// mutationsettle.mjs — three things about evolve() and the per-frame reward
// that the other test files don't cover, because they're about training
// DYNAMICS rather than single-generation correctness:
//
//   1. Mutation sigma follows sigmaGen, not the raw generation counter. This
//      file drives evolve() directly at chosen sigmaGen values and checks the
//      resulting mutation spread against MUT_SIGMA_START/FLOOR/TAU in sim.c —
//      the settle-window ARITHMETIC (when sigmaGen departs from 0 as a
//      function of the first completed lap and SETTLE_GENERATIONS) lives in
//      script.js, not here, because it needs a whole run's history that a
//      wasm instance never sees. This only proves the wasm side does the
//      right thing with whatever sigmaGen it's handed.
//      Mutation is sparse: about MUT_RATE (15%) of a child's weights move.
//   2. Elites are the stash (slot 0) then the best DISTINCT brains of the
//      generation, all unmutated, and no weight evolve() writes is ever
//      non-finite, even when a parent's is.
//   3. The per-frame progress reward no longer scales with speed — a car
//      that brakes partway through the window scores exactly the same as one
//      that never lifts, as long as it never drops below STOPPED_SPEED. It
//      used to reward raw speed every frame, which put a standing bias
//      against braking into the scoring regardless of whether the braking
//      produced a better time overall.
//
//   node tools/mutationsettle.mjs [path/to/sim.wasm]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] || join(here, '..', 'wasm', 'sim.wasm');
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports;
const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);
const i32 = (p, n) => new Int32Array(w.memory.buffer, p, n);

let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

console.log(`mutation settle / elites / braking test (${wasmPath})\n`);

// A loop sized to fit inside the world's own hard bounds (roughly -100..1300
// x, -100..1000 y — see updateCar's off-canvas check in sim.c, which fires on
// POSITION alone and doesn't care whether the track put a wall there), wide
// enough that a synthetic brain which never steers still comfortably stays on
// the road for the handful of frames each scenario runs. Corner radius is the
// standard 60 other test files use (small next to the gap between points at
// this size) — the loop's own RADIUS is a separate number, and conflating the
// two produces a corner tighter than the gap allows.
const CX = 600, CY = 450, RADIUS = 400, HALF_WIDTH = 80;
const path = Array.from({ length: 48 }, (_, i) => {
    const a = (Math.PI * 2 * i) / 48;
    return { x: Math.round(CX + Math.cos(a) * RADIUS), y: Math.round(CY + Math.sin(a) * RADIUS) };
});
const pIn = f32(w.path_in_ptr(), path.length * 4);
path.forEach((p, i) => { pIn[i * 4] = p.x; pIn[i * 4 + 1] = p.y; pIn[i * 4 + 2] = 1; pIn[i * 4 + 3] = 60; });
if (!w.track_build(w.path_in_ptr(), path.length, HALF_WIDTH, 0, 0, 0, 0, 0, w.zone_in_ptr(), 0, 0, 0)) {
    throw new Error('track_build failed');
}
const CP_N = w.track_cp_count();

const H = 5;   // stock hidden-layer size
const IN_N = 11, OUT_N = 2;

// ---------------------------------------------------------------------------
// 1. sigma follows sigmaGen, and matches the MUT_SIGMA_* formula in sim.c.
//
// The focused sub-population's brains are exactly `stash + gauss01()*sigma`
// (see evolve() in sim.c) — so with the stash zeroed, a focused child's
// weights ARE gauss01()*sigma samples directly, and gauss01() (three summed
// rnd11() draws) has unit variance by construction. A large population times
// a full brain's worth of weights gives thousands of samples, so the
// empirical standard deviation is a tight, direct estimate of sigma itself.
// ---------------------------------------------------------------------------
{
    const POP = 400;
    // focusPct high and eliteClones 0, so almost the whole population is
    // focused clones — maximising the sample count this measurement gets.
    w.set_config(10, 0.05, 0.02, 0.05, 750, 3, 0.8, H, 0);
    w.pop_init(POP, 0, H, 20260922);
    const stride = w.brain_stride();

    function measureSigma(sigmaGen) {
        // Zero the stash every time: evolve() swaps the live/next buffers, so
        // last round's bred population is now sitting where the stash lives
        // unless it's explicitly overwritten again.
        f32(w.brains_ptr() + w.stash_slot() * stride * 4, stride).fill(0);
        w.evolve(0, 1, sigmaGen);

        const focused = i32(w.car_focused_ptr(), POP);
        const brains = f32(w.brains_ptr(), POP * stride);
        // Only the touched weights are sigma samples; the rest stay exactly 0.
        let sum = 0, sumSq = 0, n = 0, total = 0;
        for (let c = 0; c < POP; c++) {
            if (!focused[c]) continue;
            for (let j = 0; j < stride; j++) {
                const v = brains[c * stride + j];
                total++;
                if (v === 0) continue;
                sum += v; sumSq += v * v; n++;
            }
        }
        const mean = sum / n;
        return { sigma: Math.sqrt(sumSq / n - mean * mean), n, rate: n / total };
    }

    const early = measureSigma(0);       // no settle elapsed: full exploration
    const late = measureSigma(100000);   // far past TAU: fully annealed

    // MUT_SIGMA_START/FLOOR from sim.c, restated here on purpose — a test
    // that read them out of the module could not tell "matches the formula"
    // from "matches whatever the formula happens to say today".
    const START = 0.5, FLOOR = 0.05;
    check(early.n > 2000, 'enough focused-clone weight samples to measure sigma',
        `${early.n} samples at sigmaGen=0`);
    check(Math.abs(early.rate - 0.15) < 0.02, 'mutation is sparse: ~15% of weights move per child',
        `${(early.rate * 100).toFixed(1)}%`);
    check(Math.abs(early.sigma - START) < START * 0.15,
        'sigmaGen=0 anneals to MUT_SIGMA_START — full exploration',
        `measured sigma ${early.sigma.toFixed(3)}, expected ~${START}`);
    check(Math.abs(late.sigma - FLOOR) < FLOOR * 0.5 || late.sigma < START * 0.3,
        'a sigmaGen far past TAU anneals down toward MUT_SIGMA_FLOOR',
        `measured sigma ${late.sigma.toFixed(3)}, expected ~${FLOOR}`);
    check(early.sigma > late.sigma * 3,
        'and the two are clearly different, not noise',
        `${early.sigma.toFixed(3)} vs ${late.sigma.toFixed(3)}`);

    // A handful of points along the actual decay curve, checked against the
    // closed form directly — not just "big" and "small".
    const TAU = 150;
    let worstRel = 0;
    for (const g of [0, 37, 75, 150, 300, 600]) {
        const expected = FLOOR + (START - FLOOR) * Math.exp(-g / TAU);
        const got = measureSigma(g).sigma;
        const rel = Math.abs(got - expected) / expected;
        if (rel > worstRel) worstRel = rel;
    }
    check(worstRel < 0.2, 'sigma tracks the exp(-sigmaGen/TAU) curve at several points',
        `worst relative error ${(worstRel * 100).toFixed(1)}%`);
}

// ---------------------------------------------------------------------------
// 2. Elites and the non-finite guard.
// ---------------------------------------------------------------------------
{
    const POP = 40, E = 6;
    w.set_config(10, 0.05, 0.02, 0.05, 750, 3, 0, H, 0);
    w.pop_init(POP, 0, H, 20260923);
    const stride = w.brain_stride();
    const brains = () => f32(w.brains_ptr(), (POP + 1) * stride);
    const live = brains();
    // Car c's brain is all c; cars 0..2 are copies of the stash (value -1)
    // and score highest, then 7 ties with 6, then descending.
    for (let c = 0; c < POP; c++) live.subarray(c * stride, (c + 1) * stride).fill(c < 3 ? -1 : c);
    f32(w.brains_ptr() + w.stash_slot() * stride * 4, stride).fill(-1);
    const ev = f32(w.ev_fitness_ptr(), POP);
    for (let c = 0; c < POP; c++) ev[c] = c < 3 ? 1000 : 500 - c;
    ev[7] = ev[6];
    live[20 * stride] = NaN; live[21 * stride + 1] = Infinity;   // poison two parents
    w.evolve(E, 1, 0);
    const next = f32(w.brains_ptr(), POP * stride);
    const val = (c) => next[c * stride + 3];
    const exact = (c) => { for (let j = 1; j < stride; j++) if (next[c * stride + j] !== next[c * stride]) return false; return true; };
    check(val(0) === -1 && exact(0), 'elite slot 0 is the stash, unmutated', `slot 0 = ${val(0)}`);
    const got = [1, 2, 3, 4, 5].map(val);
    check(JSON.stringify(got) === JSON.stringify([3, 4, 5, 6, 8]) && [1, 2, 3, 4, 5].every(exact),
        'then the best distinct brains, unmutated, skipping stash copies and same-score twins', `slots 1-5 = ${got}`);
    let bad = 0;
    for (let k = 0; k < POP * stride; k++) if (!Number.isFinite(next[k])) bad++;
    check(bad === 0, 'no non-finite weight survives evolve(), even from a poisoned parent', `${bad} non-finite`);
}

// ---------------------------------------------------------------------------
// 2.5. Nudge mode: focused clones are the stash byte-for-byte except one
// pushed weight (the throttle output bias), cycling through NUDGE_TABLE —
// never the isotropic Gaussian noise the default (nudgeMode=0) mode uses.
// ---------------------------------------------------------------------------
{
    w.set_config(10, 0.05, 0.02, 0.05, 750, 3, 0.5, H, 1);   // nudgeMode=1
    w.pop_init(20, 0, H, 20260925);
    const stride = w.brain_stride();
    const offBO = 11 * H + H * 2 + H;   // IN_N*h + h*OUT_N + h — matches offBO in sim.c
    const stash = f32(w.brains_ptr() + w.stash_slot() * stride * 4, stride);
    for (let j = 0; j < stride; j++) stash[j] = j * 0.01 + 0.001;   // distinctive, all-nonzero

    w.evolve(0, 1, 0);

    const focused = i32(w.car_focused_ptr(), 20);
    const brains = f32(w.brains_ptr(), 20 * stride);
    const NUDGE_TABLE = [0.15, -0.15, 0.35, -0.35, 0.6, -0.6];
    let n = 0, wrongWeight = 0, wrongDelta = 0, tableIdx = 0;
    for (let c = 0; c < 20; c++) {
        if (!focused[c]) continue;
        let diffs = 0, diffIdx = -1;
        for (let j = 0; j < stride; j++) {
            if (Math.abs(brains[c * stride + j] - stash[j]) > 1e-6) { diffs++; diffIdx = j; }
        }
        if (diffs !== 1 || diffIdx !== offBO + 1) wrongWeight++;
        else {
            const delta = brains[c * stride + diffIdx] - stash[diffIdx];
            if (Math.abs(delta - NUDGE_TABLE[tableIdx % NUDGE_TABLE.length]) > 1e-4) wrongDelta++;
        }
        tableIdx++; n++;
    }
    check(n >= 8, 'enough nudge clones bred to check the table', `${n} focused clones`);
    check(wrongWeight === 0, 'a nudge clone changes exactly one weight — the throttle output bias',
        wrongWeight === 0 ? 'every clone matched' : `${wrongWeight}/${n} touched the wrong weight(s)`);
    check(wrongDelta === 0, 'and the pushes cycle through NUDGE_TABLE in order',
        wrongDelta === 0 ? 'every delta matched' : `${wrongDelta}/${n} did not match the table`);
}

// ---------------------------------------------------------------------------
// 3. The per-frame reward is flat — braking doesn't cost it.
//
// Two cars, same synthetic-brain technique as carphysics.mjs: zero every
// input/hidden weight so steer/throttle are exactly the fixed output biases
// every frame, regardless of sensors. Car 0 holds full throttle the whole
// time; car 1 spends the second half of the window braking hard. Neither
// reaches a gate or crashes in the window measured, so the ENTIRE fitness
// difference between them, if any, is this one term.
// ---------------------------------------------------------------------------
{
    w.set_config(10, 0.05, 0.02, 0.3, 5000, 99, 0.15, H, 0);
    w.pop_init(2, 0, H, 20260924);
    const stride = w.brain_stride();
    const offBiasO = IN_N * H + H * OUT_N + H;
    function setBiasBrain(i, steer, throttle) {
        const b = f32(w.brains_ptr() + i * stride * 4, stride);
        b.fill(0);
        b[offBiasO] = steer; b[offBiasO + 1] = throttle;
    }
    const BIG = 20.0;

    // Asymmetric on purpose: enough accelerating frames to build a real speed
    // difference once braking starts, few enough total that neither car gets
    // anywhere near the first checkpoint (this track's gate 0 falls around
    // frame 22 for a car on this exact line — checked empirically, since
    // it depends on track geometry, not on a constant sim.c exposes) or the
    // 180-frame edge of PROGRESS_WINDOW_FRAMES.
    const ACCEL_FRAMES = 16, BRAKE_FRAMES = 4, FRAMES = ACCEL_FRAMES + BRAKE_FRAMES;
    setBiasBrain(0, 0, BIG);          // full throttle throughout
    setBiasBrain(1, 0, BIG);          // full throttle while building speed...
    w.pop_reset();
    for (let f = 0; f < ACCEL_FRAMES; f++) w.run(1);
    setBiasBrain(1, 0, -0.15);        // ...then a light brake, never to a standstill
    for (let f = 0; f < BRAKE_FRAMES; f++) w.run(1);

    w.write_fitness();
    const S = w.fitness_stride();
    const fb = f32(w.fitness_ptr(), 2 * S);
    w.write_render();
    const rb = f32(w.render_ptr(), 2 * w.render_stride());
    const speedOf = (i) => rb[i * w.render_stride() + 5];
    const crashedOf = (i) => rb[i * w.render_stride() + 1] === 1;

    check(!crashedOf(0) && !crashedOf(1), 'neither car crashes across the window',
        `car0 crashed=${crashedOf(0)}, car1 crashed=${crashedOf(1)}`);
    check(speedOf(1) > 0.05, 'the braking car never drops to a standstill',
        `car1 final speed ${speedOf(1).toFixed(3)}`);
    check(speedOf(0) > speedOf(1) * 1.3, 'and it really is braking — noticeably slower than the other car',
        `car0 ${speedOf(0).toFixed(2)} vs car1 ${speedOf(1).toFixed(2)}`);

    // Every frame is worth exactly 0.1 or exactly 0 (below STOPPED_SPEED —
    // true for the first couple of frames both cars share, before their
    // shared startup ramp gets either one above it), so no frame counted can
    // ever push a car's total past FRAMES*0.1, whichever way it drove.
    const ceiling = 0.1 * FRAMES;
    check(fb[0 * S] > 0 && fb[0 * S] <= ceiling + 1e-3, 'full-throttle fitness never exceeds the flat-per-frame ceiling',
        `got ${fb[0 * S].toFixed(4)}, ceiling ${ceiling}`);
    check(fb[1 * S] > 0 && fb[1 * S] <= ceiling + 1e-3, 'and neither does the braking car',
        `got ${fb[1 * S].toFixed(4)}, ceiling ${ceiling}`);
    // The one claim that actually matters: both cars share an identical
    // startup transient (same brain, same physics, up to the frame they
    // diverge), and the flat reward doesn't care how fast either one was
    // going after that — so braking costs literally nothing here, where the
    // old speed-proportional version would have scored car0 measurably
    // higher for never lifting.
    check(fb[0 * S] === fb[1 * S],
        'braking into a corner costs nothing next to never lifting, for equal frames alive',
        `car0 ${fb[0 * S].toFixed(4)} vs car1 ${fb[1 * S].toFixed(4)}`);
}

console.log();
if (failures) { console.error(`${failures} check(s) failed.`); process.exit(1); }
console.log('mutation settles on schedule, elites are exact and distinct, and braking is free.');
