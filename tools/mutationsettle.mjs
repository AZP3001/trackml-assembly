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
//   2. Below FEW_LAPS_THRESHOLD completed-lap generations, evolve() aims the
//      focus window at wherever the population is dying (crash_count); at or
//      above it, at wherever the current best is slowest (gate_ratio). Both
//      are poked directly through their exported pointers so the test
//      controls exactly which gate "wins" each way, rather than depending on
//      an actual population finding its way there.
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

console.log(`mutation settle / focus mode / braking test (${wasmPath})\n`);

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
    w.set_config(10, 0.05, 0.02, 0.05, 750, 3, 0.8, H);
    w.pop_init(POP, 0, H, 20260922);
    const stride = w.brain_stride();

    function measureSigma(sigmaGen) {
        // Zero the stash every time: evolve() swaps the live/next buffers, so
        // last round's bred population is now sitting where the stash lives
        // unless it's explicitly overwritten again.
        f32(w.brains_ptr() + w.stash_slot() * stride * 4, stride).fill(0);
        i32(w.crash_count_ptr(), w.max_gates()).fill(0);
        // Gate 3 "kills" every car, every time — guarantees focus_lo/hi
        // activate regardless of sigmaGen, via the failure-focus path
        // (lapCompletions=0), so this measurement never depends on the
        // gate_ratio path also being exercised correctly.
        i32(w.crash_count_ptr(), w.max_gates())[3] = 999;
        w.evolve(0, 1, sigmaGen, 0);

        const focused = i32(w.car_focused_ptr(), POP);
        const brains = f32(w.brains_ptr(), POP * stride);
        let sum = 0, sumSq = 0, n = 0;
        for (let c = 0; c < POP; c++) {
            if (!focused[c]) continue;
            for (let j = 0; j < stride; j++) {
                const v = brains[c * stride + j];
                sum += v; sumSq += v * v; n++;
            }
        }
        const mean = sum / n;
        return { sigma: Math.sqrt(sumSq / n - mean * mean), n };
    }

    const early = measureSigma(0);       // no settle elapsed: full exploration
    const late = measureSigma(100000);   // far past TAU: fully annealed

    // MUT_SIGMA_START/FLOOR from sim.c, restated here on purpose — a test
    // that read them out of the module could not tell "matches the formula"
    // from "matches whatever the formula happens to say today".
    const START = 0.5, FLOOR = 0.05;
    check(early.n > 5000, 'enough focused-clone weight samples to measure sigma',
        `${early.n} samples at sigmaGen=0`);
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
// 2. Focus mode: crash_count below FEW_LAPS_THRESHOLD, gate_ratio at/above.
//
// Gate 3 is made the clear "worst" by crash count, a gate on the far side of
// the loop the clear "worst" by pace ratio — two gates far enough apart that
// their focus spans (roughly +-18 gates at this track's size, see evolve()'s
// span calculation) can never overlap, so whichever one the window lands on
// says unambiguously which source evolve() actually read.
// ---------------------------------------------------------------------------
{
    w.set_config(10, 0.05, 0.02, 0.05, 750, 3, 0.2, H);
    w.pop_init(50, 0, H, 20260923);
    const stride = w.brain_stride();
    f32(w.brains_ptr() + w.stash_slot() * stride * 4, stride).fill(0);
    const FAR_GATE = Math.floor(CP_N / 2);   // opposite side of the loop from gate 3

    i32(w.crash_count_ptr(), w.max_gates()).fill(0);
    i32(w.crash_count_ptr(), w.max_gates())[3] = 500;
    const gr = f32(w.gate_ratio_ptr(), w.max_gates());
    gr.fill(-1);
    gr[FAR_GATE] = 0.1;   // far the slowest — every other reached gate would be -1 (unreached) or faster

    const inWindow = (g) => {
        const lo = w.focus_lo(), hi = w.focus_hi();
        if (lo < 0) return false;
        return lo <= hi ? (g >= lo && g <= hi) : (g >= lo || g <= hi);
    };

    w.evolve(0, 1, 0, 0);   // lapCompletions=0: below threshold -> crash_count
    check(inWindow(3) && !inWindow(FAR_GATE), 'below the lap threshold, focus aims at where it keeps dying',
        `window [${w.focus_lo()}, ${w.focus_hi()}] of ${CP_N} gates`);

    w.evolve(0, 1, 0, 3);   // lapCompletions=3: at threshold -> gate_ratio
    check(inWindow(FAR_GATE) && !inWindow(3), 'at the lap threshold, focus goes back to the slowest gate',
        `window [${w.focus_lo()}, ${w.focus_hi()}] of ${CP_N} gates`);

    // And with nothing dying and nothing reached either, there's nothing to
    // aim at — the window should stay inactive rather than pick an arbitrary
    // gate 0 out of an all-empty array.
    i32(w.crash_count_ptr(), w.max_gates()).fill(0);
    gr.fill(-1);
    w.evolve(0, 1, 0, 0);
    check(w.focus_lo() < 0, 'and with no data either way, the window stays off',
        `focus_lo=${w.focus_lo()}`);
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
    w.set_config(10, 0.05, 0.02, 0.3, 5000, 99, 0.15, H);
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
console.log('mutation settles on schedule, focus aims at the right thing, and braking is free.');
