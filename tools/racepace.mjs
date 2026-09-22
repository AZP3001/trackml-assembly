// racepace.mjs — the two things a racing simulation has to get right:
// cars stay on the track, and evolution selects the fast ones.
//
// Both were broken, and neither showed up in the geometry or parity tests.
//
// CONTAINMENT. The wall lookup is bucketed by the checkpoint a car is heading
// for. A car whose checkpoint index went stale was therefore handed the walls
// for a part of the track it had left, and drove straight through the ones in
// front of it. The index went stale because a gate only registered within a
// flat 50px of its middle, while a gate spans the full width of the road — so
// a car taking the outside line on any track wider than that passed gate after
// gate without registering one. Hence "you can drive through the walls on the
// outside".
//
// PACE. Reaching a gate scored the same whether it took four frames or four
// hundred, and driving slowly is far less likely to end in a wall — so the
// fittest car was the most patient one, not the quickest.
//
//   node tools/racepace.mjs [path/to/sim.wasm]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] || join(here, '..', 'wasm', 'sim.wasm');
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports;
const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);

let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

const CX = 600, CY = 450;
const polar = (steps, fn) => Array.from({ length: steps }, (_, i) => {
    const a = (Math.PI * 2 * i) / steps, r = fn(a);
    return { x: Math.round(CX + Math.cos(a) * r), y: Math.round(CY + Math.sin(a) * r) };
});

function buildTrack(path, halfWidth, startPos = null) {
    const pIn = f32(w.path_in_ptr(), path.length * 4);
    path.forEach((p, i) => { pIn[i * 4] = p.x; pIn[i * 4 + 1] = p.y; pIn[i * 4 + 2] = 1; pIn[i * 4 + 3] = 60; });
    const ok = w.track_build(w.path_in_ptr(), path.length, halfWidth,
        startPos ? 1 : 0, startPos ? startPos.x : 0, startPos ? startPos.y : 0,
        0, 0, w.zone_in_ptr(), 0, 0, 0);
    if (!ok) throw new Error('track_build failed');
    const n = w.track_centerline_count();
    return {
        n, cps: w.track_cp_count(),
        centre: f32(w.track_centerline_ptr(), n * 2).slice(),
        widths: f32(w.track_widths_ptr(), n).slice()
    };
}

// The road is every point within its local half-width of the centre line — the
// same definition the barriers are traced from. Computed here in JS rather than
// asked of the wasm, so the test is not just asking the code under test whether
// it agrees with itself.
function makeInsideTest(t) {
    return (px, py) => {
        for (let i = 0; i < t.n; i++) {
            const j = (i + 1) % t.n;
            const ax = t.centre[i * 2], ay = t.centre[i * 2 + 1];
            const bx = t.centre[j * 2], by = t.centre[j * 2 + 1];
            const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
            let s = l2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
            s = s < 0 ? 0 : s > 1 ? 1 : s;
            const ex = px - (ax + s * dx), ey = py - (ay + s * dy);
            const r = Math.max(t.widths[i], t.widths[j]);
            if (ex * ex + ey * ey < r * r) return true;
        }
        return false;
    };
}

// Run generations the way the app does: to completion, not to a frame budget.
// A crawling car needs tens of thousands of frames to finish a lap, so a
// frame-capped harness cannot see the pace problem at all.
function train({ track, halfWidth, startPos, gens, pop = 100, ttl = 750, targetLaps = 3,
                 maxSpeed = 10, watchEscapes = false, cap = 200000, seed = 4242 }) {
    const t = buildTrack(track, halfWidth, startPos);
    const inside = watchEscapes ? makeInsideTest(t) : null;
    w.set_config(maxSpeed, 0.05, 0.04, 0.2, ttl, targetLaps, 0.15, 5);
    w.pop_init(pop, 0, 5, seed);
    w.pop_randomize_brains();

    let escapes = 0, liveSamples = 0, worstOvershoot = 0;
    let last = null;
    const S = w.fitness_stride();

    for (let gen = 0; gen < gens; gen++) {
        w.pop_reset();
        let frames = 0;
        while (frames < cap) {
            if (watchEscapes) {
                w.run(1); frames += 1;
                w.write_render();
                const rb = f32(w.render_ptr(), pop * 18);
                for (let i = 0; i < pop; i++) {
                    if (rb[i * 18 + 1] === 1) continue;
                    liveSamples++;
                    const x = rb[i * 18 + 2], y = rb[i * 18 + 3];
                    if (!inside(x, y)) {
                        escapes++;
                        let d = 1e30;
                        for (let k = 0; k < t.n; k++) {
                            const o = Math.hypot(t.centre[k * 2] - x, t.centre[k * 2 + 1] - y) - t.widths[k];
                            if (o < d) d = o;
                        }
                        if (d > worstOvershoot) worstOvershoot = d;
                    }
                }
                if (w.all_crashed() === 1) break;
            } else {
                const laps = w.run(500); frames += 500;
                if (w.all_crashed() === 1 || laps >= targetLaps) break;
            }
        }
        w.write_fitness();
        const fb = f32(w.fitness_ptr(), pop * S);
        let bi = 0, bf = -Infinity;
        for (let i = 0; i < pop; i++) if (fb[i * S] > bf) { bf = fb[i * S]; bi = i; }
        let bestLapAny = Infinity, finishers = 0, totalGates = 0;
        for (let i = 0; i < pop; i++) {
            totalGates += fb[i * S + 3];
            if (fb[i * S + 1] > 0) { finishers++; if (fb[i * S + 2] > 0 && fb[i * S + 2] < bestLapAny) bestLapAny = fb[i * S + 2]; }
        }
        last = {
            fittestGates: fb[bi * S + 3], fittestLaps: fb[bi * S + 1], fittestLap: fb[bi * S + 2],
            bestLapAny, finishers, totalGates, frames, cpsPerLap: t.cps
        };
        const ev = f32(w.ev_fitness_ptr(), pop);
        for (let i = 0; i < pop; i++) ev[i] = fb[i * S];
        // sigmaGen=gen and lapCompletions=3 (FEW_LAPS_THRESHOLD in sim.c)
        // reproduce evolve()'s old single-generation-argument behaviour
        // exactly: sigma decays from generation 1 with no settle wait, and
        // the focus window aims at the slowest gate rather than the deadliest
        // one. This file is about the scoring rule and pace, not about the
        // settle-delay or failure-focus behaviour, which have their own
        // dedicated coverage — see mutationsettle.mjs.
        w.evolve(10, 1, gen, 3);
    }
    return { ...last, escapes, liveSamples, worstOvershoot, track: t };
}

console.log(`race behaviour test (${wasmPath})\n`);

// --- containment ---------------------------------------------------------
// Widths on both sides of the old 50px gate radius. The wide cases are where
// the outside line used to stop registering gates altogether.
{
    const cases = [
        ['oval  w60',  polar(24, () => 340), 60],
        ['oval  w120', polar(24, () => 340), 120],
        ['hills w60',  polar(48, a => 340 + Math.sin(a * 3) * 70), 60],
        ['gear  w40',  polar(60, a => 300 + Math.sin(a * 6) * 45), 40]
    ];
    let total = 0, samples = 0, worst = 0;
    for (const [name, path, hw] of cases) {
        const r = train({ track: path, halfWidth: hw, gens: 8, pop: 100, watchEscapes: true, cap: 1200 });
        total += r.escapes; samples += r.liveSamples;
        if (r.worstOvershoot > worst) worst = r.worstOvershoot;
        console.log(`  ${name}  ${String(r.liveSamples).padStart(7)} live samples, ${r.escapes} off the road`);
    }
    check(total === 0, 'no car is ever off the asphalt',
        total === 0 ? `${samples.toLocaleString()} live car-frames checked`
                    : `${total} escapes, worst ${worst.toFixed(1)}px past the barrier`);
}

// --- gates register across the full width of the road --------------------
// On a wide track a car on the outside line sits further from the middle of a
// gate than the old fixed radius reached, so its checkpoint index never moved:
// no laps, and the stale index cost it its wall coverage too.
{
    const r = train({ track: polar(24, () => 340), halfWidth: 140, gens: 14, pop: 100 });
    check(r.finishers > 0, 'cars complete laps on a very wide track',
        `${r.finishers} finishers, fittest covered ${r.fittestGates} gates`);
    check(r.fittestGates >= r.cpsPerLap, 'the fittest car gets at least a full lap of gates',
        `${r.fittestGates} gates vs ${r.cpsPerLap} per lap`);
}

// --- the start line is where the lap starts ------------------------------
// Cars used to be aimed at checkpoint 1 wherever the start actually was, which
// on a track with a moved start pointed them most of a lap away — and handed
// them the wall bucket for that distant part of the track.
{
    const path = polar(24, () => 340);
    buildTrack(path, 60, { x: CX - 340, y: CY });   // start on the far side
    const startCp = w.track_start_cp();
    const ncp = w.track_cp_count();
    check(startCp > 1 && startCp < ncp - 1, 'the start checkpoint follows the start line',
        `checkpoint ${startCp} of ${ncp}, not 0`);

    // Two runs, because they need different budgets: watching every frame for
    // escapes is only affordable with a short generation, and learning to
    // finish a lap from random weights is not.
    const esc = train({ track: path, halfWidth: 60, startPos: { x: CX - 340, y: CY },
                        gens: 10, pop: 100, watchEscapes: true, cap: 1500 });
    check(esc.escapes === 0, 'a moved start line does not let cars through walls',
        `${esc.liveSamples.toLocaleString()} live car-frames, ${esc.escapes} escapes`);

    // Assert the lap-counting INVARIANT rather than that a population happens
    // to finish within N generations, which is luck. A car that has covered a
    // lap's worth of gates must have been credited a lap — that is exactly what
    // breaks if laps are counted from index zero instead of the start line.
    const run = train({ track: path, halfWidth: 60, startPos: { x: CX - 340, y: CY },
                        gens: 20, pop: 120 });
    check(run.fittestGates >= run.cpsPerLap, 'a moved start still lets cars get round',
        `fittest covered ${run.fittestGates} gates of ${run.cpsPerLap} per lap`);
    if (run.fittestGates >= run.cpsPerLap) {
        const expected = Math.floor(run.fittestGates / run.cpsPerLap);
        check(run.fittestLaps >= 1 && Math.abs(run.fittestLaps - expected) <= 1,
            'and laps are counted from the start line',
            `${run.fittestGates} gates -> ${run.fittestLaps} laps (expected about ${expected})`);
    }
}

// --- pace: the scoring rule itself --------------------------------------
// Tested directly rather than through evolution. Whether a population cracks a
// hard track within N generations is bimodal and seed-dependent — six seeds on
// the narrow gear track split 4/6 and 3/6 between two builds that differ in no
// relevant way — so an assertion resting on it reports luck, not correctness.
//
// The rule is: for the SAME progress, fewer frames must score higher. That is
// what the old scoring got wrong, and it can be read straight off a single
// generation, with no evolution involved.
{
    const t = buildTrack(polar(48, a => 340 + Math.sin(a * 3) * 70), 60);
    w.set_config(10, 0.05, 0.04, 0.2, 750, 99, 0.15, 5);
    const POP = 200;
    w.pop_init(POP, 0, 5, 20260921);
    w.pop_randomize_brains();
    w.pop_reset();
    let frames = 0;
    // do-while, not while: all_crashed() reports the last completed run, and
    // before any run at all it still reads "everything crashed" — a leading
    // guard on it exits immediately and simulates nothing.
    do { w.run(50); frames += 50; } while (frames < 4000 && w.all_crashed() !== 1);
    w.write_fitness();
    const S = w.fitness_stride();
    const fb = f32(w.fitness_ptr(), POP * S);

    // Group by gates reached; within a group, more frames must not score more.
    const byGates = new Map();
    for (let i = 0; i < POP; i++) {
        const g = fb[i * S + 3];
        if (g < 1) continue;                       // never got going; nothing to compare
        if (!byGates.has(g)) byGates.set(g, []);
        // Slot 4 is the frame the car's LAST gate fell on, i.e. how long it
            // took to get that far — not how long it stayed alive afterwards.
            byGates.get(g).push({ fit: fb[i * S], frames: fb[i * S + 4] });
    }
    // Only pairs whose times differ by a clear margin. Total fitness also
    // carries a small distance-shaped term, so between two cars a frame or two
    // apart that term, not the gate timing, decides the order — including those
    // measures noise rather than the rule.
    let pairs = 0, violations = 0, worstExample = null;
    for (const [g, cars] of byGates) {
        for (let a = 0; a < cars.length; a++) for (let b = a + 1; b < cars.length; b++) {
            const slow = cars[a].frames > cars[b].frames ? cars[a] : cars[b];
            const fast = cars[a].frames > cars[b].frames ? cars[b] : cars[a];
            if (slow.frames < fast.frames * 1.25) continue;
            pairs++;
            if (slow.fit > fast.fit + 1e-3) {
                violations++;
                if (!worstExample) worstExample = { g, slow, fast };
            }
        }
    }
    const rate = pairs ? violations / pairs : 1;
    check(pairs > 20, 'enough same-progress pairs to judge',
        `${pairs} pairs across ${byGates.size} progress levels`);
    // Under the old scoring a gate paid a flat 500 whatever it cost, so for
    // equal progress the gate term could not order two cars at all and this
    // came out around chance. It is a statistical claim, not an absolute one:
    // the distance term still breaks the odd pair the other way.
    check(rate < 0.15, 'for equal progress, the quicker car scores higher',
        `${violations}/${pairs} inverted (${(rate * 100).toFixed(1)}%)` +
        (worstExample ? `, e.g. ${worstExample.slow.frames}f scored ${worstExample.slow.fit.toFixed(0)} vs ${worstExample.fast.frames}f scoring ${worstExample.fast.fit.toFixed(0)}` : ''));

    // The floor. Every gate is worth AT LEAST what it was worth before the
    // speed multiplier existed, so time can only ever add to a score, never
    // subtract from it. That is the property that keeps a slow finisher above a
    // car that crashed early — and it is precisely what a per-frame time
    // penalty, the obvious way to reward speed, would destroy.
    const BASE_GATE = 500, CRASH = 50;
    let belowFloor = 0, floorExample = null;
    for (let i = 0; i < POP; i++) {
        const gates = fb[i * S + 3], fit = fb[i * S];
        if (gates < 1) continue;
        const floor = BASE_GATE * gates - CRASH - 1;
        if (fit < floor) {
            belowFloor++;
            if (!floorExample) floorExample = `${gates} gates scored ${fit.toFixed(0)}, floor is ${floor.toFixed(0)}`;
        }
    }
    check(belowFloor === 0, 'a gate is never worth less than it used to be',
        floorExample || 'every car at or above the base rate for its progress');
}

// --- pace: and it shows up in training ----------------------------------
// A softer end-to-end check on a track that learns reliably, across several
// seeds, so it measures the scoring rather than the luck of one run.
{
    let laps = 0, worst = 0, runs = 0;
    for (const seed of [11, 4242, 31337]) {
        const r = train({ track: polar(48, a => 340 + Math.sin(a * 3) * 70), halfWidth: 60,
                          gens: 14, pop: 120, ttl: 10000, seed });
        runs++;
        if (r.fittestLaps > 0) { laps++; if (r.fittestLap > worst) worst = r.fittestLap; }
    }
    check(laps === runs, 'every seed learns to lap on an ordinary track', `${laps}/${runs}`);
    // A lap of this track flat out is around 6s. The old scoring produced
    // winners crawling round at 40 to 90 seconds on tracks like this.
    check(worst > 0 && worst < 20, 'and none of the winners is crawling',
        `slowest winning lap ${worst.toFixed(1)}s`);
}

console.log();
if (failures) { console.error(`${failures} race behaviour check(s) failed.`); process.exit(1); }
console.log('cars stay on the track, and the quick ones win.');
