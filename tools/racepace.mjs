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
    w.set_config(maxSpeed, 0.05, 0.04, 0.93, ttl, targetLaps, 0.15, 5);
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
        w.evolve(10, 1);
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

    const run = train({ track: path, halfWidth: 60, startPos: { x: CX - 340, y: CY },
                        gens: 16, pop: 100 });
    check(run.finishers > 0, 'and laps still complete from a moved start',
        `${run.finishers} finishers, fittest lap ${run.fittestLaps > 0 ? run.fittestLap.toFixed(1) + 's' : 'n/a'}`);
}

// --- pace ----------------------------------------------------------------
// The fittest car should be a quick one. Under the old scoring the winner
// could be a crawler while a much faster car sat lower down the field, because
// reaching a gate paid the same at any speed.
{
    const r = train({ track: polar(60, a => 300 + Math.sin(a * 6) * 45), halfWidth: 25,
                      gens: 22, pop: 120, ttl: 10000, seed: 777 });
    check(r.fittestLaps > 0, 'evolution finds a lap on a hard narrow track',
        `fittest completed ${r.fittestLaps} laps`);
    if (r.fittestLaps > 0) {
        const slack = r.fittestLap / r.bestLapAny;
        check(slack < 1.15, 'the fittest car is one of the fastest',
            `its lap ${r.fittestLap.toFixed(1)}s vs the field's best ${r.bestLapAny.toFixed(1)}s`);
        // A lap of this track at full speed is about 4s. Crawling used to give
        // 40s-plus here and still win.
        check(r.fittestLap < 25, 'and it is racing rather than crawling',
            `${r.fittestLap.toFixed(1)}s per lap`);
    }
}

console.log();
if (failures) { console.error(`${failures} race behaviour check(s) failed.`); process.exit(1); }
console.log('cars stay on the track, and the quick ones win.');
