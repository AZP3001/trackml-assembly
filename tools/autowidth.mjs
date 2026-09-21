// autowidth.mjs — checks the auto-width solver in sim.c.
//
// The feature: where two parts of the track run close enough together that the
// road would swallow the barrier between them, narrow the road instead. The
// slider blends between doing that only at the tight spots (local) and giving
// the whole track the narrowest width it needs anywhere (global).
//
// The two things that can go wrong are opposites, so both are tested:
//   * not narrowing a track that pinches, and
//   * narrowing a track that doesn't, which would wreck every normal track.
//
//   node tools/autowidth.mjs [path/to/sim.wasm]
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

// These mirror AUTO_GAP / AUTO_MIN_HALF in sim.c.
const GAP = 12, MIN_HALF = 11;

function build(path, halfWidth, auto = 0, blend = 0) {
    const pIn = f32(w.path_in_ptr(), path.length * 4);
    path.forEach((p, i) => {
        pIn[i * 4] = p.x; pIn[i * 4 + 1] = p.y;
        pIn[i * 4 + 2] = p.type === 'corner' ? 1 : 0;
        pIn[i * 4 + 3] = p.radius ?? 60;
    });
    const ok = w.track_build(w.path_in_ptr(), path.length, halfWidth,
        0, 0, 0, 0, 0, w.zone_in_ptr(), 0, auto, blend);
    if (!ok) throw new Error('track_build failed');
    const n = w.track_centerline_count();
    const widths = f32(w.track_widths_ptr(), n).slice();
    const centre = f32(w.track_centerline_ptr(), n * 2).slice();
    return {
        n, widths, centre, walls: w.track_wall_count(), cps: w.track_cp_count(),
        min: Math.min(...widths), max: Math.max(...widths)
    };
}

// A wedge corridor: 350px where the track runs back alongside itself only 70px
// away, then it opens out again. The parallel stretch is ~600px apart ALONG the
// track, so it is unmistakably two different parts of it rather than one corner
// curving round — exactly the case the feature is for. At a half-width of 40
// the two roads would be 80 across with 70 between their centres: overlapping,
// with no barrier left between them.
//
// The pinch is deliberately localised. A shape that is tight end to end (a
// plain paperclip) cannot distinguish local from global mode, because there is
// no un-pinched part left to stay wide.
const CORRIDOR_GAP = 70;
const corridor = [
    { x: 250, y: 200, type: 'corner' }, { x: 950, y: 200, type: 'corner' },
    { x: 950, y: 200 + CORRIDOR_GAP, type: 'corner' },
    { x: 600, y: 200 + CORRIDOR_GAP, type: 'corner' },
    { x: 250, y: 500, type: 'corner' }
];

// A plain oval — nothing anywhere near anything else. Auto width must leave it
// completely alone.
const cx = 600, cy = 450;
const oval = Array.from({ length: 32 }, (_, i) => {
    const a = (Math.PI * 2 * i) / 32;
    return { x: Math.round(cx + Math.cos(a) * 360), y: Math.round(cy + Math.sin(a) * 260), type: 'corner', radius: 60 };
});

console.log(`auto-width test (${wasmPath})\n`);

const REQ = 40;

// --- off: one width, everywhere, as before -----------------------------
{
    const r = build(corridor, REQ, 0, 0);
    check(r.min === REQ && r.max === REQ, 'auto off leaves the width alone',
        `all ${r.n} samples at ${r.min}`);
}

// --- local: narrow at the pinch, full width elsewhere ------------------
let localMin;
{
    const r = build(corridor, REQ, 1, 0);
    localMin = r.min;
    // Two roads of half-width h with 70 between their centres leave 70 - 2h of
    // barrier. Asking for GAP of it caps h at (70 - 12) / 2 = 29.
    const cap = (CORRIDOR_GAP - GAP) / 2;
    check(r.min < REQ - 1, 'a pinched track narrows', `narrowest ${r.min.toFixed(1)} (was ${REQ})`);
    check(r.min <= cap + 1.5, 'it narrows enough to leave a barrier',
        `${r.min.toFixed(1)} <= ${cap} + tolerance`);
    check(r.min >= MIN_HALF, 'it never narrows below a car', `${r.min.toFixed(1)} >= ${MIN_HALF}`);
    check(r.max > r.min + 1, 'only part of the track narrows',
        `${r.min.toFixed(1)} at the pinch, ${r.max.toFixed(1)} at the ends`);
    check(r.walls > 20 && r.cps > 5, 'the narrowed track is still a track',
        `${r.walls} walls, ${r.cps} checkpoints`);

    // The taper has to be smooth, or it shows as a notch cut into the asphalt.
    //
    // Measured as a RATE — how far the road edge moves sideways per pixel
    // travelled along the track — not as a change per sample. Per-sample would
    // just be measuring how finely the centreline happens to be resampled, and
    // would quietly pass or fail on track width rather than on smoothness.
    // 0.1 is a taper of under six degrees.
    let steepest = 0;
    for (let i = 0; i < r.n; i++) {
        const j = (i + 1) % r.n;
        const ds = Math.hypot(r.centre[j * 2] - r.centre[i * 2], r.centre[j * 2 + 1] - r.centre[i * 2 + 1]);
        if (ds < 1e-3) continue;
        const rate = Math.abs(r.widths[j] - r.widths[i]) / ds;
        if (rate > steepest) steepest = rate;
    }
    check(steepest < 0.1, 'the width tapers rather than stepping',
        `steepest taper ${steepest.toFixed(3)}px of width per px of track (${(Math.atan(steepest) * 180 / Math.PI).toFixed(1)} degrees)`);
}

// --- global: one width for the whole track -----------------------------
{
    const r = build(corridor, REQ, 1, 1);
    const spread = r.max - r.min;
    check(spread < 0.01, 'global mode gives the whole track one width',
        `every sample at ${r.min.toFixed(1)} (spread ${spread.toExponential(1)})`);
    check(Math.abs(r.min - localMin) < 0.01, 'and that width is the narrowest the track needs',
        `${r.min.toFixed(1)} matches the local minimum ${localMin.toFixed(1)}`);
}

// --- the slider in between ---------------------------------------------
{
    const half = build(corridor, REQ, 1, 0.5);
    const local = build(corridor, REQ, 1, 0);
    check(half.max < local.max && half.max > half.min,
        'a mid slider position sits between the two',
        `widest ${half.max.toFixed(1)} vs ${local.max.toFixed(1)} local / ${localMin.toFixed(1)} global`);
}

// --- a normal track must not be touched --------------------------------
{
    const off = build(oval, REQ, 0, 0);
    const on = build(oval, REQ, 1, 0);
    check(Math.abs(on.min - REQ) < 0.01 && Math.abs(on.max - REQ) < 0.01,
        'auto width leaves an ordinary track alone',
        `oval stays at ${on.min.toFixed(1)} with auto on`);
    check(on.walls === off.walls,
        'and produces identical geometry', `${on.walls} walls either way`);
}

// --- a track that genuinely crosses itself ------------------------------
// A figure eight has a real junction where the two roads are meant to merge.
// Right at the crossing the clearance reads ~0, and computing a width from
// that would floor the road at the minimum and put an absurd bottleneck in the
// middle of the junction. sim.c refuses to derive a width from a clearance
// below one half-width for exactly that reason; the approach to the crossing
// still narrows, which is what keeps the junction from opening into a plaza.
{
    const fig8 = Array.from({ length: 60 }, (_, i) => {
        const a = (Math.PI * 2 * i) / 60;
        return { x: Math.round(600 + Math.sin(a) * 380), y: Math.round(450 + Math.sin(a) * Math.cos(a) * 420) };
    });
    const off = build(fig8, REQ, 0, 0);
    const on = build(fig8, REQ, 1, 0);
    check(on.walls > 50 && on.cps === off.cps, 'a self-crossing track stays drivable',
        `${on.walls} walls, ${on.cps} checkpoints`);
    check(on.min >= MIN_HALF, 'the junction is not bottlenecked',
        `narrowest ${on.min.toFixed(1)} >= ${MIN_HALF}`);
    check(on.min < REQ && on.max >= REQ - 0.01, 'the approach to the crossing narrows, the rest does not',
        `${on.min.toFixed(1)} near the crossing, ${on.max.toFixed(1)} elsewhere`);
}

console.log();
if (failures) { console.error(`${failures} auto-width check(s) failed.`); process.exit(1); }
console.log('auto width behaves.');
