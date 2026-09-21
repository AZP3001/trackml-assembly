// checkpoints.mjs — where the gates go.
//
// Gates used to be laid down at a fixed spacing along the centre line, which
// meant a corner got one wherever the spacing happened to land. On a long
// sweeper the nearest gate could be most of the way round the bend, so progress
// through the turn was invisible to the simulation.
//
// Corners are now ANCHORS: every corner gets a gate at its apex, and the
// regular gates fill the runs between them.
//
//   node tools/checkpoints.mjs [path/to/sim.wasm]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] || join(here, '..', 'wasm', 'sim.wasm');
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports;
const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);
const i32 = (p, n) => new Int32Array(w.memory.buffer, p, n);

const CP_STRIDE = 7;          // p1x p1y p2x p2y cx cy apex
const MIN_GAP = 18;           // TRACK_CP_MIN_GAP in sim.c
const SPACING = 34;           // TRACK_CP_SPACING in sim.c

let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

function build(path, halfWidth) {
    const pIn = f32(w.path_in_ptr(), path.length * 4);
    path.forEach((p, i) => {
        pIn[i * 4] = p.x; pIn[i * 4 + 1] = p.y;
        pIn[i * 4 + 2] = p.type === 'rounded' ? 0 : 1;
        pIn[i * 4 + 3] = p.radius ?? 60;
    });
    if (!w.track_build(w.path_in_ptr(), path.length, halfWidth, 0, 0, 0, 0, 0, w.zone_in_ptr(), 0, 0, 0))
        throw new Error('track_build failed');

    const n = w.track_cp_count();
    const cf = f32(w.track_cps_ptr(), n * CP_STRIDE);
    const ci = i32(w.track_cps_ptr(), n * CP_STRIDE);
    const gates = [];
    for (let i = 0; i < n; i++) {
        gates.push({
            i,
            p1: { x: cf[i * CP_STRIDE], y: cf[i * CP_STRIDE + 1] },
            p2: { x: cf[i * CP_STRIDE + 2], y: cf[i * CP_STRIDE + 3] },
            c: { x: cf[i * CP_STRIDE + 4], y: cf[i * CP_STRIDE + 5] },
            apex: !!ci[i * CP_STRIDE + 6]
        });
    }
    const nc = w.track_centerline_count();
    return {
        gates, n,
        centre: f32(w.track_centerline_ptr(), nc * 2).slice(),
        widths: f32(w.track_widths_ptr(), nc).slice(),
        nc
    };
}

// The point of the centre line closest to a corner's vertex IS the apex of the
// arc that rounds it — the fillet is a circle, so its nearest point to the
// vertex is its midpoint. That gives a way to say where a corner's gate ought
// to be without duplicating the arc solver here.
function apexNear(t, vx, vy) {
    let best = 0, bd = Infinity;
    for (let i = 0; i < t.nc; i++) {
        const d = Math.hypot(t.centre[i * 2] - vx, t.centre[i * 2 + 1] - vy);
        if (d < bd) { bd = d; best = i; }
    }
    return { x: t.centre[best * 2], y: t.centre[best * 2 + 1] };
}

function sampleSpacing(t) {
    let sum = 0;
    for (let i = 0; i < t.nc; i++) {
        const j = (i + 1) % t.nc;
        sum += Math.hypot(t.centre[j * 2] - t.centre[i * 2], t.centre[j * 2 + 1] - t.centre[i * 2 + 1]);
    }
    return sum / t.nc;
}

const poly = (sides, r, cx = 600, cy = 450) => Array.from({ length: sides }, (_, i) => {
    const a = (Math.PI * 2 * i) / sides - Math.PI / 2;
    return { x: Math.round(cx + Math.cos(a) * r), y: Math.round(cy + Math.sin(a) * r) };
});

console.log(`checkpoint placement test (${wasmPath})\n`);

// --- every corner gets a gate, at its apex -----------------------------
{
    const cases = [
        ['square',   [{ x: 250, y: 250 }, { x: 950, y: 250 }, { x: 950, y: 650 }, { x: 250, y: 650 }], 60],
        ['triangle', [{ x: 600, y: 180 }, { x: 980, y: 720 }, { x: 220, y: 720 }], 60],
        ['hexagon',  poly(6, 330), 60],
        ['L-shape',  [{ x: 250, y: 200 }, { x: 950, y: 200 }, { x: 950, y: 700 },
                      { x: 600, y: 700 }, { x: 600, y: 450 }, { x: 250, y: 450 }], 45],
        ['pentagon', poly(5, 320), 30]
    ];
    let allCovered = true, worstOffset = 0, detail = [];
    for (const [name, path, hw] of cases) {
        const t = build(path, hw);
        const spacing = sampleSpacing(t);
        const apexes = t.gates.filter(g => g.apex);
        let covered = 0;
        for (const v of path) {
            const want = apexNear(t, v.x, v.y);
            let bd = Infinity;
            for (const g of apexes) {
                const d = Math.hypot(g.c.x - want.x, g.c.y - want.y);
                if (d < bd) bd = d;
            }
            // Gates land on centre-line samples, so "at the apex" means within
            // about one sample of it.
            if (bd <= spacing * 1.5) { covered++; if (bd > worstOffset) worstOffset = bd; }
        }
        if (covered !== path.length) allCovered = false;
        detail.push(`${name} ${covered}/${path.length}`);
        console.log(`  ${name.padEnd(9)} ${String(t.n).padStart(3)} gates, ${String(apexes.length).padStart(2)} of them corner gates, ${covered}/${path.length} corners covered`);
    }
    check(allCovered, 'every corner gets a gate', detail.join(', '));
    check(worstOffset < 40, 'and it sits at the corner apex',
        `furthest any corner gate sits from its apex: ${worstOffset.toFixed(1)}px`);
}

// --- a right-angled track is built correctly at all four corners --------
// Regression guard for acos(-0.0). The dot product of two perpendicular unit
// vectors is exactly -0.0 in one of the four orientations, and an acos that
// tests `x < 0` for its quadrant returns -pi/2 there. That made one corner of
// every rectangle collapse and its neighbour balloon into a diagonal across the
// track — and it went unnoticed because the geometry still looked like *a*
// closed loop.
{
    const path = [{ x: 250, y: 250 }, { x: 950, y: 250 }, { x: 950, y: 650 }, { x: 250, y: 650 }];
    const t = build(path, 60);
    const dists = path.map(v => {
        let bd = Infinity;
        for (let i = 0; i < t.nc; i++) bd = Math.min(bd, Math.hypot(t.centre[i * 2] - v.x, t.centre[i * 2 + 1] - v.y));
        return bd;
    });
    const spread = Math.max(...dists) - Math.min(...dists);
    check(spread < 2, 'all four corners of a rectangle are rounded alike',
        `centre line passes ${dists.map(d => d.toFixed(0)).join('/')}px from the four vertices`);
}

// --- spacing stays sane -------------------------------------------------
{
    const cases = [
        ['square',  [{ x: 250, y: 250 }, { x: 950, y: 250 }, { x: 950, y: 650 }, { x: 250, y: 650 }], 60],
        ['polar48', Array.from({ length: 48 }, (_, i) => {
            const a = (Math.PI * 2 * i) / 48, r = 340 + Math.sin(a * 3) * 70;
            return { x: Math.round(600 + Math.cos(a) * r), y: Math.round(450 + Math.sin(a) * r) };
        }), 60],
        ['narrow',  poly(9, 300), 22]
    ];
    let worstMin = Infinity, worstMax = 0;
    for (const [name, path, hw] of cases) {
        const t = build(path, hw);
        for (let i = 0; i < t.n; i++) {
            const j = (i + 1) % t.n;
            const d = Math.hypot(t.gates[j].c.x - t.gates[i].c.x, t.gates[j].c.y - t.gates[i].c.y);
            if (d < worstMin) worstMin = d;
            if (d > worstMax) worstMax = d;
        }
    }
    check(worstMin >= MIN_GAP - 1, 'gates never crowd together',
        `tightest gap ${worstMin.toFixed(1)}px (floor ${MIN_GAP})`);
    // Gates snap to centre-line samples, which are up to 30px apart, so the
    // spacing cannot be exactly even; it just must not leave a hole.
    check(worstMax <= SPACING * 2, 'and never leave a large hole',
        `widest gap ${worstMax.toFixed(1)}px (limit ${SPACING * 2})`);
}

// --- gates span the road, corner gates included -------------------------
{
    const t = build(poly(6, 330), 55);
    let worst = 0;
    for (const g of t.gates) {
        const span = Math.hypot(g.p2.x - g.p1.x, g.p2.y - g.p1.y);
        // The gate is the full width of the road at that point. With auto width
        // off that is 2 x the requested half-width everywhere.
        worst = Math.max(worst, Math.abs(span - 110));
    }
    check(worst < 0.5, 'every gate spans the full width of the road',
        `worst deviation from 110px: ${worst.toFixed(2)}px`);
}

// --- a smooth loop: every sampled vertex is a turn, so all get gates -----
{
    const t = build(poly(24, 330), 60);
    const apexes = t.gates.filter(g => g.apex).length;
    check(apexes >= 20, 'a smoothly sampled loop gets a gate on every turn',
        `${apexes} corner gates for 24 vertices`);
}

console.log();
if (failures) { console.error(`${failures} checkpoint check(s) failed.`); process.exit(1); }
console.log('gates land on the corners.');
