// Checks sim.wasm's hand-written transcendentals against the JS Math.* they
// replace. There is no libm in a freestanding wasm32 build, so sinf/cosf/atan2f
// /tanhf/acosf/exp in sim.c are ours — and a wrong polynomial coefficient there
// would not throw, it would just make cars steer slightly into walls. Run it
// from CI on every build.
//
//   node tools/mathtest.mjs [path/to/sim.wasm]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] || join(here, '..', 'wasm', 'sim.wasm');
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports;

let failures = 0;

// Everything the simulation consumes is an f32, so the bar is f32 precision
// (~1.2e-7 relative), not f64. Tolerances below are absolute unless noted.
//
// Both ends of the comparison are pinned to f32 deliberately. The ARGUMENT is
// rounded first, because the wasm export takes a float and would otherwise be
// asked about a slightly different number than Math.* got — at 2000 radians an
// f32 step is 1e-4 wide, which swamps any real error by three orders of
// magnitude. The RESULT is rounded too, since sim.c returns a float and that
// rounding is part of the contract, not an error to charge against it.
function check(name, tol, samples, jsFn, wasmFn) {
    let worst = 0, worstAt = null;
    for (const raw of samples) {
        const args = raw.map(Math.fround);
        const ref = Math.fround(jsFn(...args));
        const got = wasmFn(...args);
        const err = Math.abs(got - ref);
        if (err > worst) { worst = err; worstAt = args; }
    }
    const ok = worst <= tol;
    if (!ok) failures++;
    const at = worstAt ? ` at (${worstAt.map(v => v.toPrecision(6)).join(', ')})` : '';
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name.padEnd(10)} max|err| = ${worst.toExponential(3)}  (tol ${tol.toExponential(1)})${ok ? '' : at}`);
    return ok;
}

function linspace(a, b, n) {
    return Array.from({ length: n }, (_, i) => [a + (b - a) * (i / (n - 1))]);
}
function randRange(a, b, n, seed = 1) {
    // Deterministic LCG so a failure is reproducible.
    let s = seed >>> 0;
    return Array.from({ length: n }, () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return [a + (b - a) * (s / 4294967296)];
    });
}

console.log(`sim.wasm math self-test (${wasmPath})\n`);

// The car heading is never wrapped, so sin/cos have to survive a large argument
// that has accumulated over a long run. +-4000 rad is far past anything a real
// session reaches.
check('sin', 2e-7, [...linspace(-Math.PI * 2, Math.PI * 2, 4001), ...randRange(-4000, 4000, 20000, 7)],
    x => Math.sin(x), x => w.t_sin(x));
check('cos', 2e-7, [...linspace(-Math.PI * 2, Math.PI * 2, 4001), ...randRange(-4000, 4000, 20000, 11)],
    x => Math.cos(x), x => w.t_cos(x));

// acos feeds the corner-arc solver; its argument is a clamped dot product.
//
// SIGNED ZEROS ARE IN THE SAMPLES ON PURPOSE. The dot product of two
// perpendicular unit vectors lands on exactly -0.0 for one of the four
// orientations, and acos(-0.0) must be +pi/2 — an implementation that tests
// `x < 0` for the quadrant gets -pi/2, because -0.0 is not less than 0.0. That
// shipped, and it deleted a corner from every right-angled track. An even
// linspace never produces -0.0, so it never caught it.
{
    const samples = [...linspace(-1, 1, 20001), [0], [-0], [1], [-1]];
    // Math.fround(-0) is -0, so the negative zero survives into the call.
    check('acos', 3e-7, samples, x => Math.acos(x), x => w.t_acos(x));
    const negZero = w.t_acos(-0);
    check('acos(-0)', 3e-7, [[-0]], () => Math.PI / 2, () => negZero);
}

// tanh, the f64 one. Kept for anything that wants the full-precision version.
check('tanh', 2e-7, [...linspace(-12, 12, 20001), ...randRange(-40, 40, 5000, 13)],
    x => Math.tanh(x), x => w.t_tanh(x));

// The activation feedForward actually runs — (population x (hidden + 2)) times
// per frame, which makes it the most-executed transcendental in the project.
// It is f32 throughout and skips exp's f64 argument reduction, so it gets
// checked on its own rather than riding on the f64 one's result. Same 2e-7 bar:
// "it is only the activation" is not a reason to let it drift.
{
    const samples = [...linspace(-12, 12, 20001), ...randRange(-40, 40, 5000, 13)];
    check('tanh(nn)', 2e-7, samples, x => Math.tanh(x), x => w.t_tanh_nn(x));
    // And the four-lane form the SIMD build's hidden layer uses. Units in the
    // same layer must not disagree about their own activation depending on
    // which lane they landed in, so this is held to the stricter bar of being
    // bit-identical to the scalar one rather than merely close to Math.tanh.
    let lanesDiffer = 0;
    for (const [raw] of samples) {
        const x = Math.fround(raw);
        if (w.t_tanh_nn4(x) !== w.t_tanh_nn(x)) lanesDiffer++;
    }
    const ok = lanesDiffer === 0;
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${'tanh(x4)'.padEnd(10)} vector and scalar activations agree exactly  (${lanesDiffer} of ${samples.length} differ)`);
}

// atan2 over all four quadrants plus the axes.
{
    const pts = [];
    let s = 3;
    for (let i = 0; i < 40000; i++) {
        s = (s * 1664525 + 1013904223) >>> 0; const y = (s / 4294967296) * 2000 - 1000;
        s = (s * 1664525 + 1013904223) >>> 0; const x = (s / 4294967296) * 2000 - 1000;
        pts.push([y, x]);
    }
    for (const v of [0, 1, -1, 1e-6, -1e-6, 1000, -1000]) {
        pts.push([v, 0], [0, v], [v, v], [v, -v]);
    }
    check('atan2', 3e-7, pts, (y, x) => Math.atan2(y, x), (y, x) => w.t_atan2(y, x));
}

// exp is f64 internally (it is what tanh is built on), so it gets an f64 bar —
// relative rather than absolute, since it spans the whole double range.
{
    let worst = 0, worstAt = 0;
    for (const [x] of [...linspace(-700, 700, 20001), ...randRange(-40, 40, 20000, 17)]) {
        const want = Math.exp(x), got = w.t_exp(x);
        const rel = want === 0 ? Math.abs(got) : Math.abs((got - want) / want);
        if (rel > worst) { worst = rel; worstAt = x; }
    }
    const TOL = 1e-10;   // degree-9 truncation is ~7e-12; the rest is scaling
    const ok = worst <= TOL;
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${'exp'.padEnd(10)} max rel err = ${worst.toExponential(3)}  (tol ${TOL.toExponential(1)})${ok ? '' : ` at ${worstAt}`}`);
}

console.log();
if (failures) {
    console.error(`${failures} check(s) failed.`);
    process.exit(1);
}
console.log('all math checks passed.');
