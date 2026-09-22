// statemigrate.mjs — proves a slice of the population can be packed out of one
// wasm instance and unpacked into others mid-generation without any car
// noticing.
//
// This is what lets the worker pool change shape while a generation is
// running (the CPU-cores slider, and switching between the CPU and GPU
// backends): engine.js collects every worker's cars with pack_state(),
// re-slices the population across the new pool, and hands each new worker its
// cars with unpack_state(). If a single field were missed, or the active list
// weren't rebuilt from the crashed flags, the cars would carry on subtly
// differently — or crashed ones would come back to life — and nothing else in
// the suite would notice, because every other test runs one uninterrupted
// instance.
//
// The bar here is byte-identical: the same wasm code stepping the same state
// has to land on exactly the same bits whether or not the cars moved house
// halfway.
//
//   node tools/statemigrate.mjs [path/to/sim.wasm]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] || join(here, '..', 'wasm', 'sim.wasm');
const bytes = readFileSync(wasmPath);

let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

console.log(`state migration test (${wasmPath})\n`);

const CX = 600, CY = 450;
const path = Array.from({ length: 48 }, (_, i) => {
    const a = (Math.PI * 2 * i) / 48, r = 340 + Math.sin(a * 3) * 70;
    return { x: Math.round(CX + Math.cos(a) * r), y: Math.round(CY + Math.sin(a) * r) };
});
const POP = 64, H = 5, SEED = 20260922;

async function makeInstance() {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const w = instance.exports;
    const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);
    const u32 = (p, n) => new Uint32Array(w.memory.buffer, p, n);
    const pIn = f32(w.path_in_ptr(), path.length * 4);
    path.forEach((p, i) => { pIn[i*4] = p.x; pIn[i*4+1] = p.y; pIn[i*4+2] = 1; pIn[i*4+3] = 60; });
    w.set_config(10, 0.05, 0.02, 0.05, 750, 3, 0.2, H, 0);
    if (!w.track_build(w.path_in_ptr(), path.length, 60, 0, 0, 0, 0, 0, w.zone_in_ptr(), 0, 0, 0)) {
        throw new Error('track_build failed');
    }
    return { w, f32, u32 };
}

const WORDS = 32;
const snapshot = (inst, n) => {
    inst.w.pack_state();
    return inst.u32(inst.w.state_ptr(), n * WORDS).slice();
};
const brainsOf = (inst, n) => inst.f32(inst.w.brains_ptr(), n * inst.w.brain_stride()).slice();

// Loads `count` cars (starting at global car `start`) into a fresh instance,
// the way engine.js's 'import' does: size the slice, give it its brains, then
// overwrite every car with the packed state.
function importInto(inst, start, count, brains, state) {
    const w = inst.w;
    w.pop_init(count, start, H, 1);
    const stride = w.brain_stride();
    inst.f32(w.brains_ptr(), count * stride).set(brains.subarray(start * stride, (start + count) * stride));
    inst.u32(w.state_ptr(), count * WORDS).set(state.subarray(start * WORDS, (start + count) * WORDS));
    w.unpack_state();
}

const firstDiff = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return i; return -1; };

// A reference instance that never moves: 60 steps, then another 240.
const ref = await makeInstance();
check(ref.w.state_words() === WORDS, 'the state record is 32 words per car', `${ref.w.state_words()}`);
ref.w.pop_init(POP, 0, H, SEED);
ref.w.pop_randomize_brains();
ref.w.pop_reset();
ref.w.run(60);
const brains = brainsOf(ref, POP);
const mid = snapshot(ref, POP);
const aliveMid = ref.w.alive_count();

// ---------------------------------------------------------------------------
// 1. pack -> unpack in place is the identity, active list included.
// ---------------------------------------------------------------------------
{
    ref.w.unpack_state();
    const again = snapshot(ref, POP);
    const d = firstDiff(mid, again);
    check(d < 0, 'pack then unpack changes nothing', d < 0 ? `${POP} cars x ${WORDS} words` : `word ${d} differs`);
    check(ref.w.alive_count() === aliveMid, 'and rebuilds the same set of still-driving cars',
        `${ref.w.alive_count()} alive, expected ${aliveMid}`);
    check(aliveMid > 0 && aliveMid < POP, 'the snapshot is genuinely mid-generation (some crashed, some driving)',
        `${aliveMid}/${POP} alive after 60 steps`);
}

ref.w.run(240);
const refEnd = snapshot(ref, POP);

// ---------------------------------------------------------------------------
// 2. Moving the whole population to a fresh instance mid-run continues
//    bit-for-bit as if it had never moved.
// ---------------------------------------------------------------------------
{
    const b = await makeInstance();
    importInto(b, 0, POP, brains, mid);
    check(b.w.alive_count() === aliveMid, 'an imported slice drives exactly the cars that were still driving',
        `${b.w.alive_count()} alive`);
    b.w.run(240);
    const end = snapshot(b, POP);
    const d = firstDiff(refEnd, end);
    check(d < 0, 'moving every car to a new instance mid-generation changes nothing, bit for bit',
        d < 0 ? '240 more steps, identical' : `car ${Math.floor(d / WORDS)} word ${d % WORDS} differs`);
}

// ---------------------------------------------------------------------------
// 3. Splitting the population across two instances (a grow from one core to
//    two) also continues bit-for-bit — each car only ever depends on its own
//    state, its own brain and the track.
// ---------------------------------------------------------------------------
{
    const half = POP / 2;
    const b1 = await makeInstance(), b2 = await makeInstance();
    importInto(b1, 0, half, brains, mid);
    importInto(b2, half, POP - half, brains, mid);
    b1.w.run(240); b2.w.run(240);
    const joined = new Uint32Array(POP * WORDS);
    joined.set(snapshot(b1, half), 0);
    joined.set(snapshot(b2, POP - half), half * WORDS);
    const d = firstDiff(refEnd, joined);
    check(d < 0, 'splitting the population across two instances mid-generation changes nothing either',
        d < 0 ? `${half} + ${POP - half} cars, identical after 240 steps` : `car ${Math.floor(d / WORDS)} word ${d % WORDS} differs`);
}

// ---------------------------------------------------------------------------
// 4. track_info describes the track a second implementation will simulate.
// ---------------------------------------------------------------------------
{
    const w = ref.w;
    const t = new Uint32Array(w.memory.buffer, w.track_info(), 32);
    const tf = new Float32Array(w.memory.buffer, w.track_info(), 32);
    const cpN = t[0], bucketWalls = t[7];
    const wbs = new Int32Array(w.memory.buffer, t[10], cpN + 1);
    check(cpN === w.track_cp_count() && t[1] === w.track_start_cp() && t[3] === w.track_centerline_count(),
        'track_info agrees with the individual accessors', `${cpN} gates, start gate ${t[1]}, ${t[3]} centreline samples`);
    check(t[8] === 1 && t[9] === 1 && bucketWalls > 0 && wbs[cpN] === bucketWalls && bucketWalls % 4 === 0,
        'and describes a built road grid and wall buckets', `${bucketWalls} bucketed walls, padded to a multiple of 4`);
    check(Math.abs(tf[30] - w.sensor_len()) < 1e-6 && tf[28] > 0 && tf[29] > 0,
        'with the sensor reach, track length and widest half-width the step reads',
        `sensor ${tf[30]}, length ${tf[28].toFixed(1)}, max half-width ${tf[29].toFixed(1)}`);
}

console.log(failures ? `\n${failures} check(s) failed.` : '\ncars move between instances without noticing.');
process.exit(failures ? 1 : 0);
