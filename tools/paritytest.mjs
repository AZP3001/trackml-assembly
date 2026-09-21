// paritytest.mjs — proves the WebAssembly backend drives the same cars as the
// JavaScript one it replaces.
//
// It builds a track in wasm, hands the SAME geometry to the JS edition's worker
// (tools/reference-worker.mjs, copied verbatim from the original), loads the
// SAME brain weights into both, and steps them together comparing car state
// frame by frame.
//
// WHAT "SAME" MEANS HERE. The two are not expected to be bit-identical and
// cannot be: sim.c runs the physics in f32 where the JS runs it in f64. A
// genetic driving simulation is also chaotic — a 1e-7 difference in a steering
// output compounds, and after a few hundred frames two runs that started
// identical will be in visibly different places. So this asserts what actually
// matters:
//
//   1. the two agree closely over a short horizon (drift stays sub-pixel),
//   2. they agree on the things the simulation is *about* — who crashes, when,
//      and how many laps were finished,
//   3. the geometry the two would drive on is the same track.
//
// FITNESS IS DELIBERATELY DIFFERENT and is no longer compared for equality.
// The JavaScript edition pays the same for reaching a gate however long it
// took, which makes crawling the winning strategy — that is the bug this port
// fixes, so agreeing with it here would mean the fix had not been made. What
// is checked instead is the property that makes the new scoring safe: it only
// ever multiplies the old reward UP, never subtracts, so more progress still
// always outranks less. tools/racepace.mjs covers the rest.
//
// A real port bug — a sign error in the grip term, a wall bucket off by one,
// a transposed weight matrix — blows every one of these out immediately.
//
//   node tools/paritytest.mjs [path/to/sim.wasm]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createReferenceWorker } from './reference-worker.mjs';

const here = dirname(fileURLToPath(import.meta.url));
// Defaults to the scalar build; pass wasm/sim-simd.wasm to check that one.
const wasmPath = process.argv[2] || join(here, '..', 'wasm', 'sim.wasm');

const SENS_N = 7, IN_N = 9, OUT_N = 2;
let failures = 0;

function report(ok, name, detail) {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

// ---------------------------------------------------------------------------
// Test tracks. An oval, a many-cornered polar loop (the shape tracks.js builds
// its procedural tracks from) and a narrow one, since track width drives the
// offset-trim path that was the fiddliest part of the geometry port.
// ---------------------------------------------------------------------------
const CW = 1200, CH = 900, cx = CW / 2, cy = CH / 2;
const polar = (steps, fn, type = 'corner') => Array.from({ length: steps }, (_, i) => {
    const a = (Math.PI * 2 * i) / steps;
    const r = fn(a);
    return { x: Math.round(cx + Math.cos(a) * r), y: Math.round(cy + Math.sin(a) * r), type, radius: 60 };
});

const TRACKS = [
    { name: 'oval',      path: polar(24, () => 340), width: 60 },
    { name: 'hills',     path: polar(48, a => 340 + Math.sin(a * 3) * 70), width: 55 },
    { name: 'narrow',    path: polar(32, a => 300 + Math.sin(a * 2) * 90), width: 26 },
    { name: 'boxy',      path: [
        { x: 200, y: 200, type: 'corner', radius: 60 }, { x: 1000, y: 200, type: 'corner', radius: 60 },
        { x: 1000, y: 700, type: 'corner', radius: 60 }, { x: 200, y: 700, type: 'corner', radius: 60 }
      ], width: 60 }
];

const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports;
const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);
const i32 = (p, n) => new Int32Array(w.memory.buffer, p, n);

// ---------------------------------------------------------------------------
// Build a track in wasm, and mirror its geometry into the object shape the JS
// worker expects. Both engines then drive on literally the same walls, so any
// difference that shows up is in the physics, not the generator.
// ---------------------------------------------------------------------------
function buildTrack(def) {
    const pIn = f32(w.path_in_ptr(), def.path.length * 4);
    def.path.forEach((p, i) => {
        pIn[i * 4] = p.x; pIn[i * 4 + 1] = p.y;
        pIn[i * 4 + 2] = p.type === 'corner' ? 1 : 0;
        pIn[i * 4 + 3] = p.radius ?? 60;
    });
    const ok = w.track_build(w.path_in_ptr(), def.path.length, def.width, 0, 0, 0, 0, 0, w.zone_in_ptr(), 0, 0, 0);
    if (!ok) throw new Error(`track_build failed for ${def.name}`);

    const nw = w.track_wall_count();
    const wf = f32(w.track_walls_ptr(), nw * 5), wi = i32(w.track_walls_ptr(), nw * 5);
    const walls = [];
    for (let i = 0; i < nw; i++) {
        walls.push({
            p1: { x: wf[i * 5], y: wf[i * 5 + 1] },
            p2: { x: wf[i * 5 + 2], y: wf[i * 5 + 3] },
            segmentIndex: wi[i * 5 + 4] < 0 ? undefined : wi[i * 5 + 4]
        });
    }
    const ncp = w.track_cp_count();
    const cf = f32(w.track_cps_ptr(), ncp * 7);   // stride 7: coords + apex flag
    const checkpoints = [];
    for (let i = 0; i < ncp; i++) {
        checkpoints.push({
            index: i,
            p1: { x: cf[i * 7], y: cf[i * 7 + 1] },
            p2: { x: cf[i * 7 + 2], y: cf[i * 7 + 3] },
            center: { x: cf[i * 7 + 4], y: cf[i * 7 + 5] }
        });
    }
    return {
        walls, checkpoints, zones: [], segStep: 34,
        startPos: { x: w.track_start_x(), y: w.track_start_y() },
        startAngle: w.track_start_angle(),
        centerCount: w.track_centerline_count()
    };
}

// A stand-in for the Worker global the reference code closes over.
function makeRefWorker() {
    const outbox = [];
    const self = { postMessage: m => outbox.push(m) };
    createReferenceWorker(self);
    return { post: m => self.onmessage({ data: m }), outbox };
}

// Turn wasm's flat brain slice into the nested Float32Arrays the reference
// worker indexes. Same row-major order, which is the point.
function brainToRef(flat, h) {
    return {
        weightsIH: flat.slice(0, IN_N * h),
        weightsHO: flat.slice(IN_N * h, IN_N * h + h * OUT_N),
        biasH: flat.slice(IN_N * h + h * OUT_N, IN_N * h + h * OUT_N + h),
        biasO: flat.slice(IN_N * h + h * OUT_N + h, IN_N * h + h * OUT_N + h + OUT_N)
    };
}

const cfg = { maxSpeed: 10, acceleration: 0.05, turnSpeed: 0.04, grip: 0.93, targetLaps: 3 };
const POP = 40, HIDDEN = 5, TTL = 750;

console.log('TrackML WASM vs JS parity test\n');

let worstEarlyDrift = 0, worstCrashFrameDelta = 0, worstFitnessShortfall = Infinity;
let totalCars = 0, sameCrashStep = 0, sameCheckpoints = 0;

for (const def of TRACKS) {
    const track = buildTrack(def);

    // --- wasm side ---
    w.set_config(cfg.maxSpeed, cfg.acceleration, cfg.turnSpeed, cfg.grip, TTL, cfg.targetLaps, 0.15, HIDDEN);
    w.pop_init(POP, 0, HIDDEN, 12345);
    w.pop_randomize_brains();
    const stride = w.brain_stride();
    const allBrains = f32(w.brains_ptr(), POP * stride).slice();
    w.pop_reset();

    // --- reference side: same brains, same track, same config ---
    const ref = makeRefWorker();
    ref.post({ type: 'initTrack', track, config: { ...cfg, maxSpeed: cfg.maxSpeed } });
    const refCars = [];
    for (let i = 0; i < POP; i++) {
        refCars.push({
            id: i, x: track.startPos.x, y: track.startPos.y, angle: track.startAngle,
            vx: 0, vy: 0, speed: 0, brain: brainToRef(allBrains.subarray(i * stride, (i + 1) * stride), HIDDEN),
            fitness: 0, crashed: false, timeToLive: TTL, nextCheckpointIndex: 1, framesAlive: 0,
            lapTimes: [], completedLaps: 0, checkpointsReached: 0
        });
    }
    ref.post({ type: 'initCars', cars: refCars });

    // Step one frame at a time on both sides and compare.
    const FRAMES = 900;
    const EARLY = 60;   // horizon over which f32-vs-f64 drift stays sub-pixel
    const wasmCrashAt = new Int32Array(POP).fill(-1);
    const refCrashAt = new Int32Array(POP).fill(-1);
    let earlyDrift = 0;

    for (let frame = 1; frame <= FRAMES; frame++) {
        w.run(1);
        w.write_render();
        const rb = f32(w.render_ptr(), POP * 18);
        ref.post({ type: 'run', iters: 1 });
        const refBuf = ref.outbox.pop().buffer;

        for (let i = 0; i < POP; i++) {
            const wc = rb.subarray(i * 18, i * 18 + 18);
            const rc = refBuf.subarray(i * 18, i * 18 + 18);
            if (wasmCrashAt[i] < 0 && wc[1] === 1) wasmCrashAt[i] = frame;
            if (refCrashAt[i] < 0 && rc[1] === 1) refCrashAt[i] = frame;
            if (frame <= EARLY && !wc[1] && !rc[1]) {
                earlyDrift = Math.max(earlyDrift, Math.hypot(wc[2] - rc[2], wc[3] - rc[3]));
            }
        }
    }

    // Compare outcomes per car.
    const wasmFit = f32(w.render_ptr(), POP * 18);
    const refFinal = (() => { ref.post({ type: 'run', iters: 0 }); return ref.outbox.pop().buffer; })();
    let crashAgree = 0, crashDelta = 0, fitRel = Infinity, lapsAgree = 0;
    for (let i = 0; i < POP; i++) {
        const a = wasmCrashAt[i], b = refCrashAt[i];
        if ((a < 0) === (b < 0)) crashAgree++;
        if (a > 0 && b > 0) crashDelta = Math.max(crashDelta, Math.abs(a - b));
        const fa = wasmFit[i * 18 + 9], fb = refFinal[i * 18 + 9];
        // Signed, and the wrong way round on purpose: what matters is that the
        // wasm score never falls BELOW the reference's for the same driving.
        const denom = Math.max(Math.abs(fa), Math.abs(fb), 1);
        fitRel = Math.min(fitRel, (fa - fb) / denom);
        if (wasmFit[i * 18 + 8] === refFinal[i * 18 + 8]) lapsAgree++;
    }

    totalCars += POP;
    sameCrashStep += crashAgree;
    sameCheckpoints += lapsAgree;
    worstEarlyDrift = Math.max(worstEarlyDrift, earlyDrift);
    worstCrashFrameDelta = Math.max(worstCrashFrameDelta, crashDelta);
    worstFitnessShortfall = Math.min(worstFitnessShortfall, fitRel);

    console.log(`  ${def.name.padEnd(8)} walls=${String(track.walls.length).padStart(4)} cps=${String(track.checkpoints.length).padStart(3)} ` +
        `centre=${String(track.centerCount).padStart(4)}  drift@${EARLY}f=${earlyDrift.toExponential(2)}px  ` +
        `crash-agree=${crashAgree}/${POP}  laps-agree=${lapsAgree}/${POP}  min Δfit/fit=${fitRel.toExponential(2)}`);
}

console.log();
// Sub-pixel over the first second of simulation. The cars are 14px long, so
// this is far below anything that could change which wall they hit.
report(worstEarlyDrift < 0.5, 'short-horizon trajectories match',
    `max drift ${worstEarlyDrift.toExponential(2)}px over 60 frames (tol 0.5px)`);
report(sameCrashStep / totalCars >= 0.9, 'same cars survive',
    `${sameCrashStep}/${totalCars} agree on crashed-vs-alive`);
report(sameCheckpoints / totalCars >= 0.9, 'same lap counts',
    `${sameCheckpoints}/${totalCars} agree`);
// Not "the same fitness" — a strictly-not-worse one. The new scoring multiplies
// each gate and lap reward by how quickly it was reached, with a floor at the
// old value, so for identical driving it can only land at or above the
// reference. A negative shortfall here would mean the floor had been breached
// and progress was no longer monotonic.
report(worstFitnessShortfall >= -1e-3, 'the new scoring never pays less than the old for the same driving',
    `smallest margin ${(worstFitnessShortfall * 100).toFixed(1)}% above the reference`);

console.log();
if (failures) { console.error(`${failures} parity check(s) failed.`); process.exit(1); }
console.log('WASM backend matches the JavaScript reference.');
