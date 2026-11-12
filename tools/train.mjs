// train.mjs — headless end-to-end training run, the way the app drives it.
//
// One wasm instance plays both the master (evolve) and the worker (run), and
// the generation loop mirrors app.evolve()/Engine.evolve(): the stash takes the
// best brain whenever it beats the all-time best, sigmaGen is derived from
// run history exactly as script.js does, and a generation ends when every car is out or one has completed the target laps.
//
// Reports, per built-in track, the generation of the first completed lap and
// the generation where a car first completed the full target. Also asserts the
// guarantees learning rests on: the all-time best never goes down, and no
// weight or car state ever goes non-finite.
//
//   node tools/train.mjs [wasm] [--gens N] [--seeds N] [--tracks 0,1,2] [--quiet]
//        [--pop N --elite N --hidden N --laps N --turn X] [--require-solved]
// Defaults are the phone column of the app's settings. --require-solved fails
// the run if any track/seed doesn't finish its laps within --gens.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const wasmPath = args[0] && !args[0].startsWith('--') ? args[0] : join(here, '..', 'wasm', 'sim.wasm');
const GENS = Number(opt('--gens', 60));
const SEEDS = Number(opt('--seeds', 3));
const QUIET = args.includes('--quiet'), REQUIRE = args.includes('--require-solved');
const POP = Number(opt('--pop', 150)), ELITE = Number(opt('--elite', 15)), TARGET = Number(opt('--laps', 2)), HIDDEN = Number(opt('--hidden', 4));
const TURN = Number(opt('--turn', 0.01)), BRAKE = Number(opt('--brake', 0.01));
const MUT = [Number(opt('--mut-rate', 0.15)), Number(opt('--sigma', 0.5)), Number(opt('--sigma-floor', 0.05))];
const FOCUS = Number(opt('--focus', 0)), TTL = Number(opt('--ttl', 750));
const JSON_OUT = args.includes('--json');
// --opt "elite=30,mut-rate=0.1,sigma=0.2,sigma-floor=0.05,focus=0.2,laps=3":
// settings switched to the generation after a car first completes the target
// laps — the app's Explore -> Optimize hand-over.
const OPT = Object.fromEntries((opt('--opt', '') || '').split(',').filter(Boolean).map(kv => { const [k, v] = kv.split('='); return [k, Number(v)]; }));
const SETTLE = 15;   // app.SETTLE_GENERATIONS
const CHUNK = 2500;  // HYPER_CHUNK

// The app's own track list, built from tracks.js with a stub generator that
// just keeps the arguments.
const ctx = { Math, Array, JSON };
vm.createContext(ctx);
vm.runInContext(readFileSync(join(here, '..', 'tracks.js'), 'utf8') + '\nthis.getDefaultTracks = getDefaultTracks;', ctx);
const defs = ctx.getDefaultTracks((id, name, path, width, startPos, startAngle, zones = []) =>
    ({ id, name, path, width, startPos, startAngle, zones }), 1200, 900);
const trackIdx = (opt('--tracks', '') || defs.map((_, i) => i).join(',')).split(',').map(Number);

const mod = new WebAssembly.Module(readFileSync(wasmPath));
const ZID = { speed: 0, precision: 1, focus: 2, spawnkill: 3 };

function setup(def) {
    const w = new WebAssembly.Instance(mod, {}).exports;
    const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);
    const pIn = f32(w.path_in_ptr(), def.path.length * 4);
    def.path.forEach((p, i) => {
        pIn[i * 4] = p.x; pIn[i * 4 + 1] = p.y;
        pIn[i * 4 + 2] = p.type === 'corner' ? 1 : 0; pIn[i * 4 + 3] = p.radius !== undefined ? p.radius : 60;
    });
    const zones = def.zones || [];
    const zIn = f32(w.zone_in_ptr(), Math.max(1, zones.length * 5));
    zones.forEach((z, i) => { zIn.set([z.x, z.y, z.radius, ZID[z.type] ?? -1, z.killTimer ?? 150], i * 5); });
    const hasStart = def.startPos ? 1 : 0, hasAngle = def.startAngle != null ? 1 : 0;
    w.set_config(10, 0.05, TURN, BRAKE, TTL, TARGET, FOCUS, HIDDEN, 0);
    w.set_mutation(...MUT);
    if (!w.track_build(w.path_in_ptr(), def.path.length, def.width, hasStart, hasStart ? def.startPos.x : 0,
        hasStart ? def.startPos.y : 0, hasAngle, hasAngle ? def.startAngle : 0, w.zone_in_ptr(), zones.length, 0, 0)) {
        throw new Error(`track_build failed: ${def.name}`);
    }
    return w;
}

const allFinite = (a) => { for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false; return true; };

function train(def, seed) {
    const w = setup(def);
    const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);
    w.pop_init(POP, 0, HIDDEN, seed);
    w.pop_randomize_brains();
    w.pop_reset();
    let globalBest = -Infinity, firstLap = null, solved = null, lapGens = 0, problems = [];
    const bests = [], trace = [];
    let wallMs = 0, carSteps = 0, frames = 0, bestLapEver = Infinity;
    let elite = ELITE, target = TARGET, switchGen = null;
    for (let gen = 1; gen <= GENS; gen++) {
        let maxLaps = 0;
        const t0 = performance.now();
        for (let guard = 0; guard < 100; guard++) {
            maxLaps = w.run(CHUNK);
            carSteps += w.car_steps(); frames += w.run_iters();
            if (w.all_crashed() === 1 || maxLaps >= target) break;
        }
        w.write_fitness();
        const stride = w.fitness_stride();
        const fb = f32(w.fitness_ptr(), POP * stride);
        const ev = f32(w.ev_fitness_ptr(), POP);
        let best = -Infinity, bestIdx = 0, laps = 0;
        for (let i = 0; i < POP; i++) {
            ev[i] = fb[i * stride];
            if (ev[i] > best) { best = ev[i]; bestIdx = i; }
            laps = Math.max(laps, fb[i * stride + 1]);
            if (fb[i * stride + 1] >= 1 && fb[i * stride + 2] > 0) bestLapEver = Math.min(bestLapEver, fb[i * stride + 2]);
        }
        if (!allFinite(ev)) problems.push(`gen ${gen}: non-finite fitness`);
        w.pack_state();
        if (!allFinite(f32(w.state_ptr(), POP * w.state_words()).filter((_, k) => { const m = k % 32; return m <= 6 || m === 13 || m >= 18 && m < 31; })))
            problems.push(`gen ${gen}: non-finite car state`);
        if (laps >= 1) { if (firstLap === null) firstLap = gen; lapGens++; }
        if (laps >= TARGET && solved === null) solved = gen;
        if (solved !== null && switchGen === null && Object.keys(OPT).length) {
            switchGen = gen;
            const o = { elite: ELITE, 'mut-rate': MUT[0], sigma: MUT[1], 'sigma-floor': MUT[2], focus: FOCUS, laps: TARGET, ttl: TTL, ...OPT };
            elite = o.elite; target = o.laps;
            w.set_mutation(o['mut-rate'], o.sigma, o['sigma-floor']);
            w.set_config(10, 0.05, TURN, BRAKE, o.ttl, target, o.focus, HIDDEN, 0);
        }
        if (best > globalBest) { w.copy_brain(bestIdx, w.stash_slot()); globalBest = best; }
        bests.push(globalBest);
        const sigmaGen = firstLap === null ? 0 : Math.max(0, gen - firstLap - SETTLE);
        w.evolve(elite, 1, sigmaGen);
        if (!allFinite(f32(w.brains_ptr(), (w.max_cars() + 1) * w.brain_stride()).subarray(0, POP * w.brain_stride())))
            problems.push(`gen ${gen}: non-finite weights`);
        w.pop_reset();
        wallMs += performance.now() - t0;
        trace.push({ gen, wallMs, carSteps, frames, laps, bestLap: Number.isFinite(bestLapEver) ? bestLapEver : null });
    }
    for (let i = 1; i < bests.length; i++) if (bests[i] < bests[i - 1]) problems.push(`all-time best fell at gen ${i + 1}`);
    return { firstLap, solved, switchGen, best: globalBest, bests, problems, trace };
}

let failures = 0;
const summary = [];
console.log(`train: ${wasmPath} — pop ${POP}, elite ${ELITE}, hidden ${HIDDEN}, target ${TARGET} laps, ${GENS} gens, ${SEEDS} seeds\n`);
for (const ti of trackIdx) {
    const def = defs[ti];
    const runs = [];
    for (let s = 0; s < SEEDS; s++) {
        const r = train(def, 1000 + s * 7919);
        runs.push(r);
        for (const p of r.problems) { failures++; console.log(`FAIL  ${def.name} seed ${s}: ${p}`); }
    }
    const fmt = (v) => (v === null ? '—' : String(v)).padStart(4);
    const line = `${def.name.padEnd(22)} first lap: ${runs.map(r => fmt(r.firstLap)).join(' ')}   ${TARGET}-lap finish: ${runs.map(r => fmt(r.solved)).join(' ')}`;
    summary.push({ name: def.name, runs });
    if (!QUIET) console.log(line);
}
if (JSON_OUT) console.log('JSON ' + JSON.stringify(summary.map(s => ({ name: s.name, runs: s.runs.map(r => ({ solved: r.solved, switchGen: r.switchGen, trace: r.trace })) }))));
const solvedRuns = summary.flatMap(s => s.runs).filter(r => r.solved !== null);
const total = summary.length * SEEDS;
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
console.log(`\nsolved ${solvedRuns.length}/${total} runs; median generation to a ${TARGET}-lap finish: ${med(solvedRuns.map(r => r.solved))}; median first lap: ${med(summary.flatMap(s => s.runs).filter(r => r.firstLap !== null).map(r => r.firstLap))}`);
if (REQUIRE && solvedRuns.length < total) { failures++; console.log(`FAIL  ${total - solvedRuns.length} run(s) did not finish within ${GENS} generations`); }
console.log(failures ? `\n${failures} problem(s)` : '\nno NaN/Inf, all-time best monotone in every run');
process.exitCode = failures ? 1 : 0;
