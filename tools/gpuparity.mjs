// gpuparity.mjs — holds the WebGPU backend (gpu-worker.js) to the wasm one.
//
// The GPU build is a second implementation of the same step, in a different
// language with different arithmetic (f32 only, no f64 argument reduction),
// so it cannot match the CPU bit for bit — a car's heading is carried to the
// last f32 bit on both, but one side computes sin() through doubles. What it
// must do is compute the same function: from the SAME state, one step on each
// has to land on the same car to within f32 rounding, crash the same cars,
// pass the same gates, and stop a generation at the same frame. That is what
// this checks, directly, by running a CPU worker and a GPU worker side by
// side in a real browser and moving state between them:
//
//   1. single steps from identical state, at many points in a run, for every
//      car, across population sizes (so every thread-per-car split the GPU
//      picks gets exercised) and brain sizes up to the slider's maximum — the
//      tight bound;
//   2. how far the two drift apart over a longer uninterrupted run — loose,
//      since any chaotic system amplifies rounding differences, but it has to
//      stay the same race;
//   3. the lockstep rules: a chunk stops at the step a car completes the
//      target lap count, for every car, and reports allCrashed / maxLaps /
//      carSteps exactly the way run() in sim.c does;
//   4. splitting a car's wall scan across threads is bit-identical to not
//      splitting it (min and "any hit" don't depend on order);
//   5. import then export through the GPU worker is lossless.
//
// Headless Chromium has no hardware GPU here; --enable-unsafe-webgpu gives it
// SwiftShader's software Vulkan, which runs the same WGSL through the same
// validation, just slowly. If no adapter is available at all the test says so
// and exits cleanly rather than failing a machine that simply has no WebGPU.
//
//   npx http-server -p 8765 . &
//   node tools/gpuparity.mjs [http://127.0.0.1:8765]
import { chromium } from 'playwright';

const BASE = process.argv[2] || 'http://127.0.0.1:8765';
let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

const browser = await chromium.launch({
    ...(process.env.PLAYWRIGHT_CHROMIUM_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH } : {}),
    args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
await page.goto(`${BASE}/README.md`, { waitUntil: 'load' });

console.log(`WebGPU / wasm parity test against ${BASE}\n`);

const hasGpu = await page.evaluate(async () => !!(navigator.gpu && await navigator.gpu.requestAdapter()));
if (!hasGpu) {
    console.log('skip  no WebGPU adapter in this browser — nothing to compare');
    await browser.close();
    process.exit(0);
}

// The in-page harness: a CPU worker and a GPU worker, and helpers to drive
// them with identical messages and compare what comes back.
await page.evaluate(() => {
    const W = 32;
    function wrap(url) {
        const w = new Worker(url);
        const inbox = [], waiters = [];
        w.onmessage = e => {
            const d = e.data;
            const i = waiters.findIndex(x => x.type === d.type);
            if (i >= 0) waiters.splice(i, 1)[0].resolve(d); else inbox.push(d);
        };
        return {
            post(m, t) { w.postMessage(m, t || []); },
            next(type) {
                const i = inbox.findIndex(d => d.type === type);
                if (i >= 0) return Promise.resolve(inbox.splice(i, 1)[0]);
                return new Promise(resolve => waiters.push({ type, resolve }));
            },
            terminate() { w.terminate(); }
        };
    }

    // A deterministic stream, so every run of the test builds the same field.
    let seed = 12345;
    const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);

    const path = Array.from({ length: 48 }, (_, i) => {
        const a = (Math.PI * 2 * i) / 48, r = 340 + Math.sin(a * 3) * 70;
        return { x: Math.round(600 + Math.cos(a) * r), y: Math.round(450 + Math.sin(a) * r), type: 'corner', radius: 60 };
    });
    const at = i => ({ x: path[i].x, y: path[i].y });
    const def = {
        path, width: 60, zones: [
            { ...at(6), radius: 70, type: 'speed' },
            { ...at(14), radius: 70, type: 'precision' },
            { ...at(22), radius: 70, type: 'focus' },
            { ...at(40), radius: 50, type: 'spawnkill', killTimer: 5000 }
        ], autoWidth: false, autoWidthBlend: 0
    };
    const IN = 11;

    // Half the field is a hand-built "steer at the next gate, hold a target
    // speed" driver with its knobs jittered (they lap the track, so gates,
    // laps and the lap-target stop all get exercised); the other half is
    // random weights, which crash in every way there is.
    function makeBrains(n) {
        const H = P.H, stride = P.stride;
        const b = new Float32Array(n * stride);
        const offHO = IN * H, offBH = offHO + 2 * H;
        for (let c = 0; c < n; c++) {
            const o = c * stride;
            if (c % 2 === 0) {
                b[o + 8 * H] = 3 + rnd() * 3;
                b[o + 7 * H + 1] = -3;
                b[o + offBH + 1] = 0.6 + rnd() * 1.2;
                b[o + offHO] = 3;
                b[o + offHO + 3] = 2;
                for (let j = 0; j < 7; j++) b[o + j * H] += (rnd() - 0.5) * 0.4;
            } else {
                for (let j = 0; j < stride; j++) b[o + j] = (rnd() * 2 - 1) * 1.2;
            }
        }
        return b;
    }

    window.P = {
        W, def,
        H: 2,
        get stride() { return IN * this.H + this.H * 2 + this.H + 2; },
        get cfg() { return [10, 0.05, 0.02, 0.05, 750, 3, 0.2, this.H, 0]; },
        makeBrains,
        async boot() {
            this.mod = await WebAssembly.compile(await (await fetch('./wasm/sim.wasm')).arrayBuffer());
            this.cpu = wrap('./sim-worker.js');
            this.gpu = wrap('./gpu-worker.js');
            this.gpu2 = wrap('./gpu-worker.js');
            for (const w of [this.cpu, this.gpu, this.gpu2]) w.post({ type: 'init', module: this.mod, index: 0 });
            await this.cpu.next('ready');
            const g = await Promise.race([this.gpu.next('ready'), this.gpu.next('gpu-error')]);
            if (g.type !== 'ready') throw new Error('gpu-worker failed: ' + g.message);
            await this.gpu2.next('ready');
            return g.gpu;
        },
        setup(cfg) {
            for (const w of [this.cpu, this.gpu, this.gpu2]) w.post({ type: 'track', def: this.def, config: cfg || this.cfg });
        },
        pop(w, n, brains, extra) {
            const focused = new Int32Array(n);
            for (let c = 0; c < n; c++) focused[c] = c % 3 === 0 ? 1 : 0;
            w.post({ type: 'pop', start: 0, count: n, hidden: P.H, seed: 1, brains: brains.slice(), focused,
                     focusLo: 10, focusHi: 30, ...(extra || {}) });
        },
        async state(w) { w.post({ type: 'export' }); return (await w.next('exported')).state; },
        importState(w, n, brains, state, extra) {
            w.post({ type: 'import', start: 0, count: n, hidden: P.H, seed: 1, brains: brains.slice(),
                     focusLo: 10, focusHi: 30, state: state.slice(), gateRatio: null, ...(extra || {}) });
        },
        // A whole chunk, however long it takes — the GPU worker's per-round
        // time budget would otherwise cut SwiftShader's (slow) rounds short.
        async run(w, iters, wantRender) {
            w.post({ type: 'run', iters, wantRender: !!wantRender, budgetMs: 1e9 });
            return await w.next('done');
        },
        // Field-by-field comparison of two exported populations.
        compare(a, b, n) {
            const fa = new Float32Array(a.buffer), fb = new Float32Array(b.buffer);
            const ia = new Int32Array(a.buffer), ib = new Int32Array(b.buffer);
            const r = { crash: 0, ints: 0, roadSeg: 0, pos: 0, angle: 0, speed: 0, fit: 0, inp: 0, out: 0, alive: 0 };
            for (let c = 0; c < n; c++) {
                const o = c * W;
                if (ia[o + 7] !== ib[o + 7]) { r.crash++; continue; }
                for (const k of [8, 9, 10, 11, 12, 14, 15]) if (ia[o + k] !== ib[o + k]) { r.ints++; break; }
                if (ia[o + 16] !== ib[o + 16]) r.roadSeg++;
                if (ia[o + 7] === 0) r.alive++;
                r.pos = Math.max(r.pos, Math.abs(fa[o] - fb[o]), Math.abs(fa[o + 1] - fb[o + 1]));
                r.angle = Math.max(r.angle, Math.abs(fa[o + 2] - fb[o + 2]));
                r.speed = Math.max(r.speed, Math.abs(fa[o + 5] - fb[o + 5]));
                r.fit = Math.max(r.fit, Math.abs(fa[o + 6] - fb[o + 6]) / Math.max(1, Math.abs(fa[o + 6])));
                for (let k = 20; k < 31; k++) r.inp = Math.max(r.inp, Math.abs(fa[o + k] - fb[o + k]));
                r.out = Math.max(r.out, Math.abs(fa[o + 18] - fb[o + 18]), Math.abs(fa[o + 19] - fb[o + 19]));
            }
            return r;
        }
    };
});

const adapter = await page.evaluate(() => P.boot());
console.log(`  adapter: ${adapter}\n`);
await page.evaluate(() => P.setup());

// ---------------------------------------------------------------------------
// 1. One step from identical state, all over a run, every car.
// ---------------------------------------------------------------------------
// Stock-sized brains at every thread split, plus the largest the slider
// allows (9 hidden units), which is where an indexing slip in the network
// would show.
for (const [n, h] of [[64, 2], [200, 2], [600, 2], [100, 9], [300, 5]]) {
    const r = await page.evaluate(async ([n, h]) => {
        P.H = h;
        P.setup();
        const brains = P.makeBrains(n);
        P.pop(P.cpu, n, brains);
        const worst = { crash: 0, ints: 0, roadSeg: 0, pos: 0, angle: 0, speed: 0, fit: 0, inp: 0, out: 0, samples: 0, alive: 0 };
        let steps = 0;
        for (const until of [0, 1, 5, 20, 60, 150, 300, 600]) {
            if (until > steps) { await P.run(P.cpu, until - steps); steps = until; }
            const s = await P.state(P.cpu);
            P.importState(P.gpu, n, brains, s);
            await P.run(P.cpu, 1); await P.run(P.gpu, 1); steps++;
            const a = await P.state(P.cpu), b = await P.state(P.gpu);
            const c = P.compare(a, b, n);
            for (const k of ['crash', 'ints', 'roadSeg', 'alive']) worst[k] += c[k];
            for (const k of ['pos', 'angle', 'speed', 'fit', 'inp', 'out']) worst[k] = Math.max(worst[k], c[k]);
            worst.samples += n;
            // Re-sync the CPU to exactly where the next sample starts from.
            P.importState(P.cpu, n, brains, a);
        }
        return worst;
    }, [n, h]);
    const lanes = n <= 128 ? 8 : n <= 256 ? 4 : n <= 512 ? 2 : 1;
    check(r.crash === 0 && r.ints === 0,
        `one step from identical state crashes the same cars and passes the same gates (${n} cars, ${h} hidden, ${lanes} thread${lanes > 1 ? 's' : ''}/car)`,
        `${r.samples} car-steps compared, ${r.alive} still driving; ${r.crash} crash / ${r.ints} counter disagreements`);
    check(r.pos < 2e-3 && r.angle < 2e-5 && r.speed < 2e-4 && r.out < 2e-4 && r.inp < 2e-3 && r.fit < 2e-5,
        '  and lands on the same car to within f32 rounding',
        `worst: pos ${r.pos.toExponential(1)}px, heading ${r.angle.toExponential(1)}rad, speed ${r.speed.toExponential(1)}, ` +
        `sensors/inputs ${r.inp.toExponential(1)}, outputs ${r.out.toExponential(1)}, fitness ${r.fit.toExponential(1)} rel`);
}

await page.evaluate(() => { P.H = 2; P.setup(); });

// ---------------------------------------------------------------------------
// 2. An uninterrupted run: same race, even though rounding differences are
//    free to compound.
// ---------------------------------------------------------------------------
{
    const r = await page.evaluate(async () => {
        const n = 200, brains = P.makeBrains(n);
        P.pop(P.cpu, n, brains); P.pop(P.gpu, n, brains);
        const a0 = await P.state(P.cpu), b0 = await P.state(P.gpu);
        const start = P.compare(a0, b0, n);
        const ra = await P.run(P.cpu, 200), rb = await P.run(P.gpu, 200);
        const a = await P.state(P.cpu), b = await P.state(P.gpu);
        const fa = new Float32Array(a.buffer), fb = new Float32Array(b.buffer);
        const ia = new Int32Array(a.buffer), ib = new Int32Array(b.buffer);
        let agree = 0, errs = [];
        for (let c = 0; c < n; c++) {
            if (ia[c * 32 + 7] === ib[c * 32 + 7]) agree++;
            if (!ia[c * 32 + 7] && !ib[c * 32 + 7]) errs.push(Math.hypot(fa[c * 32] - fb[c * 32], fa[c * 32 + 1] - fb[c * 32 + 1]));
        }
        errs.sort((x, y) => x - y);
        return { startSame: start.crash + start.ints + start.pos === 0, agree, n,
                 median: errs.length ? errs[errs.length >> 1] : 0, alive: errs.length,
                 stepsA: ra.carSteps, stepsB: rb.carSteps };
    });
    check(r.startSame, 'both backends spawn the field identically (same start state, from sim.c)');
    check(r.agree >= r.n * 0.95 && r.median < 0.05,
        'over 200 uninterrupted steps they run the same race',
        `${r.agree}/${r.n} cars agree on crashed-or-not, median position gap ${r.median.toExponential(1)}px over ${r.alive} survivors`);
    check(Math.abs(r.stepsA - r.stepsB) <= r.stepsA * 0.02,
        '  and do the same amount of work', `${r.stepsA} vs ${r.stepsB} car-steps`);
}

// ---------------------------------------------------------------------------
// 3. Lockstep: the lap-target stop and the all-crashed report.
// ---------------------------------------------------------------------------
{
    const r = await page.evaluate(async () => {
        const n = 128;
        P.setup([10, 0.05, 0.02, 0.05, 750, 1, 0.2, P.H, 0]);   // target: one lap
        const brains = P.makeBrains(n);
        P.pop(P.cpu, n, brains); P.pop(P.gpu, n, brains);
        const ra = await P.run(P.cpu, 4000), rb = await P.run(P.gpu, 4000);
        const b = await P.state(P.gpu);
        const ib = new Int32Array(b.buffer);
        const frames = new Set();
        let finisher = false;
        for (let c = 0; c < n; c++) {
            const o = c * 32;
            if (ib[o + 7] === 0) frames.add(ib[o + 9]);
            if (ib[o + 11] >= 1 && ib[o + 15] === ib[o + 9]) finisher = true;
        }
        const a = await P.state(P.cpu);
        const ia = new Int32Array(a.buffer);
        let cpuStop = -1;
        for (let c = 0; c < n; c++) if (ia[c * 32 + 7] === 0) { cpuStop = ia[c * 32 + 9]; break; }
        const gpuStop = frames.size === 1 ? [...frames][0] : -1;

        // The same generation again on the GPU, but in rounds cut short by
        // its time budget, the way the app runs it in hyper mode on a slow
        // GPU: asked again and again until someone finishes. Has to stop on
        // exactly the same frame as the one whole chunk did.
        P.pop(P.gpu, n, brains);
        let rounds = 0, short = 0, sliced = null;
        while (rounds < 400) {
            P.gpu.post({ type: 'run', iters: 4000, wantRender: false });
            const r = await P.gpu.next('done');
            rounds++;
            if (!(r.maxLaps >= 1) && !r.allCrashed) short++;
            if (r.maxLaps >= 1 || r.allCrashed) { sliced = r; break; }
        }
        const sb = new Int32Array((await P.state(P.gpu)).buffer);
        let slicedStop = -1;
        for (let c = 0; c < n; c++) if (sb[c * 32 + 7] === 0) { slicedStop = sb[c * 32 + 9]; break; }

        // All-crashed: random brains only, well past the point they're all gone.
        P.setup();
        const wild = P.makeBrains(n);
        for (let c = 0; c < n; c += 2) for (let j = 0; j < P.stride; j++) wild[c * P.stride + j] = (j % 7) * 0.3 - 0.9;
        P.pop(P.cpu, n, wild); P.pop(P.gpu, n, wild);
        const ca = await P.run(P.cpu, 2500), cb = await P.run(P.gpu, 2500);
        const again = await P.run(P.gpu, 10);
        return { ra: { maxLaps: ra.maxLaps, all: ra.allCrashed, steps: ra.carSteps },
                 rb: { maxLaps: rb.maxLaps, all: rb.allCrashed, steps: rb.carSteps },
                 lockstep: frames.size, gpuStop, cpuStop, finisher, rounds, short, slicedStop,
                 ca: { all: ca.allCrashed, steps: ca.carSteps, alive: ca.alive },
                 cb: { all: cb.allCrashed, steps: cb.carSteps, alive: cb.alive },
                 again: { all: again.allCrashed, steps: again.carSteps } };
    });
    check(r.rb.maxLaps >= 1 && !r.rb.all && r.lockstep === 1 && r.finisher,
        'a chunk stops every surviving car on the step the first one completes the target laps',
        `GPU stopped at frame ${r.gpuStop} with every survivor on it; maxLaps ${r.rb.maxLaps}`);
    check(r.ra.maxLaps >= 1 && Math.abs(r.gpuStop - r.cpuStop) <= 3,
        '  the same frame the CPU build stops on', `CPU ${r.cpuStop}, GPU ${r.gpuStop}`);
    check(r.slicedStop === r.gpuStop && r.short > 0,
        '  and cut into time-budgeted rounds, it still stops on exactly that frame',
        `${r.rounds} rounds (${r.short} cut short by the budget), stopped at ${r.slicedStop}`);
    check(r.ca.all && r.cb.all && r.cb.alive === 0 && Math.abs(r.ca.steps - r.cb.steps) <= r.ca.steps * 0.02,
        'a field that crashes out is reported all-crashed, with the same step count',
        `CPU ${r.ca.steps} / GPU ${r.cb.steps} car-steps`);
    check(r.again.all && r.again.steps === 0, '  and running it again does nothing at all', `${r.again.steps} steps`);
}

// ---------------------------------------------------------------------------
// 4. Splitting a car's wall scan across threads changes nothing.
// ---------------------------------------------------------------------------
{
    const r = await page.evaluate(async () => {
        P.setup();
        const n = 96, brains = P.makeBrains(n);
        P.pop(P.cpu, n, brains);
        await P.run(P.cpu, 30);
        const s = await P.state(P.cpu);
        P.importState(P.gpu, n, brains, s, { lanes: 1 });
        P.importState(P.gpu2, n, brains, s, { lanes: 8 });
        await P.run(P.gpu, 150); await P.run(P.gpu2, 150);
        const a = await P.state(P.gpu), b = await P.state(P.gpu2);
        let diff = -1;
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { diff = i; break; }
        const alive = new Int32Array(a.buffer).filter((v, i) => i % 32 === 7 && v === 0).length;
        return { diff, alive };
    });
    check(r.diff < 0, 'eight threads per car and one land on exactly the same bits',
        r.diff < 0 ? `150 steps, ${r.alive} cars still driving` : `first difference at car ${Math.floor(r.diff / 32)} word ${r.diff % 32}`);
}

// ---------------------------------------------------------------------------
// 5. Import then export through the GPU is lossless.
// ---------------------------------------------------------------------------
{
    const r = await page.evaluate(async () => {
        const n = 150, brains = P.makeBrains(n);
        P.pop(P.cpu, n, brains);
        await P.run(P.cpu, 40);
        const s = await P.state(P.cpu);
        P.importState(P.gpu, n, brains, s);
        const back = await P.state(P.gpu);
        for (let i = 0; i < s.length; i++) if (s[i] !== back[i]) return i;
        return -1;
    });
    check(r < 0, 'cars moved onto the GPU and back come back bit-for-bit', r < 0 ? '' : `word ${r} differs`);
}

check(errors.length === 0, 'no page errors', errors.length ? errors.join(' | ') : 'clean');
await browser.close();
console.log(failures ? `\n${failures} check(s) failed.` : '\nthe GPU computes the same race as the CPU.');
process.exit(failures ? 1 : 0);
