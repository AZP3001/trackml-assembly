// e2e.mjs — drives the real page in a real browser.
//
// The unit tests prove sim.wasm computes the right thing; this proves the page
// around it actually works: the module loads, the worker pool comes up, tracks
// generate, cars drive, generations advance, and the editor still edits.
//
// Needs a server (the page fetches wasm and starts Workers, neither of which
// works over file://) and Playwright's chromium.
//
//   npx http-server -p 8765 . &
//   node tools/e2e.mjs [http://127.0.0.1:8765]
//
// PLAYWRIGHT_CHROMIUM_PATH points at an already-installed Chromium, for
// sandboxes and CI images that ship one at a version Playwright would
// otherwise insist on re-downloading. Unset, Playwright finds its own.
import { chromium } from 'playwright';

const BASE = process.argv[2] || 'http://127.0.0.1:8765';
// The phone column of DEFAULT_SETTINGS in script.js, restated here on purpose:
// a test that read the value out of the page could not tell the difference
// between "the mobile defaults are applied" and "the mobile defaults are
// whatever the page happens to be doing".
const DEFAULTS_MOBILE = { pop: 150, elite: 15, laps: 2, hidden: 4 };
let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

// --enable-unsafe-webgpu so the CPU/GPU toggle can be exercised headless
// (SwiftShader's software adapter); it changes nothing else about the page.
const browser = await chromium.launch({
    ...(process.env.PLAYWRIGHT_CHROMIUM_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH } : {}),
    args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();

const errors = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

console.log(`TrackML end-to-end test against ${BASE}\n`);

await page.goto(`${BASE}/index.html`, { waitUntil: 'load' });

// Wait for the app, not just the engine. app.init() awaits Engine.ready() and
// only THEN builds the track list and the first population, so a wait that
// stops at `Engine.workers.length > 0` returns during a window where there are
// still zero tracks and zero cars.
//
// Note the bare identifiers: engine.js and script.js are classic scripts, so
// `const Engine` lands in the global *lexical* scope and never becomes a
// property of window. `window.Engine` is undefined here even though `Engine` is
// perfectly resolvable.
const APP_READY = () => typeof Engine !== 'undefined' && Engine.master
    && Engine.workers.length > 0
    && typeof app !== 'undefined' && app.state.tracks.length > 0
    && app.state.cars.length > 0 && app.currentTrack && app.currentTrack.wallCount > 0;

await page.waitForFunction(APP_READY, null, { timeout: 30000 });

const boot = await page.evaluate(() => ({
    simd: Engine.usingSimd,
    cores: Engine.coreCount,
    workers: Engine.workers.length,
    tracks: app.state.tracks.length,
    trackName: app.currentTrack && app.currentTrack.name,
    walls: app.currentTrack && app.currentTrack.wallCount,
    cps: app.currentTrack && app.currentTrack.cpCount,
    cars: app.state.cars.length
}));
console.log(`  booted: ${boot.workers} workers, SIMD=${boot.simd}, ${boot.tracks} tracks loaded\n`);

check(boot.workers > 0, 'worker pool started', `${boot.workers} workers`);
check(boot.tracks > 5, 'built-in tracks generated', `${boot.tracks} tracks`);
check(boot.walls > 20, 'track geometry has barriers', `${boot.walls} walls, ${boot.cps} checkpoints`);
check(boot.cars > 0, 'population created', `${boot.cars} cars`);

// --- run the simulation -------------------------------------------------
// Sampled over a window rather than once. evolve() replaces every car record
// with a fresh one back on the start line, and a well-trained population turns
// generations over fast enough that a single snapshot lands on that reset and
// reports nobody moving.
await page.evaluate(() => { app.state.isRunning = true; });
const moved = await page.evaluate(async () => {
    const start = app.currentTrack.startPos;
    let peak = 0, farthest = 0, aliveOk = true, sawDrop = false;
    for (let i = 0; i < 120; i++) {
        await new Promise(r => requestAnimationFrame(r));
        let off = 0;
        for (const c of app.state.cars) {
            const d = Math.hypot(c.x - start.x, c.y - start.y);
            if (d > 5) off++;
            if (d > farthest) farthest = d;
        }
        if (off > peak) peak = off;
        // Sampled every frame, not once after an evolve — an evolve resets the
        // counter to the full population and would mask a NaN written between.
        const a = app.state.aliveCount;
        if (!Number.isFinite(a) || a < 0 || a > app.state.cars.length) aliveOk = false;
        if (a < app.state.cars.length) sawDrop = true;
    }
    return { off: peak, farthest, total: app.state.cars.length, aliveOk, sawDrop };
});
// A quarter of the field, not half: with the random brains of generation 1 a
// good fraction of cars never command any throttle at all, and those are
// eliminated within a few frames for having no momentum rather than ever
// showing up as "moving". That is the simulation working, not a stuck
// population — so the second check is the one that really says "driving".
check(moved.off > moved.total * 0.25, 'cars leave the start line',
    `peak ${moved.off}/${moved.total} moving`);
check(moved.farthest > 100, 'cars drive a real distance',
    `furthest ${Math.round(moved.farthest)}px from the line`);
check(moved.aliveOk, 'alive count stays a real number every frame');
check(moved.sawDrop, 'alive count actually tracks crashes');

// Let it evolve. Generation 1 -> 2 exercises fitness collection, selection,
// crossover in wasm and re-shipping the brains to every worker.
const genStart = await page.evaluate(() => app.state.generation);
await page.waitForFunction(g => app.state.generation > g, genStart, { timeout: 60000 });
const afterGen = await page.evaluate(() => ({
    gen: app.state.generation,
    stats: app.state.stats.length,
    best: app.state.stats.length ? app.state.stats[0].best : null,
    alive: app.state.aliveCount,
    cars: app.state.cars.length
}));
check(afterGen.gen > genStart, 'generation advanced', `gen ${genStart} -> ${afterGen.gen}`);
check(afterGen.stats > 0 && Number.isFinite(afterGen.best), 'fitness recorded', `best=${Math.round(afterGen.best)}`);
check(afterGen.cars === boot.cars, 'population size held', `${afterGen.cars} cars`);

// --- worker-count slider --------------------------------------------------
// Drives the real slider (a real 'input' event, proving the data-input
// wiring, not a hand call to Engine.setCoreCount). The pool now changes
// shape MID-GENERATION: every car is exported from the old workers and
// imported into the new ones exactly where it was (see Engine._reshape), so
// the checks are that the label moves the instant the slider does, that the
// pool lands in well under a second without the generation ending, that the
// cars carry on from where they were rather than back on the start line, and
// that training carries on afterwards.
const setCores = n => page.evaluate((v) => {
    const el = document.getElementById('cfg-coreCount');
    el.value = v;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    return document.getElementById('val-cores').textContent;
}, n);
// Positions of every still-driving car, and how many of them are still near
// the start line — the tell-tale of a population that got reset.
const snapCars = () => page.evaluate(() => {
    const s = app.currentTrack.startPos;
    return app.state.cars.map(c => ({ x: c.x, y: c.y, crashed: c.crashed, off: Math.hypot(c.x - s.x, c.y - s.y) }));
});
// A pool change happens between rounds; pause, let the round in flight land,
// reshape, then run a couple of frames and see where the cars went.
async function reshapeMidRace(label, doChange, landed) {
    await page.evaluate(() => { app.state.hyperMode = false; app.state.isRunning = true; });
    await page.waitForFunction(() => {
        const s = app.currentTrack.startPos;
        return app.state.cars.filter(c => !c.crashed && Math.hypot(c.x - s.x, c.y - s.y) > 60).length >= 5;
    }, null, { timeout: 60000 });
    await page.evaluate(() => { app.state.isRunning = false; });
    await page.waitForFunction(() => !app._pending && !Engine._runState, null, { timeout: 10000 });
    const gen0 = await page.evaluate(() => app.state.generation);
    const before = await snapCars();
    const t0 = Date.now();
    const changeResult = await doChange();
    await page.waitForFunction(landed, null, { timeout: 60000 });
    const ms = Date.now() - t0;
    // A handful of frames at 1x: each round moves a car at most Max Speed
    // (10px), so a car that carried on is within a few dozen px of where it
    // was, and one that was reset is back on the start line.
    await page.evaluate(async () => {
        app.state.isRunning = true;
        for (let i = 0; i < 4; i++) await new Promise(r => requestAnimationFrame(r));
        app.state.isRunning = false;
    });
    await page.waitForFunction(() => !app._pending && !Engine._runState, null, { timeout: 10000 });
    const after = await snapCars();
    const gen1 = await page.evaluate(() => app.state.generation);
    let carried = 0, eligible = 0;
    before.forEach((b, i) => {
        if (b.crashed || b.off < 60) return;
        eligible++;
        const a = after[i];
        if (a && (a.crashed || (Math.hypot(a.x - b.x, a.y - b.y) < 80 && a.off > 20))) carried++;
    });
    return { ms, gen0, gen1, carried, eligible, changeResult };
}

{
    const bounds = await page.evaluate(() => {
        const el = document.getElementById('cfg-coreCount');
        return { min: +el.min, max: +el.max, value: +el.value, hw: Engine.hardwareCores };
    });
    check(bounds.min === 1 && bounds.max === bounds.hw && bounds.value === boot.cores,
        'the cores slider is bounded to the hardware and starts at the stock default',
        `min ${bounds.min}, max ${bounds.max}, value ${bounds.value} (hardware ${bounds.hw})`);

    const down = await reshapeMidRace('shrink', () => setCores(1), () => Engine.workers.length === 1);
    check(down.changeResult === `1 / ${bounds.hw}`, 'the slider label follows the slider the instant it moves',
        `"${down.changeResult}"`);
    check(down.ms < 3000 && down.gen0 === down.gen1,
        'shrinking to 1 core lands mid-generation, without waiting for the generation to end',
        `pool of 1 in ${down.ms}ms, still generation ${down.gen1}`);
    check(down.eligible > 0 && down.carried === down.eligible,
        'and every car carries on from where it was instead of going back to the start line',
        `${down.carried}/${down.eligible} cars picked up where they left off`);

    const target = bounds.hw > 1 ? bounds.hw : 1;
    const up = await reshapeMidRace('grow', () => setCores(target), t => Engine.workers.length === Engine.hardwareCores);
    check(up.ms < 5000 && up.gen0 === up.gen1 && up.carried === up.eligible,
        'growing it back to every core does the same, mid-generation, cars intact',
        `${target} workers in ${up.ms}ms, ${up.carried}/${up.eligible} cars carried over`);

    await page.evaluate(() => { app.state.isRunning = true; });
    const g = await page.evaluate(() => app.state.generation);
    await page.waitForFunction(g => app.state.generation > g, g, { timeout: 60000 });
    const after = await page.evaluate(() => ({ workers: Engine.workers.length, cars: app.state.cars.length,
        badge: document.getElementById('core-count').textContent.trim() }));
    check(after.cars === boot.cars && after.workers === target,
        'training still advances afterwards, population intact', `${after.cars} cars on ${after.workers} workers`);
    check(after.badge.includes(`${target}`) && after.badge.includes('Cores'),
        'and the Compute badge reports the pool as it actually is', `"${after.badge}"`);
}

// --- CPU / GPU toggle ------------------------------------------------------
// Real clicks on the real toggle. Same mid-generation hand-over as the
// slider, but to a WebGPU compute shader — and back, both on request and on
// a lost GPU. Headless Chromium here runs WebGPU on SwiftShader (software),
// so this proves the wiring, not the speed; tools/gpuparity.mjs proves the
// GPU computes the same race.
{
    const hasGpu = await page.evaluate(async () => !!(navigator.gpu && await navigator.gpu.requestAdapter()));
    if (!hasGpu) {
        console.log('skip  no WebGPU adapter in this browser — GPU toggle not exercised');
    } else {
        // The toggle lives in the settings panel, which starts collapsed.
        await page.evaluate(() => {
            if (document.getElementById('config-panel').classList.contains('hidden')) app.toggleSettings();
        });
        const toGpu = await reshapeMidRace('gpu', () => page.click('#btn-compute-gpu'), () => Engine.mode === 'gpu');
        const ui = await page.evaluate(() => ({
            workers: Engine.workers.length,
            badge: document.getElementById('core-count').textContent.trim(),
            gpuBtn: document.getElementById('btn-compute-gpu').className,
            sliderDisabled: document.getElementById('cfg-coreCount').disabled
        }));
        check(ui.workers === 1 && /bg-blue-600/.test(ui.gpuBtn) && ui.badge.includes('GPU'),
            'the GPU toggle moves the whole population onto one WebGPU worker', `badge "${ui.badge}"`);
        check(toGpu.gen0 === toGpu.gen1 && toGpu.carried === toGpu.eligible,
            '  mid-generation, every car carrying on from where it was',
            `${toGpu.carried}/${toGpu.eligible} cars, ${toGpu.ms}ms (shader compile included)`);
        check(ui.sliderDisabled, '  and the CPU-cores slider stands down while the GPU runs');

        await page.evaluate(() => { app.state.hyperMode = true; app.state.isRunning = true; });
        const g0 = await page.evaluate(() => app.state.generation);
        await page.waitForFunction(g => app.state.generation >= g + 2, g0, { timeout: 180000 });
        await page.evaluate(async () => { await new Promise(r => setTimeout(r, 1200)); });
        const trained = await page.evaluate(() => ({ gen: app.state.generation, cars: app.state.cars.length,
            best: app.state.stats.length ? app.state.stats[0].best : null,
            rate: document.getElementById('compute-rate').textContent }));
        check(trained.cars === boot.cars && Number.isFinite(trained.best),
            'generations evolve on the GPU backend', `gen ${g0} -> ${trained.gen}, best ${Math.round(trained.best)}`);
        check(/car-steps\/s/.test(trained.rate), '  and the throughput readout measures it', `"${trained.rate}"`);
        await page.evaluate(() => { app.state.hyperMode = false; });

        // A GPU that goes away mid-race (driver reset, a phone reclaiming it)
        // must hand the race back to the CPU, not freeze it.
        const lost = await reshapeMidRace('lost', () => page.evaluate(() => Engine._onGpuLost('simulated device loss')),
            () => Engine.mode === 'cpu');
        const fell = await page.evaluate(() => ({ workers: Engine.workers.length, target: Engine._target.cores,
            status: document.getElementById('compute-status').textContent,
            cpuBtn: document.getElementById('btn-compute-cpu').className }));
        check(fell.workers === fell.target && /bg-blue-600/.test(fell.cpuBtn) && /GPU unavailable/.test(fell.status),
            'a lost GPU falls back to the CPU pool and says so', `"${fell.status}"`);
        check(lost.gen0 === lost.gen1 && lost.carried === lost.eligible,
            '  mid-generation, from the GPU\'s last state', `${lost.carried}/${lost.eligible} cars carried over`);

        await page.evaluate(() => { app.state.isRunning = true; });
        const g1 = await page.evaluate(() => app.state.generation);
        await page.waitForFunction(g => app.state.generation > g, g1, { timeout: 60000 });
        check(true, '  and training carries on there');
    }
}

// The mutation-settle/focus-mode bookkeeping (see script.js: app.evolve) ran
// at least once as part of the generation above with no page error, which is
// most of what matters — a wrong argument order into the new 4-arg
// Engine.evolve/ex.evolve would throw or silently misbehave, and the whole
// suite runs with page-error capture on. This adds the structural check that
// bare "no errors" can't: the fields exist, have sane types, and agree with
// each other and with whether a lap has actually been recorded yet — not
// whether one specifically HAS on this random population within one
// generation, which is exactly the kind of thing that's fine on one seed and
// flaky on the next.
const settle = await page.evaluate(() => ({
    firstLapGen: app._firstLapGen,
    lapCompletionCount: app._lapCompletionCount,
    settleGenerations: app.SETTLE_GENERATIONS,
    bestLapEver: app.state.bestTimes.all
}));
check(settle.firstLapGen === null || (Number.isInteger(settle.firstLapGen) && settle.firstLapGen >= 1),
    'firstLapGen is null or a real generation number', `firstLapGen=${settle.firstLapGen}`);
check(Number.isInteger(settle.lapCompletionCount) && settle.lapCompletionCount >= 0,
    'lapCompletionCount is a non-negative integer', `lapCompletionCount=${settle.lapCompletionCount}`);
check(Number.isInteger(settle.settleGenerations) && settle.settleGenerations > 0,
    'SETTLE_GENERATIONS is configured', `${settle.settleGenerations}`);
// If a lap has ever been recorded, the bookkeeping MUST agree it has —
// firstLapGen set, count at least 1 — rather than the display (bestTimes)
// and the mutation-schedule tracking (firstLapGen) disagreeing about
// whether this run has ever actually finished a lap.
check(!settle.bestLapEver || (settle.firstLapGen !== null && settle.lapCompletionCount >= 1),
    'if a lap time is on record, the settle tracking agrees a lap happened',
    `bestLapEver=${settle.bestLapEver}, firstLapGen=${settle.firstLapGen}, count=${settle.lapCompletionCount}`);

// --- hyper mode ---------------------------------------------------------
// Different code path: no render rows cross the boundary, only fitness.
const beforeHyper = await page.evaluate(() => { app.toggleHyper(); return app.state.generation; });
await page.waitForFunction(g => app.state.generation > g + 1, beforeHyper, { timeout: 60000 });
const hyper = await page.evaluate(() => {
    const g = app.state.generation;
    app.toggleHyper();
    return { gen: g, alive: app.state.aliveCount };
});
check(hyper.gen > beforeHyper + 1, 'hyper mode trains', `reached gen ${hyper.gen}`);
check(Number.isFinite(hyper.alive) && hyper.alive >= 0 && hyper.alive <= boot.cars,
    'alive count survives hyper mode', `${hyper.alive} alive`);

// --- track switching ----------------------------------------------------
await page.evaluate(() => app.switchTrack(3));
await page.waitForTimeout(500);
const switched = await page.evaluate(() => ({
    name: app.currentTrack.name, walls: app.currentTrack.wallCount, cars: app.state.cars.length
}));
check(switched.walls > 20, 'switching track rebuilds geometry', `"${switched.name}", ${switched.walls} walls`);

// --- editor -------------------------------------------------------------
// The editor regenerates the whole track through wasm on every frame it draws.
await page.evaluate(() => app.createNewTrack());
await page.waitForTimeout(300);
const edit = await page.evaluate(() => {
    editor.track.path.push({ x: 600, y: 800, type: 'rounded', radius: 60 });
    editor.updateWidth(45);
    const t = generateTrackFromPath('e2e', 'E2E', editor.track.path, editor.track.trackWidth);
    return { pts: editor.track.path.length, walls: t.wallCount, cps: t.cpCount, editing: app.state.isEditing };
});
check(edit.editing, 'editor opened');
check(edit.walls > 20 && edit.cps > 5, 'editor regenerates geometry live', `${edit.walls} walls from ${edit.pts} points`);
await page.evaluate(() => editor.cancel());

// --- brain round trip ---------------------------------------------------
const brain = await page.evaluate(() => {
    const json = Engine.brainToJSON(0);
    const bad = Engine.validateBrainJSON(json);
    return { bad, ih: json.weightsIH.length, h: json.biasH.length, o: json.biasO.length,
             finite: json.weightsIH.every(r => r.every(Number.isFinite)) };
});
check(!brain.bad && brain.ih === 11 && brain.o === 2 && brain.finite,
    'brain exports in the current JSON format', `11x${brain.h}x2${brain.bad ? ' — ' + brain.bad : ''}`);

// A brain saved before the network gained its two recurrent inputs has 9 rows.
// It still has to load — the missing rows are zero-filled, which means the old
// brain simply ignores the new inputs and drives exactly as it did.
const legacy = await page.evaluate(() => {
    const json = Engine.brainToJSON(0);
    json.weightsIH = json.weightsIH.slice(0, 9);      // pretend it is an old file
    const bad = Engine.validateBrainJSON(json);
    if (bad) return { bad };
    Engine.writeBrainJSON(Engine.master.ex.stash_slot(), json);
    const back = Engine.brainToJSON(Engine.master.ex.stash_slot());
    return {
        bad: null,
        rows: back.weightsIH.length,
        tailZeroed: back.weightsIH.slice(9).every(r => r.every(v => v === 0)),
        headKept: back.weightsIH.slice(0, 9).every((r, i) => r.every((v, k) => v === json.weightsIH[i][k]))
    };
});
check(!legacy.bad && legacy.rows === 11 && legacy.tailZeroed && legacy.headKept,
    'a brain saved by an older build still loads',
    legacy.bad || `9 rows in, ${legacy.rows} out, new inputs zeroed`);

// Write a brain into a slot and read it back — the marshalling has to be an
// exact inverse or a loaded AI drives like a different one.
const roundTrip = await page.evaluate(() => {
    const original = Engine.brainToJSON(0);
    const slot = Engine.master.ex.stash_slot();
    Engine.writeBrainJSON(slot, original);
    const back = Engine.brainToJSON(slot);
    const same = JSON.stringify(original) === JSON.stringify(back);
    return { same, weights: original.weightsIH.length * original.biasH.length };
});
check(roundTrip.same, 'brain survives a write/read round trip', `${roundTrip.weights} input weights`);

// Malformed files are rejected before anything is written, because an
// out-of-range typed-array write is dropped rather than throwing.
const rejected = await page.evaluate(() => {
    const good = Engine.brainToJSON(0);
    const cases = {
        'not a brain':      Engine.validateBrainJSON({ hello: 1 }),
        'wrong inputs':     Engine.validateBrainJSON({ ...good, weightsIH: good.weightsIH.slice(0, 3) }),
        'hidden too big':   Engine.validateBrainJSON({
                                weightsIH: Array.from({ length: 9 }, () => new Array(999).fill(0)),
                                weightsHO: Array.from({ length: 999 }, () => [0, 0]),
                                biasH: new Array(999).fill(0), biasO: [0, 0] }),
        'NaN weight':       Engine.validateBrainJSON({ ...good,
                                weightsIH: good.weightsIH.map((r, i) => i ? r : r.map(() => NaN)) })
    };
    return Object.entries(cases).filter(([, v]) => v === null).map(([k]) => k);
});
check(rejected.length === 0, 'malformed brain files are rejected',
    rejected.length ? 'accepted: ' + rejected.join(', ') : 'all 4 bad files caught');

// --- Skip Gen -----------------------------------------------------------
// The button calls app.evolve() directly, outside the run loop, which is a
// path the natural generation rollover above never takes.
const skipped = await page.evaluate(() => {
    const before = app.state.generation;
    app.evolve();
    return { before, after: app.state.generation, cars: app.state.cars.length,
             stats: app.state.stats.length };
});
check(skipped.after === skipped.before + 1 && skipped.cars > 0, 'Skip Gen evolves on demand',
    `gen ${skipped.before} -> ${skipped.after}`);

// --- zones --------------------------------------------------------------
// Zone types cross into wasm as integer ids rather than the JS edition's
// strings, so the mapping is worth asserting rather than assuming.
const zones = await page.evaluate(async () => {
    app.createNewTrack();
    editor.addZone('spawnkill');
    editor.addZone('focus');
    editor.addZone('speed');
    editor.addZone('precision');
    const t = generateTrackFromPath('z', 'Zones', editor.track.path, editor.track.trackWidth,
        editor.track.startPos, editor.track.startAngle, editor.track.zones);
    const out = { count: t.zones.length, walls: t.wallCount, types: t.zones.map(z => z.type) };
    editor.cancel();
    return out;
});
check(zones.count === 4 && zones.walls > 20, 'zones survive a track rebuild',
    `${zones.count} zones: ${zones.types.join(', ')}`);

// --- saving a track, and NOT persisting it ------------------------------
// The app deliberately keeps nothing across reloads. A track you make lasts
// the session; the code export is how you keep one.
await page.evaluate(() => {
    app.createNewTrack();
    document.getElementById('edit-name').value = 'E2E Session Track';
    editor.save();
    document.getElementById('code-modal').classList.add('hidden');
});
await page.waitForTimeout(400);
const saved = await page.evaluate(() => ({
    inList: app.state.tracks.some(t => t.name === 'E2E Session Track'),
    code: document.getElementById('code-output').value,
    storageKeys: Object.keys(localStorage).length + Object.keys(sessionStorage).length
}));
check(saved.inList, 'custom track joins the track list');
check(/^generateTrackFromPath\(/.test(saved.code), 'the editor exports code for the track',
    saved.code.slice(0, 48) + '...');
check(saved.storageKeys === 0, 'saving writes nothing to browser storage',
    `${saved.storageKeys} keys after a save`);

// Plant some storage, reload, and confirm the page wiped it and came back with
// only the built-in tracks.
await page.evaluate(() => {
    localStorage.setItem('trackml_custom_tracks_v1', '[{"id":"ghost","name":"Ghost Track"}]');
    localStorage.setItem('trackml_settings_v1', '{"populationSize":123}');
    sessionStorage.setItem('anything', 'at all');
});
await page.reload({ waitUntil: 'load' });
await page.waitForFunction(APP_READY, null, { timeout: 30000 });
const afterReload = await page.evaluate(() => ({
    local: Object.keys(localStorage).length,
    session: Object.keys(sessionStorage).length,
    sessionTrack: app.state.tracks.some(t => t.name === 'E2E Session Track'),
    ghost: app.state.tracks.some(t => t.name === 'Ghost Track'),
    tracks: app.state.tracks.length,
    pop: app.state.populationSize
}));
check(afterReload.local === 0 && afterReload.session === 0, 'reload clears browser storage',
    `${afterReload.local} local + ${afterReload.session} session keys remain`);
check(!afterReload.sessionTrack, 'a track made this session does not come back');
check(!afterReload.ghost, 'storage planted by an older build is ignored');
check(afterReload.tracks === boot.tracks, 'reload gives the built-in track list',
    `${afterReload.tracks} tracks`);
check(afterReload.pop === boot.cars, 'settings reset to defaults',
    `population ${afterReload.pop}`);

// --- session round trip ---------------------------------------------------
// Save the whole run, scramble the live state, load it back, and check the
// population and the graph came back rather than being reseeded from one brain.
const session = await page.evaluate(async () => {
    // give the graph something to hold
    app.state.stats = [{ gen: 1, best: 100, avg: 50, time: '9.99' }, { gen: 2, best: 200, avg: 90, time: '8.88' }];
    app.state.generation = 7;
    app.state.bestTimes = { gen: 8.88, all: 8.88 };
    const before = Engine.exportPopulation();
    const blob = {
        format: 'trackml-session', version: 1, generation: app.state.generation,
        settings: { populationSize: app.state.populationSize, hiddenLayers: app.state.hiddenLayers,
                    physics: { ...app.state.physics } },
        stats: app.state.stats, bestTimes: app.state.bestTimes, lapHistory: [8.88],
        population: before
    };
    // wipe the live state, then load it back
    app.state.generation = 1; app.state.stats = []; app.state.bestTimes = { gen: null, all: null };
    const bad = Engine.importPopulation(blob.population);
    const after = Engine.exportPopulation();
    app.state.generation = blob.generation;
    app.state.stats = blob.stats;
    app.updateChart();
    return {
        bad,
        brainsIdentical: before.brains === after.brains,
        stashIdentical: before.stashBrain === after.stashBrain,
        popSize: after.popSize,
        gen: app.state.generation,
        chartPoints: app.chart ? app.chart.best.length : -1
    };
});
check(!session.bad, 'a saved population loads back', session.bad || 'accepted');
check(session.brainsIdentical && session.stashIdentical,
    'every brain survives the round trip byte for byte',
    session.brainsIdentical ? 'population and all-time best both exact' : 'weights changed');
check(session.gen === 7 && session.chartPoints === 2,
    'the generation counter and the graph come back too',
    `gen ${session.gen}, ${session.chartPoints} points on the chart`);

// --- declarative event binding -------------------------------------------
// The markup carries no inline on* handlers any more, so if bindActions ever
// stopped running the whole UI would go dead silently.
const bound = await page.evaluate(() => {
    const inline = document.querySelectorAll('[onclick],[oninput],[onchange],[onmouseenter],[onmouseleave]').length;
    const declared = document.querySelectorAll('[data-click],[data-input],[data-change],[data-enter],[data-leave]').length;
    // exercise one for real: the settings panel toggle
    const panel = document.getElementById('config-panel');
    const wasHidden = panel.classList.contains('hidden');
    document.querySelector('[data-click="app.toggleSettings"]').click();
    const toggled = panel.classList.contains('hidden') !== wasHidden;
    return { inline, declared, toggled };
});
check(bound.inline === 0, 'no inline event handlers left in the markup', `${bound.inline} found`);
check(bound.declared > 50, 'handlers are declared as data attributes', `${bound.declared} bound`);
check(bound.toggled, 'and a bound handler actually fires');

// --- auto width ----------------------------------------------------------
// A wedge corridor: the track runs back alongside itself 70px away, which at a
// half-width of 60 would leave the two roads overlapping with no barrier.
const autoWidth = await page.evaluate(() => {
    const path = [
        { x: 250, y: 200, type: 'corner', radius: 60 }, { x: 950, y: 200, type: 'corner', radius: 60 },
        { x: 950, y: 270, type: 'corner', radius: 60 }, { x: 600, y: 270, type: 'corner', radius: 60 },
        { x: 250, y: 500, type: 'corner', radius: 60 }
    ];
    const spread = t => Math.max(...t.widthF32) - Math.min(...t.widthF32);
    const off   = generateTrackFromPath('aw0', 'off',    path, 40, null, null, [], { enabled: false });
    const local = generateTrackFromPath('aw1', 'local',  path, 40, null, null, [], { enabled: true, blend: 0 });
    const glob  = generateTrackFromPath('aw2', 'global', path, 40, null, null, [], { enabled: true, blend: 1 });
    return {
        offSpread: spread(off), offMin: Math.min(...off.widthF32),
        localSpread: spread(local), localMin: Math.min(...local.widthF32), localMax: Math.max(...local.widthF32),
        globSpread: spread(glob), globMin: Math.min(...glob.widthF32),
        walls: local.wallCount, autoFlag: local.autoWidth
    };
});
check(autoWidth.offSpread === 0 && autoWidth.offMin === 40, 'auto width off keeps one width',
    `all samples at ${autoWidth.offMin}`);
check(autoWidth.localMin < 35 && autoWidth.localMax > 38, 'local mode pinches only the tight stretch',
    `${autoWidth.localMin.toFixed(1)} at the pinch, ${autoWidth.localMax.toFixed(1)} elsewhere`);
check(autoWidth.globSpread < 0.01 && Math.abs(autoWidth.globMin - autoWidth.localMin) < 0.01,
    'global mode uses that width everywhere',
    `uniform ${autoWidth.globMin.toFixed(1)}`);
check(autoWidth.walls > 20 && autoWidth.autoFlag, 'the pinched track still generates barriers',
    `${autoWidth.walls} walls`);

// The editor controls have to drive it, not just the API.
const autoUI = await page.evaluate(async () => {
    app.createNewTrack();
    const cb = document.getElementById('edit-autowidth');
    const sl = document.getElementById('edit-autowidth-blend');
    cb.checked = true; cb.dispatchEvent(new Event('change'));
    const afterCheck = editor.track.autoWidth;
    sl.value = '100'; sl.dispatchEvent(new Event('input'));
    const afterSlide = editor.track.autoWidthBlend;
    const label = document.getElementById('autowidth-blend-label').textContent;
    const opts = editor.autoOpts();
    editor.cancel();
    return { afterCheck, afterSlide, label, opts };
});
check(autoUI.afterCheck === true && autoUI.afterSlide === 1,
    'the editor controls drive auto width',
    `checkbox -> ${autoUI.afterCheck}, slider -> ${autoUI.afterSlide} ("${autoUI.label}")`);

// --- image import -------------------------------------------------------
// Synthesise a hand-drawn-looking loop and push it through the real pipeline:
// threshold, skeletonize, trace, then generateTrackFromPath.
const imported = await page.evaluate(async () => {
    const c = document.createElement('canvas');
    c.width = 400; c.height = 300;
    const g = c.getContext('2d');
    g.fillStyle = '#fff'; g.fillRect(0, 0, 400, 300);
    g.strokeStyle = '#000'; g.lineWidth = 14;
    g.beginPath(); g.ellipse(200, 150, 140, 95, 0, 0, Math.PI * 2); g.stroke();
    const img = new Image();
    await new Promise(r => { img.onload = r; img.src = c.toDataURL('image/png'); });
    try {
        ImageImport.processImage(img);
        return { ok: app.state.isEditing, pts: editor.track.path.length, width: editor.track.trackWidth };
    } catch (e) {
        return { ok: false, err: String(e.message) };
    }
});
check(imported.ok && imported.pts >= 3, 'image import builds a track',
    imported.ok ? `${imported.pts} points, width ${Math.round(imported.width)}` : imported.err);
await page.evaluate(() => { if (app.state.isEditing) editor.cancel(); });

// --- corner checkpoints --------------------------------------------------
// Every corner anchors a gate at its apex, and the editor draws those in amber
// so the difference is visible rather than merely present.
{
    const gates = await page.evaluate(() => {
        const path = [
            { x: 250, y: 250, type: 'corner', radius: 60 }, { x: 950, y: 250, type: 'corner', radius: 60 },
            { x: 950, y: 650, type: 'corner', radius: 60 }, { x: 250, y: 650, type: 'corner', radius: 60 }
        ];
        const t = generateTrackFromPath('cp', 'Corners', path, 60);
        let apex = 0;
        for (let i = 0; i < t.cpCount; i++) if (t.cpApex[i]) apex++;
        // The centre line should pass the same distance from all four vertices;
        // it used not to, which deleted a corner.
        const d = path.map(v => {
            let bd = Infinity;
            for (let i = 0; i < t.centerF32.length; i += 2)
                bd = Math.min(bd, Math.hypot(t.centerF32[i] - v.x, t.centerF32[i + 1] - v.y));
            return bd;
        });
        return { total: t.cpCount, apex, spread: Math.max(...d) - Math.min(...d),
                 lazyApex: t.checkpoints.filter(c => c.apex).length };
    });
    check(gates.apex === 4, 'each corner of a square anchors a gate', `${gates.apex} corner gates of ${gates.total}`);
    check(gates.spread < 2, 'and all four corners are built alike', `spread ${gates.spread.toFixed(2)}px`);
    check(gates.lazyApex === gates.apex, 'the apex flag survives into the object view',
        `${gates.lazyApex} flagged`);
}

// --- build stamp + version number ----------------------------------------
// version.json is written by the deploy workflow, so it is absent locally.
// Both paths matter: absent must leave an honest fallback (not a stale-
// looking fake version) rather than throwing, and present must be picked up
// everywhere it's shown — otherwise the whole point (telling at a glance
// whether a push actually went live, and which version that is) is lost.
//
// The version number itself is computed by the workflow from a commit count
// (see .github/workflows/deploy.yml), never typed in by hand — that is what
// this whole feature exists to guarantee, after the predecessor project's
// history showed the hand-maintained version repeatedly going stale. This
// test only has to prove the DISPLAY side: given a version.json, does every
// place that shows a version actually show it.
{
    const noStamp = await page.evaluate(() => ({
        desktop: document.getElementById('version-tag').textContent.trim(),
        mobiles: [...document.querySelectorAll('.version-tag-compact')].map(e => e.textContent.trim())
    }));
    check(noStamp.desktop.length > 0 && !/^v?\d/i.test(noStamp.desktop),
        'version line survives a missing build stamp without faking a number',
        `shows "${noStamp.desktop}"`);
    check(noStamp.mobiles.every(t => t === ''), 'mobile headers show nothing rather than a stale version',
        JSON.stringify(noStamp.mobiles));

    const stamped = await page.evaluate(async () => {
        const real = window.fetch;
        window.fetch = (u, o) => String(u).includes('version.json')
            ? Promise.resolve({ ok: true, json: () => Promise.resolve({
                version: '21.9', commit: 'deadbeefcafe', short: 'deadbee', ref: 'main', built: '2026-09-21T09:00:00Z'
              }) })
            : real(u, o);
        app.showBuildStamp();
        await new Promise(r => setTimeout(r, 120));
        window.fetch = real;
        const el = document.getElementById('version-tag');
        return {
            text: el.textContent.trim(), title: el.title,
            mobiles: [...document.querySelectorAll('.version-tag-compact')].map(e => e.textContent.trim())
        };
    });
    check(stamped.text === 'V21.9 · build deadbee', 'the version number and build hash are shown together',
        `"${stamped.text}"`);
    check(/main/.test(stamped.title), 'and names the branch it came from', stamped.title);
    check(stamped.mobiles.length === 2 && stamped.mobiles.every(t => t === '· V21.9'),
        'the compact mobile headers pick up the same version',
        JSON.stringify(stamped.mobiles));
}

// --- the raster size and the coordinates that depend on it --------------
// On a phone the canvas no longer rasterises a fixed 1200x900; it sizes itself
// to what the display can resolve, and the scale is folded into the one view
// transform. (On a desktop it is still the full 1200x900, which is what this
// context is.) That makes the pointer maths the thing to watch: a click has to
// land on the same world point it did before, at any raster size and any zoom,
// or clicking a car selects its neighbour and dragging a track point puts it
// somewhere else.
{
    const geom = await page.evaluate(() => {
        const c = document.getElementById('sim-canvas');
        const r = c.getBoundingClientRect();
        // World -> the client coordinates a real pointer event would carry,
        // going the long way round through the element's object-contain box.
        const fit = Math.min(r.width / c.width, r.height / c.height);
        const ox = (r.width - c.width * fit) / 2, oy = (r.height - c.height * fit) / 2;
        const worldToClient = (wx, wy) => {
            const v = app._viewMatrix();
            return {
                clientX: r.left + ox + (v.z * wx + v.e) * fit,
                clientY: r.top + oy + (v.z * wy + v.f) * fit
            };
        };
        const roundTrip = (wx, wy) => {
            const p = app._toBackingPx(worldToClient(wx, wy));
            const w = app.screenToWorld(p.x, p.y);
            return Math.max(Math.abs(w.x - wx), Math.abs(w.y - wy));
        };
        const pts = [[600, 450], [100, 100], [1100, 800], [300, 620]];
        const atRest = Math.max(...pts.map(([x, y]) => roundTrip(x, y)));

        // And again zoomed in and panned off-centre, where the scale and the
        // translation are both doing work.
        app.state.view.zoom = 3.5;
        app.state.view.panX = 430; app.state.view.panY = 560;
        app._clampView();
        const zoomed = Math.max(...pts.map(([x, y]) => roundTrip(x, y)));

        // A right-button drag has to move the map by exactly the distance the
        // pointer moved over it. Held here against the world point under the
        // pointer, which must not shift at all during the drag.
        const anchor = [700, 500];
        const from = worldToClient(...anchor);
        app.startPan({ preventDefault() {}, pointerId: 7, button: 2,
                       clientX: from.clientX, clientY: from.clientY });
        app.movePan({ pointerId: 7, clientX: from.clientX + 37, clientY: from.clientY - 21 });
        const moved = app.screenToWorld(
            ...(p => [p.x, p.y])(app._toBackingPx({ clientX: from.clientX + 37, clientY: from.clientY - 21 })));
        const dragErr = Math.max(Math.abs(moved.x - anchor[0]), Math.abs(moved.y - anchor[1]));
        app.endPan({ pointerId: 7 });
        app.resetView();

        return {
            w: c.width, h: c.height, scale: app._renderScale,
            rectW: r.width, rectH: r.height,
            viewportW: app._viewportW, viewportH: app._viewportH,
            bg: app.state.bgCanvas ? app.state.bgCanvas.width : 0,
            atRest, zoomed, dragErr
        };
    });
    // The backing store is no longer forced to the world's own 4:3 — it's
    // sized (and its cover-viewport widened) to match whatever box the
    // canvas actually sits in, which on a real page is essentially never
    // exactly 4:3, so this is exactly the letterbox-bar fix landing.
    check(geom.w > 0 && geom.h > 0 && Math.abs(geom.w / geom.h - geom.rectW / geom.rectH) < 0.01,
        'the backing store is sized in whole pixels, matching the container aspect (no letterbox bars)',
        `${geom.w}x${geom.h} backing store for a ${geom.rectW.toFixed(0)}x${geom.rectH.toFixed(0)} box (scale ${geom.scale.toFixed(3)})`);
    check(geom.viewportW >= 1200 - 0.5 && geom.viewportH >= 900 - 0.5,
        'and it reveals at least the full 1200x900 world — never less',
        `viewport ${geom.viewportW.toFixed(1)}x${geom.viewportH.toFixed(1)}`);
    check(Math.abs(geom.scale - 1) < 0.01, 'and on a desktop it rasterises close to 1:1 — no mobile downscale',
        `scale ${geom.scale.toFixed(4)}`);
    check(geom.bg === geom.w, 'the cached background matches it exactly, so it blits 1:1',
        `background ${geom.bg}px wide`);
    check(geom.atRest < 0.5, 'a pointer lands on the world point it is over',
        `worst error ${geom.atRest.toExponential(2)} world px`);
    check(geom.zoomed < 0.5, 'and still does zoomed in and panned',
        `worst error ${geom.zoomed.toExponential(2)} world px`);
    check(geom.dragErr < 0.5, 'a pan drag moves the map exactly as far as the pointer',
        `world point drifted ${geom.dragErr.toExponential(2)} px under the cursor`);
}

// --- the 30/60fps repaint toggle -----------------------------------------
// The simulation itself is never gated by this (see loop()) — only how often
// the canvas is repainted. Real clicks on the real buttons, so the
// data-click wiring (which hands the handler the LITERAL STRING "30", not
// the number 30) is what's actually being proven, not a hand-written call to
// app.setRenderHz(30).
{
    const before = await page.evaluate(() => app.state.renderHz);
    await page.click('#btn-fps-30');
    const afterA = await page.evaluate(() => ({
        hz: app.state.renderHz,
        btn30: document.getElementById('btn-fps-30').className,
        btn60: document.getElementById('btn-fps-60').className
    }));
    await page.click('#btn-fps-60');
    const afterB = await page.evaluate(() => app.state.renderHz);
    check(before === 60, 'canvas repaint rate defaults to 60fps', `default ${before}fps`);
    check(afterA.hz === 30 && /bg-blue-600/.test(afterA.btn30) && !/bg-blue-600/.test(afterA.btn60),
        'the 30fps button switches the rate and highlights itself', `now ${afterA.hz}fps`);
    check(afterB === 60, 'and the 60fps button switches it back', `now ${afterB}fps`);
}

// --- camera follow-car toggle ---------------------------------------------
// Clicks the real button (proving the data-click wiring, not just a hand
// call to toggleFollowCar), then forces a frame and checks the pan actually
// landed on the spectated car — not just that the button relabelled itself.
{
    const carId = await page.evaluate(() => {
        app.resetView(); app.releaseSpectate();
        const c = app.state.cars.find(c => !c.crashed);
        app.state.spectateCarId = c.id;
        app._spectated = app._pickSpectated();
        return c.id;
    });
    const before = await page.evaluate(() => {
        app.state.view.zoom = 4;
        app.state.view.panX = 900; app.state.view.panY = 200;   // deliberately off the car
        app._clampView();
        return {
            panX: app.state.view.panX, panY: app.state.view.panY,
            followBtn: document.getElementById('btn-follow-car').className
        };
    });
    await page.click('#btn-follow-car');
    const after = await page.evaluate(() => {
        app.draw();
        const c = app.state.cars.find(c => c.id === app.state.spectateCarId);
        return {
            panX: app.state.view.panX, panY: app.state.view.panY, carX: c.x, carY: c.y,
            followBtn: document.getElementById('btn-follow-car').className
        };
    });
    await page.evaluate(() => { app.toggleFollowCar(); app.resetView(); app.releaseSpectate(); });
    check(carId !== undefined && !/bg-blue-600/.test(before.followBtn), 'follow-car toggle starts off');
    check(/bg-blue-600/.test(after.followBtn), 'clicking it highlights the button', after.followBtn);
    check(Math.abs(after.panX - before.panX) > 1 || Math.abs(after.panY - before.panY) > 1,
        'and the next frame re-centres the camera on the spectated car',
        `pan moved from (${before.panX.toFixed(1)}, ${before.panY.toFixed(1)}) to (${after.panX.toFixed(1)}, ${after.panY.toFixed(1)})`);
    check(Math.abs(after.panX - after.carX) < 0.01 && Math.abs(after.panY - after.carY) < 0.01,
        'landing exactly on the car — well inside the zoomed-in clamp bounds, so unclamped',
        `pan (${after.panX.toFixed(2)}, ${after.panY.toFixed(2)}) vs car (${after.carX.toFixed(2)}, ${after.carY.toFixed(2)})`);
}

// --- canvas actually drew something ------------------------------------
const drew = await page.evaluate(() => {
    const c = document.getElementById('sim-canvas');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    const seen = new Set();
    for (let i = 0; i < d.length; i += 4 * 997) seen.add(`${d[i]},${d[i+1]},${d[i+2]}`);
    return seen.size;
});
check(drew > 2, 'canvas is rendering', `${drew} distinct sampled colours`);

const realErrors = errors.filter(e => !/favicon|Failed to load resource/i.test(e));
check(realErrors.length === 0, 'no page errors', realErrors.length ? realErrors.slice(0, 3).join(' | ') : 'clean');

// --- the same page, told it is a phone ----------------------------------
// A second context with a phone's user agent, viewport and pixel ratio. The
// point is not that the layout reflows — that is CSS and always did — but that
// the three things a phone actually cannot afford come out smaller: the field,
// the worker pool, and the number of pixels being painted.
{
    const phone = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 ' +
                   '(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
        viewport: { width: 412, height: 915 },
        deviceScaleFactor: 2.625,
        isMobile: true, hasTouch: true
    });
    const mp = await phone.newPage();
    const mobileErrors = [];
    mp.on('pageerror', e => mobileErrors.push(String(e)));
    await mp.goto(`${BASE}/index.html`, { waitUntil: 'load' });
    await mp.waitForFunction(APP_READY, null, { timeout: 30000 });
    const m = await mp.evaluate(() => ({
        detected: IS_MOBILE,
        pop: app.state.populationSize, elite: app.state.eliteClones,
        laps: app.state.targetLaps, hidden: app.state.hiddenLayers,
        cars: app.state.cars.length,
        stride: Engine.master.ex.brain_stride(),
        workers: Engine.workers.length, cores: Engine.hardwareCores,
        canvasW: document.getElementById('sim-canvas').width,
        sliderPop: +document.getElementById('cfg-populationSize').value,
        sliderElite: +document.getElementById('cfg-eliteClones').value,
        sliderLaps: +document.getElementById('cfg-targetLaps').value,
        sliderHidden: +document.getElementById('cfg-hiddenLayers').value
    }));
    const d = DEFAULTS_MOBILE;
    check(m.detected, 'a phone is recognised as one');
    check(m.pop === d.pop && m.elite === d.elite && m.laps === d.laps && m.hidden === d.hidden,
        'phone defaults are the lighter set',
        `pop ${m.pop}, elite ${m.elite}, laps ${m.laps}, ${m.hidden} hidden units`);
    check(m.cars === d.pop, 'and the field that was actually built matches', `${m.cars} cars`);
    // 11 inputs x h, plus h x 2 out, plus h hidden biases and 2 output ones.
    check(m.stride === 11 * d.hidden + d.hidden * 2 + d.hidden + 2,
        'the smaller brain reaches wasm, not just the slider', `stride ${m.stride}`);
    check(m.workers > 0 && m.workers <= 4 && m.workers <= m.cores,
        'the worker pool is capped rather than one per logical core',
        `${m.workers} of ${m.cores}`);
    check(m.canvasW > 0 && m.canvasW < 1200, 'the canvas rasterises fewer pixels than the world',
        `${m.canvasW}px wide backing store`);
    check(m.sliderPop === m.pop && m.sliderElite === m.elite && m.sliderLaps === m.laps
        && m.sliderHidden === m.hidden,
        'and the sliders show what is actually in force');
    check(mobileErrors.length === 0, 'no page errors on mobile',
        mobileErrors.length ? mobileErrors.slice(0, 2).join(' | ') : 'clean');

    // --- touch: one-finger pan, tap-to-select, two-finger pinch-zoom ----
    // Playwright's high-level touchscreen only drives a single contact
    // point, so a pinch needs the CDP Input domain directly. This dispatches
    // the same touchstart/touchmove/touchend sequence a real gesture would;
    // Chromium turns that into the pointerdown/pointermove/pointerup events
    // _touchStart/_touchMove/_touchEnd actually run on, so a real two-finger
    // gesture is what's being proven, not a hand call to the handlers.
    const cdp = await phone.newCDPSession(mp);
    const dispatchTouch = (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts });
    await mp.evaluate(() => { app.state.isRunning = false; app.resetView(); app.releaseSpectate(); });
    const rect = await mp.evaluate(() => {
        const r = document.getElementById('sim-canvas').getBoundingClientRect();
        return { left: r.left, top: r.top, width: r.width, height: r.height };
    });

    // One-finger drag pans the camera by exactly as far as the finger moved
    // — the same formula movePan uses, checked against real dispatched touch
    // input rather than a synthetic pointer event. Zoomed in first: at zoom 1
    // (the minimum) _clampView pins pan dead centre by design — the view
    // already covers the whole map, so there is nowhere for a pan to go —
    // and a drag at zoom 1 would trivially "pass" by not moving at all.
    {
        await mp.evaluate(() => { app.state.view.zoom = 3; app._clampView(); });
        const start = { x: rect.left + rect.width * 0.5, y: rect.top + rect.height * 0.5 };
        const end = { x: start.x + 40, y: start.y - 25 };
        const before = await mp.evaluate(() => ({ panX: app.state.view.panX, panY: app.state.view.panY }));
        await dispatchTouch('touchStart', [{ x: start.x, y: start.y, id: 0 }]);
        await dispatchTouch('touchMove', [{ x: (start.x + end.x) / 2, y: (start.y + end.y) / 2, id: 0 }]);
        await dispatchTouch('touchMove', [{ x: end.x, y: end.y, id: 0 }]);
        await dispatchTouch('touchEnd', []);
        const after = await mp.evaluate(() => ({ panX: app.state.view.panX, panY: app.state.view.panY }));
        const expected = await mp.evaluate(({ sx, sy, ex, ey }) => {
            const b0 = app._toBackingPx({ clientX: sx, clientY: sy });
            const b1 = app._toBackingPx({ clientX: ex, clientY: ey });
            const z = app.state.view.zoom * app._renderScale;
            return { dx: -(b1.x - b0.x) / z, dy: -(b1.y - b0.y) / z };
        }, { sx: start.x, sy: start.y, ex: end.x, ey: end.y });
        const gotDx = after.panX - before.panX, gotDy = after.panY - before.panY;
        check(Math.abs(gotDx - expected.dx) < 0.5 && Math.abs(gotDy - expected.dy) < 0.5,
            'a one-finger touch drag pans the camera',
            `panned (${gotDx.toFixed(2)}, ${gotDy.toFixed(2)}), expected (${expected.dx.toFixed(2)}, ${expected.dy.toFixed(2)})`);
        await mp.evaluate(() => app.resetView());
    }

    // A tap that doesn't turn into a drag selects the car underneath it —
    // every car starts at the same point on the still-unstarted population,
    // so any living car under the tap is proof the hit-test ran.
    {
        const carClient = await mp.evaluate(() => {
            const c = app.state.cars.find(c => !c.crashed);
            const v = app._viewMatrix(), s = app._renderScale;
            const r = document.getElementById('sim-canvas').getBoundingClientRect();
            const cnv = document.getElementById('sim-canvas');
            const scale = Math.min(r.width / cnv.width, r.height / cnv.height);
            const offX = (r.width - cnv.width * scale) / 2, offY = (r.height - cnv.height * scale) / 2;
            const bx = v.z * c.x + v.e, by = v.z * c.y + v.f;
            return { x: r.left + offX + bx * scale, y: r.top + offY + by * scale };
        });
        await dispatchTouch('touchStart', [{ x: carClient.x, y: carClient.y, id: 0 }]);
        await dispatchTouch('touchEnd', []);
        const picked = await mp.evaluate(() => {
            const id = app.state.spectateCarId;
            return { id, valid: id !== null && !!app.state.cars[id] && !app.state.cars[id].crashed };
        });
        check(picked.valid, 'a tap that does not drag selects the car underneath it', `spectateCarId=${picked.id}`);
        await mp.evaluate(() => app.releaseSpectate());
    }

    // Two fingers spreading apart, symmetric about the canvas centre, zoom in
    // about that centre — the pinch's anchoring keeps the world point under
    // the midpoint fixed, and here the midpoint IS the view's own centre, so
    // the pan should come back out exactly where it started.
    {
        await mp.evaluate(() => app.resetView());
        const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
        const before = await mp.evaluate(() => ({ zoom: app.state.view.zoom, panX: app.state.view.panX, panY: app.state.view.panY }));
        await dispatchTouch('touchStart', [{ x: cx - 30, y: cy, id: 0 }, { x: cx + 30, y: cy, id: 1 }]);
        await dispatchTouch('touchMove', [{ x: cx - 50, y: cy, id: 0 }, { x: cx + 50, y: cy, id: 1 }]);
        await dispatchTouch('touchMove', [{ x: cx - 70, y: cy, id: 0 }, { x: cx + 70, y: cy, id: 1 }]);
        await dispatchTouch('touchEnd', []);
        const after = await mp.evaluate(() => ({ zoom: app.state.view.zoom, panX: app.state.view.panX, panY: app.state.view.panY }));
        const expectedZoom = Math.max(1, Math.min(8, before.zoom * (140 / 60)));
        check(Math.abs(after.zoom - expectedZoom) < expectedZoom * 0.05,
            'a two-finger pinch zooms', `zoom ${before.zoom.toFixed(2)} -> ${after.zoom.toFixed(2)}, expected ~${expectedZoom.toFixed(2)}`);
        check(Math.abs(after.panX - before.panX) < 1 && Math.abs(after.panY - before.panY) < 1,
            'and stays anchored on the point under the pinch — a centred pinch leaves the view centred',
            `pan drifted by (${(after.panX - before.panX).toFixed(3)}, ${(after.panY - before.panY).toFixed(3)})`);
        await mp.evaluate(() => app.resetView());
    }

    await phone.close();
}

await browser.close();
console.log();
if (failures) { console.error(`${failures} end-to-end check(s) failed.`); process.exit(1); }
console.log('end-to-end: the app works.');
