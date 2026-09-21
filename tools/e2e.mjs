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
import { chromium } from 'playwright';

const BASE = process.argv[2] || 'http://127.0.0.1:8765';
let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

const browser = await chromium.launch();
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
// good fraction of cars never command any throttle and sit on the line until
// their TTL runs out. That is the simulation working, not a stuck population —
// so the second check is the one that really says "driving".
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
check(!brain.bad && brain.ih === 9 && brain.o === 2 && brain.finite,
    'brain exports in the original JSON format', `9x${brain.h}x2${brain.bad ? ' — ' + brain.bad : ''}`);

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
check(afterReload.pop === 200, 'settings reset to defaults', `population ${afterReload.pop}`);

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

await browser.close();
console.log();
if (failures) { console.error(`${failures} end-to-end check(s) failed.`); process.exit(1); }
console.log('end-to-end: the app works.');
