// looptest.mjs — the fixed-timestep clock (clock.js) under fake hosts.
//
// World speed at 1x must be a property of wall-clock time alone: a 30, 60, 90
// or 120 Hz display, timer jitter, a repaint cap of 30 or 60, and any amount
// of toggling hyper mode or the cap must all yield the same 60 physics ticks
// per second — and a stall must never be caught up in a burst.
//
//   node tools/looptest.mjs
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const SimClock = require('../clock.js');

let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};
const fresh = () => { SimClock._last = null; SimClock._acc = 0; SimClock.resetRender(); };

// Drive the clock the way app.loop does: advance, take, maybe paint.
// `jitter` shifts each callback by up to ±jitter ms (deterministic LCG).
function host({ hz, ms, renderHz = 60, jitter = 0, hyperAt = () => false, seed = 1 }) {
    fresh();
    let ticks = 0, renders = 0, s = seed;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296) * 2 - 1;
    const frame = 1000 / hz;
    for (let k = 0; ; k++) {
        const now = 1000 + k * frame + (k ? rnd() * jitter : 0);
        if (now > 1000 + ms) break;
        const hyper = hyperAt(k);
        SimClock.advance(now, !hyper);
        if (!hyper) ticks += SimClock.take();
        if (!hyper && SimClock.renderDue(now, renderHz)) renders++;
    }
    return { ticks, renders };
}

for (const hz of [30, 60, 90, 120, 144]) {
    for (const renderHz of [30, 60]) {
        const r = host({ hz, ms: 10000, renderHz });
        check(Math.abs(r.ticks - 600) <= 1, `${hz} Hz host, ${renderHz} fps cap: 10 s = 600 ticks`, `${r.ticks} ticks`);
        const want = Math.min(renderHz, hz) * 10;
        check(Math.abs(r.renders - want) <= 2, `${hz} Hz host, ${renderHz} fps cap: ${want} paints`, `${r.renders}`);
    }
}
for (const hz of [30, 60, 120]) {
    const r = host({ hz, ms: 1000 });
    check(r.ticks === 60, `${hz} Hz host: exactly 60 ticks per 1000 ms`, `${r.ticks}`);
}
{
    const r = host({ hz: 60, ms: 10000, renderHz: 30, jitter: 2 });
    check(Math.abs(r.ticks - 600) <= 1, '60 Hz host with ±2 ms jitter: 600 ticks', `${r.ticks}`);
    check(Math.abs(r.renders - 300) <= 3, '60 Hz host with ±2 ms jitter, 30 fps cap: ~300 paints', `${r.renders}`);
}

// A 5 s stall (backgrounded tab) costs at most MAX_FRAME_MS of catch-up.
{
    fresh();
    SimClock.advance(0, true); SimClock.take();
    SimClock.advance(5000, true);
    const burst = SimClock.take();
    check(burst <= Math.ceil(SimClock.MAX_FRAME_MS / SimClock.SIM_DT), 'a 5 s stall is not fast-forwarded', `${burst} ticks`);
}

// Toggling hyper mode and the repaint cap 10 times: normal-mode time still
// yields exactly 60 ticks per second of normal-mode time, no burst on return.
{
    fresh();
    let ticks = 0, normalMs = 0, now = 0, maxPerFrame = 0;
    for (let round = 0; round < 10; round++) {
        for (const hyper of [true, false]) {
            for (let k = 0; k < 60; k++) {
                now += 1000 / 60;
                SimClock.advance(now, !hyper);
                if (!hyper) {
                    const t = SimClock.take();
                    ticks += t; normalMs += 1000 / 60; maxPerFrame = Math.max(maxPerFrame, t);
                    SimClock.renderDue(now, round % 2 ? 30 : 60);
                }
            }
        }
    }
    check(Math.abs(ticks - normalMs / SimClock.SIM_DT) <= 1, '10 hyper/fps toggles: ticks track normal-mode time only', `${ticks} ticks in ${normalMs.toFixed(0)} ms`);
    check(maxPerFrame === 1, '10 hyper/fps toggles: never more than one tick per 60 Hz frame', `max ${maxPerFrame}`);
}

console.log(failures ? `\n${failures} failure(s)` : '\nall loop timing checks passed.');
process.exit(failures ? 1 : 0);
