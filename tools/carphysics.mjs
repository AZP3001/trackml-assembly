// carphysics.mjs — proves three specific things about updateCar in sim.c:
//
//   1. A stopped car cannot turn. Steering while car_speed is ~0 used to
//      still swing the heading by 20% of turnSpeed every frame — a "parked"
//      car visibly pivoting in place.
//   2. Braking is proportional to how hard the AI presses it. The old code
//      multiplied speed by a flat 0.95 for ANY non-positive throttle, so a
//      throttle of -0.02 and -1.0 braked identically hard — brakes that
//      "instantly" snapped speed down regardless of how lightly they were
//      touched.
//   3. Turning authority is grip-limited: full at low speed, falling off as
//      roughly ref_speed/speed above TURN_GRIP_REF_SPEED — the opposite of
//      the old (0.2 + 0.8*speedFactor) curve, which made a car turn WORSE
//      the slower it went and BEST at speed.
//   4. A car with no momentum is eliminated once the spawn grace window
//      passes, whether it never launched or braked to a standstill. Heading
//      changes don't count as momentum — only speed does.
//
// Cars are driven with a synthetic brain — all input weights zero, only the
// output bias set — so steer/throttle are exactly what the test wants every
// frame, regardless of what the sensors see. That isolates updateCar's
// physics from the AI: nothing here depends on a network having learned
// anything.
//
//   node tools/carphysics.mjs [path/to/sim.wasm]
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

console.log(`car physics test (${wasmPath})\n`);

// A small, ordinary track. Every scenario below moves a car at most a few
// dozen pixels from the start line, so its size doesn't matter — it only has
// to exist so pop_init/run don't refuse to move a car with no track loaded.
const CX = 600, CY = 450;
const path = Array.from({ length: 16 }, (_, i) => {
    const a = (Math.PI * 2 * i) / 16;
    return { x: Math.round(CX + Math.cos(a) * 300), y: Math.round(CY + Math.sin(a) * 300), type: 'corner', radius: 60 };
});
const pIn = f32(w.path_in_ptr(), path.length * 4);
path.forEach((p, i) => { pIn[i * 4] = p.x; pIn[i * 4 + 1] = p.y; pIn[i * 4 + 2] = 1; pIn[i * 4 + 3] = 60; });
if (!w.track_build(w.path_in_ptr(), path.length, 60, 0, 0, 0, 0, 0, w.zone_in_ptr(), 0, 0, 0)) {
    throw new Error('track_build failed');
}

// Must track IN_N in sim.c: 7 sensors, speed, gate bearing, and the previous
// frame's two outputs.
const IN_N = 11, H = 1, OUT_N = 2;

// Writes a brain with every input/hidden weight zeroed, so its two outputs
// are exactly tanh(steerBias) and tanh(throttleBias) every frame regardless
// of sensors — a constant-output "AI" for testing the physics underneath it.
function setBiasBrain(carIndex, steerBias, throttleBias) {
    const stride = w.brain_stride();
    const b = f32(w.brains_ptr() + carIndex * stride * 4, stride);
    b.fill(0);
    const offBiasO = IN_N * H + H * OUT_N + H;   // 11*1 + 1*2 + 1 = 14
    b[offBiasO] = steerBias;
    b[offBiasO + 1] = throttleBias;
}

function readCar(carIndex) {
    w.write_render();
    const stride = w.render_stride();
    const r = f32(w.render_ptr(), (carIndex + 1) * stride);
    const o = carIndex * stride;
    return { crashed: r[o + 1] === 1, x: r[o + 2], y: r[o + 3], angle: r[o + 4], speed: r[o + 5] };
}

const BIG = 20.0; // tanh(20) saturates to 1.0 well past f32 precision
const STOPPED_SPEED = 0.05;   // mirrors sim.c

// ---------------------------------------------------------------------------
// 1. A stopped car cannot turn.
// ---------------------------------------------------------------------------
{
    w.set_config(10, 0.05, 0.5, 0.3, 1000, 99, 0.15, H);   // exaggerated turnSpeed=0.5: any leak would be obvious
    w.pop_init(1, 0, H, 1);
    setBiasBrain(0, BIG, 0);   // steer = full lock, throttle = 0 (never accelerates, never brakes)
    w.pop_reset();
    const before = readCar(0);

    // Inside the spawn grace window: still alive, still stopped, and — the
    // point of the test — the heading has not moved despite full lock.
    w.run(10);
    const inGrace = readCar(0);
    check(!inGrace.crashed, 'a stopped car survives the spawn grace window', `10 frames in`);
    check(inGrace.speed === 0, 'a car given zero throttle never gains speed', `speed ${inGrace.speed}`);
    check(inGrace.angle === before.angle, 'a car at a standstill cannot turn',
        `angle ${before.angle} -> ${inGrace.angle} over 10 frames of full steering lock`);

    // Past it: no momentum, no race.
    w.run(30);
    const after = readCar(0);
    check(after.crashed, 'a car with no momentum is eliminated once the grace window passes',
        `crashed=${after.crashed} at speed ${after.speed}`);
    check(after.angle === before.angle, 'and spinning the wheel never bought it any momentum',
        `angle unchanged at ${after.angle}`);
}

// ---------------------------------------------------------------------------
// 1b. Elimination is on SPEED, not on steering: a car that is moving keeps
//     racing however hard it is turning, and one that brakes to a stop
//     mid-track is out the same as one that never left the line.
// ---------------------------------------------------------------------------
{
    w.set_config(20, 1.0, 0.04, 0.5, 1000, 99, 0.15, H);
    w.pop_init(1, 0, H, 7);
    setBiasBrain(0, BIG, BIG);   // full steer AND full throttle
    w.pop_reset();
    w.run(1); w.run(1);          // one frame of lag, then the throttle lands
    const rolling = readCar(0);
    w.run(20);
    const stillRolling = readCar(0);
    check(!stillRolling.crashed, 'a moving car is never eliminated for turning hard',
        `speed ${stillRolling.speed.toFixed(3)} after 20 frames at full lock`);
    check(rolling.speed > STOPPED_SPEED, 'and it really was moving', `speed ${rolling.speed.toFixed(3)}`);

    // Now stamp on the brakes and watch it get eliminated when it runs out
    // of momentum, rather than sitting there until TTL.
    setBiasBrain(0, 0, -BIG);
    const framesToStop = (() => {
        for (let n = 0; n < 200; n++) {
            w.run(1);
            if (readCar(0).crashed) return n + 1;
        }
        return null;
    })();
    check(framesToStop !== null, 'braking to a standstill eliminates the car',
        framesToStop !== null ? `out after ${framesToStop} braking frames` : 'still alive after 200 frames');
}

// ---------------------------------------------------------------------------
// 2. Braking scales with throttle magnitude, not a flat snap.
// ---------------------------------------------------------------------------
{
    const BRAKE_STRENGTH = 0.3;
    w.set_config(20, 1.0, 0.04, BRAKE_STRENGTH, 1000, 99, 0.15, H);
    w.pop_init(2, 0, H, 2);
    // Both cars: full throttle, no steering, for one frame — a single big
    // jump in speed rather than a long straight-line run, so the whole test
    // stays within a couple of pixels of the start line.
    setBiasBrain(0, 0, BIG);
    setBiasBrain(1, 0, BIG);
    w.pop_reset();
    // reset_car zeroes car_out, so the first run(1) still moves on that zero
    // output and only computes ours at its end — same lag noted below. A
    // second run(1) is the one that actually applies full throttle.
    w.run(1);
    w.run(1);
    const cruise0 = readCar(0), cruise1 = readCar(1);
    check(!cruise0.crashed && !cruise1.crashed, 'both cars survive the speed-up frame');
    check(Math.abs(cruise0.speed - cruise1.speed) < 1e-5, 'both cars reach the same cruising speed before braking',
        `${cruise0.speed} vs ${cruise1.speed}`);

    // Car 0 brakes lightly (-0.1), car 1 brakes hard (-1.0). One frame to let
    // the new brain get evaluated, a second to apply it and measure — see the
    // comment on updateCar/feedForward ordering in sim.c: a brain written now
    // only steers the frame after next.
    setBiasBrain(0, 0, -0.10033534773107558);  // atanh(0.1) — light brake, throttle ≈ -0.1
    setBiasBrain(1, 0, -BIG);                  // throttle ≈ -1.0 — full brake
    w.run(1);
    const before0 = readCar(0), before1 = readCar(1);
    w.run(1);
    const after0 = readCar(0), after1 = readCar(1);

    const dec0 = before0.speed - after0.speed;
    const dec1 = before1.speed - after1.speed;
    check(!after0.crashed && !after1.crashed, 'both cars survive braking');
    check(dec1 > dec0 * 3, 'a harder brake input decelerates noticeably more than a lighter one',
        `light brake -${dec0.toFixed(4)}, hard brake -${dec1.toFixed(4)}`);
    check(dec0 > 0 && dec0 < before0.speed * 0.5, 'a light brake trims speed rather than snapping it away',
        `-${dec0.toFixed(4)} off ${before0.speed.toFixed(4)}`);

    // Exact-formula check on the hard-braking car. The brake term itself
    // removes min(brakeStrength, speed); updateCar then applies its usual
    // unconditional 0.99 drag on top (steer is 0 here, so the lateral-grip
    // term contributes nothing extra — velocity stays exactly aligned with
    // heading the whole time).
    const brakeDec1 = Math.min(BRAKE_STRENGTH, before1.speed);
    const expectedDec1 = before1.speed - (before1.speed - brakeDec1) * 0.99;
    check(Math.abs(dec1 - expectedDec1) < 1e-4, 'hard-brake deceleration matches the brake + drag formula exactly',
        `got ${dec1.toFixed(5)}, expected ${expectedDec1.toFixed(5)}`);
}

// ---------------------------------------------------------------------------
// 3. Turning authority is grip-limited: full below TURN_GRIP_REF_SPEED (3.0),
//    falling off roughly as 1/speed above it — never boosted by speed the way
//    the old (0.2 + 0.8*speedFactor) curve did.
// ---------------------------------------------------------------------------
function sampleTurnDelta(turnSpeed, jumpAccel, targetSpeedLabel) {
    w.set_config(50, jumpAccel, turnSpeed, 0.3, 1000, 99, 0.15, H);
    w.pop_init(1, 0, H, 3);
    setBiasBrain(0, 0, BIG);   // one frame of huge accel = a single clean jump to a known speed
    w.pop_reset();
    w.run(1);
    const jumped = readCar(0);

    setBiasBrain(0, BIG, 0);   // now: full steering lock, zero throttle (speed barely drifts from drag)
    w.run(1);                  // this frame still moves on the OLD (accel) output; new brain lands at its end
    const before = readCar(0);
    w.run(1);                  // this frame actually turns
    const after = readCar(0);

    check(!after.crashed, `${targetSpeedLabel}: car survives the sample`, `speed ${before.speed}`);
    const delta = after.angle - before.angle;
    const authority = Math.min(3.0 / before.speed, 1.0);
    const expected = turnSpeed * authority; // steer ≈ tanh(BIG) ≈ 1.0
    return { speed: before.speed, delta, expected };
}

{
    const TURN_SPEED = 0.1;
    const low = sampleTurnDelta(TURN_SPEED, 1.0, 'low-speed sample');   // small jump: speed stays under the 3.0 grip reference
    const high = sampleTurnDelta(TURN_SPEED, 8.0, 'high-speed sample'); // big jump: speed clears 3.0, authority should fall off

    check(low.speed < 3.0, 'low-speed sample really is below the grip-limited threshold', `speed ${low.speed.toFixed(3)}`);
    check(high.speed > 3.0, 'high-speed sample really is above the grip-limited threshold', `speed ${high.speed.toFixed(3)}`);
    check(Math.abs(low.delta - low.expected) < 1e-4, 'low-speed turn rate matches the formula exactly',
        `got ${low.delta.toFixed(5)}, expected ${low.expected.toFixed(5)}`);
    check(Math.abs(high.delta - high.expected) < 1e-4, 'high-speed turn rate matches the formula exactly',
        `got ${high.delta.toFixed(5)}, expected ${high.expected.toFixed(5)}`);
    check(low.delta > high.delta, 'a slower car turns MORE per frame than a faster one, not less',
        `${low.speed.toFixed(2)} px/f -> ${low.delta.toFixed(4)} rad/f,  ${high.speed.toFixed(2)} px/f -> ${high.delta.toFixed(4)} rad/f`);
    check(Math.abs(low.delta - TURN_SPEED) < 1e-4, 'below the grip reference speed, turning gets full authority',
        `${low.delta.toFixed(5)} vs turnSpeed ${TURN_SPEED}`);
}

console.log();
if (failures) {
    console.log(`${failures} check(s) failed.`);
    process.exit(1);
}
console.log('braking is proportional, turning is grip-limited, and a parked car cannot pivot.');
