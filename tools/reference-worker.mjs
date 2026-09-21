// reference-worker.js — started as the JS TrackML edition's simulation worker,
// copied verbatim out of script.js in AZP3001/main. Nothing in the app loads
// this; tools/paritytest.mjs runs it side by side with sim.wasm on identical
// tracks and identical brains to prove the two agree.
//
// AZP3001/main is archived and frozen — it will not change again — while this
// project's physics keeps moving (see the braking/turning/grip rework in
// updateCar below, none of which the archived edition ever had). So this file
// is no longer "re-copy it if the JS edition changes": it is now the
// independent-language mirror of sim.c's CURRENT physics, kept hand-in-hand
// with it. Its job is unchanged — an honest second implementation that a real
// wasm bug (a sign error, an off-by-one bucket, a transposed matrix) blows up
// against — it just tracks sim.c now instead of a repo that no longer moves.
//
// The one concession to running under Node: the original was a template string
// evaluated inside a real Worker, so it closed over `self`. Here it is an
// ES module that takes a `self`-alike from the caller.

export function createReferenceWorker(self) {

    const CAR_WIDTH = 14, CAR_HEIGHT = 7, SENSOR_LENGTH = 180;
    const SENSOR_ANGLES = [-Math.PI/2, -Math.PI/3, -Math.PI/6, 0, Math.PI/6, Math.PI/3, Math.PI/2];
    const SENS_LEN = SENSOR_ANGLES.length;

    let cachedTrack = null;
    let cachedConfig = null;
    let localCars = [];
    let wallsBySegment = [];

    // Pre-compute spatial wall lookup structure ONCE per track load.
    // Bucket walls by segmentIndex first (O(walls)) instead of rescanning the
    // full wall list with an O(n) .includes() check per segment (was O(segs^2)),
    // which matters a lot now that PNG-import/freehand tracks can have far more points.
    function initTrackPrecomp() {
        if (!cachedTrack) return;
        const tSegs = cachedTrack.checkpoints.length;
        // How many checkpoints back/forward a car has to look for walls. A
        // sensor reaches 180px, so this has to cover that in *pixels* — derive
        // it from the checkpoint spacing instead of hard-coding a count, which
        // silently under-covered whenever the spacing changed.
        const step = cachedTrack.segStep || 34;
        const back = Math.max(4, Math.ceil(240 / step));
        const fwd = Math.max(5, Math.ceil(260 / step));
        const bucket = new Array(tSegs);
        for (let i = 0; i < tSegs; i++) bucket[i] = [];
        const undefWalls = [];
        for (const w of cachedTrack.walls) {
            if (w.segmentIndex === undefined) undefWalls.push(w);
            else bucket[w.segmentIndex].push(w);
        }
        wallsBySegment = new Array(tSegs);
        for (let i = 0; i < tSegs; i++) {
            const set = new Set(undefWalls);
            for (let j = -back; j <= fwd; j++) {
                const seg = (i + j + tSegs * 10) % tSegs;
                for (const w of bucket[seg]) set.add(w);
            }
            wallsBySegment[i] = Array.from(set);
        }
    }

    // High performance inline intersection math (zero object allocations)
    function fastIntersect(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy) {
        const bottom = (Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay);
        if (bottom === 0) return false;
        const t = ((Dx - Cx) * (Ay - Cy) - (Dy - Cy) * (Ax - Cx)) / bottom;
        if (t < 0 || t > 1) return false;
        const u = ((Cy - Ay) * (Ax - Bx) - (Cx - Ax) * (Ay - By)) / bottom;
        if (u < 0 || u > 1) return false;
        return true;
    }

    function fastIntersectDist(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy) {
        const bottom = (Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay);
        if (bottom === 0) return 1.0;
        const t = ((Dx - Cx) * (Ay - Cy) - (Dy - Cy) * (Ax - Cx)) / bottom;
        if (t < 0 || t > 1) return 1.0;
        const u = ((Cy - Ay) * (Ax - Bx) - (Cx - Ax) * (Ay - By)) / bottom;
        if (u >= 0 && u <= 1) return t;
        return 1.0;
    }

    // Pre-allocated flat Float32Array mutation — weightsIH/weightsHO are flat
    // row-major typed arrays (row = source neuron, stride = dest layer size).
    // Flat typed arrays clone MUCH faster through postMessage than arrays-of-arrays
    // of boxed doubles, which is the dominant cost of shipping a generation to workers.
    function feedForwardCPU(c) {
        let b = c.brain, ins = c.sensorsInputs, hL = c.hL, oL = c.oL;
        const hLen = hL.length, oLen = oL.length, inLen = ins.length;
        const wIH = b.weightsIH, wHO = b.weightsHO, bH = b.biasH, bO = b.biasO;
        for (let i = 0; i < hLen; i++) {
            let sum = bH[i];
            for (let j = 0; j < inLen; j++) sum += ins[j] * wIH[j*hLen+i];
            hL[i] = Math.tanh(sum);
        }
        for (let i = 0; i < oLen; i++) {
            let sum = bO[i];
            for (let j = 0; j < hLen; j++) sum += hL[j] * wHO[j*oLen+i];
            oL[i] = Math.tanh(sum);
        }
    }

    // Fixed lateral grip — see CAR_LAT_GRIP in sim.c for why this is no
    // longer a config field.
    const CAR_LAT_GRIP = 0.93;
    // See STOPPED_SPEED / STOPPED_GRACE_FRAMES / TURN_GRIP_REF_SPEED in sim.c.
    const STOPPED_SPEED = 0.05, STOPPED_GRACE_FRAMES = 15, TURN_GRIP_REF_SPEED = 3.0;

    function updateCar(c, config) {
        if (c.crashed) return;
        c.timeToLive--; c.framesAlive++;
        if (c.timeToLive <= 0) { c.crashed = true; return; }

        const steer = c.oL[0] || 0;
        const throttle = c.oL[1] || 0;

        // Grip-limited turning: authority falls off past TURN_GRIP_REF_SPEED
        // instead of ramping UP with speed, and a car below STOPPED_SPEED
        // gets none at all — see sim.c's updateCar for the reasoning.
        if (c.speed > STOPPED_SPEED) {
            const authority = Math.min(TURN_GRIP_REF_SPEED / c.speed, 1.0);
            c.angle += steer * config.turnSpeed * authority;
        }

        const cosA = Math.cos(c.angle), sinA = Math.sin(c.angle);
        let vx = c.vx, vy = c.vy;

        if (throttle > 0) {
            vx += cosA * throttle * config.acceleration; vy += sinA * throttle * config.acceleration;
        } else if (throttle < 0) {
            // Proportional braking — see sim.c's updateCar.
            const sp = Math.sqrt(vx * vx + vy * vy);
            if (sp > 1.0e-4) {
                const dec = Math.min(-throttle * config.brakeStrength, sp);
                const k = (sp - dec) / sp;
                vx *= k; vy *= k;
            }
        }

        const latVel = vx * (-sinA) + vy * cosA;
        let grip = CAR_LAT_GRIP; if(Math.abs(latVel) > 2.5) grip *= 0.8;

        vx += (-sinA) * -latVel * grip;
        vy += cosA * -latVel * grip;
        vx *= 0.99; vy *= 0.99;
        
        let speed = Math.sqrt(vx*vx + vy*vy);
        if(speed > config.maxSpeed) { const r = config.maxSpeed/speed; vx *= r; vy *= r; speed = config.maxSpeed; }

        c.vx = vx; c.vy = vy; c.speed = speed;

        // No momentum, no race — see sim.c's updateCar.
        if (c.framesAlive > STOPPED_GRACE_FRAMES && speed < STOPPED_SPEED) { c.crashed = true; return; }

        const prevX = c.x, prevY = c.y;
        c.x += vx; c.y += vy;
        c.fitness += (speed / config.maxSpeed) * 0.1;

        if(c.x < -100 || c.x > 1300 || c.y < -100 || c.y > 1000) { c.crashed = true; return; }

        const curSeg = cachedTrack.checkpoints[c.nextCheckpointIndex] ? cachedTrack.checkpoints[c.nextCheckpointIndex].index : 0;
        const nearbyWalls = wallsBySegment[curSeg] || [];

        const hw = CAR_WIDTH/2, hh = CAR_HEIGHT/2;
        const cxs = [
            c.x + cosA*hw - sinA*hh, c.x + cosA*hw + sinA*hh,
            c.x - cosA*hw + sinA*hh, c.x - cosA*hw - sinA*hh
        ];
        const cys = [
            c.y + sinA*hw + cosA*hh, c.y + sinA*hw - cosA*hh,
            c.y - sinA*hw - cosA*hh, c.y - sinA*hw + cosA*hh
        ];

        for (let i = 0; i < nearbyWalls.length; i++) {
            let w = nearbyWalls[i];
            if (fastIntersect(prevX, prevY, c.x, c.y, w.p1.x, w.p1.y, w.p2.x, w.p2.y)) { c.crashed = true; c.fitness -= 50; return; }
            for(let j=0; j<4; j++) {
                let jn = (j+1)%4;
                if (fastIntersect(cxs[j], cys[j], cxs[jn], cys[jn], w.p1.x, w.p1.y, w.p2.x, w.p2.y)) { c.crashed = true; c.fitness -= 50; return; }
            }
        }

        let fitMult = 1.0;
        for (let i = 0; i < cachedTrack.zones.length; i++) {
            let z = cachedTrack.zones[i];
            let dx = c.x - z.x, dy = c.y - z.y;
            if (dx*dx + dy*dy < z.radius*z.radius) {
                if (z.type === 'speed' && c.speed > config.maxSpeed * 0.7) c.fitness += 2.0;
                else if (z.type === 'precision') c.fitness += 2.0;
                else if (z.type === 'focus') fitMult = 3.0;
                else if (z.type === 'spawnkill') {
                    const killTimer = (z.killTimer !== undefined ? z.killTimer : 150);
                    if (c.framesAlive >= killTimer) { c.crashed = true; return; }
                }
            }
        }

        const nCP = cachedTrack.checkpoints[c.nextCheckpointIndex];
        if (nCP) {
            let dx = c.x - nCP.center.x, dy = c.y - nCP.center.y;
            if (dx*dx + dy*dy > 160000) { c.crashed = true; return; } // off course

            const cx = (nCP.p1.x+nCP.p2.x)/2, cy = (nCP.p1.y+nCP.p2.y)/2;
            let cdx = c.x - cx, cdy = c.y - cy;
            
            if (cdx*cdx + cdy*cdy < 2500) { 
                 if (fastIntersect(prevX, prevY, c.x, c.y, nCP.p1.x, nCP.p1.y, nCP.p2.x, nCP.p2.y) || (cdx*cdx+cdy*cdy < 400)) { 
                    c.checkpointsReached++; c.nextCheckpointIndex = (c.nextCheckpointIndex + 1) % cachedTrack.checkpoints.length;
                    c.timeToLive += 150; if (c.timeToLive > 600) c.timeToLive = 600;
                    c.fitness += 500 * fitMult; 
                    if (c.nextCheckpointIndex === 0 && cachedTrack.checkpoints.length > 2) {
                        c.completedLaps++; 
                        c.lapTimes.push(c.framesAlive); 
                        c.lastLapTime = (c.lapTimes[c.lapTimes.length-1] - (c.lapTimes[c.lapTimes.length-2]||0)) / 60;
                        c.fitness += 3000 * fitMult; 
                    }
                 }
            }
        }

        for (let i = 0; i < SENS_LEN; i++) {
            let rA = c.angle + SENSOR_ANGLES[i];
            let ex = c.x + Math.cos(rA) * SENSOR_LENGTH, ey = c.y + Math.sin(rA) * SENSOR_LENGTH;
            let minT = 1.0;
            for (let j = 0; j < nearbyWalls.length; j++) {
                let w = nearbyWalls[j];
                let t = fastIntersectDist(c.x, c.y, ex, ey, w.p1.x, w.p1.y, w.p2.x, w.p2.y);
                if (t < minT) minT = t;
            }
            c.sensorsInputs[i] = 1.0 - minT;
            c.sensors[i] = 1.0 - minT; 
        }

        const tX = (nCP.p1.x + nCP.p2.x)/2, tY = (nCP.p1.y + nCP.p2.y)/2;
        let relAng = Math.atan2(tY - c.y, tX - c.x) - c.angle;
        while(relAng > Math.PI) relAng -= 2*Math.PI; while(relAng < -Math.PI) relAng += 2*Math.PI;
        
        c.sensorsInputs[SENS_LEN] = c.speed / config.maxSpeed;
        c.sensorsInputs[SENS_LEN + 1] = relAng / Math.PI;
        
        feedForwardCPU(c);
    }

    self.onmessage = function(e) {
        if(e.data.type === 'initTrack') { 
            cachedTrack = e.data.track; 
            cachedConfig = e.data.config; 
            initTrackPrecomp();
        }
        else if(e.data.type === 'initCars') { 
            localCars = e.data.cars; 
            for(let i=0; i<localCars.length; i++) {
                let c = localCars[i];
                c.hL = new Float32Array(c.brain.biasH.length);
                c.oL = new Float32Array(c.brain.biasO.length);
                c.sensorsInputs = new Float32Array(SENS_LEN + 2);
                c.sensors = new Float32Array(SENS_LEN);
            }
        }
        else if(e.data.type === 'run') {
            const iters = e.data.iters;
            let maxLaps = 0; let allCrashed = true;

            for(let i=0; i<iters; i++) {
                allCrashed = true;
                for(let c=0; c<localCars.length; c++) {
                    if(!localCars[c].crashed) {
                        updateCar(localCars[c], cachedConfig);
                        allCrashed = false;
                        if(localCars[c].completedLaps > maxLaps) maxLaps = localCars[c].completedLaps;
                    }
                }
                if(allCrashed || maxLaps >= cachedConfig.targetLaps) break;
            }

            const stride = 11 + SENS_LEN;
            const buffer = new Float32Array(localCars.length * stride);
            
            for(let i=0; i<localCars.length; i++) {
                let c = localCars[i];
                let idx = i * stride;
                buffer[idx] = c.id;
                buffer[idx+1] = c.crashed ? 1 : 0;
                buffer[idx+2] = c.x;
                buffer[idx+3] = c.y;
                buffer[idx+4] = c.angle;
                buffer[idx+5] = c.speed;
                buffer[idx+6] = c.oL[0] || 0;
                buffer[idx+7] = c.oL[1] || 0;
                buffer[idx+8] = c.completedLaps;
                buffer[idx+9] = c.fitness;
                buffer[idx+10] = c.lastLapTime || 0;
                for(let j=0; j<SENS_LEN; j++) buffer[idx+11+j] = c.sensors[j];
            }
            self.postMessage({ type: 'done', buffer, allCrashed, maxLaps }, [buffer.buffer]);
        }
    };
    return { onmessage: self.onmessage };
}
