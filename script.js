const CANVAS_WIDTH = 1200; 
const CANVAS_HEIGHT = 900; 
const CAR_WIDTH = 14;
const CAR_HEIGHT = 7;
const SENSOR_LENGTH = 180;
const SENSOR_ANGLES = [-Math.PI/2, -Math.PI/3, -Math.PI/6, 0, Math.PI/6, Math.PI/3, Math.PI/2];
const SENSOR_COUNT = SENSOR_ANGLES.length;

// Inline SVGs — avoids calling lucide.createIcons() on every toggle
const SVG_PLAY = `<svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
const SVG_PAUSE = `<svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;

// Cached DOM references — populated once in _initUICache(), used everywhere else
const ui = {};
// Dirty-check values for updateUI — skip DOM writes if unchanged
let _ui_gen = -1, _ui_alive = -1, _ui_allBest = null;

const SETTING_DESCRIPTIONS = {
    speedMultiplier: "Simulation cycles per frame. High values train extremely fast.",
    populationSize: "Number of cars per generation. Scales perfectly via multi-threading.",
    eliteClones: "Top performers copied to the next generation without mutation. Prevents regression.",
    mutationRate: "Randomness in brain evolution. 15-25% provides excellent exploratory learning.",
    hiddenLayers: "Brain complexity. More layers = smarter but heavier computation.",
    initialTTL: "Time to Live. Frames allowed before death if no checkpoint is reached.",
    targetLaps: "Laps needed to trigger the next generation automatically.",
    maxSpeed: "Top speed. Higher speeds require faster AI reaction times.",
    acceleration: "Engine power.", turnSpeed: "Steering sensitivity.", grip: "Lateral Friction. 93% is balanced."
};

// How many simulation steps to ask for per round trip to the workers.
// Hyper mode doesn't draw, so nothing is gained by coming back every frame and
// plenty is lost: the chunk is sized so a whole generation usually finishes
// inside one or two calls. sim.c stops early the moment every car in a slice
// has crashed or something has hit the lap target, so a big chunk is never
// wasted work.
const HYPER_CHUNK = 2500;

// The road surface is the centreline stroked at the full track width with a
// round join/cap — literally the region the barriers bound, so the two can
// never disagree. Geometry arrives as a flat Float32Array of x,y pairs straight
// out of wasm memory; there is no per-point object to walk.
//
// With auto width on, the road is no longer one width, so it can't be one
// stroke. Each segment is stroked at its own width instead — round caps make
// consecutive segments blend into a smooth taper, and overlapping strokes are
// exactly what you want where the track crosses itself. Segments are bucketed
// by rounded width so a few hundred of them still cost only a handful of paths.
function drawRoadSurface(ctx, t, color) {
    const c = t.centerF32;
    if (!c || c.length < 4) return;
    const n = c.length / 2;
    const w = t.widthF32;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    if (!t.autoWidth || !w || w.length !== n) {
        ctx.lineWidth = Math.max(2, t.trackWidth * 2);
        ctx.beginPath();
        ctx.moveTo(c[0], c[1]);
        for (let i = 2; i < c.length; i += 2) ctx.lineTo(c[i], c[i + 1]);
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
        return;
    }

    // Bucket by whole-pixel width. The widths were smoothed in wasm, so
    // neighbouring samples almost always land in the same bucket and each
    // bucket comes out as one path of mostly-contiguous segments.
    const buckets = new Map();
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        // The wider of the two ends, matching how the wasm side decides which
        // segment owns a point — otherwise the fill would fall a hair short of
        // the barrier on a taper.
        const lw = Math.max(2, Math.round(Math.max(w[i], w[j]) * 2));
        let b = buckets.get(lw);
        if (!b) { b = []; buckets.set(lw, b); }
        b.push(i);
    }
    for (const [lw, segs] of buckets) {
        ctx.lineWidth = lw;
        ctx.beginPath();
        for (const i of segs) {
            const j = (i + 1) % n;
            ctx.moveTo(c[i * 2], c[i * 2 + 1]);
            ctx.lineTo(c[j * 2], c[j * 2 + 1]);
        }
        ctx.stroke();
    }
    ctx.restore();
}

// Barriers, same deal: one path built from a flat [x1,y1,x2,y2,...] array.
function strokeWalls(ctx, t) {
    const w = t.wallsF32;
    if (!w) return;
    ctx.beginPath();
    for (let i = 0; i < w.length; i += 4) { ctx.moveTo(w[i], w[i + 1]); ctx.lineTo(w[i + 2], w[i + 3]); }
    ctx.stroke();
}

// --- No persistence, by design -----------------------------------------
//
// Nothing this app does survives a reload. Custom tracks, slider settings and
// the trained population all live in memory only, so every load starts from a
// clean slate. Use "Save AI" for a brain you want to keep and the code export
// in the track editor for a track.
//
// wipeStorage() doesn't just decline to write — it actively clears everything
// the page could be holding, including data written by an older build that did
// persist, and any Cache Storage or service worker left over from one. Reload
// really does mean reload.
function wipeStorage() {
    const clear = (store) => {
        try { store && store.clear(); } catch (e) { /* blocked or unavailable */ }
    };
    clear(window.localStorage);
    clear(window.sessionStorage);

    // A service worker would keep serving stale JS and wasm from its own cache
    // no matter what the network says, so retire any that's registered.
    try {
        if (navigator.serviceWorker && navigator.serviceWorker.getRegistrations) {
            navigator.serviceWorker.getRegistrations()
                .then(rs => rs.forEach(r => r.unregister()))
                .catch(() => {});
        }
    } catch (e) { /* not supported */ }

    try {
        if (window.caches && caches.keys) {
            caches.keys().then(keys => keys.forEach(k => caches.delete(k))).catch(() => {});
        }
    } catch (e) { /* not supported */ }
}

// --- Main Application ---
const app = {
    state: {
        populationSize: 200, eliteClones: 10, targetLaps: 3, mutationRate: 0.15, hiddenLayers: 5, initialTTL: 750,
        physics: { maxSpeed: 10, acceleration: 0.05, turnSpeed: 0.04, grip: 0.93 },
        tracks: [], currentTrackIndex: 1, cars: [], generation: 1, isRunning: false, speedMultiplier: 1, hyperMode: false,
        stats: [], globalBest: null, bestTimes: { gen: null, all: null }, isEditing: false, trackToEdit: null,
        bgCanvas: null, lapHistory: [], spectateCarId: null, aliveCount: 0
    },

    _initUICache: function() {
        const $ = id => document.getElementById(id);
        ui.statGen     = $('stat-generation');
        ui.statAlive   = $('stat-alive');
        ui.statGenMain  = $('stat-generation-main');
        ui.statAliveMain = $('stat-alive-main');
        ui.statAllBest  = $('stat-all-best');
        ui.statAllBestM = $('stat-all-best-m');
        ui.lapHistoryM  = $('lap-history-m');
        ui.telSteerL    = $('tel-steer-l');
        ui.telSteerR    = $('tel-steer-r');
        ui.telGas       = $('tel-gas');
        ui.telBrake     = $('tel-brake');
        ui.telSpeed     = $('tel-speed');
        ui.telSpeedVal  = $('tel-speed-val');
        ui.telemetryLabel = $('telemetry-label');
        ui.btnRelease   = $('btn-release-spectate');
        ui.canvas       = $('sim-canvas');
        ui.ctx          = ui.canvas.getContext('2d');
        ui.btnPlay      = $('btn-play');
        ui.btnHyper     = $('btn-hyper');
        ui.btnPlayM     = $('btn-play-m');
        ui.btnHyperM    = $('btn-hyper-m');
        ui.hyperBanner  = $('hyper-banner');
        ui.coreCount    = $('core-count');
        // Apply the pending Engine core label now that the element is cached
        if(ui.coreCount && Engine._pendingCoreLabel) ui.coreCount.innerHTML = Engine._pendingCoreLabel;
        // Click/tap a car to spectate it (independent of the editor's own
        // pointer handlers, which only attach while editing).
        ui.canvas.addEventListener('pointerdown', e => this.handleCanvasClick(e));
    },

    // Resolves which car telemetry/highlight follows: a manually-clicked car
    // (until it crashes or the user releases it), otherwise the fastest alive.
    getSpectatedCar: function() {
        const id = this.state.spectateCarId;
        if (id !== null) {
            const car = this.state.cars[id];
            if (car && !car.crashed) return car;
            this.state.spectateCarId = null; // selection crashed/gone — auto-revert
        }
        if (!this.state.cars.length) return null;
        return this.state.cars.reduce((p,c) => (c.fitness > p.fitness && !c.crashed ? c : p), this.state.cars[0]);
    },

    releaseSpectate: function() { this.state.spectateCarId = null; },

    handleCanvasClick: function(e) {
        if (this.state.isEditing || this.state.hyperMode || !this.state.cars.length) return;
        const rect = ui.canvas.getBoundingClientRect();
        const scale = Math.min(rect.width / CANVAS_WIDTH, rect.height / CANVAS_HEIGHT);
        const offsetX = (rect.width - CANVAS_WIDTH*scale) / 2, offsetY = (rect.height - CANVAS_HEIGHT*scale) / 2;
        const x = (e.clientX - rect.left - offsetX) / scale, y = (e.clientY - rect.top - offsetY) / scale;

        let closest = null, closestDist = 22; // hit radius, canvas units
        for (const c of this.state.cars) {
            if (c.crashed) continue;
            const d = Math.hypot(c.x - x, c.y - y);
            if (d < closestDist) { closestDist = d; closest = c; }
        }
        this.state.spectateCarId = closest ? closest.id : null;
    },

    // Async now: nothing can be built until the wasm module is compiled and the
    // worker pool has come up, because the track generator lives in there too.
    init: async function() {
        try {
            this._initUICache();
            this.showBuildStamp();
            await Engine.ready();
            if(ui.coreCount && Engine._pendingCoreLabel) ui.coreCount.innerHTML = Engine._pendingCoreLabel;
            this.resetTracks();
            this.initChart();
            if(window.lucide) lucide.createIcons();
            this.loop();
        } catch (e) {
            console.error("Init Error:", e);
            this.showFatal(e);
        }
    },

    // Show which build is actually on screen.
    //
    // version.json is written by the deploy workflow, not committed, so it says
    // what is genuinely live rather than what the source happens to claim. When
    // it is absent — running locally, or a deploy that never landed — the
    // static version in the markup stays as it is.
    showBuildStamp: function() {
        const el = document.getElementById('version-tag');
        if (!el) return;
        fetch('./version.json', { cache: 'no-store' })
            .then(r => r.ok ? r.json() : null)
            .then(v => {
                if (!v || !v.short) return;
                el.textContent = `build ${v.short}`;
                if (v.built) el.title = `deployed ${v.built} from ${v.ref || 'unknown branch'}`;
            })
            .catch(() => { /* no stamp; leave the static version alone */ });
    },

    // A failed wasm load is the one error worth explaining rather than silently
    // reloading into: opening index.html off the filesystem hits it every time,
    // and "nothing happened" is a miserable way to find that out.
    showFatal: function(e) {
        const msg = String(e && e.message || e);
        const looksLikeFileProtocol = location.protocol === 'file:';
        const el = document.createElement('div');
        el.style.cssText = 'position:fixed;inset:0;z-index:9999;background:#0f172a;color:#e2e8f0;display:flex;align-items:center;justify-content:center;padding:24px;font-family:ui-sans-serif,system-ui,sans-serif';
        el.innerHTML = `<div style="max-width:560px">
            <h2 style="font-size:20px;font-weight:700;color:#f87171;margin-bottom:12px">Couldn't start the simulation</h2>
            <p style="font-size:14px;line-height:1.6;color:#cbd5e1;margin-bottom:12px">${msg.replace(/[<>&]/g, '')}</p>
            ${looksLikeFileProtocol ? `<p style="font-size:14px;line-height:1.6;color:#fbbf24">This page is open as a local file. Browsers refuse to fetch WebAssembly and start Workers over <code>file://</code> — serve the folder over HTTP instead:</p>
            <pre style="background:#1e293b;padding:10px 12px;border-radius:8px;font-size:13px;margin-top:8px">python3 -m http.server 8000</pre>
            <p style="font-size:13px;color:#94a3b8;margin-top:8px">then open http://localhost:8000</p>` : ''}
        </div>`;
        document.body.appendChild(el);
    },

    resetTracks: function() {
        // Load tracks from the external tracks.js file
        this.state.tracks = getDefaultTracks(generateTrackFromPath, CANVAS_WIDTH, CANVAS_HEIGHT);
        this._builtInTrackIds = new Set(this.state.tracks.map(t => t.id));
        // Nothing is restored from storage: tracks you make last for the
        // session and no longer.
        this.renderTrackList();
        this.switchTrack(0);
    },

    get currentTrack() { return this.state.tracks[this.state.currentTrackIndex]; },

    // The main thread's `cars` are render records only — position, colour and
    // the last telemetry the workers sent. The brains live in wasm memory and
    // never come back across; nothing here needs them.
    initPopulation: function(loadedBrainJSON) {
        if(!this.state.tracks.length) return;
        const track = this.currentTrack;
        this.state.globalBest = null;
        const newCars = [];
        for(let i=0; i<this.state.populationSize; i++) {
            newCars.push(this._makeCarRecord(i, track, `hsl(${Math.random()*360},80%,60%)`));
        }
        this.state.cars = newCars;
        this.state.bestTimes.gen = null;
        this.state.aliveCount = newCars.length;

        Engine.initPopulation(this.state.populationSize, this.state.hiddenLayers, loadedBrainJSON);
        this.updateUI();
    },

    _makeCarRecord: function(id, track, color) {
        return {
            id, x: track.startPos.x, y: track.startPos.y, angle: track.startAngle,
            speed: 0, color, fitness: 0, crashed: false,
            sensors: new Float32Array(SENSOR_COUNT), inputs: new Float32Array(2),
            completedLaps: 0
        };
    },

    // Selection, crossover and mutation all happen inside the master wasm
    // instance — it is the only one holding every brain. All this does is hand
    // over the fitness column and rebuild the render records.
    evolve: function() {
        if(this.state.cars.length === 0) return;
        const cars = this.state.cars;
        const fitness = new Float32Array(cars.length);
        let best = -Infinity, sum = 0;
        for(let i=0; i<cars.length; i++) {
            fitness[i] = cars[i].fitness;
            sum += cars[i].fitness;
            if(cars[i].fitness > best) best = cars[i].fitness;
        }

        const res = Engine.evolve(fitness, this.state.eliteClones);
        this.state.globalBest = { fitness: res.globalBest };

        this.state.stats.push({
            gen: this.state.generation, best, avg: sum / cars.length,
            time: this.state.bestTimes.gen ? this.state.bestTimes.gen.toFixed(2) : null
        });
        this.updateChart();

        // Elite clones keep the green/lime livery so you can pick the carried-
        // forward brains out of the pack on screen.
        const track = this.currentTrack;
        const elite = Math.min(this.state.eliteClones, this.state.populationSize);
        const newCars = [];
        for(let i=0; i<this.state.populationSize; i++) {
            const color = i < elite ? (i === 0 ? '#22c55e' : '#84cc16') : `hsl(${Math.random()*360},80%,60%)`;
            newCars.push(this._makeCarRecord(i, track, color));
        }
        this.state.cars = newCars;
        this.state.generation++;
        this.state.bestTimes.gen = null;
        this.state.aliveCount = newCars.length;
        this.updateUI();
    },

    // Fold one round of worker results back into the render records. In normal
    // mode each row is the full 18-float telemetry; in hyper mode it is three
    // floats per car, since nothing is drawn.
    _applyResults: function(st) {
        const cars = this.state.cars;
        let alive = 0;
        for(const r of st.rows) {
            alive += r.alive;
            const buf = r.buffer;
            if(r.render) {
                const stride = 18;
                for(let i=0; i<r.count; i++) {
                    const idx = i * stride;
                    const c = cars[buf[idx]];
                    if(!c) continue;
                    c.crashed = buf[idx+1] === 1;
                    c.x = buf[idx+2]; c.y = buf[idx+3];
                    c.angle = buf[idx+4]; c.speed = buf[idx+5];
                    c.inputs[0] = buf[idx+6]; c.inputs[1] = buf[idx+7];
                    const prevLaps = c.completedLaps;
                    c.completedLaps = buf[idx+8];
                    c.fitness = buf[idx+9];
                    for(let j=0; j<SENSOR_COUNT; j++) c.sensors[j] = buf[idx+11+j];
                    if(c.completedLaps > prevLaps) this._recordLap(buf[idx+10]);
                }
            } else {
                // Reported by the worker rather than hard-coded, so adding a
                // field in sim.c cannot silently shift every read by one.
                const stride = r.stride || 5;
                for(let i=0; i<r.count; i++) {
                    const c = cars[r.start + i];
                    if(!c) continue;
                    c.fitness = buf[i*stride];
                    const prevLaps = c.completedLaps;
                    c.completedLaps = buf[i*stride+1];
                    c.checkpoints = buf[i*stride+3];
                    if(c.completedLaps > prevLaps) this._recordLap(buf[i*stride+2]);
                }
            }
        }
        this.state.aliveCount = alive;
    },

    _recordLap: function(time) {
        if(!(time > 0)) return;
        if(!this.state.bestTimes.gen || time < this.state.bestTimes.gen) this.state.bestTimes.gen = time;
        if(!this.state.bestTimes.all || time < this.state.bestTimes.all) this.state.bestTimes.all = time;
        this.state.lapHistory.push(time);
        if(this.state.lapHistory.length > 20) this.state.lapHistory.shift();
        this.updateLapHistory();
    },

    loop: async function() {
        if(this.state.isRunning && !this.state.isEditing) {
            const hyper = this.state.hyperMode;
            const iters = hyper ? HYPER_CHUNK : this.state.speedMultiplier;
            const st = await Engine.run(iters, !hyper);
            this._applyResults(st);
            if(st.allCrashed || st.maxLaps >= this.state.targetLaps) this.evolve();
        }

        this.updateUI();
        this.draw();
        requestAnimationFrame(this.loop.bind(this));
    },

    cacheBackgroundRender: function() {
        this.state.bgCanvas = document.createElement('canvas');
        this.state.bgCanvas.width = CANVAS_WIDTH;
        this.state.bgCanvas.height = CANVAS_HEIGHT;
        const ctx = this.state.bgCanvas.getContext('2d');
        const t = this.currentTrack;

        ctx.fillStyle = '#3a5a40'; ctx.fillRect(0,0,CANVAS_WIDTH,CANVAS_HEIGHT);
        if(!t) return;

        drawRoadSurface(ctx, t, '#343a40');
        
        // Zones are intentionally NOT rendered in normal view — only visible in the editor

        ctx.lineCap = 'round';
        ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=4; strokeWalls(ctx, t);
        ctx.strokeStyle='#ef4444'; ctx.lineWidth=4; ctx.setLineDash([15,15]); strokeWalls(ctx, t); ctx.setLineDash([]);

        if(t.cpCount > 0) {
            const cp = t.cpF32;
            ctx.strokeStyle='#ffffff'; ctx.lineWidth=6; ctx.beginPath(); ctx.moveTo(cp[0], cp[1]); ctx.lineTo(cp[2], cp[3]); ctx.stroke();
            ctx.setLineDash([6,6]); ctx.strokeStyle='#000'; ctx.stroke(); ctx.setLineDash([]);
        }
    },

    draw: function() {
        const ctx = ui.ctx; // cached — no getElementById every frame
        
        if(this.state.isEditing && this.state.trackToEdit) { 
            ctx.fillStyle = '#3a5a40'; ctx.fillRect(0,0,CANVAS_WIDTH,CANVAS_HEIGHT);
            editor.draw(ctx); 
            return; 
        }

        if(this.state.bgCanvas) ctx.drawImage(this.state.bgCanvas, 0, 0);

        if(!this.state.hyperMode) {
            // Render ALL cars persistently, no color flashing, no hiding.
            // Telemetry/highlight follows a manually-clicked car, or else
            // auto-follows the fastest alive car.
            const spectated = this.getSpectatedCar();
            const isManual = this.state.spectateCarId !== null;

            if(ui.telemetryLabel) {
                ui.telemetryLabel.textContent = spectated
                    ? (isManual ? `Spectating Car #${spectated.id} (Manual)` : 'Live Telemetry (Auto — Fastest)')
                    : 'Live Telemetry';
            }
            if(ui.btnRelease) ui.btnRelease.classList.toggle('hidden', !isManual);

            if(spectated && !spectated.crashed) {
                const i = spectated.inputs || [0,0];
                ui.telSteerL.style.width = i[0] < 0 ? Math.abs(i[0])*50 + '%' : '0%';
                ui.telSteerR.style.width = i[0] > 0 ? i[0]*50 + '%' : '0%';
                if(i[1] > 0) { ui.telGas.style.width = i[1]*100 + '%'; ui.telBrake.style.width = '0%'; }
                else { ui.telGas.style.width = '0%'; ui.telBrake.style.width = Math.abs(i[1])*100 + '%'; }
                ui.telSpeed.style.width = Math.min((spectated.speed / this.state.physics.maxSpeed)*100, 100) + '%';
                ui.telSpeedVal.textContent = Math.round(spectated.speed);
            }

            this.state.cars.forEach(c => {
                if(c.crashed) return;
                ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(c.angle); ctx.scale(1.5,1.5);
                ctx.globalAlpha = 1.0;
                ctx.fillStyle = c.color;
                ctx.fillRect(-7, -4, 14, 8);
                ctx.fillStyle='#0f172a'; ctx.fillRect(-2, -3, 4, 6);
                ctx.fillStyle='#fbbf24'; ctx.fillRect(6, -3, 1, 2); ctx.fillRect(6, 1, 1, 2);
                ctx.restore();

                if(c === spectated) {
                    // Highlight ring around the spectated car — solid cyan when
                    // manually picked, a subtle dashed ring when auto-following.
                    ctx.save();
                    ctx.strokeStyle = isManual ? '#22d3ee' : 'rgba(255,255,255,0.55)';
                    ctx.lineWidth = isManual ? 2.5 : 1.5;
                    if(!isManual) ctx.setLineDash([4,3]);
                    ctx.beginPath(); ctx.arc(c.x, c.y, 16, 0, Math.PI*2); ctx.stroke();
                    ctx.restore();

                    if(c.sensors) {
                        c.sensors.forEach((s,k) => {
                            const ang = c.angle + SENSOR_ANGLES[k];
                            ctx.strokeStyle='rgba(234,179,8,0.3)'; ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(c.x+Math.cos(ang)*SENSOR_LENGTH, c.y+Math.sin(ang)*SENSOR_LENGTH); ctx.stroke();
                            if(s>0) { const d=(1-s)*SENSOR_LENGTH; ctx.fillStyle='#f59e0b'; ctx.beginPath(); ctx.arc(c.x+Math.cos(ang)*d, c.y+Math.sin(ang)*d, 2, 0, Math.PI*2); ctx.fill(); }
                        });
                    }
                }
            });
        }
    },

    toggleRun: function() { 
        this.state.isRunning = !this.state.isRunning; 
        const running = this.state.isRunning;
        const icon = running ? SVG_PAUSE : SVG_PLAY;
        const txt = running ? 'Pause' : 'Start';
        const clsBase = "rounded-lg font-bold flex items-center justify-center gap-1.5 transition-all border text-sm ";
        const clsOn  = clsBase + "px-3 py-2 bg-red-500/20 text-red-400 border-red-500/50 hover:bg-red-500/30";
        const clsOff = clsBase + "px-3 py-2 bg-emerald-500/20 text-emerald-400 border-emerald-500/50 hover:bg-emerald-500/30";
        // Desktop sidebar button
        if(ui.btnPlay) { 
            ui.btnPlay.innerHTML = `${icon} <span>${txt}</span>`; 
            ui.btnPlay.className = (running ? clsOn : clsOff) + " flex-1 py-2";
        }
        // Mobile control-bar button
        if(ui.btnPlayM) { 
            ui.btnPlayM.innerHTML = `${icon} <span>${txt}</span>`; 
            ui.btnPlayM.className = running ? clsOn : clsOff;
        }
        // No lucide.createIcons() — inline SVGs don't need it
    },
    toggleHyper: function() { 
        this.state.hyperMode = !this.state.hyperMode; 
        const active = this.state.hyperMode;
        [ui.btnHyper, ui.btnHyperM].forEach(btn => {
            if(!btn) return;
            btn.classList.toggle('bg-yellow-500', active); 
            btn.classList.toggle('text-slate-900', active); 
            btn.classList.toggle('bg-slate-700', !active);
            btn.classList.toggle('text-slate-400', !active);
            btn.classList.toggle('border-yellow-500', active);
            btn.classList.toggle('border-slate-600', !active);
        });
        if(ui.hyperBanner) ui.hyperBanner.classList.toggle('hidden', !active);
    },

    reset: function() {
        this.state.isRunning=false; this.state.generation=1; this.state.stats=[];
        this.state.bestTimes={gen:null,all:null}; this.state.lapHistory=[]; this.state.spectateCarId=null;
        _ui_gen=-1; _ui_alive=-1; _ui_allBest=null;
        if(ui.lapHistoryM) ui.lapHistoryM.innerHTML = '<span class="text-[10px] text-slate-600 italic">No laps yet</span>';
        this.initPopulation(); 
        if(this.chart) { this.chart.data.labels = []; this.chart.data.datasets.forEach(d => d.data = []); this.chart.update(); }
        this.updateUI(); this.toggleRun(); this.toggleRun(); 
    },
    
    // Physics is pure config — pushing it doesn't need the track rebuilding.
    updatePhysics: function(k, v) { this.state.physics[k] = parseFloat(v); document.getElementById('val-'+(k==='maxSpeed'?'maxSpeed':(k==='acceleration'?'accel':(k==='turnSpeed'?'turn':'grip')))).innerText = k==='grip'?Math.round(v*100)+'%':v; Engine.pushConfig(this.state); },
    updateConfig: function(k, v) {
        this.state[k] = parseFloat(v);
        let id = 'val-'+(k==='populationSize'?'pop':k==='targetLaps'?'laps':k==='speedMultiplier'?'speed':k==='eliteClones'?'elite':k==='mutationRate'?'mut':k==='hiddenLayers'?'hidden':'ttl');
        let d = v; if(k==='speedMultiplier') d+='x'; if(k==='mutationRate') d=Math.round(v*100)+'%';
        const el = document.getElementById(id); if(el) el.innerText = d;
        // TTL, target laps and mutation rate are read inside the wasm step, so
        // they have to reach every instance. Population and hidden-layer size
        // change the shape of things and only take effect on the next reset,
        // same as the JS edition.
        Engine.pushConfig(this.state);
    },
    syncSettingsUI: function() {
        const st = this.state;
        const apply = (inputId, val, labelId, fmt) => {
            const inp = document.getElementById(inputId); if (inp) inp.value = val;
            const lbl = document.getElementById(labelId); if (lbl) lbl.innerText = fmt ? fmt(val) : val;
        };
        apply('cfg-speedMultiplier', st.speedMultiplier, 'val-speed', v => v+'x');
        apply('cfg-populationSize', st.populationSize, 'val-pop');
        apply('cfg-eliteClones', st.eliteClones, 'val-elite');
        apply('cfg-mutationRate', st.mutationRate, 'val-mut', v => Math.round(v*100)+'%');
        apply('cfg-hiddenLayers', st.hiddenLayers, 'val-hidden');
        apply('cfg-initialTTL', st.initialTTL, 'val-ttl');
        apply('cfg-targetLaps', st.targetLaps, 'val-laps');
        apply('cfg-maxSpeed', st.physics.maxSpeed, 'val-maxSpeed');
        apply('cfg-acceleration', st.physics.acceleration, 'val-accel');
        apply('cfg-turnSpeed', st.physics.turnSpeed, 'val-turn');
        apply('cfg-grip', st.physics.grip, 'val-grip', v => Math.round(v*100)+'%');
    },

    showInfo: function(k) {
        const el = document.getElementById('setting-info'); if(!el) return;
        if(k && SETTING_DESCRIPTIONS[k]) { el.innerHTML = `<span class="text-emerald-400 font-bold block mb-1 uppercase">${k.replace(/([A-Z])/g, ' $1').trim()}</span>${SETTING_DESCRIPTIONS[k]}`; el.className = "text-[10px] text-slate-300 min-h-[50px] border-t border-slate-600 pt-2 mt-2 transition-colors"; } 
        else { el.innerText = "Hover over a setting to see how it affects the AI and simulation."; el.className = "text-[10px] text-slate-500 italic min-h-[50px] border-t border-slate-600 pt-2 mt-2 transition-colors"; }
    },

    updateUI: function() {
        const gen = this.state.generation;
        // Reported by the workers rather than counted here: in hyper mode the
        // crashed flags never cross the boundary, so there is nothing to count.
        const alive = this.state.aliveCount;
        
        // Only write to DOM if value changed (dirty check — huge win on ARM)
        if(_ui_gen !== gen) {
            if(ui.statGen) ui.statGen.textContent = gen;
            if(ui.statGenMain) ui.statGenMain.textContent = gen;
            _ui_gen = gen;
        }
        if(_ui_alive !== alive) {
            if(ui.statAlive) ui.statAlive.textContent = alive;
            if(ui.statAliveMain) ui.statAliveMain.textContent = alive;
            _ui_alive = alive;
        }

        const allBestStr = this.state.bestTimes.all ? this.state.bestTimes.all.toFixed(2)+'s' : '--';
        if(_ui_allBest !== allBestStr) {
            if(ui.statAllBest) ui.statAllBest.textContent = allBestStr;
            if(ui.statAllBestM) ui.statAllBestM.textContent = allBestStr;
            _ui_allBest = allBestStr;
        }
        // No lucide.createIcons() here — that was being called EVERY frame and is
        // the #1 performance killer on ARM. Buttons now use inline SVGs instead.
    },

    updateLapHistory: function() {
        if(!ui.lapHistoryM) return;
        const hist = this.state.lapHistory;
        if(!hist.length) return;
        ui.lapHistoryM.innerHTML = hist.slice(-7).reverse().map((l,i) =>
            `<span class="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded ${i===0?'bg-emerald-500/20 text-emerald-400':'bg-slate-700/60 text-slate-400'}">${l.toFixed(2)}s</span>`
        ).join('');
    },

    // Same nested-array JSON the JS edition writes, so a brain saved there
    // loads here and vice versa.
    saveBrain: function() {
        if(!this.state.cars.length) return;
        const best = this.state.cars.reduce((p,c) => c.fitness>p.fitness?c:p);
        const json = Engine.brainToJSON(best.id);
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([JSON.stringify(json)], {type:'application/json'}));
        a.download = `trackml-g${this.state.generation}.json`;
        a.click();
    },
    loadBrain: function(inp) {
        if(!inp.files[0]) return;
        const r = new FileReader();
        r.onload = e => {
            let json;
            try { json = JSON.parse(e.target.result); }
            catch(er) { alert('That file isn\'t valid JSON.'); return; }
            const bad = Engine.validateBrainJSON(json);
            if(bad) { alert('Invalid brain file: ' + bad); return; }
            // A brain's hidden-layer size is baked into its weights, so adopt
            // it rather than trying to squeeze the file into the current slider
            // setting — that used to load garbage weights and look like a bug.
            const h = Engine.hiddenOf(json);
            if(h !== this.state.hiddenLayers) {
                this.state.hiddenLayers = h;
                this.syncSettingsUI();
            }
            this.state.isRunning=false; this.state.generation=1; this.state.stats=[];
            this.state.bestTimes={gen:null,all:null}; this.state.lapHistory=[]; this.state.spectateCarId=null;
            _ui_gen=-1; _ui_alive=-1; _ui_allBest=null;
            if(this.chart) { this.chart.data.labels = []; this.chart.data.datasets.forEach(d => d.data = []); this.chart.update(); }
            this.initPopulation(json);
            this.updateUI();
        };
        r.readAsText(inp.files[0]);
    },

    renderTrackList: function() {
        const sel = document.getElementById('track-dropdown'); sel.innerHTML = '';
        this.state.tracks.forEach((t, i) => { const opt = document.createElement('option'); opt.value = i; opt.text = t.name; opt.selected = this.state.currentTrackIndex === i; sel.appendChild(opt); });
        const search = document.getElementById('track-search');
        if (search && search.value) this.filterTrackList(search.value);
    },
    filterTrackList: function(query) {
        const sel = document.getElementById('track-dropdown'); if (!sel) return;
        const q = query.trim().toLowerCase();
        let visibleCount = 0, firstVisible = -1;
        for (const opt of sel.options) {
            const match = !q || opt.text.toLowerCase().includes(q);
            opt.style.display = match ? '' : 'none';
            if (match) { visibleCount++; if (firstVisible === -1) firstVisible = parseInt(opt.value); }
        }
        // if the currently-selected track got filtered out, jump to the first visible match
        if (visibleCount > 0 && sel.selectedOptions.length && sel.selectedOptions[0].style.display === 'none') {
            sel.value = firstVisible;
        }
    },
    
    switchTrack: function(i) {
        i = parseInt(i); if (i < 0 || i >= this.state.tracks.length) i = 0;
        this.state.currentTrackIndex = i; this.state.isRunning = false; this.state.generation = 1; this.state.globalBest = null; this.state.stats = []; this.state.spectateCarId = null; this.updateChart();
        const sel = document.getElementById('track-dropdown'); if(sel) sel.value = i;
        const t = this.state.tracks[i];
        
        this.cacheBackgroundRender();
        // Order matters: the workers rebuild the track geometry before pop_init
        // parks their cars on its start line. Worker messages stay ordered, so
        // sending them back to back is enough.
        Engine.setTrack(t, this.state);
        this.initPopulation();
        this.updateUI();
        
        // Reset play buttons using inline SVGs — no lucide needed
        const clsOff = "rounded-lg font-bold flex items-center justify-center gap-1.5 transition-all border text-sm px-3 py-2 bg-emerald-500/20 text-emerald-400 border-emerald-500/50 hover:bg-emerald-500/30";
        if(ui.btnPlay) { ui.btnPlay.innerHTML = `${SVG_PLAY} <span>Start</span>`; ui.btnPlay.className = clsOff + " flex-1 py-2"; }
        if(ui.btnPlayM) { ui.btnPlayM.innerHTML = `${SVG_PLAY} <span>Start</span>`; ui.btnPlayM.className = clsOff; }
        // No lucide.createIcons() needed here
    },

    createNewTrack: function() { 
        const t = generateTrackFromPath("custom"+Date.now(), "New Track", [{x:200,y:200},{x:1000,y:200},{x:1000,y:700},{x:200,y:700}], 60); 
        this.state.isEditing = true; this.state.trackToEdit = t; editor.init(t); this.state.isRunning = false; 
        document.getElementById('editor-controls').classList.remove('hidden'); 
        closeSidebar();
    },
    // Only the *definition* is cloned for editing — the generated geometry is
    // rebuilt from it on every editor frame anyway, and a track now carries
    // Float32Arrays that a JSON round-trip would mangle into numbered objects.
    cloneTrackDef: function(t) {
        return {
            id: t.id, name: t.name, trackWidth: t.trackWidth,
            path: t.path.map(p => ({ x: p.x, y: p.y, type: p.type, radius: p.radius })),
            startPos: { x: t.startPos.x, y: t.startPos.y },
            startAngle: t.startAngle,
            zones: (t.zones || []).map(z => Object.assign({}, z)),
            autoWidth: !!t.autoWidth,
            autoWidthBlend: t.autoWidthBlend || 0
        };
    },
    editTrack: function() {
        this.state.isEditing = true; this.state.trackToEdit = this.cloneTrackDef(this.currentTrack);
        editor.init(this.state.trackToEdit); this.state.isRunning = false;
        document.getElementById('editor-controls').classList.remove('hidden');
        closeSidebar();
    },
    duplicateTrack: function() {
        const src = this.currentTrack; if (!src) return;
        const copy = generateTrackFromPath('custom'+Date.now(), src.name + ' (Copy)', JSON.parse(JSON.stringify(src.path)), src.trackWidth, src.startPos, src.startAngle, JSON.parse(JSON.stringify(src.zones || [])), { enabled: !!src.autoWidth, blend: src.autoWidthBlend || 0 });
        this.state.isEditing = true; this.state.trackToEdit = copy; editor.init(copy); this.state.isRunning = false;
        document.getElementById('editor-controls').classList.remove('hidden');
        closeSidebar();
    },
    saveTrack: function(t) {
        const idx = this.state.tracks.findIndex(tr => tr.id === t.id);
        if(idx !== -1) this.state.tracks[idx] = t;
        else { this.state.tracks.push(t); this.state.currentTrackIndex = this.state.tracks.length-1; }
        this.state.isEditing = false;
        document.getElementById('editor-controls').classList.add('hidden');
        this.switchTrack(this.state.currentTrackIndex); this.renderTrackList();
    },
    deleteTrack: function() {
        if(confirm("Delete this track?")) {
            if(this.state.tracks.length > 1) {
                this.state.tracks.splice(this.state.currentTrackIndex, 1); this.switchTrack(0); this.renderTrackList();
            } else alert("Cannot delete last track.");
        }
    },

    chart: null,
    initChart: function() { 
        const ctx = document.getElementById('fitness-chart').getContext('2d'); 
        this.chart = new Chart(ctx, { 
            type: 'line', 
            data: { 
                labels: [], 
                datasets: [
                    {label:'Best Fitness', data:[], borderColor:'#34d399', backgroundColor:'rgba(52, 211, 153, 0.1)', borderWidth:2, pointRadius:2, fill: true},
                    {label:'Avg Fitness', data:[], borderColor:'#60a5fa', borderWidth:2, pointRadius:0}
                ] 
            }, 
            options: { 
                responsive: true, 
                maintainAspectRatio: false, 
                scales: { 
                    x: { display:false }, 
                    y: { grid: { color: '#334155' } } 
                }, 
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: { 
                    legend: { 
                        display:true, 
                        labels: { color: '#cbd5e1', boxWidth: 10, font: { size: 10 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                let val = Math.round(context.parsed.y);
                                let time = app.state.stats[context.dataIndex]?.time;
                                if(context.datasetIndex === 0 && time) return `${label}: ${val} (Lap: ${time}s)`;
                                return `${label}: ${val}`;
                            }
                        }
                    }
                } 
            } 
        }); 
    },
    updateChart: function() { 
        if(!this.chart) return; 
        this.chart.data.labels = this.state.stats.map(s => `Gen ${s.gen}`); 
        this.chart.data.datasets[0].data = this.state.stats.map(s => s.best); 
        this.chart.data.datasets[1].data = this.state.stats.map(s => s.avg); 
        this.chart.update(); 
    }
};

// --- Mobile/Desktop Track Editor ---
// --- Mobile/Desktop Track Editor ---

// --- Mobile/Desktop Track Editor ---
const editor = {
    track: null, mode: 'path', dragIndex: null, hoverIndex: null, isDraggingStart: false, isResizingZone: false, selectedZone: null, selectedIndex: null,
    isDrawing: false, drawPoints: [],

    init: function(t) {
        this.track = t;
        if (t.autoWidth === undefined) t.autoWidth = false;
        if (t.autoWidthBlend === undefined) t.autoWidthBlend = 0;
        document.getElementById('edit-name').value = t.name;
        document.getElementById('edit-width').value = t.trackWidth;
        document.getElementById('edit-angle').value = Math.round((t.startAngle || 0) * (180/Math.PI));
        document.getElementById('edit-autowidth').checked = !!t.autoWidth;
        document.getElementById('edit-autowidth-blend').value = Math.round(t.autoWidthBlend * 100);
        this.syncAutoWidthUI();
        this.setMode('path');
        const c = document.getElementById('sim-canvas');
        c.style.touchAction = 'none';
        c.onpointerdown = e => { e.preventDefault(); c.setPointerCapture(e.pointerId); this.onDown(e); };
        c.onpointermove = e => { e.preventDefault(); this.onMove(e); };
        c.onpointerup = c.onpointercancel = e => { c.releasePointerCapture(e.pointerId); this.dragIndex=null; this.isDraggingStart=false; this.isResizingZone=false; if(this.isDrawing) this.finishDrawing(); };
    },

    setMode: function(m) {
        this.mode = m; this.selectedZone = null; this.selectedIndex = null;
        this.isDrawing = false; this.drawPoints = [];
        if(document.getElementById('btn-del-point')) document.getElementById('btn-del-point').classList.add('hidden');
        if(document.getElementById('point-tools')) {
            document.getElementById('point-tools').classList.add('hidden');
            document.getElementById('point-tools').classList.remove('flex');
        }
        const ktc = document.getElementById('kill-timer-container');
        if (ktc) ktc.style.display = 'none';
        ['path','draw','zones'].forEach(x => {
            const btn = document.getElementById('btn-mode-'+x);
            if(btn) btn.className = m===x?'px-2 py-1 text-xs rounded flex items-center gap-1 bg-blue-600 text-white':'px-2 py-1 text-xs rounded flex items-center gap-1 text-slate-400';
            const tools = document.getElementById(x+'-tools');
            if(tools) tools.style.display = m===x?'flex':'none';
        });
        document.getElementById('sim-canvas').style.cursor = m === 'draw' ? 'crosshair' : 'default';
    },

    clearDrawing: function() { this.isDrawing = false; this.drawPoints = []; },

    // Ramer-Douglas-Peucker: reduce a dense freehand stroke to its essential
    // corners/curves so the resulting track path stays editable.
    rdp: function(pts, epsilon) {
        if (pts.length < 3) return pts.slice();
        const perpDist = (p, a, b) => {
            const dx = b.x-a.x, dy = b.y-a.y, len = Math.hypot(dx,dy);
            if (len === 0) return Math.hypot(p.x-a.x, p.y-a.y);
            const t = ((p.x-a.x)*dx + (p.y-a.y)*dy) / (len*len);
            const cx = a.x + t*dx, cy = a.y + t*dy;
            return Math.hypot(p.x-cx, p.y-cy);
        };
        let maxD = 0, idx = 0;
        for (let i=1; i<pts.length-1; i++) {
            const d = perpDist(pts[i], pts[0], pts[pts.length-1]);
            if (d > maxD) { maxD = d; idx = i; }
        }
        if (maxD > epsilon) {
            const left = this.rdp(pts.slice(0, idx+1), epsilon);
            const right = this.rdp(pts.slice(idx), epsilon);
            return left.slice(0, -1).concat(right);
        }
        return [pts[0], pts[pts.length-1]];
    },

    // Simplify a CLOSED freehand loop: naive RDP (first point -> last point as
    // the baseline) collapses a loop almost to nothing since those two points
    // are nearly coincident. Instead split the loop at its two most distant
    // points into two open chains, simplify each independently, then rejoin.
    simplifyClosedLoop: function(points, epsilon) {
        let pts = points.slice();
        if (pts.length > 1 && Math.hypot(pts[0].x-pts[pts.length-1].x, pts[0].y-pts[pts.length-1].y) < 20) pts.pop();
        if (pts.length < 6) return pts;

        let bestD = -1, bi = 0, bj = 1;
        const step = Math.max(1, Math.floor(pts.length/150));
        for (let i=0; i<pts.length; i+=step) for (let j=i+1; j<pts.length; j+=step) {
            const d = (pts[i].x-pts[j].x)**2 + (pts[i].y-pts[j].y)**2;
            if (d > bestD) { bestD = d; bi = i; bj = j; }
        }
        if (bi > bj) { const tmp = bi; bi = bj; bj = tmp; }

        const chainA = pts.slice(bi, bj+1);
        const chainB = pts.slice(bj).concat(pts.slice(0, bi+1));
        const simpA = this.rdp(chainA, epsilon);
        const simpB = this.rdp(chainB, epsilon);
        let merged = simpA.slice(0, -1).concat(simpB.slice(0, -1));

        // Safety cap: very detailed/jittery strokes can still leave too many
        // points for wallsBySegment / the point-tools UI to stay comfortable.
        const MAX_POINTS = 70;
        if (merged.length > MAX_POINTS) {
            const stride = Math.ceil(merged.length / MAX_POINTS);
            merged = merged.filter((_, i) => i % stride === 0);
        }
        return merged;
    },

    // Drop points that sit closer than `minGap` to the last one kept, so every
    // vertex has enough straight either side to fit a proper corner arc.
    spaceOutPoints: function(pts, minGap) {
        if (pts.length < 4) return pts.slice();
        const out = [pts[0]];
        for (let i = 1; i < pts.length; i++) {
            const last = out[out.length - 1];
            if (Math.hypot(pts[i].x - last.x, pts[i].y - last.y) >= minGap) out.push(pts[i]);
        }
        // The wrap-around back to the first point needs the same clearance.
        while (out.length > 3 && Math.hypot(out[out.length - 1].x - out[0].x, out[out.length - 1].y - out[0].y) < minGap) out.pop();
        return out.length >= 3 ? out : pts.slice();
    },

    finishDrawing: function() {
        if (!this.isDrawing) return;
        this.isDrawing = false;
        const raw = this.drawPoints;
        this.drawPoints = [];
        if (raw.length < 4) return;

        const simplified = this.simplifyClosedLoop(raw, 9);
        if (simplified.length < 3) { alert('Draw a bigger loop — that stroke was too small to build a track from.'); return; }

        // Freehand strokes land points wherever the pointer happened to be
        // sampled. Two points closer together than the road is wide leave no
        // room for the corner arcs the generator fits between them, which is
        // what turned hand-drawn loops into a chain of kinks — so thin them out
        // to at least a track-width apart first.
        const spaced = this.spaceOutPoints(simplified, Math.max(18, this.track.trackWidth * 1.1));
        if (spaced.length < 3) { alert('Draw a bigger loop — that stroke was too small to build a track from.'); return; }
        // 'corner' now means "as tight as the track width allows" rather than
        // "no rounding at all", which is exactly what a traced stroke wants.
        this.track.path = spaced.map(p => ({ x: Math.round(p.x), y: Math.round(p.y), type: 'corner', radius: 35 }));
        // A hand-drawn loop is the case auto width exists for — you can't judge
        // by eye whether two parts of the stroke left room for a barrier
        // between them. Turn it on and reflect that in the controls; the
        // checkbox is right there if it isn't wanted.
        this.track.autoWidth = true;
        const awBox = document.getElementById('edit-autowidth');
        if (awBox) awBox.checked = true;
        this.syncAutoWidthUI();
        // Re-derive a concrete start pos/angle from the new path (the old ones
        // belonged to whatever shape was there before) — computed once now
        // rather than left null, since save() reads track.startPos.x directly.
        const derived = generateTrackFromPath(this.track.id, this.track.name, this.track.path, this.track.trackWidth, null, null, [], this.autoOpts());
        this.track.startPos = derived.startPos;
        this.track.startAngle = derived.startAngle;
        document.getElementById('edit-angle').value = Math.round((derived.startAngle || 0) * (180/Math.PI));
        this.selectedIndex = null;
        this.setMode('path');
    },

    updateZoneUI: function() {
        const ktc = document.getElementById('kill-timer-container');
        if (!ktc) return;
        if (this.selectedZone && this.selectedZone.type === 'spawnkill') {
            ktc.style.display = 'flex';
            const t = this.selectedZone.killTimer !== undefined ? this.selectedZone.killTimer : 150;
            document.getElementById('kill-timer-slider').value = t;
            document.getElementById('kill-timer-val').textContent = t;
        } else {
            ktc.style.display = 'none';
        }
    },
    updateZoneKillTimer: function(val) {
        if (this.selectedZone && this.selectedZone.type === 'spawnkill') {
            this.selectedZone.killTimer = parseInt(val);
            document.getElementById('kill-timer-val').textContent = val;
        }
    },

    selectAllPoints: function() {
        if(this.track && this.track.path && this.track.path.length > 0) {
            this.selectedIndex = 'all';
            this.updatePointUI();
        }
    },

    updatePointType: function(type) {
        if(this.selectedIndex === 'all') {
            this.track.path.forEach(p => p.type = type);
            this.updatePointUI();
        } else if(this.selectedIndex !== null) {
            this.track.path[this.selectedIndex].type = type;
            this.updatePointUI();
        }
    },

    updatePointRadius: function(val) {
        if(this.selectedIndex === 'all') {
            this.track.path.forEach(p => p.radius = parseInt(val));
            document.getElementById('pt-radius-val').innerText = val;
        } else if(this.selectedIndex !== null) {
            this.track.path[this.selectedIndex].radius = parseInt(val);
            document.getElementById('pt-radius-val').innerText = val;
        }
    },

    updatePointUI: function() {
        const pt = document.getElementById('point-tools');
        if(!pt) return;
        if (this.selectedIndex !== null) {
            pt.classList.remove('hidden');
            pt.classList.add('flex');
            
            let type = 'rounded';
            let radius = 60;

            if (this.selectedIndex === 'all') {
                // If "all" is selected, hide the delete button
                if(document.getElementById('btn-del-point')) document.getElementById('btn-del-point').classList.add('hidden');
            } else {
                // If a single point is selected, grab its specific data
                const p = this.track.path[this.selectedIndex];
                type = p.type || 'rounded';
                radius = p.radius !== undefined ? p.radius : 60;
                if(document.getElementById('btn-del-point')) document.getElementById('btn-del-point').classList.remove('hidden');
            }
            
            document.getElementById('btn-pt-corner').className = type === 'corner' ? 'px-2 py-1 text-[10px] rounded bg-blue-600 text-white' : 'px-2 py-1 text-[10px] rounded text-slate-400 hover:bg-slate-700 transition-colors';
            document.getElementById('btn-pt-rounded').className = type === 'rounded' ? 'px-2 py-1 text-[10px] rounded bg-blue-600 text-white' : 'px-2 py-1 text-[10px] rounded text-slate-400 hover:bg-slate-700 transition-colors';
            
            document.getElementById('radius-container').style.display = type === 'rounded' ? 'flex' : 'none';
            document.getElementById('pt-radius-slider').value = radius;
            document.getElementById('pt-radius-val').innerText = radius;
        } else {
            pt.classList.add('hidden');
            pt.classList.remove('flex');
            if(document.getElementById('btn-del-point')) document.getElementById('btn-del-point').classList.add('hidden');
        }
    },

    save: function() { 
        const name = document.getElementById('edit-name').value || 'Custom Track';
        this.track.name = name;
        
        const cleanPath = this.track.path.map(p => ({
            x: Math.round(p.x),
            y: Math.round(p.y),
            type: p.type || 'rounded',
            radius: p.radius !== undefined ? Math.round(p.radius) : 60
        }));

        this.track.path = cleanPath;
        const t = generateTrackFromPath(this.track.id, this.track.name, cleanPath, this.track.trackWidth, this.track.startPos, this.track.startAngle, this.track.zones, this.autoOpts());
        app.saveTrack(t);

        let pathStr = JSON.stringify(cleanPath)
            .replace(/"x":/g, 'x: ').replace(/"y":/g, 'y: ')
            .replace(/"type":/g, 'type: ').replace(/"radius":/g, 'radius: ')
            .replace(/}/g, ' }').replace(/{/g, '{ ');
            
        const zonesStr = this.track.zones && this.track.zones.length ? `, ${JSON.stringify(this.track.zones)}` : '';
        const startAng = this.track.startAngle ? Number(this.track.startAngle.toFixed(4)) : 0;
        
        // Auto width is only emitted when it's on, so an ordinary track's
        // exported line stays exactly as short as it always was.
        const autoStr = this.track.autoWidth
            ? `${zonesStr ? '' : ', []'}, { enabled: true, blend: ${Number((this.track.autoWidthBlend || 0).toFixed(2))} }`
            : '';
        const code = `generateTrackFromPath("${this.track.id}", "${name}", ${pathStr}, ${this.track.trackWidth}, {x:${Math.round(this.track.startPos.x)}, y:${Math.round(this.track.startPos.y)}}, ${startAng}${zonesStr}${autoStr}),`;
        document.getElementById('code-output').value = code;
        document.getElementById('code-modal').classList.remove('hidden');
    },

    cancel: function() {
        app.state.isEditing = false; 
        document.getElementById('editor-controls').classList.add('hidden'); 
        const cv = document.getElementById('sim-canvas'); cv.onpointerdown = null; cv.onpointermove = null; cv.onpointerup = null; cv.onpointercancel = null;
    },
    updateWidth: function(v) { this.track.trackWidth = parseInt(v); },

    // The auto-width settings, in the shape generateTrackFromPath wants. Every
    // rebuild the editor does goes through this so the live preview shows the
    // same road the simulation will actually use.
    autoOpts: function() {
        return { enabled: !!this.track.autoWidth, blend: this.track.autoWidthBlend || 0 };
    },
    updateAutoWidth: function(on) {
        this.track.autoWidth = !!on;
        this.syncAutoWidthUI();
    },
    updateAutoWidthBlend: function(v) {
        this.track.autoWidthBlend = parseInt(v) / 100;
        this.syncAutoWidthUI();
    },
    syncAutoWidthUI: function() {
        const wrap = document.getElementById('autowidth-blend-wrap');
        const label = document.getElementById('autowidth-blend-label');
        const on = !!this.track.autoWidth;
        if (wrap) {
            wrap.classList.toggle('opacity-40', !on);
            wrap.classList.toggle('pointer-events-none', !on);
        }
        if (label) {
            const b = Math.round((this.track.autoWidthBlend || 0) * 100);
            label.textContent = b === 0 ? 'Pinch only' : b === 100 ? 'Whole track' : `Mix ${b}%`;
        }
    },
    updateAngle: function(v) { this.track.startAngle = parseFloat(v) * (Math.PI/180); },
    addZone: function(type) { 
        const z = { id:Date.now().toString(), x:CANVAS_WIDTH/2, y:CANVAS_HEIGHT/2, radius:80, type };
        if (type === 'spawnkill') { z.killTimer = 150; z.x = this.track.startPos.x; z.y = this.track.startPos.y; }
        this.track.zones.push(z); this.selectedZone = z; this.updateZoneUI();
    },
    deleteZone: function() { if(this.selectedZone) this.track.zones = this.track.zones.filter(z => z !== this.selectedZone); this.selectedZone = null; },
    deleteSelectedPoint: function() {
        if(this.selectedIndex !== null && this.selectedIndex !== 'all' && this.track.path.length > 3) {
            this.track.path.splice(this.selectedIndex, 1);
            this.selectedIndex = null;
            this.updatePointUI();
        }
    },

    getPos: function(e) { 
        const r = document.getElementById('sim-canvas').getBoundingClientRect(); 
        const scale = Math.min(r.width / CANVAS_WIDTH, r.height / CANVAS_HEIGHT);
        const offsetX = (r.width - (CANVAS_WIDTH * scale)) / 2;
        const offsetY = (r.height - (CANVAS_HEIGHT * scale)) / 2;
        return { x: (e.clientX - r.left - offsetX) / scale, y: (e.clientY - r.top - offsetY) / scale }; 
    },

    onDown: function(e) {
        const {x,y} = this.getPos(e);

        if (this.mode === 'draw') {
            this.isDrawing = true;
            this.drawPoints = [{x, y}];
            return;
        }

        if (this.mode === 'zones') {
            if(this.selectedZone && Math.hypot(x-(this.selectedZone.x+this.selectedZone.radius), y-this.selectedZone.y) < 30) { this.isResizingZone = true; return; }
            this.selectedZone = [...this.track.zones].reverse().find(z => Math.hypot(z.x-x, z.y-y) < z.radius) || null; 
            this.updateZoneUI();
            return;
        }
        
        if(Math.hypot(x - this.track.startPos.x, y - this.track.startPos.y) < 30) { this.isDraggingStart = true; return; }
        
        const idx = this.track.path.findIndex(p => Math.hypot(p.x-x, p.y-y) < 40);
        if(idx !== -1) { 
            this.selectedIndex = idx;
            this.dragIndex = idx;
            this.updatePointUI();
            return;
        } 
        
        let bI = -1, mD = 40;
        for(let i=0; i<this.track.path.length; i++) {
            const p1 = this.track.path[i], p2 = this.track.path[(i+1)%this.track.path.length];
            const l2 = (p1.x-p2.x)**2+(p1.y-p2.y)**2; if(l2===0) continue;
            const t = Math.max(0, Math.min(1, ((x-p1.x)*(p2.x-p1.x)+(y-p1.y)*(p2.y-p1.y))/l2));
            const d = Math.hypot(x-(p1.x+t*(p2.x-p1.x)), y-(p1.y+t*(p2.y-p1.y)));
            if(d<mD) { mD=d; bI=i; }
        }
        if(bI !== -1) { 
            this.track.path.splice(bI+1, 0, {x, y, type: 'rounded', radius: 60}); 
            this.dragIndex = bI+1; 
            this.selectedIndex = bI+1;
            this.updatePointUI();
        } else {
            this.selectedIndex = null;
            this.updatePointUI();
        }
    },

    onMove: function(e) {
        const {x,y} = this.getPos(e);
        if(this.mode === 'draw') {
            if(this.isDrawing && (e.buttons === 1 || e.type==="touchmove")) {
                const last = this.drawPoints[this.drawPoints.length-1];
                if(!last || Math.hypot(x-last.x, y-last.y) > 4) this.drawPoints.push({x,y});
            }
            return;
        }
        if(this.mode === 'zones') {
            if(this.selectedZone && (e.buttons === 1 || e.type==="touchmove")) {
                if(this.isResizingZone) this.selectedZone.radius = Math.max(30, Math.hypot(x - this.selectedZone.x, y - this.selectedZone.y)); 
                else { this.selectedZone.x = x; this.selectedZone.y = y; }
            } return;
        }
        if(this.isDraggingStart) this.track.startPos = {x,y};
        else if(this.dragIndex !== null) {
            this.track.path[this.dragIndex].x = x;
            this.track.path[this.dragIndex].y = y;
        } 
        else this.hoverIndex = this.track.path.findIndex(p => Math.hypot(p.x-x, p.y-y)<20);
    },

    draw: function(ctx) {
        const p = generateTrackFromPath(this.track.id, this.track.name, this.track.path, this.track.trackWidth, this.track.startPos, this.track.startAngle, this.track.zones, this.autoOpts());
        drawRoadSurface(ctx, p, '#343a40');
        // Light, not slate: the barrier now sits exactly on the edge of the
        // asphalt, so a near-asphalt colour made it invisible while editing.
        ctx.strokeStyle='#cbd5e1'; ctx.lineWidth=3; ctx.lineJoin='round'; ctx.lineCap='round';
        strokeWalls(ctx, p);
        ctx.strokeStyle='#38bdf8'; ctx.lineWidth=2; ctx.beginPath();
        // Regular gates in blue, corner-apex gates in amber, so it is obvious
        // at a glance that every turn got one.
        const cp = p.cpF32, apex = p.cpApex;
        for(let pass=0; pass<2; pass++) {
            ctx.strokeStyle = pass ? '#f59e0b' : '#38bdf8';
            ctx.lineWidth = pass ? 3 : 2;
            ctx.beginPath();
            for(let k=0, i=0; i<cp.length; i+=7, k++) {
                if(!!(apex && apex[k]) !== !!pass) continue;
                ctx.moveTo(cp[i], cp[i+1]); ctx.lineTo(cp[i+2], cp[i+3]);
            }
            ctx.stroke();
        }

        p.zones.forEach(z => {
            ctx.beginPath(); ctx.arc(z.x, z.y, z.radius, 0, Math.PI*2);
            let zColor = z.type==='speed' ? 'rgba(239,68,68,0.2)' : z.type==='spawnkill' ? 'rgba(249,115,22,0.25)' : 'rgba(59,130,246,0.2)';
            ctx.fillStyle = zColor;
            ctx.fill(); 
            ctx.lineWidth = z===this.selectedZone?3:1; 
            let zStroke = z.type==='speed' ? '#ef4444' : z.type==='spawnkill' ? '#f97316' : '#3b82f6';
            ctx.strokeStyle = z===this.selectedZone?'#fff':zStroke;
            if(z===this.selectedZone) ctx.setLineDash([5,5]); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.font = 'bold 13px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            let zLabel = z.type==='speed' ? 'SPEED BOOST' : z.type==='spawnkill' ? '☠ SPAWN KILL' : 'POINTS';
            ctx.fillText(zLabel, z.x, z.y);
            if (z.type === 'spawnkill') { 
                const kt = z.killTimer !== undefined ? z.killTimer : 150;
                ctx.font = '11px sans-serif'; ctx.fillStyle = 'rgba(253,186,116,0.9)';
                ctx.fillText(`kill @ ${kt}f`, z.x, z.y + 16);
            }
            if (z===this.selectedZone) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(z.x+z.radius, z.y, 8, 0, Math.PI*2); ctx.fill(); ctx.strokeStyle='#000'; ctx.lineWidth=2; ctx.stroke(); }
        });

        if(this.mode === 'draw') {
            if(this.isDrawing && this.drawPoints.length > 1) {
                ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 4; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
                ctx.beginPath(); ctx.moveTo(this.drawPoints[0].x, this.drawPoints[0].y);
                for(let i=1; i<this.drawPoints.length; i++) ctx.lineTo(this.drawPoints[i].x, this.drawPoints[i].y);
                ctx.stroke();
                ctx.fillStyle = '#22d3ee'; ctx.beginPath(); ctx.arc(this.drawPoints[0].x, this.drawPoints[0].y, 6, 0, Math.PI*2); ctx.fill();
            } else if(!this.isDrawing) {
                ctx.fillStyle = 'rgba(226,232,240,0.7)'; ctx.font = 'bold 18px sans-serif'; ctx.textAlign = 'center';
                ctx.fillText('Click & drag to draw a track loop', CANVAS_WIDTH/2, CANVAS_HEIGHT/2);
            }
        }

        if(this.mode === 'path') {
            ctx.strokeStyle='#3b82f6'; ctx.lineWidth=2; ctx.setLineDash([5,5]); ctx.beginPath();
            if(this.track.path.length>0) { ctx.moveTo(this.track.path[0].x, this.track.path[0].y); for(let i=1; i<this.track.path.length; i++) ctx.lineTo(this.track.path[i].x, this.track.path[i].y); if(this.track.path.length>2) ctx.lineTo(this.track.path[0].x, this.track.path[0].y); }
            ctx.stroke(); ctx.setLineDash([]);
            
            this.track.path.forEach((p, i) => { 
                const isSelected = (this.selectedIndex === i || this.selectedIndex === 'all');
                ctx.fillStyle = i===0 ? '#22c55e' : (isSelected ? '#ef4444' : (i===this.hoverIndex ? '#fbbf24' : '#60a5fa')); 
                
                ctx.beginPath(); 
                if (p.type === 'corner') {
                    ctx.rect(p.x - 6, p.y - 6, 12, 12); // Square for corners
                } else {
                    ctx.arc(p.x, p.y, 6, 0, Math.PI*2); // Circle for rounded
                }
                ctx.fill(); 
                
                // Add a red border to the green start point if it is selected by "Select All"
                ctx.strokeStyle = (i===0 && isSelected) ? '#ef4444' : '#fff'; 
                ctx.lineWidth=2; 
                ctx.stroke(); 
            });

            ctx.save(); ctx.translate(p.startPos.x, p.startPos.y); ctx.rotate(p.startAngle); ctx.fillStyle='rgba(34,197,94,0.5)'; ctx.fillRect(-10,-5,20,10); ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(-5, 0); ctx.lineTo(5, 0); ctx.lineTo(2, -3); ctx.moveTo(5, 0); ctx.lineTo(2, 3); ctx.stroke(); ctx.restore();
            ctx.fillStyle = this.isDraggingStart ? '#fff' : '#22c55e'; ctx.beginPath(); ctx.arc(p.startPos.x, p.startPos.y, 10, 0, Math.PI*2); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth=2; ctx.stroke();
        }
    }
};

// Before anything else: clear every trace of a previous visit.
wipeStorage();
window.onload = () => app.init();

function openSidebar() {
    document.getElementById('sidebar').classList.add('open');
    document.getElementById('sidebar-overlay').classList.add('show');
}
function closeSidebar() {
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('sidebar-overlay').classList.remove('show');
}
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('sidebar-close-btn');
    if(btn) btn.onclick = closeSidebar;
});

// --- Keyboard shortcuts: Space=play/pause, H=hyper, R=reset, Esc=cancel edit/release spectate ---
document.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return; // don't hijack typing

    if (app.state.isEditing) {
        if (e.key === 'Escape') { e.preventDefault(); editor.cancel(); }
        return;
    }

    switch (e.key) {
        case ' ':
        case 'Spacebar':
            e.preventDefault(); app.toggleRun(); break;
        case 'h': case 'H':
            app.toggleHyper(); break;
        case 'r': case 'R':
            app.reset(); break;
        case 'Escape':
            if (app.state.spectateCarId !== null) app.releaseSpectate();
            break;
    }
});
