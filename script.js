const CANVAS_WIDTH = 1200; 
const CANVAS_HEIGHT = 900; 
const CAR_WIDTH = 14;
const CAR_HEIGHT = 7;
const SENSOR_ANGLES = [-Math.PI/2, -Math.PI/3, -Math.PI/6, 0, Math.PI/6, Math.PI/3, Math.PI/2];
const SENSOR_COUNT = SENSOR_ANGLES.length;

// Inline SVGs. Every icon on the page is inline markup now and the lucide
// library is gone entirely — it was 399KB of JavaScript whose whole job was to
// swap 23 <i> tags for the same <svg> tags once, at startup.
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
    focusPct: "Fraction of the population spent as mutated clones of the current best, reward-boosted specifically through whichever stretch of track it's currently slowest on (roughly ±1 second either side). Helps it stop getting stuck taking one corner badly instead of spreading every mutation evenly over a lap that mostly already works.",
    hiddenLayers: "Brain complexity. More layers = smarter but heavier computation.",
    initialTTL: "Time to Live. Frames allowed before death if no checkpoint is reached.",
    targetLaps: "Laps needed to trigger the next generation automatically.",
    maxSpeed: "Top speed. Higher speeds require faster AI reaction times.",
    acceleration: "Engine power.", turnSpeed: "Steering sensitivity. Cars turn tightest at low speed and lose authority as they speed up, same as a real car's grip limit.",
    brakeStrength: "How hard the brake pedal bites. Braking now scales with how hard the AI presses it, instead of every negative throttle snapping speed down by the same flat amount."
};

// How many simulation steps to ask for per round trip to the workers.
// Hyper mode doesn't draw, so nothing is gained by coming back every frame and
// plenty is lost: the chunk is sized so a whole generation usually finishes
// inside one or two calls. sim.c stops early the moment every car in a slice
// has crashed or something has hit the lap target, so a big chunk is never
// wasted work.
const HYPER_CHUNK = 2500;

// Canvas repaint ceiling. The simulation is not capped by this — it steps on
// every animation frame regardless — only the drawing is.
const RENDER_HZ = 30;
const RENDER_MIN_MS = 1000 / RENDER_HZ - 1;   // -1 so a 60Hz frame clock still lands on every other frame

// Cached car sprite. The car body is 14x8 drawn at 1.5x, so 21x12 covers it
// exactly; the origin sits at the middle.
const SPRITE_W = 22, SPRITE_H = 12;
const SPRITE_CX = SPRITE_W / 2, SPRITE_CY = SPRITE_H / 2;

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

// A real checkered finish line, spanning the gate exactly (p1->p2 already
// span the road's actual width at that point, auto-narrowed sections
// included — same coordinates the car's own lap-completion test uses, see
// track.startCp). Shared by the normal view and the editor so what you see
// while building a track is what you race on.
function drawFinishLine(ctx, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 4) return;
    const CHECK = 9;   // one checker square, px
    const cols = Math.max(2, Math.round(len / CHECK));
    const cell = len / cols, rows = 2, thick = cell * rows;
    ctx.save();
    ctx.translate(x1, y1);
    ctx.rotate(Math.atan2(dy, dx));
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            ctx.fillStyle = (r + c) % 2 === 0 ? '#f8fafc' : '#0f172a';
            ctx.fillRect(c * cell, -thick / 2 + r * cell, cell + 0.5, cell + 0.5);
        }
    }
    ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1.5;
    ctx.strokeRect(0, -thick / 2, len, thick);
    ctx.restore();
}

// Same gate the car's own lap counter uses (track.startCp — not necessarily
// checkpoint 0; see the comment on it in engine.js), for whichever drawing
// pass wants to show the finish line.
function finishLineOf(t) {
    if (!t || t.cpCount <= 0) return null;
    const i = Math.min(t.startCp || 0, t.cpCount - 1) * CP_STRIDE;
    const cp = t.cpF32;
    return { x1: cp[i], y1: cp[i + 1], x2: cp[i + 2], y2: cp[i + 3] };
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
        populationSize: 500, eliteClones: 30, targetLaps: 3, focusPct: 0.20, hiddenLayers: 5, initialTTL: 750,
        physics: { maxSpeed: 10, acceleration: 0.05, turnSpeed: 0.02, brakeStrength: 0.05 },
        tracks: [], currentTrackIndex: 1, cars: [], generation: 1, isRunning: false, speedMultiplier: 1, hyperMode: false,
        stats: [], globalBest: null, bestTimes: { gen: null, all: null }, isEditing: false, trackToEdit: null,
        bgCanvas: null, lapHistory: [], spectateCarId: null, aliveCount: 0,
        // Shared by both the normal view and the editor — one canvas, one
        // pan/zoom state, so switching between them never surprises you.
        view: { zoom: 1, panX: CANVAS_WIDTH / 2, panY: CANVAS_HEIGHT / 2 }
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
        // pointer handlers, which only attach while editing). Right-button
        // drag pans instead, in both the normal view and the editor — it's
        // deliberately a different button than the editor's own left-click
        // point dragging, so the two gestures can never fight over the same
        // click.
        ui.canvas.addEventListener('pointerdown', e => this.handleCanvasClick(e));
        ui.canvas.addEventListener('pointermove', e => this.movePan(e));
        ui.canvas.addEventListener('pointerup', e => this.endPan(e));
        ui.canvas.addEventListener('pointercancel', e => this.endPan(e));
        ui.canvas.addEventListener('contextmenu', e => e.preventDefault());
        ui.canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const p = this._toBackingPx(e);
            this._zoomAt(p, Math.exp(-e.deltaY * 0.0015));
        }, { passive: false });
    },

    // ---- pan / zoom -----------------------------------------------------
    // The canvas backing store is always a fixed 1200x900 (CSS scales it to
    // fit via object-contain) and everything already draws in that space, so
    // zoom/pan is one extra transform applied once at the top of each draw
    // pass rather than a change to any drawing code: screen = view * world,
    // a uniform scale plus a translate, no rotation.
    _viewMatrix: function() {
        const v = this.state.view;
        return { z: v.zoom, e: CANVAS_WIDTH / 2 - v.panX * v.zoom, f: CANVAS_HEIGHT / 2 - v.panY * v.zoom };
    },

    // CSS pixels (from a pointer event) -> canvas backing-store pixels. The
    // one step every coordinate conversion below shares.
    _toBackingPx: function(e) {
        const r = ui.canvas.getBoundingClientRect();
        const scale = Math.min(r.width / CANVAS_WIDTH, r.height / CANVAS_HEIGHT);
        const offsetX = (r.width - CANVAS_WIDTH * scale) / 2, offsetY = (r.height - CANVAS_HEIGHT * scale) / 2;
        return { x: (e.clientX - r.left - offsetX) / scale, y: (e.clientY - r.top - offsetY) / scale };
    },

    // Backing-store pixels -> world coordinates, inverting the view matrix.
    // Every place that used to treat backing-store pixels AS world
    // coordinates (before zoom/pan existed, the two were the same thing)
    // calls this now.
    screenToWorld: function(bx, by) {
        const v = this._viewMatrix();
        return { x: (bx - v.e) / v.z, y: (by - v.f) / v.z };
    },

    _zoomAt: function(backingPt, factor) {
        const v = this.state.view;
        const z = Math.max(1, Math.min(8, v.zoom * factor));
        if(z === v.zoom) return;
        const w = this.screenToWorld(backingPt.x, backingPt.y);
        v.zoom = z;
        // Keep the same world point under the cursor after the zoom changes.
        v.panX = w.x - (backingPt.x - CANVAS_WIDTH / 2) / z;
        v.panY = w.y - (backingPt.y - CANVAS_HEIGHT / 2) / z;
        this._clampView();
        this._needsDraw = true;
    },

    // Keeps the visible viewport inside the map instead of panning off into
    // empty space beyond it. At zoom 1 (minimum) the two bounds coincide, so
    // the centre is pinned to the canvas centre — exactly the old, un-zoomed
    // behaviour.
    _clampView: function() {
        const v = this.state.view;
        const halfW = CANVAS_WIDTH / (2 * v.zoom), halfH = CANVAS_HEIGHT / (2 * v.zoom);
        v.panX = Math.max(halfW, Math.min(CANVAS_WIDTH - halfW, v.panX));
        v.panY = Math.max(halfH, Math.min(CANVAS_HEIGHT - halfH, v.panY));
    },

    resetView: function() {
        this.state.view.zoom = 1;
        this.state.view.panX = CANVAS_WIDTH / 2;
        this.state.view.panY = CANVAS_HEIGHT / 2;
        this._needsDraw = true;
    },

    zoomStep: function(factor) {
        this._zoomAt({ x: CANVAS_WIDTH / 2, y: CANVAS_HEIGHT / 2 }, factor);
    },

    _panState: null,
    startPan: function(e) {
        e.preventDefault();
        ui.canvas.setPointerCapture(e.pointerId);
        const p = this._toBackingPx(e);
        this._panState = { pointerId: e.pointerId, startBX: p.x, startBY: p.y, panX0: this.state.view.panX, panY0: this.state.view.panY };
    },
    movePan: function(e) {
        const ps = this._panState;
        if(!ps || e.pointerId !== ps.pointerId) return;
        const p = this._toBackingPx(e), v = this.state.view;
        v.panX = ps.panX0 - (p.x - ps.startBX) / v.zoom;
        v.panY = ps.panY0 - (p.y - ps.startBY) / v.zoom;
        this._clampView();
        this._needsDraw = true;
    },
    endPan: function(e) {
        if(this._panState && e.pointerId === this._panState.pointerId) this._panState = null;
    },

    // Resolves which car telemetry/highlight follows: a manually-clicked car
    // (until it crashes or the user releases it), otherwise the fastest alive.
    // The last pick, refreshed once per applied result round rather than on
    // every animation frame — it is an O(population) scan and the answer only
    // changes when the simulation does.
    _spectated: null,

    getSpectatedCar: function() { return this._spectated; },

    _pickSpectated: function() {
        const id = this.state.spectateCarId;
        if (id !== null) {
            const car = this.state.cars[id];
            if (car && !car.crashed) return car;
            this.state.spectateCarId = null; // selection crashed/gone — auto-revert
        }
        // Seeded with the first car that is actually alive. Seeding with car 0
        // regardless meant that if car 0 had crashed and no living car outscored
        // it, the telemetry panel ended up pinned to a wreck.
        const cars = this.state.cars;
        let best = null;
        for (let i = 0; i < cars.length; i++) {
            const c = cars[i];
            if (c.crashed) continue;
            if (best === null || c.fitness > best.fitness) best = c;
        }
        return best;
    },

    releaseSpectate: function() {
        this.state.spectateCarId = null;
        this._spectated = this._pickSpectated();
        this._needsDraw = true;
    },

    handleCanvasClick: function(e) {
        if(e.button === 2) { this.startPan(e); return; }
        if(e.button !== undefined && e.button !== 0) return;
        if (this.state.isEditing || this.state.hyperMode || !this.state.cars.length) return;
        const p = this._toBackingPx(e);
        const { x, y } = this.screenToWorld(p.x, p.y);

        // Hit radius in world units — the same 22 it always was at zoom=1,
        // shrinking on screen as you zoom in, which is exactly what makes
        // zooming in give more precise car selection.
        let closest = null, closestDist = 22;
        for (const c of this.state.cars) {
            if (c.crashed) continue;
            const d = Math.hypot(c.x - x, c.y - y);
            if (d < closestDist) { closestDist = d; closest = c; }
        }
        this.state.spectateCarId = closest ? closest.id : null;
        this._spectated = this._pickSpectated();
        this._needsDraw = true;
    },

    // Async now: nothing can be built until the wasm module is compiled and the
    // worker pool has come up, because the track generator lives in there too.
    init: async function() {
        try {
            this._initUICache();
            this.bindActions();
            this.showBuildStamp();
            await Engine.ready();
            if(ui.coreCount && Engine._pendingCoreLabel) ui.coreCount.innerHTML = Engine._pendingCoreLabel;
            this.resetTracks();
            this.initChart();
            this._updateStatsTable();

            this.loop();
        } catch (e) {
            console.error("Init Error:", e);
            this.showFatal(e);
        }
    },

    // Show which build is actually on screen — version number included.
    //
    // version.json is written by the deploy workflow, not committed, so it says
    // what is genuinely live rather than what the source happens to claim. When
    // it is absent — running locally, or a deploy that never landed — the
    // static fallback in the markup stays as it is.
    //
    // The version itself (e.g. "21.9") is computed by the workflow from git
    // history — a commit count since the WebAssembly port, which is V21.0 —
    // not typed in by hand anywhere. See .github/workflows/deploy.yml for why:
    // the predecessor project hand-maintained its version string and
    // repeatedly forgot to bump it. A number nobody has to remember to update
    // is a number that can't go stale.
    showBuildStamp: function() {
        fetch('./version.json', { cache: 'no-store' })
            .then(r => r.ok ? r.json() : null)
            .then(v => {
                if (!v || !v.version) return;
                const full = document.getElementById('version-tag');
                if (full) {
                    full.textContent = `V${v.version}${v.short ? ' · build ' + v.short : ''}`;
                    if (v.built) full.title = `deployed ${v.built} from ${v.ref || 'unknown branch'}`;
                }
                // The two compact mobile headers get the version alone — no
                // room there for a commit hash too.
                document.querySelectorAll('.version-tag-compact').forEach(el => {
                    el.textContent = ` · V${v.version}`;
                    if (v.built) el.title = `deployed ${v.built} from ${v.ref || 'unknown branch'}`;
                });
            })
            .catch(() => { /* no stamp; leave the static fallback alone */ });
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
        this.state.globalBest = null;
        this._resetCars();
        this.state.bestTimes.gen = null;
        this._runToken++;

        Engine.initPopulation(this.state.populationSize, this.state.hiddenLayers, loadedBrainJSON);
        this.updateUI();
    },

    // One colour per car SLOT, not per car per generation. The elite band is
    // the first `eliteClones` slots either way, so the green/lime livery still
    // marks exactly the carried-forward brains — it just stops rebuilding 500
    // colour strings every generation, and it lets the sprite cache in draw()
    // be built once for the whole run instead of thrown away each time.
    _ensureColors: function() {
        const n = this.state.populationSize;
        const elite = Math.min(this.state.eliteClones, n);
        if(this._colors && this._colors.length === n && this._colorElite === elite) return;
        const out = new Array(n);
        for(let i=0; i<n; i++) {
            out[i] = i < elite ? (i === 0 ? '#22c55e' : '#84cc16') : `hsl(${(i * 137.508) % 360},80%,60%)`;
        }
        this._colors = out;
        this._colorElite = elite;
    },

    // Resize the car array if the population changed, then reset every record
    // in place. Rebuilding 500 objects (each with two fresh typed arrays) once
    // a generation was pure churn: nothing about a car record changes shape.
    _resetCars: function() {
        const track = this.currentTrack;
        const n = this.state.populationSize;
        this._ensureColors();
        let cars = this.state.cars;
        if(!cars || cars.length !== n) {
            cars = new Array(n);
            for(let i=0; i<n; i++) cars[i] = {
                id: i, x: 0, y: 0, angle: 0, speed: 0, color: this._colors[i],
                fitness: 0, crashed: false, checkpoints: 0,
                sensors: new Float32Array(SENSOR_COUNT), inputs: new Float32Array(2),
                completedLaps: 0
            };
            this.state.cars = cars;
        }
        const sx = track ? track.startPos.x : 0, sy = track ? track.startPos.y : 0;
        const sa = track ? track.startAngle : 0;
        for(let i=0; i<n; i++) {
            const c = cars[i];
            c.color = this._colors[i];
            c.x = sx; c.y = sy; c.angle = sa;
            c.speed = 0; c.fitness = 0; c.crashed = false; c.completedLaps = 0; c.checkpoints = 0;
            c.sensors.fill(0); c.inputs[0] = 0; c.inputs[1] = 0;
        }
        this.state.aliveCount = n;
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

        // Percentile snapshot for the improvement table — cheap next to
        // breeding a whole generation, and this is the only place that ever
        // sees every car's fitness at once.
        const sorted = Array.from(fitness).sort((a, b) => b - a);
        const n1 = Math.max(1, Math.round(sorted.length * 0.01));
        const n10 = Math.max(1, Math.round(sorted.length * 0.10));
        let s1 = 0; for(let i=0; i<n1; i++) s1 += sorted[i];
        let s10 = 0; for(let i=0; i<n10; i++) s10 += sorted[i];
        const top1 = s1 / n1, top10 = s10 / n10;

        const res = Engine.evolve(fitness, this.state.eliteClones, this.state.generation);
        this.state.globalBest = { fitness: res.globalBest };

        const avg = sum / cars.length;
        this._pushStat({
            gen: this.state.generation, best, avg, top1, top10,
            time: this.state.bestTimes.gen ? this.state.bestTimes.gen.toFixed(2) : null
        });
        this.updateChart();

        this._recent.push({ gen: this.state.generation, best, avg, top1, top10 });
        if(this._recent.length > 101) this._recent.shift();
        this._updateStatsTable();

        // Elite clones keep the green/lime livery so you can pick the carried-
        // forward brains out of the pack on screen.
        this._resetCars();
        this.state.generation++;
        this.state.bestTimes.gen = null;
        this.updateUI();
    },

    // The chart's backing store. Every generation is represented somewhere in
    // here for the whole life of the run ("total", not a truncated recent
    // window) while staying capped at _statCap entries: once full, each new
    // point first tries to merge into the newest bucket, and once THAT bucket
    // is as full as every other (n === _statRes), the whole array halves its
    // resolution by merging consecutive pairs. That is the same trick a
    // real-time monitoring graph uses — the far past gets coarser instead of
    // disappearing — and it is what keeps this at a bounded, CONSTANT cost per
    // generation no matter how long the run has been going. The flat 300-point
    // window this replaced showed the same constant cost by throwing the old
    // 3/4 of the run away outright; the unbounded array it replaced (every
    // point, remapped in full on every single generation) was the actual O(n²)
    // stall that made a long session feel like it was "getting slower".
    _statCap: 300,
    _statRes: 1,
    _pushStat: function(entry) {
        entry.n = 1;
        const st = this.state.stats;
        if(st.length > 0) {
            const last = st[st.length - 1];
            if(last.n < this._statRes) {
                last.gen = entry.gen;
                last.best = Math.max(last.best, entry.best);
                last.avg = (last.avg * last.n + entry.avg) / (last.n + 1);
                if(entry.top1 !== undefined) last.top1 = (last.top1 * last.n + entry.top1) / (last.n + 1);
                if(entry.top10 !== undefined) last.top10 = (last.top10 * last.n + entry.top10) / (last.n + 1);
                if(entry.time) last.time = entry.time;
                last.n++;
                return;
            }
        }
        if(st.length >= this._statCap) {
            const merged = [];
            for(let i=0; i<st.length; i+=2) {
                const a = st[i], b = st[i+1];
                if(!b) { merged.push(a); continue; }
                const an = a.n || 1, bn = b.n || 1;
                merged.push({
                    gen: b.gen, n: an + bn,
                    best: Math.max(a.best, b.best),
                    avg: (a.avg * an + b.avg * bn) / (an + bn),
                    top1: a.top1 !== undefined ? (a.top1 * an + b.top1 * bn) / (an + bn) : undefined,
                    top10: a.top10 !== undefined ? (a.top10 * an + b.top10 * bn) / (an + bn) : undefined,
                    time: b.time || a.time
                });
            }
            this.state.stats = merged;
            this._statRes *= 2;
        }
        this.state.stats.push(entry);
    },

    // Raw, unaggregated per-generation snapshots — the last 101 only, so the
    // "improvement over the last N generations" table can read an exact value
    // N back for N up to 100. Deliberately separate from state.stats above:
    // that one coarsens on purpose to stay bounded over a whole run, which
    // would make "10 generations ago" a lie once it starts merging buckets.
    _recent: [],

    _updateStatsTable: function() {
        const el = document.getElementById('stats-table-body');
        if(!el) return;
        const r = this._recent, n = r.length;
        if(n < 2) { el.innerHTML = '<tr><td colspan="4" class="text-center text-slate-600 italic py-2">Not enough data yet</td></tr>'; return; }
        const cur = r[n - 1];
        const rows = [['Top 1%','top1'], ['Top 10%','top10'], ['Total','avg']];
        const windows = [1, 10, 100];
        el.innerHTML = rows.map(([label, key]) => {
            const cells = windows.map(w => {
                if(n <= w) return '<td class="text-slate-600 text-center">–</td>';
                const past = r[n - 1 - w][key];
                const delta = (cur[key] - past) / w;
                const cls = delta > 0 ? 'text-emerald-400' : delta < 0 ? 'text-red-400' : 'text-slate-500';
                const sign = delta > 0 ? '+' : '';
                return `<td class="text-center ${cls} font-mono">${sign}${delta.toFixed(1)}</td>`;
            }).join('');
            return `<tr><td class="text-slate-400 pr-2">${label}</td>${cells}</tr>`;
        }).join('');
    },

    // Fold one round of worker results back into the render records. In normal
    // mode each row is the full 18-float telemetry; in hyper mode it is three
    // floats per car, since nothing is drawn.
    _applyResults: function(st) {
        const cars = this.state.cars;
        let alive = 0;
        for(const r of st.rows) {
            // Read below, then returned to its worker to be refilled.
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
        for(const r of st.rows) Engine.recycle(r.index, r.buffer.buffer);
        this.state.aliveCount = alive;
        this._needsDraw = true;
        // Once per applied round, not once per rendered frame — see draw().
        this._spectated = this._pickSpectated();
    },

    _recordLap: function(time) {
        if(!(time > 0)) return;
        if(!this.state.bestTimes.gen || time < this.state.bestTimes.gen) this.state.bestTimes.gen = time;
        if(!this.state.bestTimes.all || time < this.state.bestTimes.all) this.state.bestTimes.all = time;
        this.state.lapHistory.push(time);
        if(this.state.lapHistory.length > 20) this.state.lapHistory.shift();
        this.updateLapHistory();
    },

    // Invalidation token for in-flight worker chunks. Anything that replaces
    // the population or the track bumps it, and results carrying a stale token
    // are dropped rather than written into records they no longer describe.
    _runToken: 0,
    _pending: null,
    _pendingToken: -1,
    _lastDraw: 0,
    // Set by anything that changes what the canvas should show. A paused,
    // untouched screen repaints zero times instead of sixty times a second.
    _needsDraw: true,

    loop: async function() {
        const running = this.state.isRunning && !this.state.isEditing;

        if(this._pending) {
            const st = await this._pending;
            const token = this._pendingToken;
            this._pending = null;
            if(token === this._runToken && running) {
                this._applyResults(st);
                if(st.allCrashed || st.maxLaps >= this.state.targetLaps) this.evolve();
            }
        }

        if(running && !this._pending) {
            // Start the next chunk BEFORE the UI work below rather than after
            // it. The workers used to sit idle through every updateUI/draw —
            // a hard barrier once per frame — and now they compute the next
            // chunk while the main thread paints the previous one.
            const hyper = this.state.hyperMode;
            const iters = hyper ? HYPER_CHUNK : this.state.speedMultiplier;
            this._pendingToken = this._runToken;
            this._pending = Engine.run(iters, !hyper);
        }

        this.updateUI();

        // What gets painted, and how often:
        //   * the editor draws every frame, so dragging a point stays smooth;
        //   * hyper mode draws NOTHING at all, not even the cached background,
        //     because the whole point of it is that the screen is not the
        //     output;
        //   * everything else is capped at RENDER_HZ and skipped entirely when
        //     nothing has changed. The simulation still steps on every
        //     animation frame — only the painting is throttled.
        if(this.state.isEditing) {
            this.draw();
        } else if(!this.state.hyperMode && this._needsDraw) {
            const now = performance.now();
            if(now - this._lastDraw >= RENDER_MIN_MS) {
                this._lastDraw = now;
                this._needsDraw = false;
                this.draw();
            }
        }
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

        // The gate that actually completes a lap — track.startCp, not
        // checkpoint 0 — so a track whose start was dragged away from where
        // the generator happened to begin the centreline still shows the
        // finish line where cars are actually scored on it.
        const fl = finishLineOf(t);
        if(fl) drawFinishLine(ctx, fl.x1, fl.y1, fl.x2, fl.y2);
    },

    draw: function() {
        const ctx = ui.ctx; // cached — no getElementById every frame

        if(this.state.isEditing && this.state.trackToEdit) {
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.fillStyle = '#3a5a40'; ctx.fillRect(0,0,CANVAS_WIDTH,CANVAS_HEIGHT);
            editor.draw(ctx);
            return;
        }

        // The one extra transform zoom/pan needs: everything below already
        // draws in world (1200x900) coordinates, so composing it in here once
        // is the whole change. Minimum zoom is 1 and pan is clamped to the
        // map (_clampView), so at rest this is exactly the identity transform
        // it always was.
        const v = this._viewMatrix();
        ctx.setTransform(v.z, 0, 0, v.z, v.e, v.f);

        if(this.state.bgCanvas) ctx.drawImage(this.state.bgCanvas, 0, 0);

        // Render ALL cars persistently, no color flashing, no hiding.
        // Telemetry/highlight follows a manually-clicked car, or else
        // auto-follows the fastest alive car.
        const spectated = this.getSpectatedCar();
        const isManual = this.state.spectateCarId !== null;

        this._drawTelemetry(spectated, isManual);

        // One drawImage per car instead of save/translate/rotate/scale, four
        // fillRects and a restore — roughly 3,000 canvas state changes a frame
        // at 500 cars, which was the bulk of the main thread's paint cost.
        // setTransform composes the rotation and position in one call; the
        // sprite for each livery is drawn once and cached. Composed with the
        // view matrix by hand (both are similarity transforms — no rotation
        // in the view — so this is just the product of the two), since
        // setTransform REPLACES the CTM rather than composing with it.
        const cars = this.state.cars;
        for(let i=0; i<cars.length; i++) {
            const c = cars[i];
            if(c.crashed) continue;
            const sprite = this._carSprite(c.color);
            const cs = Math.cos(c.angle), sn = Math.sin(c.angle);
            ctx.setTransform(v.z*cs, v.z*sn, -v.z*sn, v.z*cs, v.e + v.z*c.x, v.f + v.z*c.y);
            ctx.drawImage(sprite, -SPRITE_CX, -SPRITE_CY);
        }
        ctx.setTransform(v.z, 0, 0, v.z, v.e, v.f);

        if(spectated && !spectated.crashed) {
            const c = spectated;
            // Highlight ring around the spectated car — solid cyan when
            // manually picked, a subtle dashed ring when auto-following.
            ctx.save();
            ctx.strokeStyle = isManual ? '#22d3ee' : 'rgba(255,255,255,0.55)';
            ctx.lineWidth = isManual ? 2.5 : 1.5;
            if(!isManual) ctx.setLineDash([4,3]);
            ctx.beginPath(); ctx.arc(c.x, c.y, 16, 0, Math.PI*2); ctx.stroke();
            ctx.restore();

            if(c.sensors) {
                // Same reach the simulation raycast used — it follows Max
                // Speed now, so a hard-coded 180 here would draw rays that
                // stopped short of where the car can actually see.
                const len = Engine.sensorLength();
                ctx.strokeStyle = 'rgba(234,179,8,0.3)';
                for(let k=0; k<c.sensors.length; k++) {
                    const ang = c.angle + SENSOR_ANGLES[k];
                    const ca = Math.cos(ang), sa = Math.sin(ang);
                    ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(c.x + ca*len, c.y + sa*len); ctx.stroke();
                    const sv = c.sensors[k];
                    if(sv > 0) {
                        const d = (1-sv) * len;
                        ctx.fillStyle = '#f59e0b';
                        ctx.beginPath(); ctx.arc(c.x + ca*d, c.y + sa*d, 2, 0, Math.PI*2); ctx.fill();
                    }
                }
            }
        }
    },

    // A car is the same handful of rectangles every time, so each livery is
    // rasterised once into a tiny offscreen canvas and then blitted.
    _sprites: null,
    _carSprite: function(color) {
        if(!this._sprites) this._sprites = new Map();
        let sp = this._sprites.get(color);
        if(sp) return sp;
        sp = document.createElement('canvas');
        sp.width = SPRITE_W; sp.height = SPRITE_H;
        const g = sp.getContext('2d');
        g.translate(SPRITE_CX, SPRITE_CY);
        g.scale(1.5, 1.5);
        g.fillStyle = color;      g.fillRect(-7, -4, 14, 8);
        g.fillStyle = '#0f172a';  g.fillRect(-2, -3, 4, 6);
        g.fillStyle = '#fbbf24';  g.fillRect(6, -3, 1, 2); g.fillRect(6, 1, 1, 2);
        this._sprites.set(color, sp);
        return sp;
    },

    // Telemetry is six DOM writes; doing them unconditionally meant six style
    // invalidations per frame for bars that mostly had not moved a pixel.
    _tel: { label: '', manual: null, steerL: '', steerR: '', gas: '', brake: '', speed: '', speedVal: '' },
    _drawTelemetry: function(spectated, isManual) {
        const t = this._tel;
        const label = spectated
            ? (isManual ? `Spectating Car #${spectated.id} (Manual)` : 'Live Telemetry (Auto — Fastest)')
            : 'Live Telemetry';
        if(ui.telemetryLabel && label !== t.label) { ui.telemetryLabel.textContent = label; t.label = label; }
        if(ui.btnRelease && isManual !== t.manual) { ui.btnRelease.classList.toggle('hidden', !isManual); t.manual = isManual; }
        if(!spectated || spectated.crashed) return;

        const i = spectated.inputs || [0,0];
        const pct = v => Math.round(v) + '%';
        const steerL = i[0] < 0 ? pct(Math.abs(i[0])*50) : '0%';
        const steerR = i[0] > 0 ? pct(i[0]*50) : '0%';
        const gas    = i[1] > 0 ? pct(i[1]*100) : '0%';
        const brake  = i[1] > 0 ? '0%' : pct(Math.abs(i[1])*100);
        const speed  = pct(Math.min((spectated.speed / this.state.physics.maxSpeed)*100, 100));
        const speedVal = String(Math.round(spectated.speed));
        if(steerL !== t.steerL) { ui.telSteerL.style.width = steerL; t.steerL = steerL; }
        if(steerR !== t.steerR) { ui.telSteerR.style.width = steerR; t.steerR = steerR; }
        if(gas !== t.gas)       { ui.telGas.style.width = gas; t.gas = gas; }
        if(brake !== t.brake)   { ui.telBrake.style.width = brake; t.brake = brake; }
        if(speed !== t.speed)   { ui.telSpeed.style.width = speed; t.speed = speed; }
        if(speedVal !== t.speedVal) { ui.telSpeedVal.textContent = speedVal; t.speedVal = speedVal; }
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
        this._statRes = 1; this._recent = [];
        _ui_gen=-1; _ui_alive=-1; _ui_allBest=null;
        if(ui.lapHistoryM) ui.lapHistoryM.innerHTML = '<span class="text-[10px] text-slate-600 italic">No laps yet</span>';
        this.initPopulation();
        this.updateChart();
        this._updateStatsTable();
        this.updateUI(); this.toggleRun(); this.toggleRun();
    },
    
    // Physics is pure config — pushing it doesn't need the track rebuilding.
    updatePhysics: function(k, v) {
        this.state.physics[k] = parseFloat(v);
        const label = k==='maxSpeed' ? 'maxSpeed' : k==='acceleration' ? 'accel' : k==='turnSpeed' ? 'turn' : 'brakeStrength';
        const el = document.getElementById('val-' + label);
        if(el) el.innerText = v;
        Engine.pushConfig(this.state);
        // Sensor reach is derived from Max Speed, and the wall buckets a track
        // was built with are sized to that reach. Changing it without
        // rebuilding would leave longer rays looking through walls that were
        // never put in the bucket.
        if(k === 'maxSpeed' && this.currentTrack) {
            Engine.setTrack(this.currentTrack, this.state);
            this._runToken++;
            this._needsDraw = true;
        }
    },
    updateConfig: function(k, v) {
        this.state[k] = parseFloat(v);
        let id = 'val-'+(k==='populationSize'?'pop':k==='targetLaps'?'laps':k==='speedMultiplier'?'speed':k==='eliteClones'?'elite':k==='focusPct'?'focus':k==='hiddenLayers'?'hidden':'ttl');
        let d = v; if(k==='speedMultiplier') d+='x'; if(k==='focusPct') d=Math.round(v*100)+'%';
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
        apply('cfg-focusPct', st.focusPct, 'val-focus', v => Math.round(v*100)+'%');
        apply('cfg-hiddenLayers', st.hiddenLayers, 'val-hidden');
        apply('cfg-initialTTL', st.initialTTL, 'val-ttl');
        apply('cfg-targetLaps', st.targetLaps, 'val-laps');
        apply('cfg-maxSpeed', st.physics.maxSpeed, 'val-maxSpeed');
        apply('cfg-acceleration', st.physics.acceleration, 'val-accel');
        apply('cfg-turnSpeed', st.physics.turnSpeed, 'val-turn');
        apply('cfg-brakeStrength', st.physics.brakeStrength, 'val-brakeStrength');
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
    // --- whole-session save / load ----------------------------------------
    // "Save AI" keeps one brain, which is enough to show off a good lap and
    // useless for carrying on a run — reloading it reseeds the whole field
    // from mutated copies of that one network. This keeps everything: every
    // brain in the population, the generation counter, the best times, the
    // lap history and the fitness graph. Load it and the run continues as if
    // the tab had never been closed.
    saveSession: function() {
        if(!this.state.cars.length) return;
        const st = this.state;
        const session = {
            format: 'trackml-session',
            version: 1,
            savedAt: new Date().toISOString(),
            generation: st.generation,
            settings: {
                populationSize: st.populationSize, eliteClones: st.eliteClones,
                targetLaps: st.targetLaps, focusPct: st.focusPct,
                hiddenLayers: st.hiddenLayers, initialTTL: st.initialTTL,
                speedMultiplier: st.speedMultiplier,
                physics: { ...st.physics }
            },
            trackName: this.currentTrack ? this.currentTrack.name : null,
            stats: st.stats,
            bestTimes: st.bestTimes,
            lapHistory: st.lapHistory,
            population: Engine.exportPopulation()
        };
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([JSON.stringify(session)], {type:'application/json'}));
        a.download = `trackml-session-g${st.generation}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
    },

    loadSession: function(inp) {
        if(!inp.files[0]) return;
        const r = new FileReader();
        r.onload = e => {
            let j;
            try { j = JSON.parse(e.target.result); }
            catch(er) { alert("That file isn't valid JSON."); return; }
            if(!j || j.format !== 'trackml-session') {
                alert('That is not a TrackML session file. (A single saved brain goes in "Load AI".)');
                return;
            }
            const st = this.state;
            st.isRunning = false;
            const cfg = j.settings || {};
            for(const k of ['populationSize','eliteClones','targetLaps','focusPct','hiddenLayers','initialTTL','speedMultiplier']) {
                if(typeof cfg[k] === 'number') st[k] = cfg[k];
            }
            if(cfg.physics) for(const k in cfg.physics) {
                if(typeof cfg.physics[k] === 'number') st.physics[k] = cfg.physics[k];
            }
            this.syncSettingsUI();
            Engine.pushConfig(st);

            const bad = Engine.importPopulation(j.population);
            if(bad) { alert('Could not load that session: ' + bad); return; }

            st.generation = j.generation || 1;
            st.stats = Array.isArray(j.stats) ? j.stats : [];
            st.bestTimes = j.bestTimes || { gen: null, all: null };
            st.lapHistory = Array.isArray(j.lapHistory) ? j.lapHistory : [];
            st.globalBest = null;
            st.spectateCarId = null;
            // A loaded session's coarse chart buckets don't carry the exact
            // per-generation values the improvement table needs, so it starts
            // empty and rebuilds itself from here rather than show something
            // wrong. _statRes resets too: it'll re-derive naturally as new
            // generations are pushed, self-correcting via the same cap logic
            // even if that runs a little fine-grained for a while first.
            this._statRes = 1; this._recent = [];
            _ui_gen = -1; _ui_alive = -1; _ui_allBest = null;

            this._resetCars();
            this._runToken++;
            this._needsDraw = true;
            this.updateChart();
            this._updateStatsTable();
            this.updateLapHistory();
            this.updateUI();
            if(j.trackName && this.currentTrack && j.trackName !== this.currentTrack.name) {
                alert(`Session loaded. It was trained on "${j.trackName}" — the current track is "${this.currentTrack.name}".`);
            }
        };
        r.readAsText(inp.files[0]);
        inp.value = '';
    },

    // The one explicit "forget everything" control. Nothing is persisted
    // automatically — every open already starts clean — so this exists for the
    // case where something cached looks wrong and you want the browser's copy
    // of the site gone too, not just the run.
    wipeEverything: function() {
        if(!confirm('Wipe all site data and reload?\n\nThis clears browser storage and any cached copy of the app, and throws away the current run. Saved files on your computer are untouched.')) return;
        wipeStorage();
        const done = () => location.reload();
        if(window.caches && caches.keys) {
            caches.keys().then(ks => Promise.all(ks.map(k => caches.delete(k)))).then(done).catch(done);
        } else done();
    },

    // --- declarative event binding ----------------------------------------
    // The markup used to carry 82 inline on* handlers, each hard-wiring an
    // element to a global function name with nothing to catch a rename — a
    // typo failed silently at click time. Now every handler is a data-*
    // attribute naming an action, and this binds them once at startup.
    //
    // The attribute value is "root.method|arg|arg", where an argument of
    // @value, @checked, @el or @event is substituted at call time. Nothing is
    // eval'd and only the three roots below are reachable, so the page stays
    // Content-Security-Policy clean.
    bindActions: function(root) {
        const ROOTS = { app: app, editor: editor, ImageImport: typeof ImageImport !== 'undefined' ? ImageImport : null };
        const EVENTS = { click: 'click', input: 'input', change: 'change', enter: 'mouseenter', leave: 'mouseleave' };
        (root || document).querySelectorAll('[data-click],[data-input],[data-change],[data-enter],[data-leave]').forEach(el => {
            for(const key in EVENTS) {
                const spec = el.dataset[key];
                if(!spec || el['_bound_' + key]) continue;
                el['_bound_' + key] = true;
                const parts = spec.split('|');
                const [rootName, fnName] = parts[0].split('.');
                el.addEventListener(EVENTS[key], ev => {
                    const target = ROOTS[rootName];
                    const fn = target && target[fnName];
                    if(typeof fn !== 'function') { console.warn('no such action:', spec); return; }
                    const args = parts.slice(1).map(a =>
                        a === '@value' ? el.value :
                        a === '@checked' ? el.checked :
                        a === '@el' ? el :
                        a === '@event' ? ev : a);
                    fn.apply(target, args);
                });
            }
        });
    },

    toggleSettings: function() { document.getElementById('config-panel').classList.toggle('hidden'); },
    closeCodeModal: function() { document.getElementById('code-modal').classList.add('hidden'); },
    dismissBackdrop: function(el, ev) { if(ev.target === el) el.classList.add('hidden'); },
    selectText: function(el) { el.select(); },
    openSidebar: function() { openSidebar(); },
    closeSidebar: function() { closeSidebar(); },
    copyCode: function(btn) {
        const text = document.getElementById('code-output').value;
        navigator.clipboard.writeText(text).then(() => {
            btn.textContent = '\u2713 Copied!';
            setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
        });
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
            this.updateChart();
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
        this._runToken++; this._needsDraw = true;
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
    // --- fitness chart -----------------------------------------------------
    // Drawn here rather than by Chart.js, which was 208KB — half the page's
    // remaining weight — to plot two lines on a canvas this file was already
    // drawing on. Same two series, same colours, same filled area under the
    // best line, same legend and the same hover readout.
    chart: null,

    initChart: function() {
        const canvas = document.getElementById('fitness-chart');
        if(!canvas) return;
        this.chart = {
            canvas,
            ctx: canvas.getContext('2d'),
            best: [], avg: [], gens: [], times: [],
            hover: -1
        };
        canvas.addEventListener('pointermove', e => {
            const c = this.chart, rect = c.canvas.getBoundingClientRect();
            const i = this._chartIndexAt(e.clientX - rect.left, rect.width);
            if(i !== c.hover) { c.hover = i; this._drawChart(); }
        });
        canvas.addEventListener('pointerleave', () => {
            if(this.chart.hover !== -1) { this.chart.hover = -1; this._drawChart(); }
        });
        window.addEventListener('resize', () => this._drawChart());
        this._drawChart();
    },

    updateChart: function() {
        const c = this.chart;
        if(!c) return;
        const st = this.state.stats;
        c.best = st.map(s => s.best);
        c.avg = st.map(s => s.avg);
        c.gens = st.map(s => s.gen);
        c.times = st.map(s => s.time);
        this._drawChart();
    },

    // Plot area, in CSS pixels. Left gutter holds the y tick labels, the top
    // strip holds the legend.
    _chartBox: function() {
        const c = this.chart, r = c.canvas.getBoundingClientRect();
        return { x: 34, y: 20, w: Math.max(10, r.width - 44), h: Math.max(10, r.height - 30), W: r.width, H: r.height };
    },

    _chartIndexAt: function(px, width) {
        const c = this.chart, n = c.best.length;
        if(n === 0) return -1;
        const b = this._chartBox();
        if(px < b.x - 4 || px > b.x + b.w + 4) return -1;
        const t = n === 1 ? 0 : (px - b.x) / b.w;
        return Math.max(0, Math.min(n - 1, Math.round(t * (n - 1))));
    },

    // "Nice" round upper bound, so the gridlines land on readable numbers.
    _niceCeil: function(v) {
        if(!(v > 0)) return 1;
        const mag = Math.pow(10, Math.floor(Math.log10(v)));
        const f = v / mag;
        const step = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
        return step * mag;
    },

    _drawChart: function() {
        const c = this.chart;
        if(!c) return;
        const ctx = c.ctx, rect = c.canvas.getBoundingClientRect();
        if(rect.width < 2 || rect.height < 2) return;

        // Back the canvas with device pixels so the text and 2px lines stay
        // crisp on a scaled display, then work in CSS pixels throughout.
        const dpr = window.devicePixelRatio || 1;
        const pw = Math.round(rect.width * dpr), ph = Math.round(rect.height * dpr);
        if(c.canvas.width !== pw || c.canvas.height !== ph) { c.canvas.width = pw; c.canvas.height = ph; }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);

        const b = this._chartBox();
        const n = c.best.length;
        let max = 0;
        for(let i=0; i<n; i++) { if(c.best[i] > max) max = c.best[i]; if(c.avg[i] > max) max = c.avg[i]; }
        const top = this._niceCeil(max) || 1;

        const FONT = '10px "Helvetica Neue", Helvetica, Arial, sans-serif';
        ctx.font = FONT;
        ctx.textBaseline = 'middle';

        // y gridlines and ticks
        const TICKS = 4;
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = 'right';
        for(let t=0; t<=TICKS; t++) {
            const v = top * t / TICKS;
            const y = Math.round(b.y + b.h - (v / top) * b.h) + 0.5;
            ctx.beginPath(); ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y); ctx.stroke();
            ctx.fillText(v >= 1000 ? Math.round(v/1000) + 'k' : String(Math.round(v)), b.x - 6, y);
        }

        const xAt = i => n <= 1 ? b.x + b.w/2 : b.x + (i / (n - 1)) * b.w;
        const yAt = v => b.y + b.h - (Math.max(0, v) / top) * b.h;

        if(n > 0) {
            // filled area under Best, then the two lines
            ctx.beginPath();
            ctx.moveTo(xAt(0), b.y + b.h);
            for(let i=0; i<n; i++) ctx.lineTo(xAt(i), yAt(c.best[i]));
            ctx.lineTo(xAt(n-1), b.y + b.h);
            ctx.closePath();
            ctx.fillStyle = 'rgba(52, 211, 153, 0.1)';
            ctx.fill();

            const line = (data, color) => {
                ctx.beginPath();
                for(let i=0; i<n; i++) { const x = xAt(i), y = yAt(data[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
                ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
            };
            line(c.avg, '#60a5fa');
            line(c.best, '#34d399');

            // Best keeps its 2px points, as it always has
            ctx.fillStyle = '#34d399';
            for(let i=0; i<n; i++) { ctx.beginPath(); ctx.arc(xAt(i), yAt(c.best[i]), 2, 0, Math.PI*2); ctx.fill(); }
        }

        // legend, centred along the top
        const items = [['Best Fitness', '#34d399'], ['Avg Fitness', '#60a5fa']];
        ctx.textAlign = 'left';
        let wTotal = 0;
        for(const [label] of items) wTotal += 10 + 4 + ctx.measureText(label).width + 14;
        let lx = (b.W - wTotal) / 2, ly = 10;
        for(const [label, color] of items) {
            ctx.fillStyle = color; ctx.fillRect(lx, ly - 5, 10, 10);
            lx += 14;
            ctx.fillStyle = '#cbd5e1'; ctx.fillText(label, lx, ly);
            lx += ctx.measureText(label).width + 14;
        }

        // hover readout
        const hi = c.hover;
        if(hi >= 0 && hi < n) {
            const x = xAt(hi);
            ctx.strokeStyle = 'rgba(203,213,225,0.35)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(Math.round(x)+0.5, b.y); ctx.lineTo(Math.round(x)+0.5, b.y + b.h); ctx.stroke();

            const lapTime = c.times[hi];
            const lines = [
                `Gen ${c.gens[hi]}`,
                `Best: ${Math.round(c.best[hi])}${lapTime ? ` (Lap: ${lapTime}s)` : ''}`,
                `Avg: ${Math.round(c.avg[hi])}`
            ];
            let tw = 0;
            for(const l of lines) tw = Math.max(tw, ctx.measureText(l).width);
            const pad = 6, bw = tw + pad*2, bh = lines.length*13 + pad*2 - 3;
            let bx = x + 8; if(bx + bw > b.W - 2) bx = x - 8 - bw;
            let by = b.y + 4; if(by + bh > b.H) by = b.H - bh;
            ctx.fillStyle = 'rgba(15,23,42,0.92)';
            ctx.fillRect(bx, by, bw, bh);
            ctx.strokeStyle = '#334155'; ctx.strokeRect(Math.round(bx)+0.5, Math.round(by)+0.5, bw, bh);
            ctx.fillStyle = '#e2e8f0';
            lines.forEach((l, i) => ctx.fillText(l, bx + pad, by + pad + 6 + i*13));

            for(const [data, color] of [[c.best, '#34d399'], [c.avg, '#60a5fa']]) {
                ctx.fillStyle = color;
                ctx.beginPath(); ctx.arc(x, yAt(data[hi]), 3, 0, Math.PI*2); ctx.fill();
            }
        }
    }
};

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
                // If "all" is selected, hide the delete button and show the
                // whole-track resize button instead.
                if(document.getElementById('btn-del-point')) document.getElementById('btn-del-point').classList.add('hidden');
                if(document.getElementById('btn-fit-map')) document.getElementById('btn-fit-map').classList.remove('hidden');
            } else {
                // If a single point is selected, grab its specific data
                const p = this.track.path[this.selectedIndex];
                type = p.type || 'rounded';
                radius = p.radius !== undefined ? p.radius : 60;
                if(document.getElementById('btn-del-point')) document.getElementById('btn-del-point').classList.remove('hidden');
                if(document.getElementById('btn-fit-map')) document.getElementById('btn-fit-map').classList.add('hidden');
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
            if(document.getElementById('btn-fit-map')) document.getElementById('btn-fit-map').classList.add('hidden');
        }
    },

    // "Select All" + this: scales and re-centres the whole path to fill the
    // canvas, for a loop that was drawn too small (or too big) to use the map
    // well. Radii and zones scale with it, and the start position/angle are
    // re-derived from the new path rather than carried over, the same way
    // finishDrawing() already does after a freehand stroke — the old ones
    // belonged to a shape that no longer exists at this size or position.
    fitToMap: function() {
        const path = this.track.path;
        if (!path || path.length < 3) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const p of path) {
            if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
        }
        const spanX = Math.max(1, maxX - minX), spanY = Math.max(1, maxY - minY);
        // Leave room for the road's own width plus barriers either side, not
        // just the bare centreline, so a fit track doesn't fit its LINE to
        // the canvas and then run its asphalt off the edge.
        const margin = Math.max(50, this.track.trackWidth * 1.5);
        const scale = Math.min((CANVAS_WIDTH - margin * 2) / spanX, (CANVAS_HEIGHT - margin * 2) / spanY);
        if (!isFinite(scale) || scale <= 0) return;
        const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
        const tx = CANVAS_WIDTH / 2, ty = CANVAS_HEIGHT / 2;
        for (const p of path) {
            p.x = Math.round(tx + (p.x - cx) * scale);
            p.y = Math.round(ty + (p.y - cy) * scale);
            if (p.radius !== undefined) p.radius = Math.max(10, Math.round(p.radius * scale));
        }
        (this.track.zones || []).forEach(z => {
            z.x = tx + (z.x - cx) * scale;
            z.y = ty + (z.y - cy) * scale;
            z.radius = Math.max(20, z.radius * scale);
        });
        const derived = generateTrackFromPath(this.track.id, this.track.name, path, this.track.trackWidth, null, null, [], this.autoOpts());
        this.track.startPos = derived.startPos;
        this.track.startAngle = derived.startAngle;
        const angleEl = document.getElementById('edit-angle');
        if (angleEl) angleEl.value = Math.round((derived.startAngle || 0) * (180 / Math.PI));
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
        const p = app._toBackingPx(e);
        return app.screenToWorld(p.x, p.y);
    },

    onDown: function(e) {
        // Right-button is pan (app-level, see startPan) in the editor too —
        // never a new path point.
        if (e.button === 2) return;
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
        // Same view transform the normal race view uses — zoom/pan is one
        // shared state, so switching between racing and editing never resets
        // what you were looking at.
        const v = app._viewMatrix();
        ctx.setTransform(v.z, 0, 0, v.z, v.e, v.f);

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

        // Same finish line the race actually uses — see finishLineOf.
        const fl = finishLineOf(p);
        if(fl) drawFinishLine(ctx, fl.x1, fl.y1, fl.x2, fl.y2);

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
