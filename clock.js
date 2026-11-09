// clock.js — the simulation's fixed timestep and the canvas's repaint cap.
//
// The simulation steps at a fixed 60 Hz of wall-clock time, whatever the
// display refreshes at and whatever the repaint cap is set to. It used to step
// once per animation frame, so world speed WAS the frame rate: a 120 Hz iPad
// raced twice as fast as a 60 Hz phone, Safari's low-power 30 fps halved it,
// and a browser catching up a burst of frames after a stall visibly sped up.
// Physics never sees a wall-clock dt — only the number of whole ticks due.
//
// No DOM here, so tools/looptest.mjs can drive it with a fake clock.
const SimClock = {
    SIM_DT: 1000 / 60,
    // A frame gap longer than this (tab switch, GC pause, debugger) is not
    // caught up: the race pauses instead of fast-forwarding.
    MAX_FRAME_MS: 100,

    _last: null,
    _acc: 0,
    _lastRender: -Infinity,

    // Bank the time since the previous call. `accumulate` false (paused,
    // editing, hyper mode) drops the bank, so resuming never bursts.
    advance(now, accumulate) {
        const dt = this._last === null ? 0 : Math.min(Math.max(now - this._last, 0), this.MAX_FRAME_MS);
        this._last = now;
        this._acc = accumulate ? Math.min(this._acc + dt, this.MAX_FRAME_MS) : 0;
    },

    // Whole ticks now due, consumed. The epsilon keeps float drift in the
    // accumulator from turning an exact tick boundary into a missed tick.
    take() {
        const n = Math.floor((this._acc + 1e-6) / this.SIM_DT);
        this._acc = Math.max(0, this._acc - n * this.SIM_DT);
        return n;
    },

    // Repaint cap, drift-corrected: the next deadline keeps the phase of the
    // last one instead of restarting from whenever this frame happened to
    // land, so 30 on a 60 Hz screen is every other frame, not every third.
    renderDue(now, hz) {
        const interval = 1000 / hz, since = now - this._lastRender;
        if (since < interval - 0.5) return false;
        this._lastRender = since >= interval && since < 2 * interval ? now - (since % interval) : now;
        return true;
    },
    resetRender() { this._lastRender = -Infinity; }
};

if (typeof module !== 'undefined') module.exports = SimClock;
