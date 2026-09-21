# TrackML Assembly 
(Replacement for [*TrackML JS*](https://github.com/AZP3001/main))

*AI RaceTrack Evolution, with a WebAssembly backend.*
[*To Simulation*](https://azp3001.github.io/trackml-assembly/)

A 2D race-track simulation where AIs learn to navigate a racetrack through evolution algorithms.

This is the same app as [AZP3001/main](https://github.com/AZP3001/main) — same UI, same tracks, same
editor, same saved-AI format. **The only difference is the backend: the simulation, the neural
networks, the evolution and the track geometry all run in WebAssembly instead of JavaScript.**
Tracks and brains move between the two versions untouched.

---

## Overview
The simulation puts a population of cars onto a track. Each car is controlled by a neural network which processes "lidar" data to produce continuous control outputs. Only the best performers pass their "genes" (weights) to the next generation.

### The Learning Process
Each car's network produces **two analog outputs**, not four buttons — which is what lets it drive smoothly rather than in jerks:

| output | range | what it does |
| --- | --- | --- |
| Steering | -1 … +1 | full left through straight to full right |
| Throttle | -1 … +1 | positive is gas, negative is brake, and how far from zero is how hard |

It reads eleven inputs: seven distance sensors fanned out ahead of it, its own speed, the bearing to the next checkpoint, and **its own two outputs from the previous frame** — that last pair is what gives an otherwise feedforward network a short memory, so it can hold a line through a corner instead of re-deciding its steering angle from scratch sixty times a second.

---

## The WebAssembly backend

Everything that costs CPU time lives in [`wasm/sim.c`](wasm/sim.c) — a couple of thousand lines of C compiled
to a single freestanding `wasm32` module. The physics step, the wall collisions, the seven sensor
raycasts per car, the neural network, the evolution, and the whole track generator are all in there.
`script.js` keeps the UI, the editor and the drawing; it no longer does any simulation.

### How it's built

Plain **clang**, not Emscripten:

```sh
./wasm/build.sh          # needs clang >= 15 and lld
```

That produces two modules and no JavaScript glue at all:

| file | size | notes |
| --- | --- | --- |
| `wasm/sim.wasm` | ~44 KB | scalar |
| `wasm/sim-simd.wasm` | ~49 KB | adds 128-bit SIMD (`-msimd128`) |

`engine.js` probes the browser for SIMD support and loads whichever it can run. There is no libc in
the build, so `sin`, `cos`, `atan2`, `tanh`, `acos` and `exp` are implemented in `sim.c` too. They
are checked against the `Math.*` they replace over millions of samples by `tools/mathtest.mjs`; run
`npm test` for that and the rest of the suite.

Both files are committed, because GitHub Pages serves this repository as-is and they *are* the
backend. CI rebuilds them from source on every push and fails if the bytes differ, so they can't
drift from `sim.c`.

### Why not shared-memory threads

Threads inside one wasm instance need `SharedArrayBuffer`, which needs `COOP`/`COEP` response
headers, which GitHub Pages cannot send. So instead each worker gets its **own** instance with its
own memory and a contiguous slice of the population. Cars never interact, so the partition is exact,
it scales across cores the same way, and it needs no special headers.

---

## Configuration & Settings
Fine-tune the simulation and the learning process using the built-in settings.

### AI & Evolutionary Parameters
* **Pop Size** (default 500)**:** The number of agents generated per generation.
* **Elite Clones** (default 30)**:** Number of top-performing agents preserved exactly for the next generation (prevents regression).
* **Mutation Rate** (default 30%)**:** How much of each brain is randomly nudged per generation. The size of each nudge is a Gaussian that **shrinks as the run goes on**, so a high rate explores hard early and still settles down later instead of permanently kicking a working solution apart.
* **Hidden Layers:** Adjust the complexity of the AI's "brain" by changing the number of internal neurons.
* **Initial TTL (Time-To-Live):** A countdown for each agent, **reset in full every time it reaches a checkpoint**. It used to top the clock up by 150 frames and clamp it to 600, which quietly made the slider a lie — set it to 10,000 and the very first gate cut the car back to 600.
* **Target Laps:** Defines the goalpost for a successful generation before moving to the next stage of evolution.

How a generation is bred, for the curious: parents are drawn from the top fifth of the field
**rank-weighted**, so the leader parents far more often than the hundredth car rather than equally.
Crossover then works **one hidden unit at a time** — all of a unit's incoming weights, its bias and
its outgoing weights come from the same parent — because picking each weight independently splits up
groups of weights that only mean anything together, and two parents that both drive well routinely
produced a child that drove into a wall. New populations start with fan-in scaled weights instead of
a flat `[-1,1]`, which stopped the first few dozen generations being spent climbing back out of tanh
saturation.

### Physics Engine
* **Max Speed:** Maximum Speed of the Cars.
* **Acceleration:** Acceleration to the Max Speed.
* **Turn Speed:** How quickly the cars can turn — but only up to what grip allows. Below about 3 px/frame of speed a car turns at full Turn Speed; past that, available turn rate falls off roughly as `3 / speed`, the same tradeoff a real driver feels: carrying too much speed into a corner costs you the turn, so the fast line is to slow down first, not to out-steer the corner. A car with essentially no speed gets no turn authority at all — turning the wheel does nothing until it's rolling, just like a parked car.
* **Brake Strength** (default 0.2)**:** How hard the brake pedal bites when the AI's throttle output goes negative. Braking now scales with how hard it's pressed — a throttle of -0.05 barely touches the speedometer, -1.0 hauls the car down hard — rather than the old behaviour, where ANY negative throttle snapped speed down by the same flat 5% regardless of how lightly it was pressed, so brakes looked "instant" no matter what the AI actually asked for.
* **No momentum, no race.** A car that isn't moving is eliminated on the spot — whether it never commanded throttle at all or braked to a standstill mid-track. Only speed counts; spinning the heading on the spot isn't momentum (and a stopped car can't steer anyway, so it has no way back out of that state). There's a short grace window at the start line so a car gets a chance to launch. This is worth real time: with 500 random brains, generation 1 used to spend most of itself simulating cars parked on the line until their TTL expired — killing them immediately cuts the generation's live car-frames by about 73% and runs it roughly 2.8x faster.
* Lateral grip (the friction that keeps a car's velocity tracking its heading instead of drifting sideways) used to be a slider here too. It's fixed internally now: its usable range only ever canceled 80-99% of sideways slip every frame, so the two ends of that slider left a car in the same place after a couple of frames. Removed rather than kept as a knob that did effectively nothing.

### Simulation Control
* **Simulation Speed:** Adjust the simulation speed.
* **Hyper Mode:** Simulates as fast as your PC allows, and draws nothing at all while it does.

The canvas repaints at most 30 times a second, and not at all when nothing has changed — the
simulation still steps on every animation frame, only the painting is throttled. Cars are blitted
from a cached sprite per livery instead of half a dozen canvas state changes each, which at 500 cars
was most of the main thread's paint cost.

The fitness chart keeps a rolling window of the last 300 generations rather than the whole session's
history. Unbounded, it re-fed and redrew the entire history every single generation — cheap for the
first few hundred, then a stall that grew as the run went on, which is why a long session used to
feel like it was gradually slowing down even though the simulation never changed pace.

### Saving a run

**Save AI** writes out a single brain. **Save Session** writes out the whole thing — every brain in
the population, the generation counter, the best times, the lap history and the graph — and **Load
Session** picks it up exactly where it left off. Loading a single brain can only reseed the field
from mutated copies of that one network, which throws away all the diversity the run had built.

Nothing is written to browser storage: settings and custom tracks do not survive a reload, so every
visit starts clean. **Wipe All Data & Reload**, at the bottom of the settings panel, is the explicit
escape hatch — it clears storage and the browser's cached copy of the app and reloads.

---

## Spectator Mode
Click any car on the track to pin the telemetry panel and sensor overlay to it — it stays highlighted until it crashes or you click empty space / hit "Release". With nothing selected, the panel auto-follows whichever car currently has the best fitness.

## Track Editor
### How a track is built
A track is a closed centre line plus a width. The road surface is everything within that width of the centre line — the same shape you'd get by stroking the line with a fat round pen — and the barriers are traced along the outside edge of exactly that shape, as one continuous wall per side. So the road is the same width the whole way round, the walls always sit on the edge of the asphalt, and where a track crosses or runs into itself the two bits of road simply join up instead of leaving stray walls in the middle of the road.

Each point on the path is a turn, and turns are real circular arcs:
* **Rounded:** an arc of the radius you set on the slider.
* **Corner:** the tightest arc the track width still allows — sharp, but never so sharp that nothing fits through it.

A turn is never allowed to be tighter than the road is wide, and neighbouring turns share out the straight between them, so putting two points close together softens both instead of kinking the road.

The generator runs in wasm, which matters here more than it looks: the editor rebuilds the entire
track on every frame you drag a point, and the walls come back as flat `Float32Array`s the canvas
can stroke directly rather than as thousands of freshly allocated point objects.

### Auto width

A closed loop can easily come back on itself. Where two passes of the track run close together, the
road at its full width swallows the gap between them and the barrier that should separate them
disappears — the two bits of road merge into one slab. **Auto Width** fixes that by narrowing the
road there instead.

It finds, for every point on the centre line, the nearest part of the track that isn't simply
further along the road, and caps the width so a barrier still fits between the two. The awkward part
is telling "the track curving round a corner" apart from "a different part of the track"; that's
done by arc length, with a window sized to the tightest turn the generator will ever build, so an
ordinary hairpin is left alone and a genuine near miss is not.

The slider next to the checkbox decides how the narrowing is applied:

* **Local** (far left) — only the tight spots narrow, the rest of the track keeps its full width.
* **Global** (far right) — the whole track takes the narrowest width it needs anywhere, so it stays
  one even width the whole way round.
* Anywhere between blends the two.

Narrowing tapers over a couple of track widths rather than stepping, and never goes below the width
a car can physically get through. It's on by default for drawn and image-imported tracks, which are
the ones where you can't judge the clearances by eye, and off for tracks you place by hand.

Besides placing points by hand, the editor has three ways to build a track:
* **Draw:** switch to the Draw tab and drag a loop directly on the canvas — it's simplified into an editable path automatically.
* **Import from Image:** upload a PNG/JPG (a hand-drawn loop or a photo of a track layout) and it's analyzed (thresholded, skeletonized, traced) into a starting track for you to refine.
* **Duplicate:** clone the currently-selected track as a starting point for a variant.

Tracks you create last for the session. **Nothing is cached or persisted** — see below.

### Keeping a track
Hit **Save Track** and the editor hands you a `generateTrackFromPath(...)` line for it. Paste that
into `tracks.js` and the track is built into the app for everyone on the next deploy.

There is no Publish button. It used to open a prefilled GitHub issue that a workflow turned into a
pull request; posting a track directly instead would need a server to post it *to*, and this is a
static site on GitHub Pages with no backend, so that was removed rather than left as the GitHub
detour. The code export above is the replacement.

## Keyboard Shortcuts
* **Space:** Play / Pause
* **H:** Toggle Hyper Mode
* **R:** Reset
* **Esc:** Cancel track editing, or release a manual spectator selection
