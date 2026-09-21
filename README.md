# TrackML Assembly

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
The AIs have four primary control axes. To allow for more precise controls the axes are **analog (0-100%)**, allowing for more precise & smooth driving:
* **Gas**
* **Brake**
* **Steer Left**
* **Steer Right**

---

## The WebAssembly backend

Everything that costs CPU time lives in [`wasm/sim.c`](wasm/sim.c) — about 1200 lines of C compiled
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
| `wasm/sim.wasm` | ~34 KB | scalar |
| `wasm/sim-simd.wasm` | ~38 KB | adds 128-bit SIMD (`-msimd128`) |

`engine.js` probes the browser for SIMD support and loads whichever it can run. There is no libc in
the build, so `sin`, `cos`, `atan2`, `tanh`, `acos` and `exp` are implemented in `sim.c` too — see
[Testing](#testing) for how they're kept honest.

Both files are committed, because GitHub Pages serves this repository as-is and they *are* the
backend. CI rebuilds them from source on every push and fails if the bytes differ, so they can't
drift from `sim.c`.

### How it's faster

The port isn't just "the same loops, in wasm". Four things do the work:

1. **Struct-of-arrays in linear memory.** A wall is 20 contiguous bytes, not a
   `{p1:{x,y},p2:{x,y}}` object graph. A generation of brains is one flat `Float32Array`. Nothing
   in the inner loop touches anything a garbage collector has to trace.
2. **A broad-phase cull.** The JS version raycast against every wall bucketed near the car's
   checkpoint. Each car now first drops the walls that provably cannot be reached — a midpoint
   distance against a precomputed radius, no division — which shortens all twelve intersection
   loops. Behaviourally invisible; it only removes walls that could never be hit.
3. **Vectorised raycasting.** The surviving walls are gathered contiguously, so the SIMD build
   tests four of them per instruction with no branches in the loop body.
4. **Far fewer crossings.** Brains only ever flow main-thread → workers, because running a
   generation doesn't modify one. In hyper mode nothing is drawn, so workers ship three floats per
   car instead of eighteen and run thousands of steps per round trip instead of 500.

Measured with `npm run bench` (same track, same brains, same step count, single-threaded, Node 22
on x86-64 — your numbers will differ):

```
  JavaScript (reference)         929.5 ms
  WebAssembly (sim.wasm)         369.9 ms      2.5x
  WebAssembly (sim-simd.wasm)    261.3 ms      3.6x
```

So: roughly **2.5–3.6x** per core on the raw simulation, on top of which hyper mode gains again from
not marshalling render state every frame. It is not 10,000x — nothing that replaces one optimised
JIT-compiled loop with another is — but it is a large, real, measured speedup, and the
`Cores · WASM+SIMD` badge in the sidebar tells you which path you got.

### Why not shared-memory threads

Threads inside one wasm instance need `SharedArrayBuffer`, which needs `COOP`/`COEP` response
headers, which GitHub Pages cannot send. So instead each worker gets its **own** instance with its
own memory and a contiguous slice of the population. Cars never interact, so the partition is exact,
it scales across cores the same way, and it needs no special headers.

---

## Configuration & Settings
Fine-tune the simulation and the learning process using the built-in settings.

### AI & Evolutionary Parameters
* **Pop Size:** The number of agents generated per generation.
* **Elite Clones:** Number of top-performing agents preserved exactly for the next generation (prevents regression).
* **Mutation Rate:** The probability and intensity of random changes to the neural weights.
* **Hidden Layers:** Adjust the complexity of the AI's "brain" by changing the number of internal neurons.
* **Initial TTL (Time-To-Live):** A countdown timer for each agent. Agents must reach checkpoints to reset this timer, ensuring they don't just sit still.
* **Target Laps:** Defines the goalpost for a successful generation before moving to the next stage of evolution.

### Physics Engine
* **Max Speed:** Maximum Speed of the Cars.
* **Acceleration:** Acceleration to the Max Speed.
* **Turn Speed:** Controls how quickly the cars are able to turn.
* **Grip:** Doesnt do much, introduced as a fix for turning Physics.

### Simulation Control
* **Simulation Speed:** Adjust the simulation speed.
* **Hyper Mode:** Simulates as fast as your PC allows. (Doesn't render for even faster Processing)

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

---

## Staying on the track, and racing

Two things a racing sim has to get right, both of which were wrong in the
JavaScript edition and are fixed here.

### Cars can no longer drive through the barriers

The wall lookup is bucketed by the checkpoint a car is heading for, so a car
whose checkpoint index went stale was handed the walls for a part of the track
it had already left — and drove straight through the ones in front of it.

The index went stale because a gate only counted as reached within a flat 50px
of its middle, while a gate spans the *full width of the road* — up to 150px
either side. A car taking the outside line was therefore never close enough to
register one, and sailed through gate after gate. That is exactly the reported
"you can still drive through the walls on the outside". The radius now scales
with the track, so a gate registers across its whole span.

Underneath that there is now a hard backstop: **if a car's centre leaves the
asphalt, it is out immediately.** The barrier sits exactly on the edge of the
road, so leaving the road *is* crossing a barrier — and unlike a
segment-versus-segment test, this cannot be escaped by tunnelling through a
wall in one fast frame, by slipping through a seam left open where the track
crosses itself, or by reaching a stretch whose walls weren't in the bucket.
`tools/racepace.mjs` checks two million live car-frames for a car that is off
the road; before these fixes it found thousands, up to 270px past the barrier.

### Evolution now selects for speed

Reaching a gate paid the same whether it took four frames or four hundred, and
driving slowly is far less likely to end in a wall — so the fittest car was the
most patient one, not the quickest. On narrow, twisty tracks that produced
winners crawling round at 40 to 90 seconds a lap.

Gate and lap rewards are now scaled by how quickly they were reached. It is a
**multiplier on a floor, never a subtraction**, which is the part that matters:
every gate is still worth at least what it used to be, so covering more of the
track always beats covering less, and a lap is never worth less than most of a
lap. (A per-frame time penalty — the obvious fix — inverts that, and makes
crashing on purpose score better than finishing slowly.)

Measured on the same tracks and seeds, with generations run to completion the
way the app runs them:

| track | before | after |
| --- | --- | --- |
| gear, half-width 25 | 42.7s per lap | 14.7s |
| clover, half-width 25, TTL 10000 | 88.4s | 8.8s, and reached 8 generations sooner |
| ordinary tracks | 6–8s | unchanged |

---

## Nothing is cached

Every reload starts from scratch. There is no localStorage, no sessionStorage, no service worker and
no Cache Storage; the page clears all four on startup (including anything left behind by an earlier
build that did persist), sends `no-store` cache headers, and fetches the wasm module with
`cache: 'no-store'` so a reload really does re-fetch it.

Custom tracks, slider settings and the trained population therefore all live in memory only. Use
**Save AI** for a brain you want to keep, and the editor's code export for a track.

---

## Running it locally

The page fetches a `.wasm` file and starts Workers, and browsers allow neither over `file://` — so
opening `index.html` directly will show you an error telling you exactly this. Serve the folder:

```sh
python3 -m http.server 8000       # or: npm run serve
```

then open <http://localhost:8000>.

## Testing

```sh
npm run build        # compile both wasm variants
npm test             # math, parity, auto width, race behaviour
npm run test:e2e     # real browser, needs a server running
npm run bench        # wasm vs the JavaScript edition
```

* **`tools/mathtest.mjs`** — checks the hand-written `sin`/`cos`/`atan2`/`tanh`/`acos`/`exp` in
  `sim.c` against the `Math.*` they replace, over millions of samples. `acos`, `tanh` and `atan2`
  come out bit-identical at f32 precision; `sin`/`cos` are within half an ulp out to ±4000 radians
  (the car heading is never wrapped, so large arguments are real).
* **`tools/paritytest.mjs`** — the important one. It runs `sim.wasm` and the JavaScript edition's
  actual worker (`tools/reference-worker.mjs`, copied verbatim) side by side on the same tracks with
  the same brains, and compares them frame by frame. Trajectories stay within 4e-4 px over the first
  60 frames, and all 160 test cars agree on who crashed and how many laps they finished.
* **`tools/autowidth.mjs`** — the auto-width solver. Both failure modes are tested, since they're
  opposites: not narrowing a track that pinches, and narrowing one that doesn't (which would wreck
  every normal track). Also covers the local/global slider, the smoothness of the taper, and a
  self-crossing figure eight.
* **`tools/racepace.mjs`** — the two behavioural properties above: no car is
  ever off the asphalt, gates register across the full width of the road, a
  moved start line still works, and the fittest car is one of the fastest. It
  runs generations to completion rather than to a frame budget — a crawling car
  needs tens of thousands of frames to finish a lap, so a frame-capped harness
  cannot see the pace problem at all.
* **`tools/e2e.mjs`** — drives the real page in Chromium: module loads, workers come up, tracks
  generate, cars drive, generations advance, hyper mode trains, the editor edits, brains export,
  auto width responds to its controls, and a reload comes back with storage empty.

The two engines are **not** bit-identical and can't be — `sim.c` runs the physics in `f32` where the
JS runs it in `f64`, and a genetic driving sim is chaotic enough that a 1e-7 difference eventually
separates two runs. What the tests assert is that they agree over a short horizon and agree on
outcomes, which is what "works the same" actually means here.

Fitness is the one place they deliberately disagree, for the reason above; agreeing with the
JavaScript edition's scoring would mean the crawling bug had not been fixed. The parity test checks
instead that the new scoring never pays *less* than the old for identical driving, which is the
invariant that keeps progress monotonic.

## Repository layout

```
index.html              markup, unchanged from the JS edition apart from the script tags
script.js               UI, editor, persistence, drawing — no simulation
engine.js               the wasm backend: loader, SIMD detection, worker pool, geometry facade
sim-worker.js           hosts one wasm instance and simulates one slice of the population
tracks.js               the built-in and community tracks
image-import.js         PNG/JPG -> track pipeline
wasm/sim.c              the entire compute backend
wasm/build.sh           clang -> sim.wasm + sim-simd.wasm
wasm/*.wasm             committed build output (CI verifies it matches the source)
tools/                  math, parity, auto-width, end-to-end and benchmark harnesses
external/               tailwind, chart.js, lucide — vendored, no CDN at runtime
```
