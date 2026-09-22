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
| `wasm/sim.wasm` | ~49 KB | scalar |
| `wasm/sim-simd.wasm` | ~53 KB | adds 128-bit SIMD (`-msimd128`) |

`engine.js` probes the browser for SIMD support and loads whichever it can run. There is no libc in
the build, so `sin`, `cos`, `atan2`, `tanh`, `acos` and `exp` are implemented in `sim.c` too. They
are checked against the `Math.*` they replace over millions of samples by `tools/mathtest.mjs`; run
`npm test` for that and the rest of the suite.

Each module reserves **6 MB** of linear memory, not the round 16 MB it used to. That number is
multiplied by one instance per core plus the master, so on an eight-core machine it was a hundred
and forty megabytes of address space — nearly all of it two brain buffers sized for a hidden layer
three times larger than the slider can ask for. Tracks that need more arena than the 6 MB covers
grow the memory themselves; nothing on either side of the boundary holds a view across a call that
can grow it.

Both files are committed, because GitHub Pages serves this repository as-is and they *are* the
backend. CI rebuilds them from source on every push and fails if the bytes differ, so they can't
drift from `sim.c`.

### What a step costs

Everything below runs per living car per frame, so it is the only code in the project where the
shape of the data matters more than the algorithm. Two changes moved more than the rest put
together:

* **The seven sensor rays no longer call `sin`/`cos`.** Their angles are fixed offsets from the
  car's own heading, which the step already has a sine and cosine for, so the whole fan comes out of
  the angle-addition identities in four multiplies — and the ±90° pair for free. That was seven
  argument reductions and fourteen polynomial evaluations per car per frame, which made
  trigonometry, not raycasting, the largest single line item in the step.
* **Walls behind the car are dropped before the rays are cast, not during.** The fan spans ±90°
  about the heading, so every point of every ray has a forward projection of at least zero and a
  wall lying entirely behind the car cannot be hit by any of them. One dot product in the broad
  phase — which runs once — halves the work of the seven raycasts that follow it. The readings are
  bit-for-bit the ones the full set produced; `tools/e2e.mjs` and the unit suite both ran against
  the culled and unculled builds to confirm it.

The rest is the same story at a smaller scale: the intersection tests range-check against the
denominator instead of dividing by it (a miss, which is almost every test, now costs no division at
all), the network's weights are walked in the order they are stored rather than across it and four
hidden units at a time, the activation is an f32 series instead of an f64 one, the road test starts
from the centreline segment that claimed the car last frame, the bearing to the next gate is
unwrapped with one `floor` instead of a `while` loop that got longer the further a car had driven,
and `run` keeps a compacted list of the cars still driving rather than reading the whole population
to find them. Together, on the same work: **2.0x** on the SIMD module, **1.8x** on the scalar one.

### Why not shared-memory threads

Threads inside one wasm instance need `SharedArrayBuffer`, which needs `COOP`/`COEP` response
headers, which GitHub Pages cannot send. So instead each worker gets its **own** instance with its
own memory and a contiguous slice of the population. Cars never interact, so the partition is exact,
it scales across cores the same way, and it needs no special headers.

The slices are **not equal**. Every round ends at a barrier — nothing is drawn or bred until the
last worker reports — so on cores of different speeds an equal split means everyone waits for the
slowest, and a phone's little cores are a third the speed of its big ones. Each worker times its own
`run` and reports the exact number of car-steps it did, which gives a real throughput figure rather
than one distorted by how many of its cars happened to still be alive; the master keeps a rolling
average of it and resizes the slices at each generation boundary, where every worker is idle and
about to be handed a fresh slice anyway. Before any timings exist this is exactly the old even
split.

---

## Configuration & Settings
Fine-tune the simulation and the learning process using the built-in settings.

Four of them start in a different place on a phone or a tablet — see
[On a phone or tablet](#on-a-phone-or-tablet) below. They are the same sliders with the same range
either way; only where the handle starts differs.

### AI & Evolutionary Parameters
* **Pop Size** (default 500, up to 2000; **150** on mobile)**:** The number of agents generated per generation.
* **Elite Clones** (default 30; **15** on mobile)**:** Number of top-performing agents preserved exactly for the next generation (prevents regression). Fewer than a handful and a generation can occasionally lose ground it had already made — around 30 is enough that the population doesn't "forget" a solution it found. It tracks the population rather than being an absolute: a sixth of a 150-car field is the same share of it that 30 is of 500.
* **Focus %** (default 20%)**:** Fraction of the population spent each generation as mutated clones of the current best, with their reward specifically boosted through whichever stretch of track needs it most (roughly ±1 second either side). *Which* stretch depends on how the run is going: until the population has finished a lap in several separate generations, that's wherever cars are actually dying — one corner killing every run over and over is a far bigger problem than a corner it already gets through a bit slowly, and the old version, which only ever looked at gates the current best had actually reached, couldn't see a corner it never got past at all. Once finishing is no longer the bottleneck, it goes back to targeting the slowest stretch, same as before — going fast where it's *already* near top speed has little room left to improve.
* **Hidden Layers** (default 5; **4** on mobile)**:** Adjust the complexity of the AI's "brain" by changing the number of internal neurons. The one fewer on mobile is not a rounding-down — the hidden layer is evaluated four units at a time on the SIMD build, so five units cost two passes and four cost one.
* **Initial TTL (Time-To-Live):** A countdown for each agent, **reset in full every time it reaches a checkpoint**. It used to top the clock up by 150 frames and clamp it to 600, which quietly made the slider a lie — set it to 10,000 and the very first gate cut the car back to 600.
* **Target Laps** (default 3; **2** on mobile)**:** Defines the goalpost for a successful generation before moving to the next stage of evolution.

How a generation is bred, for the curious: parents are drawn from the top fifth of the field
**rank-weighted**, so the leader parents far more often than the hundredth car rather than equally.
Crossover then works **one hidden unit at a time** — all of a unit's incoming weights, its bias and
its outgoing weights come from the same parent — because picking each weight independently splits up
groups of weights that only mean anything together, and two parents that both drive well routinely
produced a child that drove into a wall. New populations start with fan-in scaled weights instead of
a flat `[-1,1]`, which stopped the first few dozen generations being spent climbing back out of tanh
saturation.

There used to be a separate **Mutation Rate** slider controlling how much of each brain got nudged.
It's gone: every weight is mutated on every generation now, and the *size* of the nudge is a Gaussian
that **shrinks as the run goes on** (annealing from wide exploration early to fine polishing later).
Once that curve already controls how big a mutation is, gating *whether* a weight gets one at all on a
coin flip was a second knob doing overlapping work — dropping it is one less setting to tune, not a
missing feature.

That shrinking curve doesn't start counting from generation 1 any more. It used to, which meant a
genuinely hard track — one the population hadn't finished even once after hundreds of generations —
had its mutations ground down toward fine-polishing size anyway, right when it most needed to keep
trying new things to break through at all. Now the clock doesn't start until the population completes
a lap for the first time, and then waits a further ~15 generations past that before it starts
counting down, so one lucky early lap doesn't immediately start shrinking the exploration that found
it. A track that clicks quickly anneals on close to the same schedule as before; a hard one keeps
exploring for as long as it actually takes.

Every setting's slider is centred on its own default — nudge it either direction from there rather
than starting near one end of the range.

### Physics Engine
* **Max Speed:** Maximum Speed of the Cars.
* **Acceleration** (default 0.05)**:** Acceleration to the Max Speed.
* **Turn Speed** (default 0.02)**:** How quickly the cars can turn — but only up to what grip allows. Below about 3 px/frame of speed a car turns at full Turn Speed; past that, available turn rate falls off roughly as `3 / speed`, the same tradeoff a real driver feels: carrying too much speed into a corner costs you the turn, so the fast line is to slow down first, not to out-steer the corner. A car with essentially no speed gets no turn authority at all — turning the wheel does nothing until it's rolling, just like a parked car.
* **Brake Strength** (default 0.05)**:** How hard the brake pedal bites when the AI's throttle output goes negative. Braking now scales with how hard it's pressed — a throttle of -0.05 barely touches the speedometer, -1.0 hauls the car down hard — rather than the old behaviour, where ANY negative throttle snapped speed down by the same flat 5% regardless of how lightly it was pressed, so brakes looked "instant" no matter what the AI actually asked for.
* **No momentum, no race.** A car that isn't moving is eliminated on the spot — whether it never commanded throttle at all or braked to a standstill mid-track. Only speed counts; spinning the heading on the spot isn't momentum (and a stopped car can't steer anyway, so it has no way back out of that state). There's a short grace window at the start line so a car gets a chance to launch. This is worth real time: with 500 random brains, generation 1 used to spend most of itself simulating cars parked on the line until their TTL expired — killing them immediately cuts the generation's live car-frames by about 73% and runs it roughly 2.8x faster.
* **Braking is never punished on its own.** Being alive and making recent progress pays a flat reward every frame — it used to scale with how fast the car happened to be going that instant, which sounds like rewarding speed but actually rewards never lifting off the throttle for any reason, including the correct one: braking into a corner to carry more speed out of it. The AI is free to trade a slower entry for a faster exit whenever that produces the better actual lap time, because it's *actual lap time* — real elapsed frames to the next gate and to the finish — that the checkpoint and lap bonuses reward, not the speedometer reading at any given instant.
* Lateral grip (the friction that keeps a car's velocity tracking its heading instead of drifting sideways) used to be a slider here too. It's fixed internally now: its usable range only ever canceled 80-99% of sideways slip every frame, so the two ends of that slider left a car in the same place after a couple of frames. Removed rather than kept as a knob that did effectively nothing.

### Simulation Control
* **Simulation Speed:** Adjust the simulation speed.
* **Hyper Mode:** Simulates as fast as your PC allows, and draws nothing at all while it does.

The canvas repaints at most 60 times a second by default (a button next to the zoom controls drops it
to 30), and not at all when nothing has changed — the simulation still steps on every animation
frame, only the painting is throttled. Cars are blitted from a cached sprite per livery instead of
half a dozen canvas state changes each, which at 500 cars was most of the main thread's paint cost.

On a phone, how many pixels that paint covers is decided at load rather than fixed. Everything draws
in a 1200x900 world and always will, and on a desktop that is still exactly what gets rasterised —
the browser downsamples it to the window, which is free supersampling on a machine that will not
miss the pixels. A phone shows the canvas in a box around 380px wide, so the same million pixels are
painted to display about a tenth of that, on the thread that also has to keep the simulation fed;
there the raster follows the display instead. The scale is a factor on the single view transform, so
no drawing code knows it exists; `tools/e2e.mjs` holds the pointer maths to it, since a click has to
land on the same world point at any raster size and any zoom.

The two panels under the graph — the fitness chart and the improvement table — are coalesced rather
than redrawn per generation. In hyper mode generations turn over several times a second, and a
canvas repaint plus an `innerHTML` reparse at that rate is work done for nobody: the numbers are all
still there on the next tick.

Sensor readings cross from the workers for every car but are unpacked for one. Seven floats per car
per frame were being copied into the render records to be read for the single car the overlay
follows; that one car now takes them straight out of the buffer they arrived in, before it goes back
to its worker.

The fitness chart shows the **whole run**, not a truncated recent window — every generation is
represented somewhere on it for as long as the session lasts. It stays fast anyway: once it has 300
points, each new generation first tries to merge into the newest one, and once that one is as full as
the rest, the whole array halves its resolution by merging consecutive pairs. So the far past gets
coarser instead of disappearing, exactly like a real monitoring graph — the cost per generation stays
constant no matter how long the run has been going, which is the actual fix for the "gets slower the
longer I leave it running" problem: redrawing the *entire* unbounded history every generation was an
O(n²) stall over a session, and simply throwing away everything past a fixed window (an earlier,
cruder fix) traded "total" away to get the same constant cost.

Next to the graph is a small table of average fitness improvement per generation — for the top 1%,
the top 10%, and the whole population — over the last 1, 10 and 100 generations, so you can see at a
glance whether a run is still climbing or has plateaued at each of those timescales.

### Saving a run

**Save AI** writes out a single brain. **Save Session** writes out the whole thing — every brain in
the population, the generation counter, the best times, the lap history and the graph — and **Load
Session** picks it up exactly where it left off. Loading a single brain can only reseed the field
from mutated copies of that one network, which throws away all the diversity the run had built.

Nothing is written to browser storage: settings and custom tracks do not survive a reload, so every
visit starts clean. **Wipe All Data & Reload**, at the bottom of the settings panel, is the explicit
escape hatch — it clears storage and the browser's cached copy of the app and reloads.

---

## On a phone or tablet

The page checks once, at load, whether it is running on a phone or a tablet, and if it is, four
things start somewhere different. Nothing is taken away — every one of them is still the same
slider with the same range, and every one can be pushed back up.

| | desktop | phone / tablet |
| --- | --- | --- |
| Pop Size | 500 | **150** |
| Elite Clones | 30 | **15** |
| Target Laps | 3 | **2** |
| Hidden Layers | 5 | **4** |

Underneath those, three things the sliders don't show:

* **The worker pool is capped**, at roughly the fast half of the reported cores and never more than
  four. `navigator.hardwareConcurrency` counts logical cores, and on a phone that count is a lie
  about what they are worth: eight cores usually means four fast and four slow, differing by a
  factor of three. Every round ends at a barrier, so a slice handed to a slow core sets the pace for
  the whole pool — past a point, *adding* cores makes it slower. Each worker also carries its own
  wasm instance, which is memory a tab does not have to spare and heat a phone cannot shed. The core
  readout in the sidebar shows both numbers ("4/8 Cores") when they differ.
* **The canvas rasterises fewer pixels**, sized to the display rather than to the 1200x900 world —
  see the rendering note under Simulation Control. Desktop is untouched.

The repaint ceiling itself is not part of this table — it defaults to 60 frames a second on every
device now, with a 30 next to it (bottom-left of the canvas) for anyone who would rather spend that
half of the frame budget on training. See [Zoom & Pan](#zoom--pan) for pinch-to-zoom and one-finger
panning on a touchscreen.

Together, one generation on a phone-shaped workload: **143 ms before, 18 ms after** — about half of
that from the settings and half from the module itself.

How the check works, because getting this wrong in the other direction would be worse than not
doing it: Chromium's `userAgentData.mobile` is the only honest answer and only exists for phones;
iPadOS ships a desktop user-agent string and gives itself away by being a "Macintosh" with a
touchscreen; anything else is asked whether its primary input is a finger and whether it has no
hover, which a tablet on an unfamiliar browser still answers correctly. A desktop that somehow
matched would get a smaller starting population and four sliders to put back, which is the mild
failure of the two.

---

## Spectator Mode
Click any car on the track to pin the telemetry panel and sensor overlay to it — it stays highlighted until it crashes or you click empty space / hit "Release". With nothing selected, the panel auto-follows whichever car currently has the best fitness.

## Zoom & Pan
Scroll to zoom in and out of the map (centred on the cursor), or use the +/−/1:1 buttons over the
bottom-right corner of the canvas. Right-click and drag to pan around while zoomed in. On a
touchscreen, pinch with two fingers to zoom — the pinch's midpoint stays under your fingers as you
zoom and pan together in the same gesture, the way a map app's does — and drag with one finger to
pan; a tap that doesn't turn into a drag still selects the car underneath it, same as a mouse click.
(Panning by touch is the race view only — while editing a track, single-finger touch drags path
points instead, so pan there with the +/-/1:1 buttons, Fit to Map, or a mouse if one's handy.) This
works the same way in the race view and the editor — it's one shared view, so switching between them
never resets what you were looking at. Zooming in doesn't just make things bigger: car-selection and
point-editing precision scale with it too, since a fixed distance in track units covers fewer screen
pixels the further in you are.

The canvas repaint rate — not the simulation, which always steps at full speed regardless — is 60
frames a second by default, with a 30 next to it (bottom-left of the canvas) for whoever would rather
spend that half of the frame budget on training instead of painting.

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

### Finish line
The finish line drawn on the map is the actual gate that completes a lap — the same one `updateCar`
in `sim.c` checks — not just the first checkpoint the generator happened to lay down. On a track whose
start line has been dragged away from where the centreline generation began, those used to be two
different gates, so the drawn line and the one cars were actually scored on could be nowhere near each
other. It's rendered as a real checkered banner spanning the road's actual width at that point (the
same auto-width-aware coordinates the lap check itself uses), in both the normal view and the editor.

### Resizing a track to fit the map
Click **Select All** in the Path tab, then **Fit to Map**, and the whole track — every point, its
corner radii, and any zones — is rescaled and re-centred to fill the canvas with a sensible margin.
Useful for a loop that was drawn (or imported) too small, or dragged off to one side, to make good use
of the space.

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
