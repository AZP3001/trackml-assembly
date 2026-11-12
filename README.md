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
own memory and its own slice of the population. Cars never interact, so the partition is exact,
it scales across cores the same way, and it needs no special headers. Every logical core gets a
worker, on phones and tablets too.

What runs at the same time: every worker steps its own cars in parallel; the main thread sends the
next round *before* it draws the last one, so painting overlaps simulation; and breeding (on the
main thread's own wasm instance) is a fraction of a millisecond once per generation. A browser
can't pin a thread to a particular core, so "which work goes to which core" is decided the way
it can be: by how much work each worker is handed.

The slices are **not equal**. Every round ends at a barrier — nothing is drawn or bred until the
last worker reports — so on cores of different speeds an equal split means everyone waits for the
slowest, and a phone's little cores are a third the speed of its big ones. Each worker times its own
`run` and reports the exact number of car-steps it did, which gives a real throughput figure rather
than one distorted by how many of its cars happened to still be alive; the master keeps a rolling
average of it and resizes the slices at each generation boundary, where every worker is idle and
about to be handed a fresh slice anyway. Before any timings exist the split is even. That is also
why every core is used: a little core gets a little slice and finishes with the rest, so it adds
throughput rather than setting the pace.

The slices are also **dealt**, not cut into blocks. Breeding puts the elites — the cars that drive
the longest — at the front of the field, so worker 0 used to be left driving them while every other
core sat idle at the barrier. Each car now goes to the worker furthest below its speed-weighted
share, which interleaves the field. Measured over a training run with four workers in hyper mode,
core utilisation went from 57-74% to 74-81% on the harder tracks (The Snake, Neon District,
Monaco).

### Changing the pool mid-race

The **CPU Cores** slider and the **CPU / GPU** toggle (Settings → Compute Engine) both change the
pool while a generation is running, and neither sends anyone back to the start line. New workers are
spawned and caught up on the track in the background while the old pool keeps racing; then the one
round in flight is allowed to land, every worker hands back its cars exactly as they are
(`pack_state` in `sim.c` — every per-car field, 32 words a car), and the population is re-sliced and
handed to the new pool (`unpack_state`), which carries on from precisely that state. Only that
hand-over pauses the race — a message round trip, typically a few milliseconds — so a change lands
within a frame or two. `tools/statemigrate.mjs` holds it to byte-identical: cars moved to a new
instance, or split across two, finish exactly where an instance that never moved them does.

**As a CPU benchmark.** Under the car-steps/s readout (Settings → Compute Engine) the run is also
averaged, counting **Hyper mode only** (at 1x the number is just 60 steps a second per car): over the
whole run and over generations 1–100, 100–1000 and 1000+, with a graph of one point per generation
(the dashed line marks generation 100). Early generations are short and mostly crashes, so they
measure overhead as much as raw speed; the later windows are the sustained number, after a phone or
laptop has had time to heat up and throttle. It starts over on Reset, a new track, or changing CPU
cores / GPU; leaving Hyper only pauses it. For comparing devices, use the same track, Pop Size and
core count.

At 1x speed the race is paced by wall-clock time — a fixed 60 steps a second (`clock.js`), on a 30,
60 or 120 Hz screen alike and whatever the repaint cap — so a faster backend doesn't make it visibly
faster; the difference shows in hyper mode, and in the **car-steps/s** readout under the slider,
which measures what the pool is actually doing.

A generation ends at the step the first car anywhere reaches the target lap count. Each worker only
stops for its own cars, so a slice that ran on past that step (in hyper mode, up to a whole
2500-step round) is rewound to the start of the round and replayed to exactly that step — the
simulation is deterministic, so no slice ever scores frames the others didn't get, and the result
doesn't depend on how many cores the field was split over.

## The GPU backend (WebGPU)

With the toggle on **GPU**, one `gpu-worker.js` replaces the CPU pool. It speaks the same message
protocol as `sim-worker.js`, so the engine treats it as a pool of one.

**What runs on the GPU, and why only that.** A simulation step is strictly sequential for a car —
sensors feed the network, the network drives the physics, the physics decides the next sensors — so
moving any single piece (say, just the neural network, which is ~65 multiply-adds a car) to the GPU
would need a CPU↔GPU round trip every step, and that costs far more than the work moved. The only
split that pays is the whole step: wall raycasts, the network, physics, gates, laps and scoring all
run in one WGSL compute shader, the population stays resident in GPU memory for the whole
generation, and the only traffic is a tiny uniform write going in and one readback of the car state
per frame (per chunk in hyper mode) coming out. Evolution stays on the CPU — it runs once a
generation, needs every car's fitness, and that readback is already happening. Track geometry is
still generated by `sim.c` (the GPU worker hosts a wasm instance for that, and for the start-of-run
car state) and uploaded once per track.

**Lockstep.** A generation ends the moment any car completes the target lap count, for every car at
the same step. So each step is its own dispatch — consecutive dispatches are the one global barrier a
GPU offers — and a car that finishes records the frame it did; no car steps past it. Hyper mode's
long chunks are submitted in batches with a pipelined probe of that flag, so once a generation is
over the rest of the chunk isn't even submitted.

**Keeping a small population busy.** A phone's 150 cars aren't nearly enough threads for a GPU, and
each car's wall scan is the long pole of a step. So up to eight threads share a car: they split its
wall bucket, keep their own ray minima, and combine through workgroup memory. The split is chosen
from the population size to keep roughly two thousand threads in flight, and it's bit-identical
whichever is chosen.

**Numbers.** WGSL has no f64, and its built-in trig is only loosely specified — while a car's
heading is never wrapped. The trig is therefore hand-written in f32 (Cody-Waite reduction and
minimax polynomials), and the network's `tanh` is the same f32 routine `sim.c` uses.
`tools/gpuparity.mjs` runs a CPU worker and a GPU worker side by side and holds them to the same
race: from identical state, one step on each crashes the same cars, passes the same gates and lands
within f32 rounding; a generation stops on the same frame; and moving cars onto the GPU and back is
lossless.

**Which devices.** Anything with WebGPU: Chrome and Edge on desktop and Android, Safari on macOS,
iOS and iPadOS 26, Firefox on Windows. Older Android phones that only have OpenGL ES get it through
WebGPU's compatibility mode, which the shader is kept within (four storage buffers). If there's no
WebGPU, the toggle says so and the race stays on the CPU; if the GPU is lost mid-race (a driver
reset, a phone reclaiming it in the background), the race moves back to the CPU from the GPU's last
state. When it pays off: large populations and hyper mode. With a small field a multi-core CPU can
be just as fast, which is why CPU is the default — the steps/s readout tells you which wins on your
machine.

---

## Configuration & Settings
Fine-tune the simulation and the learning process using the built-in settings.

Two of them start in a different place on a phone or a tablet — see
[On a phone or tablet](#on-a-phone-or-tablet) below. They are the same sliders with the same range
either way; only where the handle starts differs.

### Training profiles

Finishing a track and making it fast are different jobs, so there are two profiles (**Goal** in the
settings). A run starts in **Explore**; the first generation in which a car completes the track hands
over to **Optimize** by itself. Reset or a new track goes back to Explore, and either button can be
pressed at any time. Every slider still works on top of whichever profile is active.

| | Explore — just finish it | Optimize — lap time |
| --- | --- | --- |
| Target Laps | **1**: a generation ends at the first finished lap | **2**: the second is a flying lap, the one worth timing |
| Elite Clones | 10% of the field, so a lucky line isn't lost | 3%: more room for new lines beat protecting old ones |
| Mutation | 15% of weights, sigma 0.5 | same |

How they were chosen: `tools/train.mjs` runs the real wasm headless and records, per generation,
total car-steps (CPU work — divide by the car-steps/s readout for seconds on your device), simulated
frames (how long you'd watch at 1x) and best lap. Settings were varied one at a time over 9–13
tracks x 3 seeds.

**Exploring** (cost to the first finished lap; unfinished runs counted at twice their total):

| | finished | CPU (M car-steps) | race time at 1x |
| --- | --- | --- | --- |
| stock (150 cars, 10% elites, 15% mutation, sigma 0.5, 1 lap) | 26/27 | **0.21** | **4.4 min** |
| 50 / 100 / 250 / 400 cars | 15 / 19 / 26 / 25 | 0.31 / 0.31 / 0.32 / 0.42 | 7.4 / 5.9 / 4.6 / 4.8 |
| 3 / 6 / 8 hidden units | 22 / 24 / 24 | 0.32 / 0.24 / 0.25 | 5.9 / 5.2 / 5.9 |
| 5% / 30% / 50% of weights mutated | 22 / 23 / 24 | 0.35 / 0.31 / 0.28 | 4.9 / 6.6 / 6.8 |
| sigma 0.25 / 1.0 | 23 / 26 | 0.31 / 0.22 | 4.5 / 5.0 |
| 2 laps per generation | 26 | 0.25 | 6.4 |

**Optimizing** (geometric-mean best lap over 9 tracks x 3 seeds, after the same Explore start):

| after the hand-over | best lap at 5M car-steps | at 10M |
| --- | --- | --- |
| **2 laps, 3% elites, sigma 0.5** | **24.0 s** | **22.4 s** |
| 2 laps, 10% elites, sigma 0.5 | 24.9 s | 23.1 s |
| sigma 0.2 / 0.1 / 0.05 | 26.1 / 27.9 / 30.3 s | |
| 20% elites, sigma 0.2 | 26.7 s | |
| 30% / 50% Focus % mutants of the best (sigma 0.2 / 0.1) | 26.6 / 29.0 s | |
| 3 laps per generation, sigma 0.2 | 27.3 s | |
| 80 / 300 cars | 28.8 / 28.1 s | 27.6 / 25.9 s |

Smaller, more "careful" mutations were worse for lap time at every budget measured, which is why
the mutation size doesn't change between the profiles.

**Population per device** (13 tracks including the four hardest, Explore phase):

| | finished | CPU (M car-steps) | race time at 1x |
| --- | --- | --- | --- |
| 150 cars, 4 hidden — **phone** | 35/39 | **0.26** | 4.9 min |
| 150 cars, 5 hidden | 32/39 | 0.33 | 5.9 min |
| 300 cars, 4 hidden — **computer** | **36/39** | 0.32 | **3.7 min** |
| 300 cars, 5 hidden | 35/39 | 0.42 | 4.9 min |
| 500 cars, 5 hidden (the old desktop default) | 35/39 | 0.43 | 4.0 min |

A phone is bound by CPU, and 150 cars is the least work per finished track. A computer has the
throughput to spend ~25% more work for the fastest finish in race time and the best hit rate on hard
tracks.

### AI & Evolutionary Parameters
* **Goal: Explore / Optimize:** which training profile is active — see [Training profiles](#training-profiles). Explore hands over to Optimize by itself.
* **Pop Size** (default 300, up to 2000; **150** on mobile)**:** The number of agents generated per generation.
* **Elite Clones** (set by the profile: 10% of Pop Size while exploring, 3% while optimizing)**:** Number of top-performing agents preserved exactly for the next generation (prevents regression): the all-time best, then the best *distinct* brains of the generation just run. These used to be that many copies of the all-time best — but the simulation is deterministic, so every copy drove the identical race and all but one slot was wasted. Fewer than a handful and a generation can occasionally lose ground it had already made — around 30 is enough that the population doesn't "forget" a solution it found. It tracks the population rather than being an absolute: a sixth of a 150-car field is the same share of it that 30 is of 500.
* **Focus %** (default 0%)**:** Fraction of the population bred as mutants of the all-time best instead of by crossover. It used to come with a fitness boost through a "weakest stretch" window — but the boosted score was compared raw against the all-time best, so a boosted car could take the champion's place with a score no honest run could beat, and the champion then never updated again. The boost is gone; in testing (`tools/train.mjs`), leaving this at 0 learned fastest.
* **Hidden Layers** (default 4)**:** Adjust the complexity of the AI's "brain" by changing the number of internal neurons. Four beat 3, 5, 6 and 8 in testing, and the SIMD build evaluates the hidden layer four units at a time, so a fifth costs a whole second pass.
* **Initial TTL (Time-To-Live):** A countdown for each agent, **reset in full every time it reaches a checkpoint**. It used to top the clock up by 150 frames and clamp it to 600, which quietly made the slider a lie — set it to 10,000 and the very first gate cut the car back to 600.
* **Target Laps** (set by the profile: 1 while exploring, 2 while optimizing)**:** Defines the goalpost for a successful generation before moving to the next stage of evolution.

How a generation is bred, for the curious: parents are drawn from the top fifth of the field
**rank-weighted**, so the leader parents far more often than the hundredth car rather than equally.
Crossover then works **one hidden unit at a time** — all of a unit's incoming weights, its bias and
its outgoing weights come from the same parent — because picking each weight independently splits up
groups of weights that only mean anything together, and two parents that both drive well routinely
produced a child that drove into a wall. New populations start with fan-in scaled weights instead of
a flat `[-1,1]`, which stopped the first few dozen generations being spent climbing back out of tanh
saturation.

Each child has about 15% of its weights mutated — the JS edition's default rate — by a Gaussian nudge
that **shrinks as the run goes on** (annealing from wide exploration early to fine polishing later).
For a while every weight was mutated on every generation instead; at the starting nudge size that
moved each weight about 4.5x as far per generation as the JS edition did, which turned most of the
bred field back into near-random brains. Across the 25 built-in tracks (`tools/train.mjs`, phone
settings, 30 generations, 3 seeds) going back to sparse mutation, distinct elites and no focus boost
took the solved runs from 55/75 to 65/75; Monaco went from never to generation 4-5.

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
to 30), and not at all when nothing has changed — the simulation steps at its fixed 60 Hz either
way, only the painting is throttled. Cars are blitted from a cached sprite per livery instead of
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

The page checks once, at load, whether it is running on a phone or a tablet, and if it is, the
population starts smaller (see [Training profiles](#training-profiles) for why). Nothing is taken away — every one of them is still the same
slider with the same range, and every one can be pushed back up.

| | desktop | phone / tablet |
| --- | --- | --- |
| Pop Size | 300 | **150** |
| Elite Clones | 30 → 9 | **15 → 5** (Explore → Optimize) |
| Hidden Layers | 4 | 4 |

Underneath those, two things the sliders don't show:

* **Held sideways, it's the desktop layout**: settings in a panel beside the track instead of a
  drawer over it — any landscape screen 640px or wider (an iPad, or a phone on its side). On a phone
  held sideways the panel is narrower, the chart row is left out to keep the track tall, and
  Start / Hyper / Reset stay above the track. Portrait keeps the drawer.
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

The canvas repaint rate — not the simulation, which always steps at a fixed 60 Hz regardless — is 60
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
