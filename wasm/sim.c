// sim.c — the entire TrackML compute backend, compiled to freestanding wasm32.
//
// This is the only thing that differs from the JavaScript edition of TrackML:
// the simulation, the neural nets, the evolution and the track geometry all
// live in here instead of in a JS Web Worker. The UI, the editor, the track
// format and the on-disk brain format are byte-for-byte the same, so tracks
// and saved AIs move between the two versions untouched.
//
// Built with plain clang (`--target=wasm32 -nostdlib`), NOT Emscripten, so the
// output is one self-contained ~40KB .wasm with no JS glue and no runtime to
// boot. There is no libc here: everything below, down to sinf and the
// allocator, is in this file.
//
// LAYOUT NOTE. Every hot array is a struct-of-arrays of f32/i32 in linear
// memory. That is the whole point of the port: the JS version spends its time
// chasing pointers through `{p1:{x,y},p2:{x,y}}` wall objects and boxing car
// state into the GC heap, and it re-ships every brain through postMessage on
// every generation. Here a wall is 20 contiguous bytes, a generation of brains
// is one flat Float32Array, and the per-step inner loop touches nothing the
// engine has to trace.

#define WASM_EXPORT(name) __attribute__((export_name(#name))) name

// ---------------------------------------------------------------------------
// Limits. These mirror the UI slider maxima in index.html — population tops out
// at 2000 and the hidden layer at 9 — with headroom so a hand-edited setting
// can't walk off the end of a static array.
//
// MAX_HIDDEN is not free headroom: the two brain buffers are sized
// MAX_CARS x BRAIN_MAX floats apiece and BRAIN_MAX is linear in it, so they
// were 7.4MB of the module's 8MB of static data — reserved in EVERY instance,
// and the app runs one per core plus the master. On an eight-core phone that
// was over a hundred megabytes of address space set aside for hidden units the
// slider cannot ask for. 16 is still most of a factor of two above the
// slider's maximum, and engine.js validates an imported brain against
// max_hidden() before writing a single weight, so a file from somewhere else
// is refused with a message rather than silently truncated.
// ---------------------------------------------------------------------------
#define MAX_CARS     2048
#define MAX_HIDDEN   16
#define SENS_N       7
// Absolute checkpoint-index ceiling for the per-gate speed telemetry used by
// the focused-learning feature below. Real tracks stay far under this — at
// the generator's normal gate spacing (TRACK_CP_SPACING, 34px) it would take
// an ~8700px loop to reach it — so this is headroom, not a real limit.
#define MAX_GATES    256
// Sensors, speed, angle to the next gate, and the car's own two outputs from
// the previous frame. That last pair is what gives an otherwise feedforward
// network a memory: this frame's decision can depend on the last one, so the
// chain out(t) -> out(t+1) -> ... carries state forward without the cost of a
// full recurrent hidden layer (which would add h*h weights instead of 2*h).
// In practice it is what lets a car hold a steady line through a corner
// instead of rediscovering its steering angle from scratch every frame.
#define IN_N         (SENS_N + 4)
#define OUT_N        2
#define BRAIN_MAX    (IN_N * MAX_HIDDEN + MAX_HIDDEN * OUT_N + MAX_HIDDEN + OUT_N)
#define RENDER_STRIDE 18   // id,crashed,x,y,angle,speed,out0,out1,laps,fit,lap,7 sensors

#define CAR_W        14.0f
#define CAR_H        7.0f
// Sensor reach is derived from top speed rather than fixed, because what a
// driver actually needs is a constant amount of TIME to react, not a constant
// number of pixels. At the old fixed 180px a car at Max Speed 25 covered the
// whole sensor range in seven frames and physically could not see a corner
// coming. The floor keeps the default (Max Speed 10) at exactly the 180px it
// has always been, so nothing about the stock settings changes.
#define SENSOR_LOOKAHEAD_FRAMES 18.0f
#define SENSOR_LEN_MIN   180.0f
#define SENSOR_LEN_MAX   500.0f

#define TRACK_MAX_SEG      30.0f
#define TRACK_CP_SPACING   34.0f
#define TRACK_CP_MIN_GAP   18.0f   // closest two gates may sit, px
#define TRACK_WALL_MAXLEN  110.0f

#define PI_F  3.14159265358979323846f
#define PI_D  3.14159265358979323846

typedef unsigned int   u32;
typedef unsigned long long u64;
typedef int            i32;

// ---------------------------------------------------------------------------
// Freestanding libc bits. clang lowers small memcpy/memset to loads and stores,
// but it is free to emit a call for a non-constant size, so the symbols have to
// exist or the link fails.
// ---------------------------------------------------------------------------
void *memcpy(void *d, const void *s, unsigned long n) {
    unsigned char *dp = (unsigned char *)d; const unsigned char *sp = (const unsigned char *)s;
    while (n--) *dp++ = *sp++;
    return d;
}
void *memset(void *d, int c, unsigned long n) {
    unsigned char *dp = (unsigned char *)d;
    while (n--) *dp++ = (unsigned char)c;
    return d;
}

// ---------------------------------------------------------------------------
// Math. sqrt/abs/floor/ceil/min/max are single wasm instructions; the
// transcendentals are not, so they are implemented here rather than imported
// from JS — a call out to Math.sin per sensor per car per step would cost more
// than the physics it feeds.
//
// The reductions run in f64 (free on wasm, and it keeps the argument reduction
// honest for the unbounded car heading, which is never wrapped). tools/mathtest
// checks every one of these against the JS Math.* it replaces.
// ---------------------------------------------------------------------------
static inline float  absf(float x)    { return __builtin_fabsf(x); }
static inline float  sqrtf_(float x)  { return __builtin_sqrtf(x); }
static inline double sqrtd_(double x) { return __builtin_sqrt(x); }
static inline float  floorf_(float x) { return __builtin_floorf(x); }
static inline double floord_(double x){ return __builtin_floor(x); }
static inline float  ceilf_(float x)  { return __builtin_ceilf(x); }
static inline float  minf(float a, float b) { return a < b ? a : b; }
static inline float  maxf(float a, float b) { return a > b ? a : b; }
static inline int    mini(int a, int b) { return a < b ? a : b; }
static inline int    maxi(int a, int b) { return a > b ? a : b; }
static inline float  hypotf_(float x, float y) { return sqrtf_(x * x + y * y); }
static inline float  clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// sin/cos on |r| <= pi/4, minimax-truncated Taylor. Error < 1e-11 in f64, far
// below what an f32 result can even represent.
static double sin_poly(double x) {
    double x2 = x * x;
    return x * (1.0 + x2 * (-1.66666666666666667e-1 + x2 * (8.33333333333333333e-3 +
           x2 * (-1.98412698412698413e-4 + x2 * (2.75573192239858907e-6 +
           x2 * -2.50521083854417188e-8)))));
}
static double cos_poly(double x) {
    double x2 = x * x;
    return 1.0 + x2 * (-0.5 + x2 * (4.16666666666666667e-2 +
           x2 * (-1.38888888888888889e-3 + x2 * (2.48015873015873016e-5 +
           x2 * -2.75573192239858907e-7))));
}
// pi/2 to double, plus the bit that did not fit — used to build pi/2 in atan.
#define PIO2      1.57079632679489661926e0
#define PIO2_TAIL 6.12323399573676603587e-17
// Cody-Waite reduction constants. PIO2_R1 carries only 33 significant bits, so
// n * PIO2_R1 stays *exact* for every quadrant count we can reach (n up to 2^20
// keeps the product inside 53 bits) and all the rounding lands in the small R2
// term. Using the full-precision pi/2 here instead looks equivalent and is not:
// it costs a bit of r per quadrant, which at a few thousand radians of
// accumulated car heading is an error of 1e-4 in the result.
#define PIO2_R1 1.57079632673412561417e+00
#define PIO2_R2 6.07710050650619224932e-11
static void sincos_d(double x, double *so, double *co) {
    double q = x * (2.0 / PI_D);
    double n = floord_(q + 0.5);
    double r = (x - n * PIO2_R1) - n * PIO2_R2;
    long long k = (long long)n & 3;
    double s = sin_poly(r), c = cos_poly(r);
    switch ((int)k) {
        case 0: *so =  s; *co =  c; break;
        case 1: *so =  c; *co = -s; break;
        case 2: *so = -s; *co = -c; break;
        default:*so = -c; *co =  s; break;
    }
}
static float sinf_(float x) { double s, c; sincos_d((double)x, &s, &c); return (float)s; }
static float cosf_(float x) { double s, c; sincos_d((double)x, &s, &c); return (float)c; }

// atan on |x| <= tan(pi/8) — the Cephes P/Q rational, |err| < 1e-17.
// Two details that are easy to get wrong and silent when you do: Q carries an
// *implicit leading 1* (it is degree 5 to P's degree 4), and both are evaluated
// highest-coefficient-first. Getting either backwards still produces a smooth,
// plausible-looking curve — it is just a different function, off by 1.5e-2.
static double atan_unit(double x) {
    double z = x * x;
    double p = -8.750608600031904e-1;
    p = p * z + -1.615753718733365e1;
    p = p * z + -7.500855792314705e1;
    p = p * z + -1.228866684490136e2;
    p = p * z + -6.485021904942025e1;
    double q = z + 2.485846490142306e1;
    q = q * z + 1.650270098316989e2;
    q = q * z + 4.328810604912903e2;
    q = q * z + 4.853903996359137e2;
    q = q * z + 1.945506571482614e2;
    return x + x * z * (p / q);
}
static double atan_d(double x) {
    double ax = x < 0 ? -x : x;
    double y;
    if (ax > 2.414213562373095) {          // tan(3pi/8)
        ax = -1.0 / ax;
        y = PIO2 + (atan_unit(ax) + PIO2_TAIL);
    } else if (ax > 0.4142135623730950) {  // tan(pi/8)
        ax = (ax - 1.0) / (ax + 1.0);
        y = (PI_D / 4.0) + (atan_unit(ax) + 0.5 * PIO2_TAIL);
    } else {
        y = atan_unit(ax);
    }
    return x < 0 ? -y : y;
}
static inline int signbit_d(double x) { union { double d; u64 u; } v; v.d = x; return (int)(v.u >> 63); }

// Full IEEE-754 atan2, signed zeros included, because that is what Math.atan2
// does and the two editions are meant to agree: atan2(0, -0) is pi, not 0.
static float atan2f_(float yf, float xf) {
    double y = (double)yf, x = (double)xf;
    if (y == 0.0) {
        if (x > 0.0 || (x == 0.0 && !signbit_d(x))) return signbit_d(y) ? -0.0f : 0.0f;
        return signbit_d(y) ? (float)-PI_D : (float)PI_D;
    }
    if (x == 0.0) return y > 0.0 ? (float)(PI_D / 2.0) : (float)(-PI_D / 2.0);
    double a = atan_d(y / x);
    if (x > 0.0) return (float)a;
    return (float)(y > 0.0 ? a + PI_D : a - PI_D);
}

// 2^n by writing the exponent field directly — no loop, no table.
static double pow2i(int n) {
    if (n < -1022) return 0.0;
    if (n > 1023) return 1.0e308 * 1.0e308;
    union { u64 u; double d; } v;
    v.u = ((u64)(n + 1023)) << 52;
    return v.d;
}
#define LN2_HI 6.93147180369123816490e-01
#define LN2_LO 1.90821492927058770002e-10
static double exp_d(double x) {
    if (x > 709.0) return 1.0e308 * 1.0e308;
    if (x < -745.0) return 0.0;
    double n = floord_(x * 1.44269504088896340736 + 0.5);
    double r = (x - n * LN2_HI) - n * LN2_LO;   // |r| <= ln2/2
    // Degree 9 leaves a truncation error of r^10/10! ~ 7e-12 at the worst |r|.
    // Overkill for tanh, which lands in an f32 — but two extra FMAs is a cheap
    // price for an exp that is right rather than merely right enough.
    double p = 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    return p * pow2i((int)n);
}
// The activation, so it runs (population x hidden+output) times per step.
// Saturating early is not just a speed trick: exp(2x) overflows long before
// tanh stops being 1.0 to f32 precision.
static float tanhf_(float x) {
    if (x > 9.011f)  return 1.0f;
    if (x < -9.011f) return -1.0f;
    double e = exp_d(2.0 * (double)x);
    return (float)((e - 1.0) / (e + 1.0));
}

// The activation again, but in f32 and without exp_d's f64 argument reduction,
// its branches or its degree-9 polynomial. This is what feedForward calls —
// (hidden + 2) times per car per frame, which makes it the most-executed
// transcendental in the project by an order of magnitude — while tanhf_ above
// stays exactly as it was for the t_tanh export and its 2e-7 bar.
//
//   tanh(x) = (e^2x - 1) / (e^2x + 1),  e^y = 2^(y * log2e)
//
// The 2^n is built by writing the exponent field of a float directly, and the
// fractional part is a degree-7 Taylor of 2^f on |f| <= 1/2 (truncation
// ~5e-9 relative, an order below what an f32 can hold). The error of the whole
// thing is dominated by the quotient, whose sensitivity to a relative error in
// e is 2e/(e+1)^2 — maximal at e = 1, where it is half an f32 epsilon. So the
// result is good to ~1e-7 ABSOLUTE across the whole line, which is the bar
// that matters for a number feeding a [-1,1] control output. tools/mathtest
// checks it against Math.tanh on every build, same as the rest.
#define NN_LOG2E_X2 2.885390081777926814f   // 2 * log2(e)
static inline float exp2_poly(float f) {
    // 2^f = e^(f ln2), Taylor to f^7.
    float p = 1.5252733804059840e-5f;
    p = p * f + 1.5403530393381609e-4f;
    p = p * f + 1.3333558146428443e-3f;
    p = p * f + 9.6181291076284772e-3f;
    p = p * f + 5.5504108664821580e-2f;
    p = p * f + 2.4022650695910071e-1f;
    p = p * f + 6.9314718055994531e-1f;
    return p * f + 1.0f;
}
static inline float pow2_fast(float n) {
    union { u32 u; float f; } v;
    v.u = (u32)(((i32)n + 127) << 23);
    return v.f;
}
static float nn_tanh(float x) {
    // Past this the f32 result is exactly +-1 anyway, and clamping here is
    // what keeps the exponent write below inside the float range.
    if (x > 9.011f)  return 1.0f;
    if (x < -9.011f) return -1.0f;
    float y = x * NN_LOG2E_X2;
    float n = __builtin_nearbyintf(y);          // one wasm instruction: f32.nearest
    float e = exp2_poly(y - n) * pow2_fast(n);
    return (e - 1.0f) / (e + 1.0f);
}

static float acosf_(float x) {
    if (x >= 1.0f) return 0.0f;
    if (x <= -1.0f) return PI_F;
    double d = (double)x;
    // NEGATIVE ZERO. `d < 0.0` is false for -0.0, so the quadrant correction
    // below was skipped while the division still yielded -infinity — and
    // acos(-0.0) came back as -pi/2 instead of +pi/2.
    //
    // This is not a curiosity. A dot product of two perpendicular unit vectors
    // lands on exactly -0.0 for one of the four orientations (0*-1 + -1*0), so
    // every right-angled corner facing that way reported a negative interior
    // angle. The tangent length solved from it was wrong, the corner's arc
    // collapsed, and its neighbour's ballooned to swallow the straight between
    // them: on a plain rectangle one corner simply vanished and was replaced by
    // a diagonal across the track.
    if (d == 0.0) return (float)(PI_D / 2.0);          // true for +0.0 and -0.0
    double a = atan_d(sqrtd_(1.0 - d * d) / d);
    return (float)(d < 0.0 ? a + PI_D : a);
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift128+). The JS version leans on Math.random, whose
// stream is per-realm and unseedable; seeding here means a worker's slice of a
// generation is reproducible, which is what makes a mutation bug debuggable.
// ---------------------------------------------------------------------------
static u64 rng_s0 = 0x9E3779B97F4A7C15ull, rng_s1 = 0xBF58476D1CE4E5B9ull;
static void rng_seed(u32 seed) {
    u64 z = 0x9E3779B97F4A7C15ull ^ ((u64)seed * 0x2545F4914F6CDD1Dull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    rng_s0 = z ^ (z >> 31);
    rng_s1 = z * 0x2545F4914F6CDD1Dull + 0x9E3779B97F4A7C15ull;
    if (rng_s0 == 0 && rng_s1 == 0) rng_s0 = 1;
}
static inline u64 rng_next(void) {
    u64 x = rng_s0, y = rng_s1;
    rng_s0 = y;
    x ^= x << 23;
    rng_s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rng_s1 + y;
}
// Uniform [0,1) with 24 bits of mantissa — the same resolution Math.random()
// gives once it is squeezed into an f32 weight.
static inline float rnd01(void) { return (float)(rng_next() >> 40) * (1.0f / 16777216.0f); }
static inline float rnd11(void) { return rnd01() * 2.0f - 1.0f; }

// ---------------------------------------------------------------------------
// Bump allocator over everything above __heap_base. Track geometry is the only
// thing allocated from it and it is rebuilt wholesale on every edit, so the
// whole arena is reset per build and nothing is ever individually freed.
// Car state and brains are static, not arena, precisely so a track rebuild
// mid-session cannot land on top of a population.
// ---------------------------------------------------------------------------
extern unsigned char __heap_base;
static u32 arena_base = 0, arena_top = 0;

static void arena_reset(void) {
    if (!arena_base) arena_base = ((u32)(unsigned long)&__heap_base + 15u) & ~15u;
    arena_top = arena_base;
}
static void *arena_alloc(u32 bytes) {
    u32 p = (arena_top + 15u) & ~15u;
    u32 end = p + bytes;
    u32 have = (u32)__builtin_wasm_memory_size(0) << 16;
    if (end > have) {
        u32 need = ((end - have) + 65535u) >> 16;
        if (__builtin_wasm_memory_grow(0, need + 16) == (unsigned long)-1) return 0;
    }
    arena_top = end;
    return (void *)(unsigned long)p;
}

// ---------------------------------------------------------------------------
// Track geometry
//
// A direct port of the JS generator, same model: the road surface is every
// point within `width` of the centreline (i.e. exactly what stroking the line
// with a round pen paints), and the barriers are the boundary of that shape.
// Offsets that fold back through the road get thrown away rather than clamped,
// so the road is the full width everywhere and no wall can sit on the asphalt.
// ---------------------------------------------------------------------------
typedef struct { float x, y; } Vec2;
typedef struct { float x1, y1, x2, y2; i32 seg; } Wall;      // 20 bytes
// 28 bytes. `apex` marks a gate placed at the middle of a corner rather than
// by the regular spacing — the editor draws those differently, and it is what
// makes the feature visible rather than merely present.
typedef struct { float p1x, p1y, p2x, p2y, cx, cy; i32 apex; } Checkpoint;
typedef struct { float x, y, radius; i32 type; float killTimer; } Zone; // 20 bytes

static Vec2       *tk_center;    static i32 tk_center_n;
static Wall       *tk_walls;     static i32 tk_wall_n;
static Checkpoint *tk_cps;       static i32 tk_cp_n;
static Zone       *tk_zones;     static i32 tk_zone_n;
static float       tk_start_x, tk_start_y, tk_start_angle, tk_width;
static i32         tk_start_cp;     // checkpoint nearest the start line
static float       tk_cp_min_step;  // tightest gap between two gates, px
static float       tk_len;          // centreline length, px

// Uniform grid over the centreline, CSR-packed (counts -> prefix sums -> fill)
// instead of the JS Map<"gx,gy", []>. Same 3x3 neighbourhood query, but the
// lookup is two array reads rather than a string concat and a hash probe.
static i32   *grid_start; static i32 *grid_items;
static i32    grid_nx, grid_ny; static float grid_cell, grid_ox, grid_oy;

// The centreline sample nearest a given distance along the track. cum[] is
// sorted, so this is a binary search plus a look at the other neighbour.
static i32 sampleAtDistance(const float *cum, i32 len, float d) {
    if (d <= 0.0f) return 0;
    if (d >= cum[len - 1]) return len - 1;
    i32 lo = 0, hi = len - 1;
    while (lo < hi) {
        i32 mid = (lo + hi + 1) >> 1;
        if (cum[mid] <= d) lo = mid; else hi = mid - 1;
    }
    if (lo + 1 < len && (cum[lo + 1] - d) < (d - cum[lo])) return lo + 1;
    return lo;
}

static void _trackTol(float dist, float *flat, float *sag) {
    float t = dist * 0.015f;
    *flat = clampf(t, 0.15f, 0.6f);
    *sag  = clampf(t * 1.5f, 0.2f, 0.9f);
}

static float _pointSegDist2(float px, float py, float ax, float ay, float bx, float by) {
    float dx = bx - ax, dy = by - ay;
    float l2 = dx * dx + dy * dy;
    float t = l2 > 0.0f ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0.0f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    float ex = px - (ax + t * dx), ey = py - (ay + t * dy);
    return ex * ex + ey * ey;
}

static void buildCentreGrid(Vec2 *pts, i32 n, float cell) {
    float minx = pts[0].x, maxx = pts[0].x, miny = pts[0].y, maxy = pts[0].y;
    for (i32 i = 1; i < n; i++) {
        if (pts[i].x < minx) minx = pts[i].x; if (pts[i].x > maxx) maxx = pts[i].x;
        if (pts[i].y < miny) miny = pts[i].y; if (pts[i].y > maxy) maxy = pts[i].y;
    }
    // Pad by three cells: queries are offset points sitting up to `dist` off the
    // line, and each reads the 3x3 block around itself.
    grid_cell = cell;
    grid_ox = minx - cell * 3.0f;
    grid_oy = miny - cell * 3.0f;
    grid_nx = (i32)((maxx - minx) / cell) + 7;
    grid_ny = (i32)((maxy - miny) / cell) + 7;
    if (grid_nx < 1) grid_nx = 1;
    if (grid_ny < 1) grid_ny = 1;

    i32 cells = grid_nx * grid_ny;
    grid_start = (i32 *)arena_alloc((u32)(cells + 1) * 4);
    i32 *counts = (i32 *)arena_alloc((u32)(cells + 1) * 4);
    if (!grid_start || !counts) { grid_items = 0; return; }
    for (i32 i = 0; i <= cells; i++) counts[i] = 0;

    // Pass 1: count. Segments are at most maxSeg long against a cell of at least
    // 16px, so a segment's bbox covers a handful of cells at worst.
    i32 total = 0;
    for (i32 pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            grid_start[0] = 0;
            for (i32 i = 0; i < cells; i++) grid_start[i + 1] = grid_start[i] + counts[i];
            total = grid_start[cells];
            grid_items = (i32 *)arena_alloc((u32)(total > 0 ? total : 1) * 4);
            if (!grid_items) return;
            for (i32 i = 0; i <= cells; i++) counts[i] = 0;
        }
        for (i32 i = 0; i < n; i++) {
            Vec2 a = pts[i], b = pts[(i + 1) % n];
            i32 x0 = (i32)floorf_((minf(a.x, b.x) - grid_ox) / cell);
            i32 x1 = (i32)floorf_((maxf(a.x, b.x) - grid_ox) / cell);
            i32 y0 = (i32)floorf_((minf(a.y, b.y) - grid_oy) / cell);
            i32 y1 = (i32)floorf_((maxf(a.y, b.y) - grid_oy) / cell);
            if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
            if (x1 >= grid_nx) x1 = grid_nx - 1; if (y1 >= grid_ny) y1 = grid_ny - 1;
            for (i32 gx = x0; gx <= x1; gx++) for (i32 gy = y0; gy <= y1; gy++) {
                i32 k = gy * grid_nx + gx;
                if (pass == 0) counts[k]++;
                else grid_items[grid_start[k] + counts[k]++] = i;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Variable width
//
// The road used to be one number wide everywhere. It is now a per-centreline-
// sample half-width in tk_w[], which is what lets the track pinch in where two
// passes of it run close together instead of merging into one slab.
//
// Everything downstream that used to compare a distance against the single
// `dist` now has to ask "how wide is the road *here*", so the inside-the-road
// test below replaces the plain distance query: a point is inside the road when
// some segment's own half-width reaches it, and the depth it reaches by is what
// the offset trim wants to know.
// ---------------------------------------------------------------------------
static float *tk_w;          // per-sample half-width
static float  tk_w_max;      // the widest it gets, for grid sizing
static float  tk_w_min;      // the narrowest, for approximation budgets

// How deep inside the road this point sits, and the half-width of the segment
// that claims it. Negative depth means outside the asphalt altogether.
static float roadDepth(float px, float py, float *rOut) {
    if (!grid_items || !tk_w) { if (rOut) *rOut = tk_width; return -1.0e30f; }
    i32 gx = (i32)floorf_((px - grid_ox) / grid_cell);
    i32 gy = (i32)floorf_((py - grid_oy) / grid_cell);
    float best = -1.0e30f, bestR = tk_width;
    i32 xlo = maxi(gx - 1, 0), xhi = mini(gx + 1, grid_nx - 1);
    i32 ylo = maxi(gy - 1, 0), yhi = mini(gy + 1, grid_ny - 1);
    i32 n = tk_center_n;
    for (i32 ix = xlo; ix <= xhi; ix++) for (i32 iy = ylo; iy <= yhi; iy++) {
        i32 k = iy * grid_nx + ix;
        for (i32 s = grid_start[k]; s < grid_start[k + 1]; s++) {
            i32 i = grid_items[s];
            i32 j = (i + 1) % n;
            Vec2 a = tk_center[i], b = tk_center[j];
            // The segment's width is taken as the wider of its two ends, so a
            // taper never leaves a sliver of unclaimed road between samples.
            float r = maxf(tk_w[i], tk_w[j]);
            float d = sqrtf_(_pointSegDist2(px, py, a.x, a.y, b.x, b.y));
            float depth = r - d;
            if (depth > best) { best = depth; bestR = r; }
        }
    }
    if (rOut) *rOut = bestR;
    return best;
}

// Is this point on the asphalt at all? Same union-of-discs question as
// roadDepth, but it only needs a yes or no, so it skips the square roots and
// bails on the first segment that claims the point.
static inline i32 segClaims(i32 i, float px, float py) {
    i32 j = i + 1; if (j >= tk_center_n) j = 0;
    Vec2 a = tk_center[i], b = tk_center[j];
    float r = maxf(tk_w[i], tk_w[j]);
    return _pointSegDist2(px, py, a.x, a.y, b.x, b.y) < r * r;
}
// The only caller is a car, which is asked this every frame and carries an
// enormous hint: it moves a few pixels per frame, so the centreline segment
// that claimed it last frame almost always claims it again this one. Trying
// that segment and its immediate neighbours first turns the usual case from a
// 3x3 grid walk over a few dozen segments into a handful of point-to-segment
// distances, and the answer is identical either way — it is the same
// union-of-discs test, only asked in a better order.
#define ROAD_HINT_SPAN 2
static i32 insideRoad(float px, float py, i32 *hint) {
    if (!grid_items || !tk_w) return 1;
    i32 n = tk_center_n;
    i32 hs = *hint;
    if (hs >= 0 && hs < n) {
        for (i32 d = -ROAD_HINT_SPAN; d <= ROAD_HINT_SPAN; d++) {
            i32 i = hs + d; if (i < 0) i += n; else if (i >= n) i -= n;
            if (segClaims(i, px, py)) { *hint = i; return 1; }
        }
    }
    // Miss: the car has moved further than the hint window, or it is off the
    // road entirely. Fall back to the grid and re-seed the hint from whatever
    // claims it.
    i32 gx = (i32)floorf_((px - grid_ox) / grid_cell);
    i32 gy = (i32)floorf_((py - grid_oy) / grid_cell);
    i32 xlo = maxi(gx - 1, 0), xhi = mini(gx + 1, grid_nx - 1);
    i32 ylo = maxi(gy - 1, 0), yhi = mini(gy + 1, grid_ny - 1);
    for (i32 ix = xlo; ix <= xhi; ix++) for (i32 iy = ylo; iy <= yhi; iy++) {
        i32 k = iy * grid_nx + ix;
        for (i32 s = grid_start[k]; s < grid_start[k + 1]; s++) {
            i32 i = grid_items[s];
            if (segClaims(i, px, py)) { *hint = i; return 1; }
        }
    }
    return 0;
}

// Distance from sample i to the nearest part of the track that isn't simply
// "further along the road" — used to decide how much room the road actually
// has at that point.
//
// The arc-length exclusion window is the whole trick. Excluding too little and
// every corner reads as a collision with itself; excluding too much and a
// hairpin's other leg goes unnoticed. The window is sized to the tightest turn
// the generator will build (minRadius = 1.1 * width + 4): half a circle of that
// radius is about 3.5 widths of arc, so anything inside ~4 widths of arc is the
// road curving normally and is left alone. Past that, a close approach is two
// genuinely different parts of the track and the road should give way.
static void computeClearance(i32 n, float reqW, float *segLen, float *out) {
    float window = reqW * 4.0f + 15.0f;
    for (i32 i = 0; i < n; i++) {
        float px = tk_center[i].x, py = tk_center[i].y;
        float best = 1.0e30f;
        // Walk outward from i in both directions, skipping the window, and stop
        // once the remaining track is all on the far side of the loop.
        float fwd = 0.0f;
        for (i32 k = 1; k < n; k++) {
            i32 j = (i + k) % n;
            fwd += segLen[(i + k - 1) % n];
            if (fwd <= window) continue;
            // The same pair is reached from the other end too; stopping at the
            // halfway point keeps this O(n^2/2) instead of O(n^2).
            if (k > n / 2) break;
            float dx = tk_center[j].x - px, dy = tk_center[j].y - py;
            float d2 = dx * dx + dy * dy;
            if (d2 < best) best = d2;
        }
        float bwd = 0.0f;
        for (i32 k = 1; k < n; k++) {
            i32 j = (i - k + n * 2) % n;
            bwd += segLen[(i - k + n * 2) % n];
            if (bwd <= window) continue;
            if (k > n / 2) break;
            float dx = tk_center[j].x - px, dy = tk_center[j].y - py;
            float d2 = dx * dx + dy * dy;
            if (d2 < best) best = d2;
        }
        out[i] = best >= 1.0e30f ? 1.0e30f : sqrtf_(best);
    }
}

// Turn clearances into half-widths.
//
// mode: 0 = off (flat requested width), 1 = auto.
// blend: 0 = narrow only where it's needed, 1 = one uniform width for the whole
//        track (the narrowest any point needs). Anything between tapers toward
//        uniform, which is the slider in the editor.
#define AUTO_MIN_HALF  11.0f   // a car is 14x7; below this it cannot get through
#define AUTO_GAP       12.0f   // barrier-to-barrier gap left between two passes

// A cyclic box blur, in place, via a running sum — O(n) per pass regardless of
// the window size, which matters because the window here is large on purpose.
static void boxBlur(float *v, float *tmp, i32 n, i32 r) {
    if (r < 1 || n < 3) return;
    if (r > (n - 1) / 2) r = (n - 1) / 2;
    if (r < 1) return;
    float inv = 1.0f / (float)(2 * r + 1);
    float sum = 0.0f;
    for (i32 k = -r; k <= r; k++) sum += v[(k + n * 2) % n];
    for (i32 i = 0; i < n; i++) {
        tmp[i] = sum * inv;
        sum -= v[(i - r + n * 2) % n];
        sum += v[(i + r + 1 + n * 2) % n];
    }
    for (i32 i = 0; i < n; i++) v[i] = tmp[i];
}

static void computeWidths(i32 n, float reqW, i32 mode, float blend, float *clear, float *segLen) {
    if (mode == 0) {
        for (i32 i = 0; i < n; i++) tk_w[i] = reqW;
        tk_w_max = reqW; tk_w_min = reqW;
        return;
    }
    for (i32 i = 0; i < n; i++) {
        float c = clear[i];
        float w = reqW;
        // Two passes closer together than a single half-width have merged into
        // a junction. Pinching there doesn't restore a wall between them, it
        // just puts a notch in the middle of a crossing — so leave those alone.
        if (c > reqW && c < 2.0f * reqW + AUTO_GAP) {
            w = (c - AUTO_GAP) * 0.5f;
            if (w > reqW) w = reqW;
            if (w < AUTO_MIN_HALF) w = AUTO_MIN_HALF;
        }
        tk_w[i] = w;
    }

    // Erode, then blur.
    //
    // The erosion pass widens each narrow stretch along the track so the pinch
    // begins before the tight bit rather than at it; the blur then turns the
    // resulting steps into a taper. The order matters — blurring first would
    // average the narrow point back open and undo the whole thing.
    //
    // Both windows are sized in PIXELS OF TRACK, not in samples, and converted
    // through the average sample spacing. Sizing them in samples instead ties
    // how gentle the taper looks to how finely the centreline happened to be
    // resampled, which varies with track width — the same pinch then tapers
    // smoothly on one track and steps visibly on another.
    float perim = 0.0f;
    for (i32 i = 0; i < n; i++) perim += segLen[i];
    float avgSeg = maxf(perim / (float)n, 0.5f);
    float taper = maxf(reqW * 2.5f, 40.0f);          // how long the ramp should be
    i32 blurR = (i32)(taper / avgSeg + 0.5f);
    blurR = maxi(1, mini(blurR, n / 6));
    i32 erodeR = blurR + 2;                          // keep the minimum after blurring

    float *tmp = (float *)arena_alloc((u32)n * 4);
    if (tmp) {
        for (i32 i = 0; i < n; i++) {
            float m = tk_w[i];
            for (i32 k = -erodeR; k <= erodeR; k++) {
                float v = tk_w[(i + k + n * 2) % n];
                if (v < m) m = v;
            }
            tmp[i] = m;
        }
        for (i32 i = 0; i < n; i++) tk_w[i] = tmp[i];
        // Two box passes make a triangular kernel — smooth enough that the
        // asphalt reads as a taper rather than a series of steps.
        boxBlur(tk_w, tmp, n, blurR);
        boxBlur(tk_w, tmp, n, blurR);
    }

    if (blend > 0.0f) {
        float gmin = tk_w[0];
        for (i32 i = 1; i < n; i++) if (tk_w[i] < gmin) gmin = tk_w[i];
        for (i32 i = 0; i < n; i++) tk_w[i] = tk_w[i] + (gmin - tk_w[i]) * blend;
    }

    tk_w_max = tk_w[0]; tk_w_min = tk_w[0];
    for (i32 i = 1; i < n; i++) {
        if (tk_w[i] > tk_w_max) tk_w_max = tk_w[i];
        if (tk_w[i] < tk_w_min) tk_w_min = tk_w[i];
    }
}

// A growable Vec2 list over the arena. Doubling and copying wastes arena space,
// but the arena is thrown away wholesale on the next build so it never leaks.
typedef struct { Vec2 *p; i32 n, cap; } VecList;
static void vl_init(VecList *v, i32 cap) { v->p = (Vec2 *)arena_alloc((u32)cap * 8); v->n = 0; v->cap = v->p ? cap : 0; }
static void vl_push(VecList *v, float x, float y) {
    if (v->n >= v->cap) {
        i32 nc = v->cap * 2 + 16;
        Vec2 *np = (Vec2 *)arena_alloc((u32)nc * 8);
        if (!np) return;
        for (i32 i = 0; i < v->n; i++) np[i] = v->p[i];
        v->p = np; v->cap = nc;
    }
    v->p[v->n].x = x; v->p[v->n].y = y; v->n++;
}

// Per-vertex arc solve, ported straight across. Two rules keep it drivable:
// every vertex gets at least minRadius (so a turn is never tighter than the
// road is wide), and neighbouring arcs share out the straight between them (so
// two close-together points soften each other instead of kinking).
typedef struct { float ax, ay, bx, by, phi, turn, T, lb; i32 sign; i32 ok; } VtxArc;

// `apexOut` collects the midpoint of every corner arc this builds — the apex of
// the turn. They are positions rather than sample indices because the polyline
// is resampled at an even spacing afterwards, which renumbers everything; the
// caller maps each position back to its nearest final sample.
static i32 buildCentreline(const float *path, i32 n, float width, VecList *out,
                           Vec2 *apexOut, i32 *apexCount, i32 apexCap) {
    float minRadius = width * 1.1f + 4.0f;
    float tol_flat, tol_sag; _trackTol(width, &tol_flat, &tol_sag);

    VtxArc *V = (VtxArc *)arena_alloc((u32)n * sizeof(VtxArc));
    if (!V) return 0;

    for (i32 i = 0; i < n; i++) {
        float cx = path[i * 4], cy = path[i * 4 + 1];
        i32 pi = (i - 1 + n) % n, ni = (i + 1) % n;
        float ax = path[pi * 4] - cx, ay = path[pi * 4 + 1] - cy;
        float bx = path[ni * 4] - cx, by = path[ni * 4 + 1] - cy;
        float al = hypotf_(ax, ay), bl = hypotf_(bx, by);
        // The arena is not zeroed, so a skipped vertex still needs inert values:
        // the share-the-straight pass below indexes neighbours unconditionally.
        if (al < 1e-6f || bl < 1e-6f) { V[i].ok = 0; V[i].T = 0.0f; V[i].lb = 1.0f; continue; }
        ax /= al; ay /= al; bx /= bl; by /= bl;
        float cosv = clampf(ax * bx + ay * by, -1.0f, 1.0f);
        float phi = acosf_(cosv);        // interior angle at this vertex
        float turn = PI_F - phi;         // how far the heading swings through it
        float asked = path[i * 4 + 3];
        i32 type = (i32)path[i * 4 + 2];  // 0 = rounded, 1 = corner
        float want = type == 1 ? minRadius : maxf(asked, minRadius);
        float T = 0.0f;
        if (turn > 0.02f) {
            double s, c; sincos_d((double)(phi * 0.5f), &s, &c);
            float tanHalf = (float)(s / (c == 0.0 ? 1e-12 : c));
            T = minf(5000.0f, want / maxf(1e-4f, tanHalf));
        }
        float cross = (-ax) * by - (-ay) * bx;   // which way the heading turns
        V[i].ax = ax; V[i].ay = ay; V[i].bx = bx; V[i].by = by;
        V[i].phi = phi; V[i].turn = turn; V[i].T = T;
        V[i].sign = cross >= 0.0f ? 1 : -1; V[i].lb = bl; V[i].ok = 1;
    }

    for (i32 pass = 0; pass < 4; pass++) {
        for (i32 i = 0; i < n; i++) {
            VtxArc *a = &V[i], *b = &V[(i + 1) % n];
            if (!a->ok || !b->ok) continue;
            float sum = a->T + b->T, room = a->lb * 0.98f;
            if (sum > room && sum > 1e-6f) { float s = room / sum; a->T *= s; b->T *= s; }
        }
    }

    VecList raw; vl_init(&raw, n * 8 + 64);
    for (i32 i = 0; i < n; i++) {
        VtxArc *v = &V[i];
        float cx = path[i * 4], cy = path[i * 4 + 1];
        if (!v->ok) continue;
        // `push` in the JS keeps a 0.25px minimum gap; same here.
        #define PUSH(X, Y) do { \
            float _x = (X), _y = (Y); \
            if (raw.n == 0 || hypotf_(_x - raw.p[raw.n-1].x, _y - raw.p[raw.n-1].y) > 0.25f) vl_push(&raw, _x, _y); \
        } while (0)
        if (v->T < 0.75f || v->turn <= 0.02f) { PUSH(cx, cy); continue; }
        double sh, ch; sincos_d((double)(v->phi * 0.5f), &sh, &ch);
        float r = v->T * (float)(sh / (ch == 0.0 ? 1e-12 : ch));
        float Ax = cx + v->ax * v->T, Ay = cy + v->ay * v->T;
        float Bx = cx + v->bx * v->T, By = cy + v->by * v->T;
        if (!(r > 0.5f)) { PUSH(Ax, Ay); PUSH(Bx, By); continue; }
        // Arc centre: r from the incoming tangent point, square to the incoming
        // heading, on whichever side the path turns toward.
        float hx = -v->ax, hy = -v->ay;
        float ox = Ax + (-hy) * (float)v->sign * r, oy = Ay + hx * (float)v->sign * r;
        float a0 = atan2f_(Ay - oy, Ax - ox);
        float sweep = (float)v->sign * v->turn;
        float step = clampf(2.0f * acosf_(maxf(0.0f, 1.0f - tol_flat / r)), 0.06f, 0.5f);
        i32 steps = maxi(2, (i32)ceilf_(absf(sweep) / step));
        for (i32 k = 0; k <= steps; k++) {
            float a = a0 + sweep * ((float)k / (float)steps);
            PUSH(ox + cosf_(a) * r, oy + sinf_(a) * r);
        }
        // Halfway through the sweep, which is the apex of the turn — the point
        // on the corner closest to its inside edge.
        if (apexOut && apexCount && *apexCount < apexCap) {
            float am = a0 + sweep * 0.5f;
            apexOut[*apexCount].x = ox + cosf_(am) * r;
            apexOut[*apexCount].y = oy + sinf_(am) * r;
            (*apexCount)++;
        }
        PUSH(Bx, By);
        #undef PUSH
    }
    if (raw.n > 1) {
        if (hypotf_(raw.p[0].x - raw.p[raw.n-1].x, raw.p[0].y - raw.p[raw.n-1].y) < 0.25f) raw.n--;
    }
    if (raw.n < 3) return 0;

    // Resample evenly. A long bare chord has no samples on it, and the offset
    // trim below only notices another part of the track crossing a chord by the
    // samples sitting on it — so a narrow track needs correspondingly fine ones.
    float maxSeg = maxf(8.0f, minf(TRACK_MAX_SEG, width * 1.2f));
    vl_init(out, raw.n * 3 + 64);
    for (i32 i = 0; i < raw.n; i++) {
        Vec2 p = raw.p[i], q = raw.p[(i + 1) % raw.n];
        float d = hypotf_(q.x - p.x, q.y - p.y);
        i32 steps = maxi(1, (i32)ceilf_(d / maxSeg));
        for (i32 k = 0; k < steps; k++) {
            float t = (float)k / (float)steps;
            vl_push(out, p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t);
        }
    }
    return out->n;
}

// One side of the raw offset outline. side = +1 is left of travel, -1 right.
// Outside-of-turn joins get a real arc so the barrier hugs the same rounded
// corner the asphalt has; inside-of-turn joins get the true miter, which is
// what keeps the road full width through a corner.
typedef struct { float x, y; i32 ci; } OffPt;
typedef struct { OffPt *p; i32 n, cap; } OffList;
static void ol_push(OffList *v, float x, float y, i32 ci) {
    if (v->n >= v->cap) {
        i32 nc = v->cap * 2 + 32;
        OffPt *np = (OffPt *)arena_alloc((u32)nc * sizeof(OffPt));
        if (!np) return;
        for (i32 i = 0; i < v->n; i++) np[i] = v->p[i];
        v->p = np; v->cap = nc;
    }
    v->p[v->n].x = x; v->p[v->n].y = y; v->p[v->n].ci = ci; v->n++;
}

// `dist` is now per-sample (tk_w), so each offset point steps out by however
// wide the road is at the sample it belongs to.
static void offsetOutline(Vec2 *pts, i32 n, i32 side, OffList *raw) {
    raw->p = (OffPt *)arena_alloc((u32)(n * 2 + 64) * sizeof(OffPt));
    raw->n = 0; raw->cap = raw->p ? n * 2 + 64 : 0;

    for (i32 i = 0; i < n; i++) {
        float dist = tk_w[i];
        float flat, sag; _trackTol(dist, &flat, &sag);
        float arcStep = clampf(2.0f * acosf_(maxf(0.0f, 1.0f - flat / dist)), 0.08f, 0.5f);
        Vec2 prev = pts[(i - 1 + n) % n], curr = pts[i], next = pts[(i + 1) % n];
        float ix = curr.x - prev.x, iy = curr.y - prev.y; float il = hypotf_(ix, iy);
        float ox = next.x - curr.x, oy = next.y - curr.y; float ol = hypotf_(ox, oy);
        if (il < 1e-6f || ol < 1e-6f) continue;
        ix /= il; iy /= il; ox /= ol; oy /= ol;
        float s = (float)side;
        float n1x = -iy * s, n1y = ix * s;
        float n2x = -oy * s, n2y = ox * s;
        float cross = ix * oy - iy * ox;
        float dot = ix * ox + iy * oy;
        if (cross * s < -1e-9f) {
            float a0 = atan2f_(n1y, n1x);
            float sweep = atan2f_(n2y, n2x) - a0;
            while (sweep >  PI_F) sweep -= PI_F * 2.0f;
            while (sweep < -PI_F) sweep += PI_F * 2.0f;
            i32 steps = maxi(1, (i32)ceilf_(absf(sweep) / arcStep));
            for (i32 k = 0; k <= steps; k++) {
                float a = a0 + sweep * ((float)k / (float)steps);
                ol_push(raw, curr.x + cosf_(a) * dist, curr.y + sinf_(a) * dist, i);
            }
        } else {
            float m = 1.0f + dot;
            if (m > 0.08f) {
                float k = dist / m;
                ol_push(raw, curr.x + (n1x + n2x) * k, curr.y + (n1y + n2y) * k, i);
            } else {
                // Doubling back on itself — a miter would run off to infinity,
                // so bevel and let the trim sort it out.
                ol_push(raw, curr.x + n1x * dist, curr.y + n1y * dist, i);
                ol_push(raw, curr.x + n2x * dist, curr.y + n2y * dist, i);
            }
        }
    }
}

typedef struct { Wall *p; i32 n, cap; } WallList;
static void wl_push(WallList *v, float x1, float y1, float x2, float y2, i32 seg) {
    if (v->n >= v->cap) {
        i32 nc = v->cap * 2 + 64;
        Wall *np = (Wall *)arena_alloc((u32)nc * sizeof(Wall));
        if (!np) return;
        for (i32 i = 0; i < v->n; i++) np[i] = v->p[i];
        v->p = np; v->cap = nc;
    }
    v->p[v->n].x1 = x1; v->p[v->n].y1 = y1; v->p[v->n].x2 = x2; v->p[v->n].y2 = y2;
    v->p[v->n].seg = seg; v->n++;
}

// Drop every offset point that landed inside the road — which only happens
// where the outline folded back through itself — and chain the survivors.
static void trimOutlineToWalls(OffList *raw, float dist, i32 *cpOfSample, WallList *outw) {
    i32 *keep = (i32 *)arena_alloc((u32)(raw->n + 1) * 4);
    if (!keep) return;
    i32 m = 0;
    for (i32 i = 0; i < raw->n; i++) {
        // "Is this offset point inside the road?" — which with a variable width
        // is no longer a fixed distance from the centreline but whether any
        // segment's own half-width reaches it.
        if (roadDepth(raw->p[i].x, raw->p[i].y, 0) <= 0.75f) keep[m++] = i;
    }
    if (m < 3) return;

    // Merge runs of near-collinear segments inline (the JS does this in a second
    // simplifyWalls pass). `dropped` holds every point the current run has
    // swallowed: checking only the newest one lets the error creep up over a
    // long run and quietly narrow the road.
    float flat, sag; _trackTol(tk_w_min > 0.0f ? tk_w_min : dist, &flat, &sag);
    float sag2 = sag * sag;
    #define MAX_DROPPED 4096
    static Vec2 dropped[MAX_DROPPED];
    i32 ndrop = 0;

    for (i32 k = 0; k < m; k++) {
        OffPt a = raw->p[keep[k]], b = raw->p[keep[(k + 1) % m]];
        float dx = b.x - a.x, dy = b.y - a.y;
        float segLen = hypotf_(dx, dy);
        i32 bridged = ((keep[(k + 1) % m] - keep[k] + raw->n) % raw->n) != 1;
        if (bridged && segLen > dist) {
            // Joining two survivors straight across is right for a corner whose
            // inside edge got trimmed, but wrong where the track crosses or
            // touches itself — there the chord would wall off open road.
            i32 cuts = 0;
            for (i32 s = 1; s <= 3 && !cuts; s++) {
                float t = (float)s / 4.0f;
                // Originally `d < r * 0.548`; expressed as a depth so it reads
                // the local half-width rather than one global one.
                float r, depth = roadDepth(a.x + dx * t, a.y + dy * t, &r);
                if (depth > r * 0.4523f) cuts = 1;
            }
            if (cuts) { ndrop = 0; continue; }
        }
        i32 seg = cpOfSample[a.ci];
        Wall *last = outw->n > 0 ? &outw->p[outw->n - 1] : 0;
        if (last && last->x2 == a.x && last->y2 == a.y && last->seg == seg) {
            float merged = hypotf_(b.x - last->x1, b.y - last->y1);
            if (merged <= TRACK_WALL_MAXLEN && ndrop < MAX_DROPPED) {
                i32 ok = 1;
                for (i32 d = 0; d < ndrop && ok; d++) {
                    if (_pointSegDist2(dropped[d].x, dropped[d].y, last->x1, last->y1, b.x, b.y) > sag2) ok = 0;
                }
                if (ok && _pointSegDist2(a.x, a.y, last->x1, last->y1, b.x, b.y) <= sag2) {
                    dropped[ndrop].x = a.x; dropped[ndrop].y = a.y; ndrop++;
                    last->x2 = b.x; last->y2 = b.y;
                    continue;
                }
            }
        }
        ndrop = 0;
        wl_push(outw, a.x, a.y, b.x, b.y, seg);
    }
    #undef MAX_DROPPED
}

// ---------------------------------------------------------------------------
// Wall lookup: for each checkpoint, the walls a car sitting there could
// possibly touch or see.
//
// Built once per track load and stored PRE-GATHERED and struct-of-arrays: the
// walls of a bucket sit contiguously, duplicated across the buckets that share
// them, rather than as indices into the master wall list. Duplication costs a
// few hundred KB on a big track and buys a per-step loop that streams straight
// down memory instead of chasing an index vector — which is what makes both the
// broad-phase cull and the vectorised raycast below worth anything.
//
// bmx/bmy/br2 are the broad phase: a wall's midpoint and the squared radius
// (SENSOR_LEN + half its length)^2 beyond which no ray from a car at that
// distance can possibly reach it.
// ---------------------------------------------------------------------------
// Defined with the rest of the config, below; the bucket window needs it here.
float sensor_len(void);

static i32   *wbs_start;
static float *b_x1, *b_y1, *b_dx, *b_dy, *b_mx, *b_my, *b_r2, *b_hl;

static void buildWallBuckets(void) {
    i32 segs = tk_cp_n;
    if (segs <= 0) { wbs_start = 0; return; }
    // How far back and forward a car must be able to see walls: its sensor
    // reach plus its own travel and some slack. Both follow sensor_len(), so
    // raising Max Speed widens the window to match the longer rays — otherwise
    // the extra reach would look straight through walls that were never put in
    // the bucket. set_config is always called before track_build, and the JS
    // side rebuilds the track when Max Speed changes, which is what keeps
    // these in step.
    const float WALL_BACK_PX = sensor_len() + 60.0f;
    const float WALL_FWD_PX  = sensor_len() + 80.0f;

    // The window is measured in PIXELS OF TRACK, walked gate by gate, rather
    // than as a gate count derived from an assumed spacing.
    //
    // Gate spacing is not uniform — corner gates are anchored to the apex of
    // each turn, so they sit wherever the corners are. Converting a pixel reach
    // into a gate count needs some single spacing to divide by, and there isn't
    // one: divide by the average and the tight clusters under-cover, which is
    // precisely the stale-bucket bug that let cars through walls; divide by the
    // tightest gap and every bucket on the track inflates to suit one outlier,
    // roughly doubling the work the per-step cull does. Walking the actual
    // distances gives each gate exactly the window it needs.
    float *gap = (float *)arena_alloc((u32)segs * 4);
    i32 *nBack = (i32 *)arena_alloc((u32)segs * 4);
    i32 *nFwd  = (i32 *)arena_alloc((u32)segs * 4);
    if (!gap || !nBack || !nFwd) { wbs_start = 0; return; }
    for (i32 i = 0; i < segs; i++) {
        i32 j = (i + 1) % segs;
        gap[i] = hypotf_(tk_cps[j].cx - tk_cps[i].cx, tk_cps[j].cy - tk_cps[i].cy);
        if (gap[i] < 0.5f) gap[i] = 0.5f;
    }
    for (i32 i = 0; i < segs; i++) {
        float d = 0.0f; i32 k = 0;
        while (k < segs - 1 && d < WALL_BACK_PX) { d += gap[((i - k - 1) % segs + segs) % segs]; k++; }
        nBack[i] = maxi(4, k);
        d = 0.0f; k = 0;
        while (k < segs - 1 && d < WALL_FWD_PX) { d += gap[(i + k) % segs]; k++; }
        nFwd[i] = maxi(5, k);
        if (nBack[i] > segs - 1) nBack[i] = segs - 1;
        if (nFwd[i] > segs - 1) nFwd[i] = segs - 1;
    }

    i32 *cnt = (i32 *)arena_alloc((u32)(segs + 1) * 4);
    i32 *bstart = (i32 *)arena_alloc((u32)(segs + 1) * 4);
    if (!cnt || !bstart) { wbs_start = 0; return; }
    for (i32 i = 0; i <= segs; i++) cnt[i] = 0;

    i32 nUndef = 0;
    for (i32 i = 0; i < tk_wall_n; i++) {
        if (tk_walls[i].seg < 0 || tk_walls[i].seg >= segs) nUndef++;
        else cnt[tk_walls[i].seg]++;
    }
    bstart[0] = 0;
    for (i32 i = 0; i < segs; i++) bstart[i + 1] = bstart[i] + cnt[i];
    i32 *bitems = (i32 *)arena_alloc((u32)(tk_wall_n > 0 ? tk_wall_n : 1) * 4);
    i32 *undef = (i32 *)arena_alloc((u32)(nUndef > 0 ? nUndef : 1) * 4);
    if (!bitems || !undef) { wbs_start = 0; return; }
    for (i32 i = 0; i <= segs; i++) cnt[i] = 0;
    i32 u = 0;
    for (i32 i = 0; i < tk_wall_n; i++) {
        i32 s = tk_walls[i].seg;
        if (s < 0 || s >= segs) undef[u++] = i;
        else bitems[bstart[s] + cnt[s]++] = i;
    }

    // Each wall belongs to exactly one bucket, so a window gathers no
    // duplicates and the JS version's per-segment Set isn't needed.
    //
    // Buckets are padded out to a multiple of four with degenerate zero-length
    // walls. Those make the intersection test's denominator exactly zero, which
    // it already rejects, so the vector path can run four lanes at a time off
    // the end of a bucket without a scalar tail or a single branch.
    wbs_start = (i32 *)arena_alloc((u32)(segs + 1) * 4);
    if (!wbs_start) return;
    i32 total = 0;
    for (i32 i = 0; i < segs; i++) {
        wbs_start[i] = total;
        i32 n = nUndef;
        for (i32 j = -nBack[i]; j <= nFwd[i]; j++) {
            i32 seg = ((i + j) % segs + segs) % segs;
            n += cnt[seg];
        }
        total += (n + 3) & ~3;
    }
    wbs_start[segs] = total;
    if (total <= 0) total = 4;

    u32 bytes = (u32)total * 4;
    b_x1 = (float *)arena_alloc(bytes); b_y1 = (float *)arena_alloc(bytes);
    b_dx = (float *)arena_alloc(bytes); b_dy = (float *)arena_alloc(bytes);
    b_mx = (float *)arena_alloc(bytes); b_my = (float *)arena_alloc(bytes);
    b_r2 = (float *)arena_alloc(bytes); b_hl = (float *)arena_alloc(bytes);
    if (!b_x1 || !b_y1 || !b_dx || !b_dy || !b_mx || !b_my || !b_r2 || !b_hl) { wbs_start = 0; return; }

    i32 w = 0;
    for (i32 i = 0; i < segs; i++) {
        i32 base = w;
        for (i32 j = 0; j < nUndef; j++) {
            Wall *wl = &tk_walls[undef[j]];
            b_x1[w] = wl->x1; b_y1[w] = wl->y1;
            b_dx[w] = wl->x2 - wl->x1; b_dy[w] = wl->y2 - wl->y1;
            w++;
        }
        for (i32 j = -nBack[i]; j <= nFwd[i]; j++) {
            i32 seg = ((i + j) % segs + segs) % segs;
            for (i32 k = 0; k < cnt[seg]; k++) {
                Wall *wl = &tk_walls[bitems[bstart[seg] + k]];
                b_x1[w] = wl->x1; b_y1[w] = wl->y1;
                b_dx[w] = wl->x2 - wl->x1; b_dy[w] = wl->y2 - wl->y1;
                w++;
            }
        }
        while ((w - base) & 3) { b_x1[w] = 0.0f; b_y1[w] = 0.0f; b_dx[w] = 0.0f; b_dy[w] = 0.0f; w++; }
        for (i32 k = base; k < w; k++) {
            b_mx[k] = b_x1[k] + b_dx[k] * 0.5f;
            b_my[k] = b_y1[k] + b_dy[k] * 0.5f;
            float hl = hypotf_(b_dx[k], b_dy[k]) * 0.5f;
            b_hl[k] = hl;
            float r = sensor_len() + hl;
            b_r2[k] = r * r;
        }
    }
}

// Input staging. JS writes the path and zone list straight into these before
// calling track_build. They are static rather than arena-allocated on purpose:
// track_build's first act is to reset the arena, which would land on top of its
// own input if the caller had staged it there.
#define MAX_PATH_PTS 4096
#define MAX_ZONES    256
static float path_in[MAX_PATH_PTS * 4];
static float zone_in[MAX_ZONES * 5];

__attribute__((export_name("path_in_ptr"))) i32 path_in_ptr(void) { return (i32)(unsigned long)path_in; }
__attribute__((export_name("zone_in_ptr"))) i32 zone_in_ptr(void) { return (i32)(unsigned long)zone_in; }
__attribute__((export_name("max_path_pts"))) i32 max_path_pts(void) { return MAX_PATH_PTS; }
__attribute__((export_name("max_zones"))) i32 max_zones_(void) { return MAX_ZONES; }

// ---------------------------------------------------------------------------
// track_build — the whole generator. `path` is n quadruples of
// {x, y, type (0 rounded / 1 corner), radius}; `zones` is nz quintuples of
// {x, y, radius, type, killTimer}. Deterministic, so every worker rebuilding
// from the same path gets byte-identical geometry and the population stays in
// sync without shipping any of it across postMessage.
// ---------------------------------------------------------------------------
__attribute__((export_name("track_build")))
i32 track_build(const float *path, i32 n_in, float width,
                i32 has_start, float sx, float sy,
                i32 has_angle, float sang,
                const float *zones, i32 nz,
                i32 auto_width, float auto_blend) {
    arena_reset();
    tk_center = 0; tk_center_n = 0; tk_walls = 0; tk_wall_n = 0;
    tk_cps = 0; tk_cp_n = 0; tk_zones = 0; tk_zone_n = 0;
    wbs_start = 0; grid_items = 0; tk_w = 0;
    tk_width = width; tk_w_max = width;
    tk_start_x = 100.0f; tk_start_y = 100.0f; tk_start_angle = 0.0f;

    tk_zone_n = nz;
    if (nz > 0) {
        tk_zones = (Zone *)arena_alloc((u32)nz * sizeof(Zone));
        if (!tk_zones) { tk_zone_n = 0; }
        else for (i32 i = 0; i < nz; i++) {
            tk_zones[i].x = zones[i * 5]; tk_zones[i].y = zones[i * 5 + 1];
            tk_zones[i].radius = zones[i * 5 + 2];
            tk_zones[i].type = (i32)zones[i * 5 + 3];
            tk_zones[i].killTimer = zones[i * 5 + 4];
        }
    }
    if (n_in < 3) return 0;

    // Drop coincident points, and un-close an explicitly-closed path.
    float *path2 = (float *)arena_alloc((u32)n_in * 16);
    if (!path2) return 0;
    i32 n = 0;
    for (i32 i = 0; i < n_in; i++) {
        if (n > 0 && hypotf_(path[i * 4] - path2[(n - 1) * 4], path[i * 4 + 1] - path2[(n - 1) * 4 + 1]) <= 1.0f) continue;
        path2[n * 4] = path[i * 4]; path2[n * 4 + 1] = path[i * 4 + 1];
        path2[n * 4 + 2] = path[i * 4 + 2]; path2[n * 4 + 3] = path[i * 4 + 3];
        n++;
    }
    if (n > 2 && hypotf_(path2[0] - path2[(n - 1) * 4], path2[1] - path2[(n - 1) * 4 + 1]) < 5.0f) n--;
    if (n < 3) return 0;

    float dist = maxf(4.0f, width);
    VecList cl;
    // The centreline is built at the requested width and never changes: auto
    // width only decides how far the asphalt reaches either side of it. That
    // ordering is what makes the feature cheap — the corner radii were already
    // solved for the full width, so narrowing can only ever add clearance.
    Vec2 *apex = (Vec2 *)arena_alloc((u32)(n + 2) * sizeof(Vec2));
    i32 nApex = 0;
    if (!buildCentreline(path2, n, dist, &cl, apex, &nApex, n + 2) || cl.n < 3) return 0;
    tk_center = cl.p; tk_center_n = cl.n;

    i32 len = tk_center_n;

    // --- per-sample width ---
    tk_w = (float *)arena_alloc((u32)len * 4);
    if (!tk_w) return 0;
    if (auto_width) {
        float *segLen = (float *)arena_alloc((u32)len * 4);
        float *clear = (float *)arena_alloc((u32)len * 4);
        if (!segLen || !clear) { auto_width = 0; }
        else {
            for (i32 i = 0; i < len; i++) {
                Vec2 a = tk_center[i], b = tk_center[(i + 1) % len];
                segLen[i] = hypotf_(b.x - a.x, b.y - a.y);
            }
            computeClearance(len, dist, segLen, clear);
            computeWidths(len, dist, 1, clampf(auto_blend, 0.0f, 1.0f), clear, segLen);
        }
    }
    if (!auto_width) computeWidths(len, dist, 0, 0.0f, 0, 0);

    // The grid backs the inside-the-road test, so its cells have to be at least
    // as big as the widest the road ever gets.
    buildCentreGrid(tk_center, len, maxf(tk_w_max, 16.0f));

    // --- checkpoint gates ---
    //
    // Two rules put a gate down: the regular spacing along the centreline, and
    // the apex of every corner. The corner gates are what let progress be
    // measured through a turn rather than only on the straights either side of
    // it — on a long sweeper the nearest regular gate can be most of the way
    // round the bend.
    i32 *cpOfSample = (i32 *)arena_alloc((u32)len * 4);
    Checkpoint *cps = (Checkpoint *)arena_alloc((u32)(len + 2) * sizeof(Checkpoint));
    i32 *isApex = (i32 *)arena_alloc((u32)len * 4);
    if (!cpOfSample || !cps || !isApex) return 0;
    for (i32 i = 0; i < len; i++) isApex[i] = 0;

    // Each recorded apex position lands on whichever final sample is nearest to
    // it. The resample above renumbered everything, so this is the only way back.
    for (i32 a = 0; a < nApex; a++) {
        i32 best = 0; float bestD = 1.0e30f;
        for (i32 i = 0; i < len; i++) {
            float dx = tk_center[i].x - apex[a].x, dy = tk_center[i].y - apex[a].y;
            float d = dx * dx + dy * dy;
            if (d < bestD) { bestD = d; best = i; }
        }
        isApex[best] = 1;
    }

    // Corners are ANCHORS; the regular gates fill in between them.
    //
    // Laying gates down at a fixed spacing and then trying to squeeze an extra
    // one in at each apex does not work: wherever a regular gate happens to
    // fall just short of a corner, the apex gate is too close to keep and the
    // corner silently goes without one. On a square that lost half of them.
    // Anchoring on the corners first and dividing each run between them into
    // equal steps gives every corner a gate by construction, and keeps the
    // spacing even either side of it.
    float *cum = (float *)arena_alloc((u32)len * 4);
    i32 *anchor = (i32 *)arena_alloc((u32)(len + 2) * 4);
    i32 *gateAt = (i32 *)arena_alloc((u32)len * 4);
    i32 *gateApex = (i32 *)arena_alloc((u32)len * 4);
    if (!cum || !anchor || !gateAt || !gateApex) return 0;
    for (i32 i = 0; i < len; i++) { gateAt[i] = 0; gateApex[i] = 0; }

    cum[0] = 0.0f;
    for (i32 i = 1; i < len; i++)
        cum[i] = cum[i-1] + hypotf_(tk_center[i].x - tk_center[i-1].x, tk_center[i].y - tk_center[i-1].y);
    float total = cum[len-1] + hypotf_(tk_center[0].x - tk_center[len-1].x, tk_center[0].y - tk_center[len-1].y);
    if (total < 1.0f) total = 1.0f;
    tk_len = total;

    // Anchors, in order, thinned so two never sit closer than the minimum gap.
    // A densely traced import can put several arcs within a few pixels, and
    // crowded gates cost more than the extra precision is worth.
    i32 na = 0;
    for (i32 i = 0; i < len; i++) {
        if (!isApex[i]) continue;
        if (na > 0 && cum[i] - cum[anchor[na-1]] < TRACK_CP_MIN_GAP) continue;
        anchor[na++] = i;
    }
    // The wrap-around pair needs the same clearance as every other.
    if (na > 1 && (total - cum[anchor[na-1]] + cum[anchor[0]]) < TRACK_CP_MIN_GAP) na--;
    // No corner worth anchoring (a path of near-straight vertices): fall back
    // to plain even spacing from the first sample.
    if (na == 0) { anchor[0] = 0; na = 1; }

    for (i32 a = 0; a < na; a++) {
        i32 ia = anchor[a];
        gateAt[ia] = 1;
        gateApex[ia] = isApex[ia] ? 1 : 0;

        i32 ib = anchor[(a + 1) % na];
        float L = (na == 1) ? total
                            : (ib > ia ? cum[ib] - cum[ia] : total - cum[ia] + cum[ib]);
        // Round rather than ceil: ceil biases every run short, so a run barely
        // over the spacing gets split into two cramped halves.
        i32 steps = (i32)(L / TRACK_CP_SPACING + 0.5f);
        if (steps < 1) steps = 1;

        for (i32 k = 1; k < steps; k++) {
            // The run from the last anchor back to the first wraps past the end
            // of the sample list, so the target distance wraps with it.
            float target = cum[ia] + L * ((float)k / (float)steps);
            if (target >= total) target -= total;
            i32 at = sampleAtDistance(cum, len, target);
            // Nudge forward off an occupied sample rather than giving up on the
            // gate. Samples sit up to maxSeg apart and gates about
            // TRACK_CP_SPACING apart, which are close enough that two targets
            // land on the same sample fairly often; dropping one then leaves a
            // double-width gap exactly where the spacing was meant to be even.
            for (i32 t = 0; t < 3 && gateAt[at]; t++) at = (at + 1) % len;
            if (!gateAt[at]) gateAt[at] = 1;
        }
    }

    i32 ncp = 0;
    float sinceGate = 0.0f;
    tk_cp_min_step = TRACK_CP_SPACING;
    i32 firstGate = -1;
    for (i32 i = 0; i < len; i++) {
        if (i > 0) sinceGate += hypotf_(tk_center[i].x - tk_center[i-1].x, tk_center[i].y - tk_center[i-1].y);
        if (gateAt[i]) {
            Vec2 prev = tk_center[(i - 1 + len) % len], next = tk_center[(i + 1) % len], c = tk_center[i];
            float tx = next.x - prev.x, ty = next.y - prev.y;
            float tl = hypotf_(tx, ty); if (tl == 0.0f) tl = 1.0f;
            tx /= tl; ty /= tl;
            // The gate spans the road as it is *here*, so a narrowed stretch
            // gets a narrowed gate rather than one poking through the barrier.
            float hw_ = tk_w[i];
            cps[ncp].p1x = c.x - ty * hw_; cps[ncp].p1y = c.y + tx * hw_;
            cps[ncp].p2x = c.x + ty * hw_; cps[ncp].p2y = c.y - tx * hw_;
            cps[ncp].cx = c.x; cps[ncp].cy = c.y;
            cps[ncp].apex = gateApex[i];
            if (ncp > 0 && sinceGate < tk_cp_min_step) tk_cp_min_step = sinceGate;
            if (firstGate < 0) firstGate = i;
            sinceGate = 0.0f;
            ncp++;
        }
        cpOfSample[i] = ncp - 1;
    }
    // The gap that closes the loop counts too.
    if (ncp > 1 && firstGate >= 0) {
        float wrap = sinceGate + cum[firstGate];
        if (wrap < tk_cp_min_step) tk_cp_min_step = wrap;
    }
    if (tk_cp_min_step < 1.0f) tk_cp_min_step = 1.0f;
    // Samples before the first gate belong to the last one, going round.
    for (i32 i = 0; i < len && cpOfSample[i] < 0; i++) cpOfSample[i] = ncp - 1;
    tk_cps = cps; tk_cp_n = ncp;

    // --- barriers: the boundary of the stroked road, minus any fold-back ---
    WallList wl; wl.p = 0; wl.n = 0; wl.cap = 0;
    for (i32 s = 0; s < 2; s++) {
        i32 side = s == 0 ? 1 : -1;
        OffList raw;
        offsetOutline(tk_center, len, side, &raw);
        trimOutlineToWalls(&raw, dist, cpOfSample, &wl);
    }
    // Walls long enough to be a bridging chord belong to no one checkpoint, so
    // leave them unbucketed and they get checked everywhere.
    for (i32 i = 0; i < wl.n; i++) {
        if (hypotf_(wl.p[i].x2 - wl.p[i].x1, wl.p[i].y2 - wl.p[i].y1) > TRACK_WALL_MAXLEN + 10.0f) wl.p[i].seg = -1;
    }
    tk_walls = wl.p; tk_wall_n = wl.n;

    // --- start line ---
    tk_start_x = has_start ? sx : floorf_(tk_center[0].x + 0.5f);
    tk_start_y = has_start ? sy : floorf_(tk_center[0].y + 0.5f);
    // A start point left outside the asphalt spawns the whole field into a
    // wall; pull it back onto the nearest bit of centreline.
    // Originally `d > r * 0.9`, i.e. within a tenth of the edge; as a depth so
    // it reads the local half-width.
    if (has_start && roadDepth(tk_start_x, tk_start_y, 0) < tk_w[0] * 0.1f) {
        i32 best = 0; float bestD = 1.0e30f;
        for (i32 i = 0; i < len; i++) {
            float dx = tk_center[i].x - tk_start_x, dy = tk_center[i].y - tk_start_y;
            float d = dx * dx + dy * dy;
            if (d < bestD) { bestD = d; best = i; }
        }
        tk_start_x = floorf_(tk_center[best].x + 0.5f);
        tk_start_y = floorf_(tk_center[best].y + 0.5f);
    }
    // Default heading: the direction the track runs AT THE START LINE.
    //
    // This used to read the tangent at centreline sample 0 regardless of where
    // the start actually was. On a track whose start had been dragged to the
    // far side of the loop that is close to a reversed heading, so the whole
    // field spawned pointing backwards down the road and drove straight into
    // the barrier behind them. Same family of bug as aiming them at checkpoint
    // 1 wherever the start was: a start that isn't sample 0 was simply not
    // considered.
    if (has_angle) {
        tk_start_angle = sang;
    } else {
        i32 si = 0; float sd = 1.0e30f;
        for (i32 i = 0; i < len; i++) {
            float dx = tk_center[i].x - tk_start_x, dy = tk_center[i].y - tk_start_y;
            float d = dx * dx + dy * dy;
            if (d < sd) { sd = d; si = i; }
        }
        Vec2 a = tk_center[(si - 1 + len) % len], b = tk_center[(si + 1) % len];
        tk_start_angle = atan2f_(b.y - a.y, b.x - a.x);
    }

    // Which checkpoint the start line actually sits on.
    //
    // Cars used to be handed nextCheckpointIndex = 1 no matter where the start
    // was. On a track whose start had been dragged elsewhere that aimed them at
    // a checkpoint most of a lap away — and since the wall lookup is keyed on
    // the checkpoint a car is heading for, it also handed them the walls from
    // the wrong part of the track entirely.
    tk_start_cp = 0;
    {
        float best = 1.0e30f;
        for (i32 i = 0; i < tk_cp_n; i++) {
            float dx = tk_cps[i].cx - tk_start_x, dy = tk_cps[i].cy - tk_start_y;
            float d = dx * dx + dy * dy;
            if (d < best) { best = d; tk_start_cp = i; }
        }
    }

    buildWallBuckets();
    return 1;
}

// Geometry accessors — JS wraps these in typed-array views over the module's
// memory, so nothing is copied out unless the caller actually asks for it.
__attribute__((export_name("track_centerline_ptr"))) i32 track_centerline_ptr(void) { return (i32)(unsigned long)tk_center; }
// One half-width per centreline sample — what the canvas needs to stroke a road
// that changes width along its length.
__attribute__((export_name("track_widths_ptr"))) i32 track_widths_ptr(void) { return (i32)(unsigned long)tk_w; }
__attribute__((export_name("track_centerline_count"))) i32 track_centerline_count(void) { return tk_center_n; }
__attribute__((export_name("track_walls_ptr"))) i32 track_walls_ptr(void) { return (i32)(unsigned long)tk_walls; }
__attribute__((export_name("track_wall_count"))) i32 track_wall_count(void) { return tk_wall_n; }
__attribute__((export_name("track_cps_ptr"))) i32 track_cps_ptr(void) { return (i32)(unsigned long)tk_cps; }
__attribute__((export_name("track_cp_count"))) i32 track_cp_count(void) { return tk_cp_n; }
__attribute__((export_name("track_start_x"))) float track_start_x(void) { return tk_start_x; }
__attribute__((export_name("track_start_y"))) float track_start_y(void) { return tk_start_y; }
__attribute__((export_name("track_start_angle"))) float track_start_angle(void) { return tk_start_angle; }
__attribute__((export_name("track_start_cp"))) i32 track_start_cp(void) { return tk_start_cp; }

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static float cfg_maxSpeed = 10.0f, cfg_accel = 0.05f, cfg_turnSpeed = 0.02f, cfg_brakeStrength = 0.05f;
static i32   cfg_initialTTL = 750, cfg_targetLaps = 3;
// Fraction of the population spawned each generation as mutated clones of the
// stash, reward-boosted specifically through whichever stretch of track the
// stash is currently slowest on (see evolve()/inFocusWindow() below). Not a
// probability like the old mutation-rate knob was — that one is gone; see the
// comment on MUT_SIGMA_* near evolve().
static float cfg_focusPct = 0.20f;
static i32   cfg_hidden = 5;

// Lateral grip used to cancel sideways slip and keep the car's velocity
// tracking its heading (see updateCar). It used to be a user-facing slider,
// but its usable range (0.8-0.99) canceled 80-99% of any sideways velocity
// every single frame either way — a few frames in, the two ends of the slider
// left the car in the same place. Fixed rather than exposed, so nothing about
// turning or braking below depends on a knob that never did anything.
static const float CAR_LAT_GRIP = 0.93f;

// Sensor reach for the current top speed. Everything that needs it — the
// per-frame raycasts, the wall-bucket window the track is built with, and the
// sensor overlay the UI draws — reads it from here so the three can never
// disagree. Exported because script.js draws the rays and must use the same
// number the simulation raycast used.
// Recomputed by set_config rather than per call: updateCar wants it once per
// car per frame and buildWallBuckets once per wall, and it is a multiply and
// two compares off a config value that changes a handful of times a session.
static float cfg_sensorLen = SENSOR_LEN_MIN;
__attribute__((export_name("sensor_len")))
float sensor_len(void) { return cfg_sensorLen; }
static void recompute_sensor_len(void) {
    float r = cfg_maxSpeed * SENSOR_LOOKAHEAD_FRAMES;
    cfg_sensorLen = r < SENSOR_LEN_MIN ? SENSOR_LEN_MIN : (r > SENSOR_LEN_MAX ? SENSOR_LEN_MAX : r);
}

__attribute__((export_name("set_config")))
void set_config(float maxSpeed, float accel, float turnSpeed, float brakeStrength,
                i32 initialTTL, i32 targetLaps, float focusPct, i32 hidden) {
    cfg_maxSpeed = maxSpeed; cfg_accel = accel; cfg_turnSpeed = turnSpeed; cfg_brakeStrength = brakeStrength;
    cfg_initialTTL = initialTTL; cfg_targetLaps = targetLaps;
    cfg_focusPct = focusPct;
    cfg_hidden = hidden < 1 ? 1 : (hidden > MAX_HIDDEN ? MAX_HIDDEN : hidden);
    recompute_sensor_len();
}

// Absolute checkpoint-index window the focused sub-population's reward is
// boosted inside — see evolve(). -1 means inactive (no stash yet, or the
// window would cover the whole lap). Set on the master by evolve() and
// shipped to every worker instance each generation, exactly like the brains.
static i32 focus_lo = -1, focus_hi = -1;
__attribute__((export_name("focus_lo"))) i32 focus_lo_(void) { return focus_lo; }
__attribute__((export_name("focus_hi"))) i32 focus_hi_(void) { return focus_hi; }
__attribute__((export_name("set_focus_window")))
void set_focus_window(i32 lo, i32 hi) { focus_lo = lo; focus_hi = hi; }

// Per-gate pace of car 0's run this generation, as idealCp/dtCp (>1 means
// faster than the track's "ideal" pace at Max Speed, <1 slower; -1 = not
// reached this run). Car 0 mirrors the stash exactly whenever there is one —
// same deterministic track, same weights, same starting state — so this is
// effectively free telemetry on the current all-time-best brain, gathered by
// the ordinary run instead of a dedicated evaluation pass. Populated in
// updateCar, read by evolve() (after the master's copy is refreshed from
// worker 0 in engine.js — the master itself never simulates a car).
static float gate_ratio[MAX_GATES];
__attribute__((export_name("gate_ratio_ptr"))) i32 gate_ratio_ptr(void) { return (i32)(unsigned long)gate_ratio; }
__attribute__((export_name("max_gates"))) i32 max_gates(void) { return MAX_GATES; }
static void reset_gate_ratio(void) { for (i32 g = 0; g < MAX_GATES; g++) gate_ratio[g] = -1.0f; }

// How many cars, THIS GENERATION, died while heading for gate g — the other
// half of what evolve() needs to aim the focus window, and the half
// gate_ratio structurally cannot provide: a gate a car never reaches leaves
// no ratio behind (it stays at its -1 reset value), so a corner the whole
// population dies at is invisible to gate_ratio no matter how often it kills
// them. This is population-wide (every worker's every car, not just car 0 —
// there is no "the stash" worth mirroring yet in the phase this is for) and
// summed across workers by engine.js before evolve() reads it, the same way
// gate_ratio is refreshed into master memory first, just added instead of
// copied. Reset with gate_ratio, once per generation.
static i32 crash_count[MAX_GATES];
__attribute__((export_name("crash_count_ptr"))) i32 crash_count_ptr(void) { return (i32)(unsigned long)crash_count; }
static void reset_crash_count(void) { for (i32 g = 0; g < MAX_GATES; g++) crash_count[g] = 0; }
// recordCrash() itself lives further down, by run() — it reads car_nextCP,
// which is not declared until the population arrays below.

static inline i32 inFocusWindow(i32 g) {
    if (focus_lo < 0) return 0;
    if (focus_lo <= focus_hi) return g >= focus_lo && g <= focus_hi;
    return g >= focus_lo || g <= focus_hi;   // window wraps past gate 0
}

// How much harder the focused sub-population's reward hits inside its
// window. It is a multiplier on top of the ordinary reward, never a
// replacement for it — a focused car still scores normally everywhere else,
// so nothing stops it finishing the rest of the lap while it explores the
// one stretch selection is currently prioritising there.
static const float FOCUS_BOOST = 3.0f;

// ---------------------------------------------------------------------------
// Population. Struct-of-arrays; nothing here is ever allocated per generation.
// ---------------------------------------------------------------------------
static float car_x[MAX_CARS], car_y[MAX_CARS], car_angle[MAX_CARS];
static float car_vx[MAX_CARS], car_vy[MAX_CARS], car_speed[MAX_CARS], car_fitness[MAX_CARS];
static i32   car_crashed[MAX_CARS], car_ttl[MAX_CARS], car_frames[MAX_CARS];
static i32   car_nextCP[MAX_CARS], car_laps[MAX_CARS], car_cpReached[MAX_CARS];
static float car_lastLap[MAX_CARS]; static i32 car_prevLapFrame[MAX_CARS];
static i32   car_lastCpFrame[MAX_CARS];   // for scoring how fast each gate was reached
// Which centreline segment claimed this car last frame — see insideRoad.
static i32   car_roadSeg[MAX_CARS];
// Set by evolve() (on the master) for the slots it breeds as focused clones,
// then shipped to each worker alongside its slice of the brains every
// generation — reset_car() below never touches it, only evolve() does.
static i32   car_focused[MAX_CARS];
static float car_out[MAX_CARS * OUT_N];
static float car_in[MAX_CARS * IN_N];
// Rounded up to a whole number of vectors and aligned, because feedForward
// writes the hidden layer four units at a time.
static float hidden_scratch[(MAX_HIDDEN + 3) & ~3] __attribute__((aligned(16)));

__attribute__((export_name("car_focused_ptr")))
i32 car_focused_ptr(void) { return (i32)(unsigned long)car_focused; }

// Brains, flat and contiguous: [weightsIH (IN_N*h) | weightsHO (h*OUT_N) | biasH (h) | biasO (OUT_N)]
// per car, one car after another. Same row-major order the JS edition uses, so
// the JSON import/export code needs no changes.
//
// One slot past the live population is the stash: the all-time best brain lives
// there so evolve() can clone from it without it being bred over, and "Load AI"
// parks an imported brain there before seeding.
#define STASH_SLOT MAX_CARS
static float brains_bufA[(MAX_CARS + 1) * BRAIN_MAX];
static float brains_bufB[(MAX_CARS + 1) * BRAIN_MAX];
static float *brains = brains_bufA;
static float *brains_next = brains_bufB;
static float render_buf[MAX_CARS * RENDER_STRIDE];
static float fitness_buf[MAX_CARS * 5];

static i32 pop_n = 0, pop_id_offset = 0, brain_stride_v = 0;
// The cars still driving, compacted. A generation starts with every car in
// here and empties it as they crash; the alternative — testing car_crashed on
// every slot on every iteration — reads the whole population to find the
// handful still moving, and a generation spends most of its steps in exactly
// that state. At 500 cars and a 2500-step hyper chunk that was over a million
// pointless loads and branches per chunk.
static i32 active[MAX_CARS];
static i32 active_n = 0;
static void rebuild_active(void) {
    active_n = 0;
    for (i32 i = 0; i < pop_n; i++) if (!car_crashed[i]) active[active_n++] = i;
}

__attribute__((export_name("brain_stride")))
i32 brain_stride(void) { return brain_stride_v; }
__attribute__((export_name("brains_ptr")))
i32 brains_ptr(void) { return (i32)(unsigned long)brains; }
__attribute__((export_name("render_ptr")))
i32 render_ptr(void) { return (i32)(unsigned long)render_buf; }
__attribute__((export_name("render_stride")))
i32 render_stride(void) { return RENDER_STRIDE; }
__attribute__((export_name("fitness_ptr")))
i32 fitness_ptr(void) { return (i32)(unsigned long)fitness_buf; }
__attribute__((export_name("max_cars")))
i32 max_cars(void) { return MAX_CARS; }
__attribute__((export_name("max_hidden")))
i32 max_hidden(void) { return MAX_HIDDEN; }
__attribute__((export_name("stash_slot")))
i32 stash_slot(void) { return STASH_SLOT; }

static i32 stride_for(i32 h) { return IN_N * h + h * OUT_N + h + OUT_N; }

// Reset a car to the start line without touching its brain.
static void reset_car(i32 i) {
    car_x[i] = tk_start_x; car_y[i] = tk_start_y; car_angle[i] = tk_start_angle;
    car_vx[i] = 0.0f; car_vy[i] = 0.0f; car_speed[i] = 0.0f;
    car_fitness[i] = 0.0f; car_crashed[i] = 0;
    car_ttl[i] = cfg_initialTTL; car_frames[i] = 0;
    car_nextCP[i] = tk_cp_n > 0 ? (tk_start_cp + 1) % tk_cp_n : 0;
    car_laps[i] = 0; car_cpReached[i] = 0;
    car_lastLap[i] = 0.0f; car_prevLapFrame[i] = 0; car_lastCpFrame[i] = 0;
    car_roadSeg[i] = -1;
    for (i32 k = 0; k < OUT_N; k++) car_out[i * OUT_N + k] = 0.0f;
    for (i32 k = 0; k < IN_N; k++) car_in[i * IN_N + k] = 0.0f;
}

// Allocate the slice this instance owns. `id_offset` is the global id of its
// first car, so render rows carry ids the main thread can index directly.
__attribute__((export_name("pop_init")))
i32 pop_init(i32 count, i32 id_offset, i32 hidden, u32 seed) {
    if (count > MAX_CARS) count = MAX_CARS;
    if (count < 0) count = 0;
    pop_n = count; pop_id_offset = id_offset;
    cfg_hidden = hidden < 1 ? 1 : (hidden > MAX_HIDDEN ? MAX_HIDDEN : hidden);
    brain_stride_v = stride_for(cfg_hidden);
    rng_seed(seed);
    for (i32 i = 0; i < pop_n; i++) { reset_car(i); car_focused[i] = 0; }
    rebuild_active();
    reset_gate_ratio();
    reset_crash_count();
    focus_lo = -1; focus_hi = -1;
    return pop_n;
}

// Fill every brain with fresh uniform [-1,1] weights.
// Fan-in scaled (Xavier) initialisation. Filling every weight from a flat
// [-1,1] pushed a 9-input tanh unit straight into saturation, where it barely
// responds to its inputs at all — so the first few dozen generations were
// spent climbing back out of that rather than learning to drive.
__attribute__((export_name("pop_randomize_brains")))
void pop_randomize_brains(void) {
    i32 h = cfg_hidden, stride = brain_stride_v;
    i32 offHO = IN_N * h, offBH = offHO + h * OUT_N, offBO = offBH + h;
    float aIH = sqrtf_(6.0f / (float)(IN_N + h));
    float aHO = sqrtf_(6.0f / (float)(h + OUT_N));
    for (i32 c = 0; c < pop_n; c++) {
        float *b = &brains[c * stride];
        for (i32 j = 0; j < offHO; j++) b[j] = rnd11() * aIH;
        for (i32 j = offHO; j < offBH; j++) b[j] = rnd11() * aHO;
        // Small rather than zero, so two cars never start out identical.
        for (i32 j = offBH; j < offBO + OUT_N; j++) b[j] = rnd11() * 0.1f;
    }
}

__attribute__((export_name("pop_reset")))
void pop_reset(void) {
    for (i32 i = 0; i < pop_n; i++) reset_car(i);
    rebuild_active();
    // A fresh run per generation, so last generation's per-gate pace can't
    // leak into this one's — car 0 might crash early and never overwrite the
    // entries a slower or luckier previous run left behind. crash_count needs
    // the same fresh start, for the same reason.
    reset_gate_ratio();
    reset_crash_count();
}

// ---------------------------------------------------------------------------
// The step. Everything below runs per car per frame, so it is the only code in
// the project where the shape of the data actually matters.
// ---------------------------------------------------------------------------
// Do two segments cross?
//
// The textbook form divides out the denominator and then checks the two
// parameters against [0,1]. Both divisions are pure waste for a test that only
// wants a yes or no: t = tn/bottom lies in [0,1] exactly when tn lies between
// 0 and bottom, and flipping the signs of everything when bottom is negative
// turns that into one unsigned range check per parameter. A miss — which is
// what almost every call is — now costs no division at all, and a float
// division is an order of magnitude dearer than the multiply it replaces.
static inline i32 fastIntersect(float Ax, float Ay, float Bx, float By,
                                float Cx, float Cy, float Dx, float Dy) {
    float abx = Bx - Ax, aby = By - Ay;
    float cdx = Dx - Cx, cdy = Dy - Cy;
    float bottom = cdy * abx - cdx * aby;
    if (bottom == 0.0f) return 0;
    float acx = Ax - Cx, acy = Ay - Cy;
    float tn = cdx * acy - cdy * acx;
    float un = acy * abx - acx * aby;
    if (bottom < 0.0f) { bottom = -bottom; tn = -tn; un = -un; }
    return tn >= 0.0f && tn <= bottom && un >= 0.0f && un <= bottom;
}
// Per-car broad-phase scratch, refilled every step. `sc_` holds the walls a
// sensor could reach; `nc_` the much smaller set the car could physically touch
// this frame. Both are struct-of-arrays and padded to a multiple of four.
// Per-car broad-phase capacity. A bucket is a SUBSET of one track's walls,
// and the densest track any test or import has produced has ~350 walls in
// total, so this is an order of magnitude of headroom rather than the two
// orders 16384 gave. That mattered because these are eight static arrays
// living in every wasm instance: 512KB per instance x (one per worker plus
// the master) was several megabytes of almost entirely untouched memory on a
// machine with a few cores. The bounds checks below are unchanged.
#define MAX_SCRATCH 4096
// Four floats of slack past the cap: the vectorised gather below writes a full
// 128-bit group and then advances the cursor by however many of its four lanes
// it actually kept, so the last accepted wall can put three dead floats past
// the write position.
#define SCRATCH_SLACK 4
static float sc_x1[MAX_SCRATCH + SCRATCH_SLACK], sc_y1[MAX_SCRATCH + SCRATCH_SLACK];
static float sc_dx[MAX_SCRATCH + SCRATCH_SLACK], sc_dy[MAX_SCRATCH + SCRATCH_SLACK];
static float nc_x1[MAX_SCRATCH + SCRATCH_SLACK], nc_y1[MAX_SCRATCH + SCRATCH_SLACK];
static float nc_dx[MAX_SCRATCH + SCRATCH_SLACK], nc_dy[MAX_SCRATCH + SCRATCH_SLACK];
static i32 sc_n, nc_n;

#ifdef __wasm_simd128__
typedef float v4f __attribute__((vector_size(16)));
typedef int   v4i __attribute__((vector_size(16)));
typedef signed char v16i8 __attribute__((vector_size(16)));
static inline v4f vsplat(float x) { return (v4f){ x, x, x, x }; }
// Unaligned 128-bit load/store. wasm has no alignment requirement on
// v128.load, but casting a float* straight to a v4f* claims an alignment the
// pointer may not have, and a brain's hidden block starts wherever `stride`
// puts it. Going through a packed struct is the portable way to say
// "128 bits, any address".
typedef struct { v4f v; } __attribute__((packed, aligned(1))) v4f_unaligned;
static inline v4f vload(const float *p) { return ((const v4f_unaligned *)p)->v; }
static inline void vstore(float *p, v4f x) { ((v4f_unaligned *)p)->v = x; }
// Branchless lane select: mask lanes are all-ones or all-zeros from a compare.
static inline v4f vsel(v4i m, v4f a, v4f b) { return (v4f)((m & (v4i)a) | (~m & (v4i)b)); }
// Left-packing: given four lanes and a mask saying which to keep, move the
// kept ones to the front so they can be written out as a run. wasm has no
// compress instruction, but it has a full dynamic byte shuffle, and there are
// only sixteen possible masks — so the shuffle pattern is a table lookup and
// the whole compaction is one swizzle and one store per array.
static const v16i8 PACK_LANES[16] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0 },
    { 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 6, 7, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 0, 0, 0, 0 },
    { 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0 },
    { 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
};
static inline void packStore(float *dst, v4f v, v16i8 pattern) {
    vstore(dst, (v4f)__builtin_wasm_swizzle_i8x16((v16i8)v, pattern));
}

// nn_tanh, four at a time, and to the same f32 result: the hidden layer runs
// four lanes wide while the two outputs stay scalar, and a network whose units
// disagreed about their own activation by an ulp depending on where they sat
// in the layer would be a genuinely nasty bug to find. Branch-free, so the
// saturation the scalar version does with an early return is a min/max pair
// here — past +-9.011 the quotient already lands on exactly +-1 in f32.
static inline v4f nn_tanh4(v4f x) {
    const v4f lim = vsplat(9.011f), one = vsplat(1.0f);
    v4f xc = __builtin_wasm_min_f32x4(__builtin_wasm_max_f32x4(x, -lim), lim);
    v4f y = xc * vsplat(NN_LOG2E_X2);
    v4f n = __builtin_wasm_nearest_f32x4(y);
    v4f f = y - n;
    v4f p = vsplat(1.5252733804059840e-5f);
    p = p * f + vsplat(1.5403530393381609e-4f);
    p = p * f + vsplat(1.3333558146428443e-3f);
    p = p * f + vsplat(9.6181291076284772e-3f);
    p = p * f + vsplat(5.5504108664821580e-2f);
    p = p * f + vsplat(2.4022650695910071e-1f);
    p = p * f + vsplat(6.9314718055994531e-1f);
    p = p * f + one;
    v4i sc = ((v4i)__builtin_convertvector(n, v4i) + (v4i){127,127,127,127}) << 23;
    v4f e = p * (v4f)sc;
    return (e - one) / (e + one);
}
#endif

// Nearest hit along a ray, as a fraction of its length, over a contiguous run
// of walls. This is the hot spot of the whole program — seven of these per
// living car per frame — so it is written to have no branches in the loop body
// and, where the target allows it, to do four walls at a time.
static float raycast(float Ax, float Ay, float Bx, float By,
                     const float *x1, const float *y1, const float *dx, const float *dy, i32 n) {
    const float abx = Bx - Ax, aby = By - Ay;
#ifdef __wasm_simd128__
    const v4f vAx = vsplat(Ax), vAy = vsplat(Ay);
    const v4f vabx = vsplat(abx), vaby = vsplat(aby);
    const v4f one = vsplat(1.0f), zero = vsplat(0.0f);
    const v4i signbit = (v4i){ (int)0x80000000, (int)0x80000000, (int)0x80000000, (int)0x80000000 };
    v4f best = one;
    for (i32 i = 0; i < n; i += 4) {
        v4f cx = vload(x1 + i), cy = vload(y1 + i);
        v4f ddx = vload(dx + i), ddy = vload(dy + i);
        v4f bottom = ddy * vabx - ddx * vaby;
        v4f acx = vAx - cx, acy = vAy - cy;
        v4f tn = ddx * acy - ddy * acx;
        v4f un = acy * vabx - acx * vaby;
        // Sign-normalise instead of dividing, the same way the scalar path
        // does: multiplying tn, un and bottom through by sign(bottom) leaves
        // the two range tests unchanged and costs three bit operations. The
        // sign flip is an xor of the denominator's sign bit, and |bottom| is
        // that bit cleared.
        v4i sgn = (v4i)bottom & signbit;
        v4f ab = (v4f)((v4i)bottom & ~signbit);
        tn = (v4f)((v4i)tn ^ sgn);
        un = (v4f)((v4i)un ^ sgn);
        v4i m = (ab != zero) & (tn >= zero) & (tn <= ab) & (un >= zero) & (un <= ab);
        // The division is the single dearest instruction in the loop and four
        // walls out of four miss on the overwhelming majority of iterations,
        // so it is worth a branch to skip it. The branch is about as
        // predictable as a branch gets: a ray crosses two or three walls out
        // of the hundred-odd it is tested against.
        if (__builtin_wasm_any_true_v128((v16i8)m)) {
            v4f cand = vsel(m, tn / ab, one);
            best = __builtin_wasm_min_f32x4(cand, best);
        }
    }
    float b0 = best[0] < best[1] ? best[0] : best[1];
    float b1 = best[2] < best[3] ? best[2] : best[3];
    return b0 < b1 ? b0 : b1;
#else
    float best = 1.0f;
    for (i32 i = 0; i < n; i++) {
        float ddx = dx[i], ddy = dy[i];
        float bottom = ddy * abx - ddx * aby;
        if (bottom == 0.0f) continue;
        float acx = Ax - x1[i], acy = Ay - y1[i];
        float tn = ddx * acy - ddy * acx;
        float un = acy * abx - acx * aby;
        if (bottom < 0.0f) { bottom = -bottom; tn = -tn; un = -un; }
        // Same sign-normalised range test as segmentHits, for the same
        // reason — but here the fraction itself is wanted, so the division
        // survives. It now happens only for the handful of walls a ray
        // actually crosses rather than for every wall it is compared against.
        if (tn < 0.0f || tn > bottom || un < 0.0f || un > bottom) continue;
        float t = tn / bottom;
        if (t < best) best = t;
    }
    return best;
#endif
}

// Does this segment cross any wall in the run? Same math, but it can stop at
// the first hit, so it stays scalar — a crash ends the car's frame anyway.
// Five of these per living car per frame (the travel segment plus the four
// sides of the body), and like fastIntersect above they now contain no
// division whatsoever: the range check is done against the denominator
// instead of dividing by it.
static i32 segmentHits(float Ax, float Ay, float Bx, float By,
                       const float *x1, const float *y1, const float *dx, const float *dy, i32 n) {
    const float abx = Bx - Ax, aby = By - Ay;
    for (i32 i = 0; i < n; i++) {
        float ddx = dx[i], ddy = dy[i];
        float bottom = ddy * abx - ddx * aby;
        if (bottom == 0.0f) continue;
        float acx = Ax - x1[i], acy = Ay - y1[i];
        float tn = ddx * acy - ddy * acx;
        float un = acy * abx - acx * aby;
        if (bottom < 0.0f) { bottom = -bottom; tn = -tn; un = -un; }
        if (tn >= 0.0f && tn <= bottom && un >= 0.0f && un <= bottom) return 1;
    }
    return 0;
}

// The network. Eleven inputs, one hidden layer, two outputs — small enough
// that how the loops walk the weights costs more than the arithmetic does.
//
// weightsIH is stored [input][hidden], so ONE INPUT'S weights to every hidden
// unit are contiguous. The old loop nest had hidden on the outside and input
// on the inside, which walked that array with a stride of h and touched a
// different cache line for every one of the 11 multiply-adds a unit needs.
// Accumulating across the hidden layer instead — a running total per unit,
// one input's row added into all of them at a time — reads every weight in
// order, and on the SIMD build does four units per instruction.
//
// No masking on the tail: the k loop rounds h up to a multiple of four and
// the extra lanes accumulate whatever weights sit after the row. Those lanes
// are never read back (only hidden_scratch[0..h-1] is), and the reads stay
// inside this car's own brain — the furthest is wIH[(IN_N-1)*h + h + 3],
// which is offHO + 3, and the brain runs to offHO + 3h + 2 past that.
static void feedForward(i32 i) {
    const float *in = &car_in[i * IN_N];
    const float *b = &brains[i * brain_stride_v];
    i32 h = cfg_hidden;
    const float *wIH = b;
    const float *wHO = b + IN_N * h;
    const float *bH  = wHO + h * OUT_N;
    const float *bO  = bH + h;
#ifdef __wasm_simd128__
    i32 hv = (h + 3) & ~3;
    for (i32 k = 0; k < hv; k += 4) {
        v4f acc = vload(bH + k);
        for (i32 j = 0; j < IN_N; j++) acc += vsplat(in[j]) * vload(wIH + j * h + k);
        vstore(hidden_scratch + k, nn_tanh4(acc));
    }
#else
    for (i32 k = 0; k < h; k++) hidden_scratch[k] = bH[k];
    for (i32 j = 0; j < IN_N; j++) {
        float v = in[j];
        const float *row = wIH + j * h;
        for (i32 k = 0; k < h; k++) hidden_scratch[k] += v * row[k];
    }
    for (i32 k = 0; k < h; k++) hidden_scratch[k] = nn_tanh(hidden_scratch[k]);
#endif
    // Two outputs, so the same transpose applies in the other direction:
    // weightsHO is stored [hidden][output], and walking it a hidden unit at a
    // time reads the pair for that unit side by side instead of striding down
    // one output column and then the other.
    float s0 = bO[0], s1 = bO[1];
    for (i32 j = 0; j < h; j++) {
        float v = hidden_scratch[j];
        s0 += v * wHO[j * OUT_N];
        s1 += v * wHO[j * OUT_N + 1];
    }
    float *out = &car_out[i * OUT_N];
    out[0] = nn_tanh(s0);
    out[1] = nn_tanh(s1);
}

// Below this a car counts as not moving at all: it cannot steer (no grip to
// turn against without roll) and it is out of the race (see updateCar). One
// threshold rather than two, because it is one idea — "this car has no
// momentum" — and the two rules would be incoherent apart.
static const float STOPPED_SPEED = 0.05f;
// How long a car is allowed to have no momentum before it is eliminated.
// reset_car zeroes car_out, so EVERY car's first frame moves on zero throttle
// whatever its brain would ask for, and a real launch needs a few more frames
// to build measurable speed. Short enough that a car which never commands
// throttle is gone almost immediately instead of idling out its whole TTL.
static const i32 STOPPED_GRACE_FRAMES = 15;
// How long after its last gate a car keeps earning the per-frame speed reward.
// Generous — a car moving at any reasonable pace clears a gate far inside
// this — so it never penalises going fast, it only stops paying a car that has
// stopped making progress.
static const i32 PROGRESS_WINDOW_FRAMES = 180;
// See the steering block inside updateCar for the physics this implements.
static const float TURN_GRIP_REF_SPEED = 3.0f;

static void updateCar(i32 i) {
    car_ttl[i]--; car_frames[i]++;
    if (car_ttl[i] <= 0) { car_crashed[i] = 1; return; }

    float steer = car_out[i * OUT_N];
    float throttle = car_out[i * OUT_N + 1];

    // Turning is grip-limited, not speed-boosted. Holding a steer angle at
    // speed v asks the tires for lateral acceleration proportional to v times
    // the yaw rate, and a tire only has so much of that to give before it
    // slides instead of turning — so the available yaw rate falls off as
    // roughly 1/v once past TURN_GRIP_REF_SPEED. Below STOPPED_SPEED there is
    // no rolling for the tires to grip at all, so a stopped car gets zero
    // authority: turning the wheel does nothing until it is moving, exactly
    // like a real parked car.
    if (car_speed[i] > STOPPED_SPEED) {
        float authority = minf(TURN_GRIP_REF_SPEED / car_speed[i], 1.0f);
        car_angle[i] += steer * cfg_turnSpeed * authority;
    }

    double sd, cd; sincos_d((double)car_angle[i], &sd, &cd);
    float sinA = (float)sd, cosA = (float)cd;
    float vx = car_vx[i], vy = car_vy[i];

    if (throttle > 0.0f) {
        vx += cosA * throttle * cfg_accel; vy += sinA * throttle * cfg_accel;
    } else if (throttle < 0.0f) {
        // Braking scales with how hard the pedal is pressed, not a flat
        // snap — a throttle of -0.05 should barely touch the speedometer, and
        // -1.0 should haul the car down hard. The old code multiplied speed
        // by a flat 0.95 for ANY non-positive throttle, so -0.01 and -1.0
        // braked identically hard and looked like an instant stop either way.
        float sp = sqrtf_(vx * vx + vy * vy);
        if (sp > 1.0e-4f) {
            float dec = minf(-throttle * cfg_brakeStrength, sp);   // never reverses the car
            float k = (sp - dec) / sp;
            vx *= k; vy *= k;
        }
    }

    float latVel = vx * (-sinA) + vy * cosA;
    float grip = CAR_LAT_GRIP; if (absf(latVel) > 2.5f) grip *= 0.8f;

    vx += (-sinA) * -latVel * grip;
    vy += cosA * -latVel * grip;
    vx *= 0.99f; vy *= 0.99f;

    float speed = sqrtf_(vx * vx + vy * vy);
    if (speed > cfg_maxSpeed) { float r = cfg_maxSpeed / speed; vx *= r; vy *= r; speed = cfg_maxSpeed; }

    car_vx[i] = vx; car_vy[i] = vy; car_speed[i] = speed;

    // No momentum, no race. A car sitting still is either parked on the line
    // having never asked for throttle, or it has braked to a standstill
    // somewhere on track — and since a stopped car cannot steer either, it
    // has no way back out of that state. It used to sit there burning frames
    // until its TTL ran out, which at 500 cars is most of a generation spent
    // simulating cars that are not going anywhere.
    //
    // Speed only: spinning the heading on the spot is not momentum, and the
    // steering block above will not turn a stopped car anyway.
    // Same cost as hitting a wall, deliberately. If stopping were free it
    // would be the cheap way out of a corner a car could not make — brake to
    // a standstill instead of crashing and keep the fitness. Neither is a
    // way to score.
    if (car_frames[i] > STOPPED_GRACE_FRAMES && speed < STOPPED_SPEED) {
        car_crashed[i] = 1; car_fitness[i] -= 50.0f; return;
    }

    float prevX = car_x[i], prevY = car_y[i];
    car_x[i] += vx; car_y[i] += vy;
    // Being alive and still making progress pays — but flatly, not by how
    // fast. This used to scale with (speed/maxSpeed), which reads as
    // "rewarding going fast" but actually rewards "never slowing down for any
    // reason" — including the reason that's correct racing technique: braking
    // INTO a corner to carry more speed OUT of it. Every frame that trade
    // costs a bit of this term even when it wins the corner, so evolution had
    // a standing bias against the brake pedal that had nothing to do with lap
    // time. The checkpoint and lap bonuses below are the term that actually
    // measures lap time — real elapsed frames to get somewhere, brakes and
    // all — so they are where "fast" should be judged, and this one has no
    // business re-judging it a second, cruder way.
    //
    // What this term is actually FOR is the gating around it, not its size:
    // paid unconditionally, it used to reward time spent alive full stop, so
    // a car circling a wide piece of track banked fitness forever without
    // passing a single gate. Gating it on having reached a gate recently is
    // what makes the go-nowhere loop worth nothing (a moving car is always
    // inside the window; a looping one falls out of it after
    // PROGRESS_WINDOW_FRAMES and is worth nothing per frame from then on) —
    // that gate does the whole job by itself, whether the per-frame amount
    // tracks speed or not. Still requires actual motion (STOPPED_SPEED), so a
    // parked car earns nothing just for sitting inside the window.
    if (car_frames[i] - car_lastCpFrame[i] < PROGRESS_WINDOW_FRAMES && speed > STOPPED_SPEED) {
        float progressMult = (car_focused[i] && inFocusWindow(car_nextCP[i])) ? FOCUS_BOOST : 1.0f;
        car_fitness[i] += 0.1f * progressMult;
    }

    if (car_x[i] < -100.0f || car_x[i] > 1300.0f || car_y[i] < -100.0f || car_y[i] > 1000.0f) { car_crashed[i] = 1; return; }
    if (tk_cp_n <= 0 || !wbs_start) { car_crashed[i] = 1; return; }

    // Hard containment: off the asphalt is out, immediately.
    //
    // The wall tests below are the detailed ones — they catch the car's body
    // clipping a barrier, which happens first and is what normally ends a run.
    // This is the backstop underneath them, and it cannot be escaped by any of
    // the ways a segment-versus-segment test can be: tunnelling through a wall
    // in one fast frame, slipping through the seam where the barrier is left
    // open because the track crosses itself, or driving into a stretch whose
    // walls were not in the lookup bucket. The barrier sits exactly on the edge
    // of the road, so "the centre is off the road" is "the centre is past a
    // barrier", whether or not any particular wall segment noticed.
    if (!insideRoad(car_x[i], car_y[i], &car_roadSeg[i])) { car_crashed[i] = 1; car_fitness[i] -= 50.0f; return; }

    i32 nxt = car_nextCP[i];
    i32 curSeg = (nxt >= 0 && nxt < tk_cp_n) ? nxt : 0;
    i32 wStart = wbs_start[curSeg], wEnd = wbs_start[curSeg + 1];

    float hw = CAR_W * 0.5f, hh = CAR_H * 0.5f;
    float cx_ = car_x[i], cy_ = car_y[i];
    float cxs[4] = {
        cx_ + cosA * hw - sinA * hh, cx_ + cosA * hw + sinA * hh,
        cx_ - cosA * hw + sinA * hh, cx_ - cosA * hw - sinA * hh
    };
    float cys[4] = {
        cy_ + sinA * hw + cosA * hh, cy_ + sinA * hw - cosA * hh,
        cy_ - sinA * hw - cosA * hh, cy_ - sinA * hw + cosA * hh
    };

    // --- broad phase ---------------------------------------------------
    // The bucket is a coarse cull by checkpoint; most of what it holds is still
    // nowhere near this car. One cheap pass over it — a midpoint distance
    // against a precomputed radius, no division — splits it into the walls a
    // sensor could reach and the far smaller set the car could touch this
    // frame. Every wall dropped here is one that provably cannot be hit, so the
    // result is identical and the twelve intersection loops below get shorter.
    //
    // Both tests are conservative supersets: if any point of a wall lies within
    // radius R of the car, then its midpoint lies within R + half its length,
    // which is exactly what is compared.
    {
        float collR = speed + 8.0f;
        i32 sn = 0, nn = 0;
        i32 w = wStart;
#ifdef __wasm_simd128__
        // Four walls per iteration. The three tests are the same three; what
        // the vector path adds is that the survivors are compacted into the
        // scratch arrays with a swizzle rather than one conditional store at a
        // time, which is what the scalar version spends most of its time on
        // once the tests themselves are this cheap.
        //
        // Buckets are padded to a multiple of four by buildWallBuckets, so the
        // scalar tail below normally runs zero times; it is there because
        // nothing else in this file assumes that padding either.
        {
            const v4f vcx = vsplat(cx_), vcy = vsplat(cy_);
            const v4f vcos = vsplat(cosA), vsin = vsplat(sinA);
            const v4f vcollR = vsplat(collR);
            for (; w + 4 <= wEnd && sn <= MAX_SCRATCH - 4 && nn <= MAX_SCRATCH - 4; w += 4) {
                v4f ddx = vcx - vload(b_mx + w), ddy = vcy - vload(b_my + w);
                v4f d2 = ddx * ddx + ddy * ddy;
                v4f hl = vload(b_hl + w);
                v4i near = d2 <= vload(b_r2 + w);
                v4i msen = near & ((ddx * vcos + ddy * vsin) <= hl);
                v4f cr = vcollR + hl;
                v4i mcol = near & (d2 <= cr * cr);
                i32 bs = __builtin_wasm_bitmask_i32x4(msen);
                i32 bc = __builtin_wasm_bitmask_i32x4(mcol);
                if (bs | bc) {
                    v4f x1 = vload(b_x1 + w), y1 = vload(b_y1 + w);
                    v4f dxv = vload(b_dx + w), dyv = vload(b_dy + w);
                    if (bs) {
                        v16i8 pat = PACK_LANES[bs];
                        packStore(sc_x1 + sn, x1, pat); packStore(sc_y1 + sn, y1, pat);
                        packStore(sc_dx + sn, dxv, pat); packStore(sc_dy + sn, dyv, pat);
                        sn += __builtin_popcount((unsigned)bs);
                    }
                    if (bc) {
                        v16i8 pat = PACK_LANES[bc];
                        packStore(nc_x1 + nn, x1, pat); packStore(nc_y1 + nn, y1, pat);
                        packStore(nc_dx + nn, dxv, pat); packStore(nc_dy + nn, dyv, pat);
                        nn += __builtin_popcount((unsigned)bc);
                    }
                }
            }
        }
#endif
        for (; w < wEnd; w++) {
            float ddx = cx_ - b_mx[w], ddy = cy_ - b_my[w];
            float d2 = ddx * ddx + ddy * ddy;
            if (d2 > b_r2[w]) continue;
            float hl = b_hl[w];
            // A third cull, for the sensor set only, and free next to the
            // seven raycasts it shortens. The fan spans -90 to +90 degrees
            // about the heading, so EVERY point of every ray has a forward
            // projection of at least zero: a wall lying entirely behind the
            // car cannot be hit by any of them. Projections along the wall
            // stay within half its length of its midpoint's, so
            // "midpoint at least -halfLength forward" is the exact
            // conservative form of that, and on an ordinary track it drops
            // about half the bucket. (Not applied to the collision set: the
            // car's own body extends backwards, so it can still clip a wall
            // behind its centre.)
            if (ddx * cosA + ddy * sinA <= hl && sn < MAX_SCRATCH) {
                sc_x1[sn] = b_x1[w]; sc_y1[sn] = b_y1[w];
                sc_dx[sn] = b_dx[w]; sc_dy[sn] = b_dy[w];
                sn++;
            }
            float cr = collR + hl;
            if (d2 <= cr * cr && nn < MAX_SCRATCH) {
                nc_x1[nn] = b_x1[w]; nc_y1[nn] = b_y1[w];
                nc_dx[nn] = b_dx[w]; nc_dy[nn] = b_dy[w];
                nn++;
            }
        }
        // Pad with zero-length walls so the vector raycast can overrun the end
        // safely; a zero delta yields a zero denominator, which is rejected.
        while (sn & 3) { sc_x1[sn] = 0.0f; sc_y1[sn] = 0.0f; sc_dx[sn] = 0.0f; sc_dy[sn] = 0.0f; sn++; }
        sc_n = sn; nc_n = nn;
    }

    if (segmentHits(prevX, prevY, cx_, cy_, nc_x1, nc_y1, nc_dx, nc_dy, nc_n)) {
        car_crashed[i] = 1; car_fitness[i] -= 50.0f; return;
    }
    for (i32 j = 0; j < 4; j++) {
        i32 jn = (j + 1) & 3;
        if (segmentHits(cxs[j], cys[j], cxs[jn], cys[jn], nc_x1, nc_y1, nc_dx, nc_dy, nc_n)) {
            car_crashed[i] = 1; car_fitness[i] -= 50.0f; return;
        }
    }

    float fitMult = 1.0f;
    for (i32 z = 0; z < tk_zone_n; z++) {
        Zone *zn = &tk_zones[z];
        float dx = cx_ - zn->x, dy = cy_ - zn->y;
        if (dx * dx + dy * dy < zn->radius * zn->radius) {
            if (zn->type == 0 && car_speed[i] > cfg_maxSpeed * 0.7f) car_fitness[i] += 2.0f;
            else if (zn->type == 1) car_fitness[i] += 2.0f;
            else if (zn->type == 2) fitMult = 3.0f;
            else if (zn->type == 3) {
                i32 killTimer = (i32)zn->killTimer;
                if (car_frames[i] >= killTimer) { car_crashed[i] = 1; return; }
            }
        }
    }

    // The checkpoint captured here is the one relAng is measured against below,
    // even when the car passes it this frame — matching the JS exactly.
    // The gate the car was heading for when the frame began. relAng below is
    // measured against THIS one even if the car goes on to cross it, which is
    // the behaviour the network was trained against and the JS edition has.
    Checkpoint *relCP = (nxt >= 0 && nxt < tk_cp_n) ? &tk_cps[nxt] : 0;
    Checkpoint *nCP = relCP;
    // Gates are checked in a LOOP, because one frame can cross more than one.
    //
    // Corner gates sit as little as TRACK_CP_MIN_GAP apart while a car covers
    // up to maxSpeed in a frame, so at speed it can pass two or three at once.
    // Registering only the first leaves the index trailing behind the car, and
    // a trailing index is the failure that let cars drive through walls: the
    // wall lookup is bucketed by the gate a car is heading for. The bound is a
    // safety stop, not an expected limit.
    for (i32 pass = 0; pass < 8 && nCP; pass++) {
        float dx = cx_ - nCP->cx, dy = cy_ - nCP->cy;
        if (pass == 0 && dx * dx + dy * dy > 160000.0f) { car_crashed[i] = 1; return; }  // off course

        float mx = (nCP->p1x + nCP->p2x) * 0.5f, my = (nCP->p1y + nCP->p2y) * 0.5f;
        float cdx = cx_ - mx, cdy = cy_ - my;
        // How close to the middle of the gate the car has to be before it is
        // even worth testing whether it crossed.
        //
        // This used to be a flat 50px, while a gate spans the full width of the
        // road — up to 150 either side. A car taking the outside line was
        // therefore never close enough to the middle to register, and sailed
        // through gate after gate without its checkpoint index advancing. That
        // is not a scoring curiosity: the wall lookup is bucketed by the
        // checkpoint a car is heading for, so a car with a stale index is
        // handed the walls for a part of the track it left long ago and drives
        // straight through the ones actually in front of it. Sizing the radius
        // to the gate (plus a frame of travel and half a car) makes the test
        // reach the whole gate at any track width.
        float gateR = tk_w_max + speed + 8.0f;
        if (cdx * cdx + cdy * cdy >= gateR * gateR) break;
        if (!(fastIntersect(prevX, prevY, cx_, cy_, nCP->p1x, nCP->p1y, nCP->p2x, nCP->p2y)
              || (cdx * cdx + cdy * cdy < 400.0f))) break;
        {
            {
                car_cpReached[i]++;
                car_nextCP[i] = (car_nextCP[i] + 1) % tk_cp_n;
                // Reaching a gate resets the clock to the full Initial TTL.
                // It used to add 150 and clamp to 600, which quietly made the
                // slider a lie: set it to 10,000 and the very first gate cut
                // the car back to 600 frames. Now the setting means what it
                // says for the whole run.
                car_ttl[i] = cfg_initialTTL;

                // Scoring a gate purely on having reached it makes crawling the
                // winning strategy: the reward is the same however long it took,
                // and going slowly is far less likely to end in a wall. That is
                // what produced cars pottering round for two and a half minutes
                // a lap. The reward now scales with how quickly the gate came.
                //
                // It is a MULTIPLIER on a floor, never a subtraction, and that
                // matters: every gate is still worth at least the old 500, so
                // reaching more of them always beats reaching fewer, and a lap
                // is never worth less than most of a lap. A per-frame time
                // penalty would have inverted that and made crashing on purpose
                // score better than finishing slowly.
                // Measured from the track itself, not from the nominal gate
                // spacing. Corner gates made the gate count depend on how many
                // turns a track has, and an "ideal" derived from a fixed
                // spacing then drifted with it — inflating the target time,
                // saturating the speed bonus and flattening out the very
                // gradient that stops cars crawling.
                float meanGap = tk_len / (float)maxi(tk_cp_n, 1);
                float idealCp = meanGap / maxf(cfg_maxSpeed, 0.001f);
                float dtCp = (float)(car_frames[i] - car_lastCpFrame[i]);
                car_lastCpFrame[i] = car_frames[i];

                // The gate just reached, as an absolute checkpoint index (the
                // increment above already moved car_nextCP on, so this frame's
                // gate is one behind it) — NOT `nxt`, which stays fixed at
                // whichever gate this frame started heading for even as `nCP`
                // walks past several in one pass.
                i32 gAbs = (car_nextCP[i] - 1 + tk_cp_n) % tk_cp_n;
                // Recorded only for car 0 — see the comment on gate_ratio
                // above for why that alone is enough to track the stash's pace.
                if (i == 0 && gAbs >= 0 && gAbs < MAX_GATES) gate_ratio[gAbs] = idealCp / maxf(dtCp, idealCp);

                float gateMult = (car_focused[i] && inFocusWindow(gAbs)) ? FOCUS_BOOST : 1.0f;
                car_fitness[i] += 500.0f * (1.0f + 3.0f * (idealCp / maxf(dtCp, idealCp))) * fitMult * gateMult;

                // A lap is complete when the car is back at the gate it
                // started from — not when the index happens to wrap past zero,
                // which only coincided with the start line on tracks whose
                // start had never been moved.
                if (car_nextCP[i] == (tk_start_cp + 1) % tk_cp_n && tk_cp_n > 2) {
                    car_laps[i]++;
                    float lapFrames = (float)(car_frames[i] - car_prevLapFrame[i]);
                    car_lastLap[i] = lapFrames / 60.0f;
                    car_prevLapFrame[i] = car_frames[i];
                    float idealLap = tk_len / maxf(cfg_maxSpeed, 0.001f);
                    car_fitness[i] += 3000.0f * (1.0f + 3.0f * (idealLap / maxf(lapFrames, idealLap))) * fitMult;
                }
            }
        }
        // On to the next gate, in case this frame crossed that one too.
        nCP = &tk_cps[car_nextCP[i]];
    }

    // Sensors. The inner loop is the hot spot of the whole program: seven rays
    // against every wall in the window, for every living car, every frame.
    //
    // The readings live in car_in and nowhere else. They used to be written
    // twice — once here and once into a parallel car_sensors array that only
    // the sensor overlay read — which was seven redundant stores per living
    // car per frame for a copy that was always identical.
    float *in = &car_in[i * IN_N];
    const float senLen = cfg_sensorLen;

    // Seven ray directions, from the heading the car already has sin/cos for.
    //
    // This used to be seven more sincos_d calls — seven Cody-Waite argument
    // reductions and fourteen polynomial evaluations per living car per frame,
    // which made trigonometry, not raycasting, the single largest line item in
    // the step. But the seven offsets are compile-time constants, so
    //
    //     sin(a +- b) = sin a cos b +- cos a sin b
    //     cos(a +- b) = cos a cos b -+ sin a sin b
    //
    // turns all seven into four multiplies and eight adds off sd/cd, with the
    // +-90 degree pair falling out for free (cos 90 = 0, sin 90 = 1) and the
    // straight-ahead ray being sd/cd themselves. The +-30 and +-60 pairs share
    // their four products because cos 60 = sin 30 and sin 60 = cos 30, so the
    // two offsets are the same two numbers swapped.
    //
    // The products are done in f64 like the reduction they replace, so each
    // direction is still correct to the last bit an f32 can hold.
    const double C30 = 0.86602540378443864676;   // cos 30 = sin 60
    const double S30 = 0.5;                      // sin 30 = cos 60
    const double p = sd * C30, q = cd * S30, r = cd * C30, t = sd * S30;
    const float rcos[SENS_N] = {
        (float)sd,        // -90
        (float)(q + p),   // -60
        (float)(r + t),   // -30
        (float)cd,        //   0
        (float)(r - t),   // +30
        (float)(q - p),   // +60
        (float)-sd        // +90
    };
    const float rsin[SENS_N] = {
        (float)-cd,       // -90
        (float)(t - r),   // -60
        (float)(p - q),   // -30
        (float)sd,        //   0
        (float)(p + q),   // +30
        (float)(t + r),   // +60
        (float)cd         // +90
    };
    for (i32 k = 0; k < SENS_N; k++) {
        float ex = cx_ + rcos[k] * senLen, ey = cy_ + rsin[k] * senLen;
        float minT = raycast(cx_, cy_, ex, ey, sc_x1, sc_y1, sc_dx, sc_dy, sc_n);
        in[k] = 1.0f - minT;
    }

    // Speed and bearing to the next gate. Written unconditionally: leaving the
    // previous frame's values in place when there is no gate to aim at fed the
    // network stale numbers without ever saying so.
    if (relCP) {
        float tX = (relCP->p1x + relCP->p2x) * 0.5f, tY = (relCP->p1y + relCP->p2y) * 0.5f;
        float relAng = atan2f_(tY - cy_, tX - cx_) - car_angle[i];
        // The heading is never wrapped (see the note on sincos_d), so after a
        // few laps car_angle is tens of radians and this difference can be
        // many whole turns from the branch the network wants. Two `while`
        // loops unwound it one turn at a time — a loop in the per-car,
        // per-frame path whose trip count grows with how long the car has
        // been driving, and with Turn Speed wound up it grows fast. One
        // floor does the whole reduction at once. The test in front of it is
        // what a lap of an ordinary track hits on almost every frame, and
        // costs a compare.
        if (relAng > PI_F || relAng < -PI_F) {
            double relD = (double)relAng;
            relD -= (2.0 * PI_D) * floord_((relD + PI_D) * (1.0 / (2.0 * PI_D)));
            relAng = (float)relD;
        }
        in[SENS_N] = car_speed[i] / cfg_maxSpeed;
        in[SENS_N + 1] = relAng / PI_F;
    } else {
        in[SENS_N] = 0.0f;
        in[SENS_N + 1] = 0.0f;
    }

    // The car's own last decision, read before feedForward overwrites it.
    in[SENS_N + 2] = car_out[i * OUT_N];
    in[SENS_N + 3] = car_out[i * OUT_N + 1];

    feedForward(i);
}

// ---------------------------------------------------------------------------
// run — step the whole slice `iters` times, stopping early once every car here
// has crashed or something has hit the lap target. Returns the best lap count
// seen; all_crashed() reports the other half.
//
// This is the call the JS edition made once per animation frame with 500 iters
// and a full postMessage round trip on each side. Here it is one call with no
// marshalling, so hyper mode can run thousands of steps per crossing.
// ---------------------------------------------------------------------------
static i32 last_all_crashed = 1;
// Cars actually stepped by the last run() — the exact count of updateCar calls,
// not the population size. engine.js divides the worker's wall-clock time by it
// to get a real throughput figure, which is the only honest way to compare two
// workers whose slices died off at different rates. Costs one add per
// iteration.
static i32 last_car_steps = 0;
__attribute__((export_name("car_steps")))
i32 car_steps(void) { return last_car_steps; }

// See crash_count above — this is where every car's crash becomes visible
// exactly once, whichever of updateCar's several crash sites caused it, so it
// is the one place that needs to know about all of them rather than each of
// them needing to know about this.
static inline void recordCrash(i32 i) {
    i32 g = car_nextCP[i];
    if (g >= 0 && g < MAX_GATES) crash_count[g]++;
}

__attribute__((export_name("run")))
i32 run(i32 iters) {
    i32 maxLaps = 0;
    i32 allCrashed = 1;
    i32 steps = 0;
    for (i32 it = 0; it < iters; it++) {
        i32 n = active_n;
        allCrashed = (n == 0);
        if (allCrashed) break;
        steps += n;
        // Compacted in place as it goes: a car that crashes in its own
        // update simply is not written back.
        i32 keep = 0;
        for (i32 a = 0; a < n; a++) {
            i32 c = active[a];
            updateCar(c);
            if (car_laps[c] > maxLaps) maxLaps = car_laps[c];
            if (car_crashed[c]) recordCrash(c); else active[keep++] = c;
        }
        active_n = keep;
        if (maxLaps >= cfg_targetLaps) break;
    }
    last_all_crashed = allCrashed;
    last_car_steps = steps;
    return maxLaps;
}

__attribute__((export_name("all_crashed")))
i32 all_crashed(void) { return last_all_crashed; }

// Pack the render rows the main thread reads. Skipped entirely in hyper mode,
// which is a good part of why hyper mode is so much faster here.
__attribute__((export_name("write_render")))
void write_render(void) {
    for (i32 i = 0; i < pop_n; i++) {
        float *r = &render_buf[i * RENDER_STRIDE];
        r[0] = (float)(pop_id_offset + i);
        r[1] = car_crashed[i] ? 1.0f : 0.0f;
        r[2] = car_x[i]; r[3] = car_y[i]; r[4] = car_angle[i]; r[5] = car_speed[i];
        r[6] = car_out[i * OUT_N]; r[7] = car_out[i * OUT_N + 1];
        r[8] = (float)car_laps[i]; r[9] = car_fitness[i]; r[10] = car_lastLap[i];
        for (i32 k = 0; k < SENS_N; k++) r[11 + k] = car_in[i * IN_N + k];
    }
}

// Fitness alone, for the evolve step — far cheaper than a full render pass when
// all the main thread wants is who won. Hyper mode never draws a car, so this
// is the only thing it ships back per generation.
__attribute__((export_name("write_fitness")))
void write_fitness(void) {
    for (i32 i = 0; i < pop_n; i++) {
        fitness_buf[i * 5]     = car_fitness[i];
        fitness_buf[i * 5 + 1] = (float)car_laps[i];
        fitness_buf[i * 5 + 2] = car_lastLap[i];
        // Gates passed: the progress measure that doesn't quantise to whole
        // laps, so "nearly all the way round" is distinguishable from "barely
        // started" both here and on screen.
        fitness_buf[i * 5 + 3] = (float)car_cpReached[i];
        // The frame its most recent gate fell on — the time it took to get as
        // far as it got. Total frames alive is the wrong measure of pace: a car
        // can reach three gates briskly and then mill about for another four
        // hundred frames without reaching a fourth, which says nothing about
        // how quickly it covered those three.
        fitness_buf[i * 5 + 4] = (float)car_lastCpFrame[i];
    }
}
__attribute__((export_name("fitness_stride")))
i32 fitness_stride(void) { return 5; }

__attribute__((export_name("alive_count")))
i32 alive_count(void) { return active_n; }

// ---------------------------------------------------------------------------
// Evolution. Selection needs the whole population, which is spread across
// workers, so the main thread's instance owns it: it holds every brain, breeds
// the next generation here, and hands each worker back its slice.
// ---------------------------------------------------------------------------
static i32 order[MAX_CARS];
static float ev_fitness[MAX_CARS];

// JS writes the gathered per-car fitness straight into this array rather than
// making one call per car.
__attribute__((export_name("ev_fitness_ptr")))
i32 ev_fitness_ptr(void) { return (i32)(unsigned long)ev_fitness; }

// Descending sort of the fitness index, by heapsort: no recursion, no scratch
// memory, O(n log n). The insertion sort this replaces was O(n^2) — at 500
// cars that is up to ~125,000 comparisons sitting on the critical path between
// every pair of generations, and it got worse with every car added.
//
// The heap is a MIN-heap on fitness, so repeatedly moving the smallest to the
// back leaves the array ordered best-first, which is what everything below
// expects.
static void sift_down(i32 lo, i32 hi) {
    i32 root = lo;
    for (;;) {
        i32 child = 2 * root + 1;
        if (child > hi) break;
        if (child + 1 <= hi && ev_fitness[order[child + 1]] < ev_fitness[order[child]]) child++;
        if (ev_fitness[order[root]] <= ev_fitness[order[child]]) break;
        i32 t = order[root]; order[root] = order[child]; order[child] = t;
        root = child;
    }
}
static void sort_by_fitness(void) {
    for (i32 i = 0; i < pop_n; i++) order[i] = i;
    if (pop_n < 2) return;
    for (i32 start = pop_n / 2 - 1; start >= 0; start--) sift_down(start, pop_n - 1);
    for (i32 end = pop_n - 1; end > 0; end--) {
        i32 t = order[0]; order[0] = order[end]; order[end] = t;
        sift_down(0, end - 1);
    }
}

// Rank-weighted draw from the top of the sorted order. Squaring a uniform
// concentrates the pick near the front, so the generation's best car parents
// several times more often than the weakest car in the pool. The old code drew
// uniformly from the top 20%, which gave the 1st and the 100th car identical
// odds and threw away most of the fitness signal it had just computed.
static inline i32 pick_parent(i32 poolSize) {
    float r = rnd01();
    i32 idx = (i32)(r * r * (float)poolSize);
    if (idx >= poolSize) idx = poolSize - 1;
    if (idx < 0) idx = 0;
    return order[idx];
}

// A rough standard normal: three uniforms summed is mean 0, variance 1, and
// costs three cheap xorshift draws and no transcendentals. Good enough for a
// mutation kernel, where the only thing that matters is that small nudges are
// common and large ones are rare — which a flat uniform never gave us.
static inline float gauss01(void) { return rnd11() + rnd11() + rnd11(); }

// How hard mutation hits, as a function of how long the run has been going.
// A perturbation sized for generation 1 is far too coarse by generation 500:
// it keeps kicking a working solution apart instead of refining it. This
// decays from START toward FLOOR with a half-life of about TAU generations,
// so early training explores and late training polishes.
//
// "How long the run has been going" is NOT the raw generation counter —
// that's `sigmaGen` below, not `generation`, and the two can differ a lot.
// A track the population hasn't finished even once yet has learned nothing
// this curve should be polishing, and the raw generation count decays sigma
// toward FLOOR regardless: a hard track can sit at zero laps for hundreds of
// generations, by which point exploration has been ground down to almost
// nothing exactly when the population most needs to keep trying new things
// to break through. sigmaGen instead holds at 0 (so sigma sits at
// MUT_SIGMA_START, full exploration) until the population has completed a
// lap for the first time AND then settled for a further stretch of
// generations past that — engine.js/script.js track both and do the actual
// arithmetic, since only they see every generation's result across the whole
// run; this file only ever sees whatever they hand it. Once past that point
// sigmaGen counts up from 0 the same way the old raw generation count did, so
// a run that finds its feet quickly anneals on close to the same schedule as
// before.
#define MUT_SIGMA_START  0.5f
#define MUT_SIGMA_FLOOR  0.05f
#define MUT_SIGMA_TAU    150.0f

// Node-level crossover, then annealed Gaussian mutation. The elite clones come
// from the stash (the all-time best), not from this generation's winner, which
// is what stops a bad generation regressing.
//
// Crossover works a hidden unit at a time — all of a unit's incoming weights,
// its bias and its outgoing weights are taken from the SAME parent. Picking
// each weight independently (what this used to do) splits up groups of weights
// that only mean anything together, so two parents that both drive well
// routinely produced a child that drove into a wall.
//
// Every weight is mutated, always — there is no per-weight probability here
// any more. There used to be (the old "Mutation Rate" setting), sitting
// alongside sigma, which already controls how BIG each nudge is and already
// anneals it from exploring to polishing over the run. A rate on top of that
// just skipped some weights on a coin flip; once sigma has shrunk toward its
// floor late in a run, touching every weight with a tiny nudge and touching a
// random 30% of them with the same tiny nudge land in essentially the same
// place, so the extra knob was a second control for the one job sigma
// already does alone. Dropping it also means one less setting to tune.
// A run "reliably" finishes once it has completed a lap this many separate
// generations, not just once — one lucky lap is easy to get from a fluke
// line through one easy corner and says nothing about the rest of the track.
// Below this, evolve() points the focus window at wherever cars are actually
// dying (crash_count); at and above it, survival is no longer the
// bottleneck and it goes back to pointing at wherever the current best is
// slowest (gate_ratio) — see the branch below.
#define FEW_LAPS_THRESHOLD 3

__attribute__((export_name("evolve")))
void evolve(i32 eliteClones, i32 hasGlobalBest, i32 sigmaGen, i32 lapCompletions) {
    sort_by_fitness();
    i32 stride = brain_stride_v;
    i32 h = cfg_hidden;
    i32 offHO = IN_N * h, offBH = offHO + h * OUT_N, offBO = offBH + h;
    i32 written = 0;

    for (i32 c = 0; c < pop_n; c++) car_focused[c] = 0;

    // Where to aim the focus window — one of two different questions
    // depending on how far the run has actually gotten.
    //
    // Below FEW_LAPS_THRESHOLD: "where does it keep dying?" gate_ratio
    // cannot answer this — a gate the population never reaches leaves no
    // ratio behind at all, so the corner actually killing every run is
    // invisible to it no matter how lethal it is. crash_count answers the
    // right question instead: population-wide (every car that crashed this
    // generation, not just car 0), summed across every worker by engine.js
    // and written into this instance's copy right before this call, exactly
    // like gate_ratio is — just added instead of copied, since this one has
    // no single car worth mirroring.
    //
    // At or above FEW_LAPS_THRESHOLD: survival is no longer the problem, so
    // this goes back to the original question — where is the current best
    // slowest? — using car 0's per-gate pace this generation (gate_ratio),
    // refreshed from worker 0 the same way. Car 0 mirrors the stash exactly
    // whenever there is one, since the track and the starting state are both
    // deterministic.
    i32 worst = -1;
    i32 gateLimit = mini(tk_cp_n, MAX_GATES);
    if (lapCompletions < FEW_LAPS_THRESHOLD) {
        i32 worstCount = 0;
        for (i32 g = 0; g < gateLimit; g++) {
            if (crash_count[g] > worstCount) { worstCount = crash_count[g]; worst = g; }
        }
    } else {
        float worstRatio = 1.0e30f;
        for (i32 g = 0; g < gateLimit; g++) {
            if (gate_ratio[g] >= 0.0f && gate_ratio[g] < worstRatio) { worstRatio = gate_ratio[g]; worst = g; }
        }
    }
    focus_lo = -1; focus_hi = -1;
    if (hasGlobalBest && worst >= 0 && tk_cp_n > 2) {
        float meanGap = tk_len / (float)maxi(tk_cp_n, 1);
        float idealCp = meanGap / maxf(cfg_maxSpeed, 0.001f);
        // +-1 second either side of the worst gate, in gates rather than
        // frames — TRACK_CP_SPACING gates take idealCp frames apiece at Max
        // Speed, so 60 frames (one second at the fixed 60fps this project
        // has always assumed for lap times) is 60/idealCp of them.
        i32 span = (i32)(60.0f / maxf(idealCp, 1.0f) + 0.5f);
        if (span < 1) span = 1;
        if (span * 2 + 1 < tk_cp_n) {
            focus_lo = ((worst - span) % tk_cp_n + tk_cp_n) % tk_cp_n;
            focus_hi = (worst + span) % tk_cp_n;
        }
        // else the window would already cover the whole lap — nothing left
        // to prioritise over anything else, so leave it inactive.
    }

    if (hasGlobalBest) {
        i32 clones = mini(eliteClones, pop_n);
        if (clones < 0) clones = 0;
        const float *src = &brains[STASH_SLOT * stride];
        for (i32 k = 0; k < clones; k++) {
            float *dst = &brains_next[k * stride];
            for (i32 j = 0; j < stride; j++) dst[j] = src[j];
        }
        written = clones;
    }

    float sigma = MUT_SIGMA_FLOOR + (MUT_SIGMA_START - MUT_SIGMA_FLOOR)
                  * (float)exp_d(-(double)(sigmaGen < 0 ? 0 : sigmaGen) / (double)MUT_SIGMA_TAU);

    // The focused sub-population: mutated clones of the stash, same sigma as
    // everything else, flagged so updateCar can boost their reward through
    // the window above. Only spawned once there is both a stash to clone and
    // an actual weak spot to aim at — on an early generation, or a track
    // short enough that the window would cover the whole lap, this is 0 and
    // every slot breeds normally below.
    i32 focusCount = 0;
    if (hasGlobalBest && focus_lo >= 0) {
        focusCount = (i32)(cfg_focusPct * (float)pop_n + 0.5f);
        if (focusCount > pop_n - written) focusCount = pop_n - written;
        if (focusCount < 0) focusCount = 0;
        const float *stashSrc = &brains[STASH_SLOT * stride];
        for (i32 k = 0; k < focusCount; k++) {
            i32 idx = written + k;
            float *dst = &brains_next[idx * stride];
            for (i32 j = 0; j < stride; j++) dst[j] = stashSrc[j] + gauss01() * sigma;
            car_focused[idx] = 1;
        }
        written += focusCount;
    }

    i32 poolSize = maxi(2, pop_n / 5);
    if (poolSize > pop_n) poolSize = pop_n;
    for (i32 i = written; i < pop_n; i++) {
        const float *p1 = &brains[pick_parent(poolSize) * stride];
        const float *p2 = &brains[pick_parent(poolSize) * stride];
        float *dst = &brains_next[i * stride];

        for (i32 k = 0; k < h; k++) {
            const float *src = rnd01() < 0.5f ? p1 : p2;
            for (i32 j = 0; j < IN_N; j++) dst[j * h + k] = src[j * h + k];
            for (i32 m = 0; m < OUT_N; m++) dst[offHO + k * OUT_N + m] = src[offHO + k * OUT_N + m];
            dst[offBH + k] = src[offBH + k];
        }
        for (i32 m = 0; m < OUT_N; m++) {
            const float *src = rnd01() < 0.5f ? p1 : p2;
            dst[offBO + m] = src[offBO + m];
        }

        for (i32 j = 0; j < stride; j++) dst[j] += gauss01() * sigma;
    }

    // Swap the buffers rather than copying the whole population back. Only the
    // stash has to travel, because it lives past the end of the live slots and
    // the next generation still has to be able to clone from it.
    float *tmp = brains; brains = brains_next; brains_next = tmp;
    const float *stashSrc = &brains_next[STASH_SLOT * stride];
    float *stashDst = &brains[STASH_SLOT * stride];
    for (i32 j = 0; j < stride; j++) stashDst[j] = stashSrc[j];
}

// Copy a brain between slots — used to park this generation's winner in the
// stash, and to pull a stashed/imported brain back into the live population.
__attribute__((export_name("copy_brain")))
void copy_brain(i32 from, i32 to) {
    i32 stride = brain_stride_v;
    if (from < 0 || to < 0 || from > STASH_SLOT || to > STASH_SLOT || from == to) return;
    const float *s = &brains[from * stride];
    float *d = &brains[to * stride];
    for (i32 j = 0; j < stride; j++) d[j] = s[j];
}

// Seed the whole population from the stashed brain: car 0 gets it verbatim, the
// rest get mutated copies. This is what "Load AI" does in the JS edition too.
__attribute__((export_name("seed_from_stash")))
void seed_from_stash(void) {
    i32 stride = brain_stride_v;
    const float *src = &brains[STASH_SLOT * stride];
    for (i32 i = 0; i < pop_n; i++) {
        float *dst = &brains[i * stride];
        if (i == 0) { for (i32 j = 0; j < stride; j++) dst[j] = src[j]; continue; }
        for (i32 j = 0; j < stride; j++) dst[j] = src[j] + rnd11() * 0.5f;
    }
}


// ---------------------------------------------------------------------------
// Math self-test hooks — tools/mathtest.mjs drives these against JS Math so a
// bad polynomial coefficient shows up at build time rather than as a car that
// mysteriously steers into a wall.
// ---------------------------------------------------------------------------
__attribute__((export_name("t_sin")))   float t_sin(float x)   { return sinf_(x); }
__attribute__((export_name("t_cos")))   float t_cos(float x)   { return cosf_(x); }
__attribute__((export_name("t_atan2"))) float t_atan2(float y, float x) { return atan2f_(y, x); }
__attribute__((export_name("t_tanh")))  float t_tanh(float x)  { return tanhf_(x); }
// The f32 activation feedForward actually runs. Held to its own bar in
// tools/mathtest so "it is only the activation" can never quietly become
// "and nobody checked it".
__attribute__((export_name("t_tanh_nn"))) float t_tanh_nn(float x) { return nn_tanh(x); }
// And the four-lane form the hidden layer actually runs on the SIMD build, so
// "the two agree" is a checked claim rather than a comment. On the scalar
// build there is no vector path and this is the scalar one.
__attribute__((export_name("t_tanh_nn4"))) float t_tanh_nn4(float x) {
#ifdef __wasm_simd128__
    return nn_tanh4(vsplat(x))[0];
#else
    return nn_tanh(x);
#endif
}
__attribute__((export_name("t_acos")))  float t_acos(float x)  { return acosf_(x); }
__attribute__((export_name("t_exp")))   double t_exp(double x) { return exp_d(x); }
