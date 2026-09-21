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
// at 500 and the hidden layer at 25 — with headroom so a hand-edited setting
// can't walk off the end of a static array.
// ---------------------------------------------------------------------------
#define MAX_CARS     512
#define MAX_HIDDEN   32
#define SENS_N       7
#define IN_N         (SENS_N + 2)
#define OUT_N        2
#define BRAIN_MAX    (IN_N * MAX_HIDDEN + MAX_HIDDEN * OUT_N + MAX_HIDDEN + OUT_N)
#define RENDER_STRIDE 18   // id,crashed,x,y,angle,speed,out0,out1,laps,fit,lap,7 sensors

#define CAR_W        14.0f
#define CAR_H        7.0f
#define SENSOR_LEN   180.0f

#define TRACK_MAX_SEG      30.0f
#define TRACK_CP_SPACING   34.0f
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
static float acosf_(float x) {
    if (x >= 1.0f) return 0.0f;
    if (x <= -1.0f) return PI_F;
    double d = (double)x;
    return (float)(atan_d(sqrtd_(1.0 - d * d) / d) + (d < 0.0 ? PI_D : 0.0));
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
typedef struct { float p1x, p1y, p2x, p2y, cx, cy; } Checkpoint; // 24 bytes
typedef struct { float x, y, radius; i32 type; float killTimer; } Zone; // 20 bytes

static Vec2       *tk_center;    static i32 tk_center_n;
static Wall       *tk_walls;     static i32 tk_wall_n;
static Checkpoint *tk_cps;       static i32 tk_cp_n;
static Zone       *tk_zones;     static i32 tk_zone_n;
static float       tk_start_x, tk_start_y, tk_start_angle, tk_width;

// Uniform grid over the centreline, CSR-packed (counts -> prefix sums -> fill)
// instead of the JS Map<"gx,gy", []>. Same 3x3 neighbourhood query, but the
// lookup is two array reads rather than a string concat and a hash probe.
static i32   *grid_start; static i32 *grid_items;
static i32    grid_nx, grid_ny; static float grid_cell, grid_ox, grid_oy;

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

static i32 buildCentreline(const float *path, i32 n, float width, VecList *out) {
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
static i32   *wbs_start;
static float *b_x1, *b_y1, *b_dx, *b_dy, *b_mx, *b_my, *b_r2, *b_hl;

static void buildWallBuckets(void) {
    i32 segs = tk_cp_n;
    if (segs <= 0) { wbs_start = 0; return; }
    // A sensor reaches 180px, so the window has to cover that in *pixels* —
    // derive it from checkpoint spacing rather than hard-coding a count.
    float step = TRACK_CP_SPACING;
    i32 back = maxi(4, (i32)ceilf_(240.0f / step));
    i32 fwd  = maxi(5, (i32)ceilf_(260.0f / step));

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
        for (i32 j = -back; j <= fwd; j++) {
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
        for (i32 j = -back; j <= fwd; j++) {
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
            float r = SENSOR_LEN + hl;
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
    if (!buildCentreline(path2, n, dist, &cl) || cl.n < 3) return 0;
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

    // --- checkpoint gates, evenly spaced along the centreline ---
    i32 *cpOfSample = (i32 *)arena_alloc((u32)len * 4);
    Checkpoint *cps = (Checkpoint *)arena_alloc((u32)(len + 2) * sizeof(Checkpoint));
    if (!cpOfSample || !cps) return 0;
    i32 ncp = 0;
    float acc = TRACK_CP_SPACING;
    for (i32 i = 0; i < len; i++) {
        if (i > 0) acc += hypotf_(tk_center[i].x - tk_center[i-1].x, tk_center[i].y - tk_center[i-1].y);
        if (acc >= TRACK_CP_SPACING) {
            acc -= TRACK_CP_SPACING;
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
            ncp++;
        }
        cpOfSample[i] = ncp - 1;
    }
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
    tk_start_angle = has_angle ? sang
        : atan2f_(tk_center[1].y - tk_center[0].y, tk_center[1].x - tk_center[0].x);

    buildWallBuckets();
    return 1;
}

// Geometry accessors — JS wraps these in typed-array views over the module's
// memory, so nothing is copied out unless the caller actually asks for it.
__attribute__((export_name("track_centerline_ptr"))) i32 track_centerline_ptr(void) { return (i32)(unsigned long)tk_center; }
// One half-width per centreline sample — what the canvas needs to stroke a road
// that changes width along its length.
__attribute__((export_name("track_widths_ptr"))) i32 track_widths_ptr(void) { return (i32)(unsigned long)tk_w; }
__attribute__((export_name("track_width_max"))) float track_width_max(void) { return tk_w_max; }
__attribute__((export_name("track_centerline_count"))) i32 track_centerline_count(void) { return tk_center_n; }
__attribute__((export_name("track_walls_ptr"))) i32 track_walls_ptr(void) { return (i32)(unsigned long)tk_walls; }
__attribute__((export_name("track_wall_count"))) i32 track_wall_count(void) { return tk_wall_n; }
__attribute__((export_name("track_cps_ptr"))) i32 track_cps_ptr(void) { return (i32)(unsigned long)tk_cps; }
__attribute__((export_name("track_cp_count"))) i32 track_cp_count(void) { return tk_cp_n; }
__attribute__((export_name("track_start_x"))) float track_start_x(void) { return tk_start_x; }
__attribute__((export_name("track_start_y"))) float track_start_y(void) { return tk_start_y; }
__attribute__((export_name("track_start_angle"))) float track_start_angle(void) { return tk_start_angle; }

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static float cfg_maxSpeed = 10.0f, cfg_accel = 0.05f, cfg_turnSpeed = 0.04f, cfg_grip = 0.93f;
static i32   cfg_initialTTL = 750, cfg_targetLaps = 3;
static float cfg_mutationRate = 0.15f;
static i32   cfg_hidden = 5;

__attribute__((export_name("set_config")))
void set_config(float maxSpeed, float accel, float turnSpeed, float grip,
                i32 initialTTL, i32 targetLaps, float mutationRate, i32 hidden) {
    cfg_maxSpeed = maxSpeed; cfg_accel = accel; cfg_turnSpeed = turnSpeed; cfg_grip = grip;
    cfg_initialTTL = initialTTL; cfg_targetLaps = targetLaps;
    cfg_mutationRate = mutationRate;
    cfg_hidden = hidden < 1 ? 1 : (hidden > MAX_HIDDEN ? MAX_HIDDEN : hidden);
}

// ---------------------------------------------------------------------------
// Population. Struct-of-arrays; nothing here is ever allocated per generation.
// ---------------------------------------------------------------------------
static float car_x[MAX_CARS], car_y[MAX_CARS], car_angle[MAX_CARS];
static float car_vx[MAX_CARS], car_vy[MAX_CARS], car_speed[MAX_CARS], car_fitness[MAX_CARS];
static i32   car_crashed[MAX_CARS], car_ttl[MAX_CARS], car_frames[MAX_CARS];
static i32   car_nextCP[MAX_CARS], car_laps[MAX_CARS], car_cpReached[MAX_CARS];
static float car_lastLap[MAX_CARS]; static i32 car_prevLapFrame[MAX_CARS];
static float car_sensors[MAX_CARS * SENS_N];
static float car_out[MAX_CARS * OUT_N];
static float car_in[MAX_CARS * IN_N];
static float hidden_scratch[MAX_HIDDEN];

// Brains, flat and contiguous: [weightsIH (IN_N*h) | weightsHO (h*OUT_N) | biasH (h) | biasO (OUT_N)]
// per car, one car after another. Same row-major order the JS edition uses, so
// the JSON import/export code needs no changes.
//
// One slot past the live population is the stash: the all-time best brain lives
// there so evolve() can clone from it without it being bred over, and "Load AI"
// parks an imported brain there before seeding.
#define STASH_SLOT MAX_CARS
static float brains[(MAX_CARS + 1) * BRAIN_MAX];
static float render_buf[MAX_CARS * RENDER_STRIDE];
static float fitness_buf[MAX_CARS * 3];

static i32 pop_n = 0, pop_id_offset = 0, brain_stride_v = 0;

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
    car_nextCP[i] = 1; car_laps[i] = 0; car_cpReached[i] = 0;
    car_lastLap[i] = 0.0f; car_prevLapFrame[i] = 0;
    for (i32 k = 0; k < SENS_N; k++) car_sensors[i * SENS_N + k] = 0.0f;
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
    for (i32 i = 0; i < pop_n; i++) reset_car(i);
    return pop_n;
}

// Fill every brain with fresh uniform [-1,1] weights.
__attribute__((export_name("pop_randomize_brains")))
void pop_randomize_brains(void) {
    i32 total = pop_n * brain_stride_v;
    for (i32 i = 0; i < total; i++) brains[i] = rnd11();
}

__attribute__((export_name("pop_reset")))
void pop_reset(void) { for (i32 i = 0; i < pop_n; i++) reset_car(i); }

__attribute__((export_name("car_count")))
i32 car_count(void) { return pop_n; }

// ---------------------------------------------------------------------------
// The step. Everything below runs per car per frame, so it is the only code in
// the project where the shape of the data actually matters.
// ---------------------------------------------------------------------------
static inline i32 fastIntersect(float Ax, float Ay, float Bx, float By,
                                float Cx, float Cy, float Dx, float Dy) {
    float bottom = (Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay);
    if (bottom == 0.0f) return 0;
    float t = ((Dx - Cx) * (Ay - Cy) - (Dy - Cy) * (Ax - Cx)) / bottom;
    if (t < 0.0f || t > 1.0f) return 0;
    float u = ((Cy - Ay) * (Ax - Bx) - (Cx - Ax) * (Ay - By)) / bottom;
    if (u < 0.0f || u > 1.0f) return 0;
    return 1;
}
// Per-car broad-phase scratch, refilled every step. `sc_` holds the walls a
// sensor could reach; `nc_` the much smaller set the car could physically touch
// this frame. Both are struct-of-arrays and padded to a multiple of four.
#define MAX_SCRATCH 16384
static float sc_x1[MAX_SCRATCH], sc_y1[MAX_SCRATCH], sc_dx[MAX_SCRATCH], sc_dy[MAX_SCRATCH];
static float nc_x1[MAX_SCRATCH], nc_y1[MAX_SCRATCH], nc_dx[MAX_SCRATCH], nc_dy[MAX_SCRATCH];
static i32 sc_n, nc_n;

#ifdef __wasm_simd128__
typedef float v4f __attribute__((vector_size(16)));
typedef int   v4i __attribute__((vector_size(16)));
static inline v4f vsplat(float x) { return (v4f){ x, x, x, x }; }
// Branchless lane select: mask lanes are all-ones or all-zeros from a compare.
static inline v4f vsel(v4i m, v4f a, v4f b) { return (v4f)((m & (v4i)a) | (~m & (v4i)b)); }
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
    v4f best = one;
    for (i32 i = 0; i < n; i += 4) {
        v4f cx = *(const v4f *)(x1 + i), cy = *(const v4f *)(y1 + i);
        v4f ddx = *(const v4f *)(dx + i), ddy = *(const v4f *)(dy + i);
        v4f bottom = ddy * vabx - ddx * vaby;
        v4f acx = vAx - cx, acy = vAy - cy;
        // A zero denominator makes these inf or NaN; every comparison below is
        // false for NaN, so the lane falls out through the same mask that
        // rejects an out-of-range hit. No special case needed.
        v4f t = (ddx * acy - ddy * acx) / bottom;
        v4f u = (acy * vabx - acx * vaby) / bottom;
        v4i m = (bottom != zero) & (t >= zero) & (t <= one) & (u >= zero) & (u <= one);
        v4f cand = vsel(m, t, one);
        best = vsel(cand < best, cand, best);
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
        float t = (ddx * acy - ddy * acx) / bottom;
        if (t < 0.0f || t > 1.0f) continue;
        float u = (acy * abx - acx * aby) / bottom;
        if (u < 0.0f || u > 1.0f) continue;
        if (t < best) best = t;
    }
    return best;
#endif
}

// Does this segment cross any wall in the run? Same math, but it can stop at
// the first hit, so it stays scalar — a crash ends the car's frame anyway.
static i32 segmentHits(float Ax, float Ay, float Bx, float By,
                       const float *x1, const float *y1, const float *dx, const float *dy, i32 n) {
    const float abx = Bx - Ax, aby = By - Ay;
    for (i32 i = 0; i < n; i++) {
        float ddx = dx[i], ddy = dy[i];
        float bottom = ddy * abx - ddx * aby;
        if (bottom == 0.0f) continue;
        float acx = Ax - x1[i], acy = Ay - y1[i];
        float t = (ddx * acy - ddy * acx) / bottom;
        if (t < 0.0f || t > 1.0f) continue;
        float u = (acy * abx - acx * aby) / bottom;
        if (u >= 0.0f && u <= 1.0f) return 1;
    }
    return 0;
}

static void feedForward(i32 i) {
    const float *in = &car_in[i * IN_N];
    const float *b = &brains[i * brain_stride_v];
    i32 h = cfg_hidden;
    const float *wIH = b;
    const float *wHO = b + IN_N * h;
    const float *bH  = wHO + h * OUT_N;
    const float *bO  = bH + h;
    for (i32 k = 0; k < h; k++) {
        float sum = bH[k];
        for (i32 j = 0; j < IN_N; j++) sum += in[j] * wIH[j * h + k];
        hidden_scratch[k] = tanhf_(sum);
    }
    float *out = &car_out[i * OUT_N];
    for (i32 k = 0; k < OUT_N; k++) {
        float sum = bO[k];
        for (i32 j = 0; j < h; j++) sum += hidden_scratch[j] * wHO[j * OUT_N + k];
        out[k] = tanhf_(sum);
    }
}

static void updateCar(i32 i) {
    car_ttl[i]--; car_frames[i]++;
    if (car_ttl[i] <= 0) { car_crashed[i] = 1; return; }

    float steer = car_out[i * OUT_N];
    float throttle = car_out[i * OUT_N + 1];

    float speedFactor = minf(car_speed[i] / 4.0f, 1.0f);
    car_angle[i] += steer * cfg_turnSpeed * (0.2f + 0.8f * speedFactor);

    double sd, cd; sincos_d((double)car_angle[i], &sd, &cd);
    float sinA = (float)sd, cosA = (float)cd;
    float vx = car_vx[i], vy = car_vy[i];

    if (throttle > 0.0f) { vx += cosA * throttle * cfg_accel; vy += sinA * throttle * cfg_accel; }
    else { vx *= 0.95f; vy *= 0.95f; }

    float latVel = vx * (-sinA) + vy * cosA;
    float grip = cfg_grip; if (absf(latVel) > 2.5f) grip *= 0.8f;

    vx += (-sinA) * -latVel * grip;
    vy += cosA * -latVel * grip;
    vx *= 0.99f; vy *= 0.99f;

    float speed = sqrtf_(vx * vx + vy * vy);
    if (speed > cfg_maxSpeed) { float r = cfg_maxSpeed / speed; vx *= r; vy *= r; speed = cfg_maxSpeed; }

    car_vx[i] = vx; car_vy[i] = vy; car_speed[i] = speed;
    float prevX = car_x[i], prevY = car_y[i];
    car_x[i] += vx; car_y[i] += vy;
    car_fitness[i] += (speed / cfg_maxSpeed) * 0.1f;

    if (car_x[i] < -100.0f || car_x[i] > 1300.0f || car_y[i] < -100.0f || car_y[i] > 1000.0f) { car_crashed[i] = 1; return; }
    if (tk_cp_n <= 0 || !wbs_start) { car_crashed[i] = 1; return; }

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
        for (i32 w = wStart; w < wEnd; w++) {
            float ddx = cx_ - b_mx[w], ddy = cy_ - b_my[w];
            float d2 = ddx * ddx + ddy * ddy;
            if (d2 > b_r2[w]) continue;
            if (sn < MAX_SCRATCH) {
                sc_x1[sn] = b_x1[w]; sc_y1[sn] = b_y1[w];
                sc_dx[sn] = b_dx[w]; sc_dy[sn] = b_dy[w];
                sn++;
            }
            float cr = collR + b_hl[w];
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
    Checkpoint *nCP = (nxt >= 0 && nxt < tk_cp_n) ? &tk_cps[nxt] : 0;
    if (nCP) {
        float dx = cx_ - nCP->cx, dy = cy_ - nCP->cy;
        if (dx * dx + dy * dy > 160000.0f) { car_crashed[i] = 1; return; }  // off course

        float mx = (nCP->p1x + nCP->p2x) * 0.5f, my = (nCP->p1y + nCP->p2y) * 0.5f;
        float cdx = cx_ - mx, cdy = cy_ - my;
        if (cdx * cdx + cdy * cdy < 2500.0f) {
            if (fastIntersect(prevX, prevY, cx_, cy_, nCP->p1x, nCP->p1y, nCP->p2x, nCP->p2y)
                || (cdx * cdx + cdy * cdy < 400.0f)) {
                car_cpReached[i]++;
                car_nextCP[i] = (nxt + 1) % tk_cp_n;
                car_ttl[i] += 150; if (car_ttl[i] > 600) car_ttl[i] = 600;
                car_fitness[i] += 500.0f * fitMult;
                if (car_nextCP[i] == 0 && tk_cp_n > 2) {
                    car_laps[i]++;
                    car_lastLap[i] = (float)(car_frames[i] - car_prevLapFrame[i]) / 60.0f;
                    car_prevLapFrame[i] = car_frames[i];
                    car_fitness[i] += 3000.0f * fitMult;
                }
            }
        }
    }

    // Sensors. The inner loop is the hot spot of the whole program: seven rays
    // against every wall in the window, for every living car, every frame.
    float *sens = &car_sensors[i * SENS_N];
    float *in = &car_in[i * IN_N];
    static const float SENSOR_ANGLES[SENS_N] = {
        -PI_F / 2.0f, -PI_F / 3.0f, -PI_F / 6.0f, 0.0f, PI_F / 6.0f, PI_F / 3.0f, PI_F / 2.0f
    };
    for (i32 k = 0; k < SENS_N; k++) {
        float rA = car_angle[i] + SENSOR_ANGLES[k];
        double rs, rc; sincos_d((double)rA, &rs, &rc);
        float ex = cx_ + (float)rc * SENSOR_LEN, ey = cy_ + (float)rs * SENSOR_LEN;
        float minT = raycast(cx_, cy_, ex, ey, sc_x1, sc_y1, sc_dx, sc_dy, sc_n);
        sens[k] = 1.0f - minT;
        in[k] = 1.0f - minT;
    }

    if (nCP) {
        float tX = (nCP->p1x + nCP->p2x) * 0.5f, tY = (nCP->p1y + nCP->p2y) * 0.5f;
        float relAng = atan2f_(tY - cy_, tX - cx_) - car_angle[i];
        while (relAng >  PI_F) relAng -= 2.0f * PI_F;
        while (relAng < -PI_F) relAng += 2.0f * PI_F;
        in[SENS_N] = car_speed[i] / cfg_maxSpeed;
        in[SENS_N + 1] = relAng / PI_F;
    }

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

__attribute__((export_name("run")))
i32 run(i32 iters) {
    i32 maxLaps = 0;
    i32 allCrashed = 1;
    for (i32 it = 0; it < iters; it++) {
        allCrashed = 1;
        for (i32 c = 0; c < pop_n; c++) {
            if (!car_crashed[c]) {
                updateCar(c);
                allCrashed = 0;
                if (car_laps[c] > maxLaps) maxLaps = car_laps[c];
            }
        }
        if (allCrashed || maxLaps >= cfg_targetLaps) break;
    }
    last_all_crashed = allCrashed;
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
        for (i32 k = 0; k < SENS_N; k++) r[11 + k] = car_sensors[i * SENS_N + k];
    }
}

// Fitness alone, for the evolve step — far cheaper than a full render pass when
// all the main thread wants is who won. Hyper mode never draws a car, so this
// is the only thing it ships back per generation.
__attribute__((export_name("write_fitness")))
void write_fitness(void) {
    for (i32 i = 0; i < pop_n; i++) {
        fitness_buf[i * 3]     = car_fitness[i];
        fitness_buf[i * 3 + 1] = (float)car_laps[i];
        fitness_buf[i * 3 + 2] = car_lastLap[i];
    }
}

__attribute__((export_name("alive_count")))
i32 alive_count(void) {
    i32 n = 0;
    for (i32 i = 0; i < pop_n; i++) if (!car_crashed[i]) n++;
    return n;
}

// ---------------------------------------------------------------------------
// Evolution. Selection needs the whole population, which is spread across
// workers, so the main thread's instance owns it: it holds every brain, breeds
// the next generation here, and hands each worker back its slice.
// ---------------------------------------------------------------------------
static i32 order[MAX_CARS];
static float ev_fitness[MAX_CARS];
static float brains_next[MAX_CARS * BRAIN_MAX];

// JS writes the gathered per-car fitness straight into this array rather than
// making one call per car.
__attribute__((export_name("ev_fitness_ptr")))
i32 ev_fitness_ptr(void) { return (i32)(unsigned long)ev_fitness; }

__attribute__((export_name("get_order")))
i32 get_order(i32 rank) { return (rank >= 0 && rank < pop_n) ? order[rank] : 0; }

// Descending insertion sort on the fitness index. The population tops out at
// 500 and is nearly sorted generation to generation, so this beats the setup
// cost of anything cleverer.
static void sort_by_fitness(void) {
    for (i32 i = 0; i < pop_n; i++) order[i] = i;
    for (i32 i = 1; i < pop_n; i++) {
        i32 k = order[i]; float f = ev_fitness[k];
        i32 j = i - 1;
        while (j >= 0 && ev_fitness[order[j]] < f) { order[j + 1] = order[j]; j--; }
        order[j + 1] = k;
    }
}

// Uniform crossover of two parents drawn from the top 20%, then per-weight
// mutation — the same scheme as the JS edition, minus the object churn. The
// elite clones come from the stash (the all-time best), not from this
// generation's winner, which is what stops a bad generation regressing.
__attribute__((export_name("evolve")))
void evolve(i32 eliteClones, i32 hasGlobalBest) {
    sort_by_fitness();
    i32 stride = brain_stride_v;
    i32 written = 0;

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

    i32 poolSize = maxi(2, pop_n / 5);
    if (poolSize > pop_n) poolSize = pop_n;
    for (i32 i = written; i < pop_n; i++) {
        const float *p1 = &brains[order[(i32)(rnd01() * (float)poolSize) % poolSize] * stride];
        const float *p2 = &brains[order[(i32)(rnd01() * (float)poolSize) % poolSize] * stride];
        float *dst = &brains_next[i * stride];
        for (i32 j = 0; j < stride; j++) {
            float v = rnd01() < 0.5f ? p1[j] : p2[j];
            if (rnd01() < cfg_mutationRate) v += rnd11() * 0.5f;
            dst[j] = v;
        }
    }
    i32 total = pop_n * stride;
    for (i32 i = 0; i < total; i++) brains[i] = brains_next[i];
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
        for (i32 j = 0; j < stride; j++) {
            float v = src[j];
            if (rnd01() < cfg_mutationRate) v += rnd11() * 0.5f;
            dst[j] = v;
        }
    }
}

__attribute__((export_name("seed_rng")))
void seed_rng(u32 s) { rng_seed(s); }

// ---------------------------------------------------------------------------
// Math self-test hooks — tools/mathtest.mjs drives these against JS Math so a
// bad polynomial coefficient shows up at build time rather than as a car that
// mysteriously steers into a wall.
// ---------------------------------------------------------------------------
__attribute__((export_name("t_sin")))   float t_sin(float x)   { return sinf_(x); }
__attribute__((export_name("t_cos")))   float t_cos(float x)   { return cosf_(x); }
__attribute__((export_name("t_atan2"))) float t_atan2(float y, float x) { return atan2f_(y, x); }
__attribute__((export_name("t_tanh")))  float t_tanh(float x)  { return tanhf_(x); }
__attribute__((export_name("t_acos")))  float t_acos(float x)  { return acosf_(x); }
__attribute__((export_name("t_exp")))   double t_exp(double x) { return exp_d(x); }
