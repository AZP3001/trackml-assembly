#!/usr/bin/env bash
# Build sim.c into a freestanding wasm32 module.
#
# No Emscripten: plain clang with the wasm32 backend LLVM already ships, plus
# wasm-ld. That means no JS glue file, no runtime to boot, and an output you can
# read with wasm-objdump. Ubuntu/Debian: `apt install clang lld`.
#
# The build is reproducible — same compiler, same bytes — so CI can rebuild and
# diff against the committed wasm/sim.wasm to prove the two are in sync.
set -euo pipefail

cd "$(dirname "$0")"

CC="${CC:-clang}"

if ! command -v "$CC" >/dev/null 2>&1; then
    echo "error: $CC not found. Install clang (>= 15) and lld." >&2
    exit 1
fi

# Two binaries from one source. sim-simd.wasm turns on the 128-bit SIMD
# proposal, which lets the raycast test four walls per instruction; sim.wasm is
# the scalar fallback for engines that would refuse to instantiate the other.
# engine.js probes for SIMD support and picks one. The #ifdef that splits them
# is __wasm_simd128__, set by -msimd128.
build() {
    local out="$1"; shift
    "$CC" \
        --target=wasm32 \
        -O3 \
        -fno-builtin \
        -fvisibility=hidden \
        -nostdlib \
        -Wall -Wextra \
        -Wno-unused-parameter \
        -Wl,--no-entry \
        -Wl,--initial-memory=4194304 \
        -Wl,-z,stack-size=262144 \
        -Wl,--strip-all \
        "$@" \
        -o "$out" \
        sim.c
    echo "built $out ($(wc -c < "$out") bytes)"
}

# -fno-builtin matters more than it looks: without it clang recognises the byte
# loops in memcpy/memset as the very functions they implement and rewrites them
# into calls to themselves. There is no -ffast-math either — the geometry leans
# on exact float comparisons (`if (bottom == 0.0f)`), and reassociating those
# would quietly change which walls a ray hits.
build sim.wasm
build sim-simd.wasm -msimd128
