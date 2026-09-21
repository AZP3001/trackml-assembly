#!/usr/bin/env bash
# Build external/tailwind.css from the classes actually used in the page.
#
# The page used to load Tailwind's browser build (tailwind.js, 407KB), which is
# a CSS compiler that scans the DOM and generates a stylesheet at runtime, on
# every single page load. Tailwind's own docs call that a development-only
# path. This produces the same CSS once, ahead of time, as a file the browser
# can cache.
#
# Re-run it whenever a class is added to the markup or to a class string built
# in JS — CI checks that the committed CSS matches what this produces.
set -euo pipefail
cd "$(dirname "$0")/.."
npx --yes tailwindcss@3.4.17 \
    --config tools/css/tailwind.config.js \
    --input  tools/css/input.css \
    --output external/tailwind.css \
    --minify
echo "built external/tailwind.css ($(wc -c < external/tailwind.css) bytes)"
