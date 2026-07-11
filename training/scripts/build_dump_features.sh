#!/bin/sh
# Build the vendored RNNoise feature extractor without downloading upstream data.
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
vendor="$root/training/vendor/xiph-rnnoise"
output=${1:-"$vendor/dump_features"}

mkdir -p "$(dirname "$output")"
clang -O3 -DTRAINING -I"$vendor" -I"$vendor/src" -I"$vendor/include" \
  "$vendor/src/dump_features.c" \
  "$vendor/src/denoise.c" \
  "$vendor/src/pitch.c" \
  "$vendor/src/celt_lpc.c" \
  "$vendor/src/kiss_fft.c" \
  "$vendor/src/parse_lpcnet_weights.c" \
  "$vendor/src/rnnoise_tables.c" \
  -lm -o "$output"

test -x "$output"
"$output" 2>&1 | grep -q '^usage:'
printf 'built %s from training/vendor/xiph-rnnoise\n' "$output"
