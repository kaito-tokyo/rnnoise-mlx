#!/bin/sh
# Generate train/eval RNNoise feature files with the pinned dump_features binary.
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 <dump_features> <prepared-dir> <output-dir> [train-count] [eval-count]" >&2
  exit 2
fi

dump_features=$1
prepared=$2
output=$3
train_count=${4:-10000}
eval_count=${5:-500}

[ -x "$dump_features" ] || { echo "not executable: $dump_features" >&2; exit 1; }
mkdir -p "$output"

"$dump_features" -rir_list "$prepared/train_rir_list.txt" \
  "$prepared/train_speech.pcm" "$prepared/train_background.pcm" \
  "$prepared/train_foreground.pcm" "$output/train.f32" "$train_count"

"$dump_features" -rir_list "$prepared/eval_rir_list.txt" \
  "$prepared/eval_speech.pcm" "$prepared/eval_background.pcm" \
  "$prepared/eval_foreground.pcm" "$output/eval.f32" "$eval_count"

python3 - "$output/train.f32" "$train_count" "$output/eval.f32" "$eval_count" <<'PY'
from pathlib import Path
import sys

bytes_per_sequence = 2000 * 98 * 4
for path, count in zip(sys.argv[1::2], map(int, sys.argv[2::2])):
    actual = Path(path).stat().st_size
    expected = count * bytes_per_sequence
    if actual != expected:
        raise SystemExit(f"unexpected size for {path}: {actual}, expected {expected}")
    print(f"verified {path}: {actual} bytes")
PY
