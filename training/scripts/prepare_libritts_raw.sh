#!/bin/sh
# Deterministically convert a LibriTTS-R subset to RNNoise input PCM.
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 <libritts-root> <output.pcm> [clip-count]" >&2
  exit 2
fi
input_root=$1
output=$2
clip_count=${3:-1000}
manifest="${output%.pcm}.clips.txt"
tmp_manifest=$(mktemp)
trap 'rm -f "$tmp_manifest"' EXIT

find "$input_root" -type f -name '*.wav' -print | LC_ALL=C sort |
  awk 'BEGIN { srand(141) } { printf "%.17f\t%s\n", rand(), $0 }' |
  LC_ALL=C sort -n | head -n "$clip_count" | cut -f2- > "$tmp_manifest"
actual_count=$(wc -l < "$tmp_manifest" | tr -d ' ')
[ "$actual_count" -gt 0 ] || { echo "no WAV files found below $input_root" >&2; exit 1; }
mkdir -p "$(dirname "$output")"
: > "$output"
while IFS= read -r wav; do
  ffmpeg -v error -nostdin -i "$wav" -ar 48000 -ac 1 -f s16le - >> "$output"
  dd if=/dev/zero bs=9600 count=1 2>/dev/null >> "$output"
done < "$tmp_manifest"
mv "$tmp_manifest" "$manifest"
trap - EXIT
printf 'converted %s clips to %s\n' "$actual_count" "$output"
