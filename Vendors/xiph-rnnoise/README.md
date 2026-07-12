# Vendored Xiph RNNoise feature extractor

This directory contains only the C feature-extraction path needed to create
the 98-float-per-frame `.f32` files consumed by this repository's MLX trainer.
It is vendored from Xiph RNNoise revision
`70f1d256acd4b34a572f999a05c87bf00b67730d`.

The vendored source includes these upstream components:

- `dump_features.c`, `denoise.c`, `pitch.c`, `celt_lpc.c`, `kiss_fft.c`
- `parse_lpcnet_weights.c` and `rnnoise_tables.c`
- the headers required by those translation units

It deliberately does not include upstream model weights, the old Keras/HDF5
training scripts, or the regular RNNoise runtime library. The vendored
`rnnoise_data.h` contains architecture and shape declarations only.

## Build

From this directory on macOS:

```sh
clang -O3 -DTRAINING -I. -Isrc -Iinclude \
  src/dump_features.c src/denoise.c src/pitch.c src/celt_lpc.c \
  src/kiss_fft.c src/parse_lpcnet_weights.c src/rnnoise_tables.c \
  -lm -o dump_features
```

The executable accepts raw 48 kHz mono little-endian PCM and emits 98
little-endian float32 values per frame:

```sh
./dump_features -rir_list rir_list.txt \
  speech.pcm background_noise.pcm foreground_noise.pcm features.f32 10000
```

The RIR FFT requires the `fstride[MAXFACTORS + 1]` bound in `src/kiss_fft.c`.
The vendor copy also exposes `rnnoise_process_frame_with_callback`, which keeps
the upstream analysis, pitch filtering, and synthesis path while obtaining the
32 neural gains from the external C/BNNS model. Both changes are maintained
directly in the vendored source rather than as build-time patches.

## Provenance and licensing

The original Xiph RNNoise source is distributed under the BSD-3-Clause terms
in [`COPYING`](COPYING). Copyright notices in individual source files are
preserved. `parse_lpcnet_weights.c` retains its separate Amazon copyright
notice. Original source: <https://github.com/xiph/rnnoise>.
