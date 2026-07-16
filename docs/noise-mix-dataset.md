# DNS/MKA/Multi-Pressure noise mixture

The production-oriented noise population uses DNS5 Freesound, MKA, and the
Multi-Pressure Keyboard dataset. MUSAN music and speech are excluded. At most
eight automatically qualified files from MUSAN's annotated background list are
retained as a small background supplement. OpenSLR RIRs remain spatial
augmentation and are not treated as noise.

## Audit and render

The audit is deterministic (seed 141), preserves the source files, removes MKA
duplicates, applies the documented waveform gates, and splits source identities
90/10 within each category.

```sh
python -m rnnoise_mlx.tools.prepare_noise_mix audit \
  --corpus /Users/umireon/Datasets/rnnoise-mlx-base-corpus \
  --output data/noise-mix/audit.json

python -m rnnoise_mlx.tools.prepare_noise_mix render \
  --audit data/noise-mix/audit.json \
  --output data/noise-mix/prepared \
  --train-hours 1 --eval-hours 0.1
```

The audit emits the full JSON report plus train/eval JSONL manifests. Rendering
emits 48 kHz mono signed 16-bit background and foreground PCM streams and a
provenance manifest containing input paths, weights, allocated samples, and
SHA-256 checksums.

Background duration is 90% DNS `fan` and 10% curated MUSAN. Foreground duration
is 60% DNS, 25% MKA, and 15% Multi-Pressure. Multi-Pressure events retain their
recorded level, rotate High/Medium/Low, and receive deterministic 0.2--1.2 second
gaps.

## Foreground frequency experiment

`generate_features` preserves the upstream one-in-eight default and exposes the
comparison setting explicitly:

```sh
python -m rnnoise_mlx.tools.generate_features \
  Vendors/xiph-rnnoise/dump_features data/prepared data/features \
  --foreground-probability-denominator 4
```

Use identical speech PCM, speech offsets, RIR list, feature seed, model seed,
sequence count, and update count when comparing denominator 8 and 4. Feature
manifests record the selected denominator.

Build the fixed five-SNR WAV population from one held-out clean utterance with:

```sh
python -m rnnoise_mlx.tools.build_noise_evaluation_set \
  --audit data/noise-mix/audit.json --clean held-out-clean.wav \
  --output data/noise-mix/evaluation
```

The manifest covers every DNS class, every MKA device/VoIP path, all three
pressure levels, a rejected MUSAN example, and an unmodified clean-only case.

## First full audit result

The 2026-07-17 local audit examined 13,334 candidates and accepted 13,183. It
retained 2,791 MKA clips after excluding 77 clips for clipping, two for DC
offset, and seven exact duplicates. Seven MUSAN files passed the automatic
background gates. DNS `mislabeled` clips were excluded. The generated one-hour
train and six-minute evaluation PCM streams decoded successfully and matched
the requested sample allocations exactly.
