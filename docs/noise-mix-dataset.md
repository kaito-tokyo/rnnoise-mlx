# DNS/MKA/Multi-Pressure noise mixture

The production-oriented noise population uses DNS5 Freesound, MKA, and the
Multi-Pressure Keyboard dataset. MUSAN music and speech are excluded. At most
eight automatically qualified files from MUSAN's annotated background list are
retained as a small background supplement. OpenSLR RIRs remain spatial
augmentation and are not treated as noise.

## Source versions and licenses

- **DNS Challenge 5 Freesound noise**: the ICASSP 2023 DNS Challenge 5
  distribution from the
  [Microsoft DNS-Challenge repository](https://github.com/microsoft/DNS-Challenge).
  The upstream dataset license list states that only CC0 Freesound files were
  selected; those files are licensed under
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
- **Multi-Keyboard Acoustic (MKA) Datasets**: Mendeley Data version 3,
  DOI [`10.17632/bpt2hvf8n3.3`](https://doi.org/10.17632/bpt2hvf8n3.3),
  licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Audio dataset of keyboard keystrokes** (the Multi-Pressure source):
  Zenodo version v8, DOI
  [`10.5281/zenodo.19453177`](https://doi.org/10.5281/zenodo.19453177),
  licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Preserve each downloaded archive filename and checksum with the release
provenance. CC BY sources require attribution to the creators in distributions
and accompanying model documentation.

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
