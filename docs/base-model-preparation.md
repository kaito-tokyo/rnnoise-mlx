# Preparing data for a base model

This procedure generates RNNoise training features from LibriTTS-R and
RIRS_NOISES plus the curated DNS/MKA/Multi-Pressure mixture documented in
[`noise-mix-dataset.md`](noise-mix-dataset.md). Original archives remain
unchanged; extracted and generated files are stored under `data/`.

## 1. Extract the archives

Ensure that sufficient disk space is available. The extracted corpus may
require more than 100 GB.

```text
data/corpus/
  LibriTTS_R/train-clean-100/
  LibriTTS_R/train-clean-360/
  LibriTTS_R/dev-clean/
  LibriTTS_R/test-clean/
  musan/noise/                 # only the automatically curated background subset
  DNS5/
  MKA/
  multi_pressure_keyboard/
  RIRS_NOISES/
```

```sh
mkdir -p data/corpus
tar -xzf /Users/umireon/Datasets/train_clean_100.tar.gz -C data/corpus
tar -xzf /Users/umireon/Datasets/train_clean_360.tar.gz -C data/corpus
tar -xzf /Users/umireon/Datasets/dev_clean.tar.gz -C data/corpus
tar -xzf /Users/umireon/Datasets/test_clean.tar.gz -C data/corpus
tar -xzf /Users/umireon/Datasets/musan.tar.gz -C data/corpus
unzip -q /Users/umireon/Datasets/rirs_noises.zip -d data/corpus
```

## 2. Create deterministic manifests and PCM streams

Seed 141 and a SHA-256 path hash reserve 10% of MUSAN and RIR files for
evaluation. Official corpus boundaries are preserved: train-clean is used for
training, dev-clean for development evaluation, and test-clean only for final
evaluation.

```sh
.venv/bin/python -m rnnoise_mlx.tools.prepare_base_dataset manifests \
  --corpus data/corpus --output data/manifests --include-train-360

.venv/bin/python -m rnnoise_mlx.tools.prepare_base_dataset render \
  --corpus data/corpus --manifests data/manifests --output data/prepared \
  --speech-limit 40000 --noise-limit 1500 --rir-limit 512 --workers 8

.venv/bin/python -m rnnoise_mlx.tools.prepare_noise_mix audit \
  --corpus data/corpus --output data/noise-mix/audit.json
.venv/bin/python -m rnnoise_mlx.tools.prepare_noise_mix render \
  --audit data/noise-mix/audit.json --output data/noise-mix/prepared \
  --train-hours 1 --eval-hours 0.1

cp data/noise-mix/prepared/train_background.pcm data/prepared/train_background.pcm
cp data/noise-mix/prepared/train_foreground.pcm data/prepared/train_foreground.pcm
cp data/noise-mix/prepared/eval_background.pcm data/prepared/eval_background.pcm
cp data/noise-mix/prepared/eval_foreground.pcm data/prepared/eval_foreground.pcm
```

The final four commands replace only the base renderer's MUSAN noise streams.
They retain its prepared speech and RIR files while ensuring feature generation
reads the curated DNS/MKA/Multi-Pressure streams from `data/prepared`.

Manifest entries are ordered by a seeded hash, so limits do not introduce a
filename-order speaker or subset bias. `dump_features` transforms every RIR at
startup and retains it in memory, so the standard command limits each split to
512 RIRs. FFmpeg conversion is parallelized while output remains in manifest
order; lower `--workers` if memory pressure is visible. Use `--limit 10` only
to validate the pipeline; that output does not contain enough audio for
training.

`manifest.json` records the seed, input root, and group counts. Preserve all
manifests with model provenance. Rendering requires FFmpeg and produces 48 kHz
mono s16le audio plus 48 kHz mono little-endian float32 RIR files.

## 3. Build `dump_features`

The required upstream code is vendored under `Vendors/xiph-rnnoise`. Its README
records the source revision, license, dependencies, and applied 65536-point RIR
FFT fix. No upstream neural weights are vendored.

```sh
.venv/bin/python -m rnnoise_mlx.tools.build_dump_features
```

## 4. Generate and verify features

Start with 256 training and 64 evaluation sequences for pipeline validation.
Each sequence has 2,000 frames and each frame has 98 little-endian float32
values. First generate deterministic speech-window offsets, then pass them to
`dump_features`; the script verifies the exact feature output size.

```sh
.venv/bin/python -m rnnoise_mlx.tools.select_speech_offsets \
  data/prepared/train_speech.pcm data/offsets/train.txt \
  --count 256 --seed 141
.venv/bin/python -m rnnoise_mlx.tools.select_speech_offsets \
  data/prepared/eval_speech.pcm data/offsets/eval.txt \
  --count 64 --seed 142

.venv/bin/python -m rnnoise_mlx.tools.generate_features \
  Vendors/xiph-rnnoise/dump_features data/prepared data/features \
  --train-count 256 --eval-count 64 --speech-offsets data/offsets
```

The selector treats each split as one concatenated PCM stream and uses exactly
one SplitMix64 draw per sequence. Clip boundaries are not special. Preserve the
offset manifests and their adjacent metadata JSON files with model provenance.
The deterministic contract covers speech selection only; RNNoise noise/RIR and
signal augmentation remain downstream random transformations.

Increase the training count to at least 10,000 for model candidates. Upstream
recommends 200,000 or more, so scale gradually after measuring storage and
generation time. 10,000 sequences require about 7.84 GB; 200,000 require about
156.8 GB.

## 5. Smoke training

```sh
.venv/bin/python -m rnnoise_mlx.training.train data/features/train.f32 runs/base-smoke \
  --eval-features data/features/eval.f32 \
  --batch-size 8 --sequence-length 2000 \
  --segmented-tbptt-length 100 --segmented-tbptt-state carry \
  --max-updates 320 --seed 141 \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-experiment "$MLFLOW_EXPERIMENT" --mlflow-run-name base-smoke
```

Verify 320 completed updates, finite losses, improved evaluation loss, and
identical evaluation after reload in `training.json`. This validates the
pipeline, not release audio quality. Retrain promising configurations with
segment length 500 and at least 10,000 updates, then run fixed-WAV listening
tests.

Feature generation writes `train.manifest.json` and `eval.manifest.json` next
to the bulk `.f32` files. Training uploads these manifests, `run-config.json`,
`model-config.json`, `training.json`, the final SafeTensors model, and every
complete resumable checkpoint to MLflow. Bulk feature and corpus files remain
in the shared dataset store. Pass additional small selection or mix manifests
with repeated `--provenance-artifact PATH` options; files larger than 16 MiB
are rejected to prevent accidental corpus uploads.

Feature augmentation uses `splitmix64-domain-v1`. Random values are derived
from the global seed, absolute sequence index, domain, and a domain-local
counter. The fixed domains are: input offsets (1), speech start (2), gains and
mute choices (3), response filters (4), low-pass cutoff (5), RIR selection
(6), clipping (7), and quantization (8). `-sequence_start` therefore permits
parallel or resumed generation without changing any sequence bytes.

Publish completed files into the immutable shared store only after generation:

```sh
python -m rnnoise_mlx.tools.feature_store publish train.f32 \
  /Users/umireon/Datasets/rnnoise-mlx-features/v1/train/generation-000 \
  --sequence-count 10000
python -m rnnoise_mlx.tools.feature_store verify \
  /Users/umireon/Datasets/rnnoise-mlx-features/v1/train/generation-000
```

Publishing takes a per-generation lock, verifies size and SHA-256 in a
temporary directory, atomically renames it into place, and rebuilds
`v1/index.json`. The immutable generation ID is derived from the feature
SHA-256; seed and sequence metadata come from the required adjacent generation
manifest. Re-publishing the same valid generation reuses it.

## Provenance checklist

- Archive filenames and SHA-256 values
- Dataset versions, source references, licenses, and required attributions for
  LibriTTS-R, MUSAN, RIRS_NOISES, DNS Challenge 5 Freesound, MKA version 3,
  and the Multi-Pressure Keyboard dataset version v8
- Pinned RNNoise source revision
- Every manifest and `manifest.json`
- Feature counts, training command, seed, and repository commit
- `training.json` and fixed-WAV listening-test results
