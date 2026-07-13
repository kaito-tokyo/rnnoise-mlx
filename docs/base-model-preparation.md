# Preparing data for a base model

This procedure generates RNNoise training features from LibriTTS-R, MUSAN, and
RIRS_NOISES without using upstream weights or outputs. Original archives remain
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
  musan/{noise,music,speech}/
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
```

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

## Provenance checklist

- Archive filenames and SHA-256 values
- Dataset versions and licenses for LibriTTS-R, MUSAN, and RIRS_NOISES
- Pinned RNNoise source revision
- Every manifest and `manifest.json`
- Feature counts, training command, seed, and repository commit
- `training.json` and fixed-WAV listening-test results
