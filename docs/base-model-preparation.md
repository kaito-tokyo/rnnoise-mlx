# Preparing data for a base model

This procedure generates RNNoise training features from LibriTTS-R, MUSAN, and
RIRS_NOISES without using upstream weights or outputs. Original archives remain
unchanged; extracted and generated files are stored under `training/data/`.

## 1. Extract the archives

Ensure that sufficient disk space is available. The extracted corpus may
require more than 100 GB.

```text
training/data/corpus/
  LibriTTS_R/train-clean-100/
  LibriTTS_R/train-clean-360/
  LibriTTS_R/dev-clean/
  LibriTTS_R/test-clean/
  musan/{noise,music,speech}/
  RIRS_NOISES/
```

```sh
mkdir -p training/data/corpus
tar -xzf /Users/umireon/Datasets/train_clean_100.tar.gz -C training/data/corpus
tar -xzf /Users/umireon/Datasets/train_clean_360.tar.gz -C training/data/corpus
tar -xzf /Users/umireon/Datasets/dev_clean.tar.gz -C training/data/corpus
tar -xzf /Users/umireon/Datasets/test_clean.tar.gz -C training/data/corpus
tar -xzf /Users/umireon/Datasets/musan.tar.gz -C training/data/corpus
unzip -q /Users/umireon/Datasets/rirs_noises.zip -d training/data/corpus
```

## 2. Create deterministic manifests and PCM streams

Seed 141 and a SHA-256 path hash reserve 10% of MUSAN and RIR files for
evaluation. Official corpus boundaries are preserved: train-clean is used for
training, dev-clean for development evaluation, and test-clean only for final
evaluation.

```sh
python3 training/scripts/prepare_base_dataset.py manifests \
  --corpus training/data/corpus --output training/data/manifests --include-train-360

python3 training/scripts/prepare_base_dataset.py render \
  --corpus training/data/corpus --manifests training/data/manifests --output training/data/prepared \
  --speech-limit 40000 --noise-limit 1500 --rir-limit 512
```

Manifest entries are ordered by a seeded hash, so limits do not introduce a
filename-order speaker or subset bias. `dump_features` transforms every RIR at
startup and retains it in memory, so the standard command limits each split to
512 RIRs. Use `--limit 10` only to validate the pipeline; that output does not
contain enough audio for training.

`manifest.json` records the seed, input root, and group counts. Preserve all
manifests with model provenance. Rendering requires FFmpeg and produces 48 kHz
mono s16le audio plus 48 kHz mono little-endian float32 RIR files.

## 3. Build `dump_features`

The required upstream code is vendored under `training/vendor/xiph-rnnoise`. Its README
records the source revision, license, dependencies, and applied 65536-point RIR
FFT fix. No upstream neural weights are vendored.

```sh
training/scripts/build_dump_features.sh
```

## 4. Generate and verify features

Start with 256 training and 64 evaluation sequences for pipeline validation.
Each sequence has 2,000 frames and each frame has 98 little-endian float32
values. The script verifies the exact output size.

```sh
training/scripts/generate_features.sh training/vendor/xiph-rnnoise/dump_features \
  training/data/prepared training/data/features 256 64
```

Increase the training count to at least 10,000 for model candidates. Upstream
recommends 200,000 or more, so scale gradually after measuring storage and
generation time. 10,000 sequences require about 7.84 GB; 200,000 require about
156.8 GB.

## 5. Smoke training

```sh
.venv/bin/rnnoise-mlx-train training/data/features/train.f32 runs/base-smoke \
  --eval-features training/data/features/eval.f32 \
  --batch-size 8 --sequence-length 2000 \
  --segmented-tbptt-length 100 --segmented-tbptt-state carry \
  --max-updates 320 --seed 141
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
