# Mac mini training handoff

## Objective

Train a moderately practical RNNoise-family baseline on the M2 Pro Mac mini.
The current 320-update models are pipeline checks, not production-quality
models.

## Target machine

- Mac mini `Mac14,12`
- Apple M2 Pro: 10 CPU cores (6 performance, 4 efficiency)
- 32 GB unified memory
- Keep the machine awake with `caffeinate` during long training.

## Dataset policy

- Clean speech: LibriTTS-R `train-clean-100` / `dev-clean`, CC BY 4.0.
- Noise: MUSAN, CC BY 4.0. Xiph `background_noise_v2` and
  `foreground_noise` are excluded from release-model training because their
  audio provenance and redistribution terms are not sufficiently explicit.
- RIR: currently Xiph `measured_rirs-v2`; its licensing must be resolved or it
  must be replaced before calling the resulting model distributable.
- Upstream model weights are build-only inputs for `dump_features`; they must
  not initialize, supervise, distill, select, or otherwise influence released
  neural weights.
- LDTX `side-track-2.mp4` recordings are not included yet. They are staged for
  subjective evaluation and possible later adaptation.

## Completed work

- MUSAN 930 WAV files classified by stationarity/transient metrics.
- Deterministic file-level MUSAN train/eval split with zero overlap.
- Split counts:
  - train background: 82
  - train foreground: 194
  - eval background: 18
  - eval foreground: 29
- RIR crash fixed:
  - FFT `fstride` one-element overflow fixed.
  - RIR input/FFT buffers moved from stack to heap.
  - sequence-sized weighted-RMS buffer moved from stack to heap.
- AddressSanitizer RIR smoke generation passed.
- RIR train/eval feature generation passed at 256/64 sequence scale.
- Full RIR train feature generation completed: 10,000 sequences.
- Repository tests for classification/splitting/PCM preparation pass.

## Files to transfer

`data/`, `runs/`, and `.venv/` are Git-ignored. A Git clone alone will not
transfer generated features or corpus assets. Copy the feature files explicitly
with Finder, `rsync`, or another binary-safe transfer method.

Transfer the repository and these four generated files while preserving their
names and order:

| file | bytes | SHA-256 |
|---|---:|---|
| `data/features/musan-practical/train.part0.f32` | 1,960,000,000 | `57ec7b92de3df891f9f3a0d616acd2f2897782fb64052f02d3018c64ec0f563b` |
| `data/features/musan-practical/train.part1.f32` | 1,960,000,000 | `c5e7315f5947fe4da17b750ad51f32e8fe510fd29e8bc2767a2e46ff9fb2097a` |
| `data/features/musan-practical/train.part2.f32` | 1,960,000,000 | `f396ba9c7217a169a5dceb428517eac7f8d93a58b728be562ed811f7fc400044` |
| `data/features/musan-practical/train.part3.f32` | 1,960,000,000 | `39eb8dbe2c524e4387615c4146d4278d3589fce23c227765472f83b24b39a573` |

Each part contains exactly 2,500 complete sequences. Each sequence is 2,000
frames × 98 little-endian float32 values (784,000 bytes). Total: 10,000
sequences and 7,840,000,000 bytes.

The eval 500-sequence feature file has not been generated yet. Either generate
it before transfer or transfer the source data and patched `dump_features`.

## Mac mini setup

Do not copy `.venv` between machines. From the repository root on the Mac mini:

```sh
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

MLX must report the GPU outside a restricted/headless sandbox:

```sh
.venv/bin/python -c 'import mlx.core as mx; print(mx.default_device()); print(mx.array([1]) + 1)'
```

Expected device: `Device(gpu, 0)`.

## Join train features

Join parts in numeric order and verify the exact byte size:

```sh
mkdir -p data/features/musan-practical
cat data/features/musan-practical/train.part0.f32 \
    data/features/musan-practical/train.part1.f32 \
    data/features/musan-practical/train.part2.f32 \
    data/features/musan-practical/train.part3.f32 \
    > data/features/musan-practical/train-rir-10000.f32
stat -f '%z' data/features/musan-practical/train-rir-10000.f32
```

Expected size: `7840000000` bytes.

## Required eval generation

Use only the held-out MUSAN eval PCM files:

```sh
data/tools/rnnoise-upstream/dump_features \
  -rir_list data/training/musan-baseline/rir-list.txt \
  data/corpora/libritts-r/dev-clean-250clips-48k.pcm \
  data/training/musan-practical/eval-background.pcm \
  data/training/musan-practical/eval-foreground.pcm \
  data/features/musan-practical/eval-rir-500.f32 \
  500
```

Expected size: `392000000` bytes.

## Benchmark first

Run 320 updates on the Mac mini to measure its actual speed:

```sh
caffeinate -dimsu .venv/bin/rnnoise-mlx-train \
  data/features/musan-practical/train-rir-10000.f32 \
  runs/musan-practical-m2pro-benchmark \
  --eval-features data/features/musan-practical/eval-rir-500.f32 \
  --batch-size 8 \
  --sequence-length 2000 \
  --segmented-tbptt-length 100 \
  --segmented-tbptt-state carry \
  --max-updates 320
```

Record `updates_per_second` from `training.json` before estimating the long
run. Do not extrapolate from the M1 Max once the M2 Pro measurement exists.

## Practical baseline run

After the benchmark passes, run 20,000 updates:

```sh
caffeinate -dimsu .venv/bin/rnnoise-mlx-train \
  data/features/musan-practical/train-rir-10000.f32 \
  runs/musan-practical-rir-20k \
  --eval-features data/features/musan-practical/eval-rir-500.f32 \
  --batch-size 8 \
  --sequence-length 2000 \
  --segmented-tbptt-length 100 \
  --segmented-tbptt-state carry \
  --max-updates 20000
```

Do not start the 20,000-update run unless the eval file has the expected size,
the benchmark finishes without NaN/Inf, outputs remain in `[0, 1]`, and model
save/reload evaluation matches.

## Acceptance gates

The model is only a candidate for practical use after all of these hold:

1. Training finishes without NaN/Inf.
2. Held-out eval total/gain/VAD loss improves from initialization.
3. SafeTensors reload matches the trained evaluation.
4. A fixed LDTX voice/noise recording sounds better than the no-RIR 320-update
   baseline without unacceptable speech damage.
5. Unseen speakers and held-out noise are checked, not only the user's voice.
6. RIR licensing is resolved before redistribution.
7. Model provenance records dataset versions, licenses, split manifest, patch,
   seeds/configuration, hashes, and evaluation results.

If loss is still improving at 20,000 updates and subjective quality is useful,
continue from a verified checkpoint toward 75,000 updates. Do not call the
model production-ready based on loss alone.

## Relevant repository artifacts

- `scripts/classify_noise.py`
- `scripts/split_noise_manifest.py`
- `scripts/prepare_noise_pcm.py`
- `patches/rnnoise-rir-macos.patch`
- `data/corpora/musan/noise-classification.csv`
- `data/training/musan-baseline/split-manifest.csv`
- `docs/licensing.md`
