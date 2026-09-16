# rnnoise-mlx

An RNNoise-family noise suppressor for Apple Silicon and macOS. Training uses
[MLX](https://github.com/ml-explore/mlx), real-time neural inference targets
BNNS, and feature extraction and gain application use the upstream RNNoise DSP.

Release models produced by this repository are trained only from independently
generated data. Xiph/upstream models may be used as validation baselines, but
their weights, outputs, and distillation targets must not influence release
artifacts.

## Status

- `src/rnnoise_mlx/tests/`: Python tests excluded from the distribution package
- `src/rnnoise_mlx/`: installable Python package for training, conversion, and validation
- `src/rnnoise_mlx/tools/`: Python modules for data preparation, conversion, and validation
- `Sources/RNNoiseBNNS/`: C API for stateful BNNSGraph inference
- `Vendors/xiph-rnnoise/`: vendored feature-extraction sources and provenance
- `models/`: model contract between MLX and BNNS
- `docs/`: reproducibility, evaluation, and licensing documentation

## Training

```sh
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m rnnoise_mlx.training.train data/features/train.f32 runs/smoke \
  --batch-size 8 \
  --sequence-length 2000 \
  --max-updates 320 \
  --segmented-tbptt-length 100 \
  --segmented-tbptt-state carry \
  --eval-features data/features/eval.f32
```

Each `features.f32` frame matches upstream `dump_features`: 98 float32 values
containing 65 input features, 32 target gains, and one VAD target.

Use a TBPTT segment length of 100 for rapid corpus and quality screening. Loss
is a failure-detection signal, not a substitute for listening tests. Promote
promising candidates to segment length 500 with continuous Conv/GRU state,
`stop_gradient` at segment boundaries, and one optimizer update after all 2,000
frames.

```sh
.venv/bin/python -m rnnoise_mlx.training.train data/features/train.f32 runs/promoted-500 \
  --batch-size 8 \
  --sequence-length 2000 \
  --segmented-tbptt-length 500 \
  --segmented-tbptt-state carry \
  --eval-features data/features/eval.f32 \
  --max-updates 320
```

The current corpus-cleaning and promoted-training parameters are documented in
[the CJK two-stage procedure](docs/cjk-two-stage-training.md).

### Training on Notebook

Notebook training uses two explicit phases. The preflight cell must run first;
it validates the feature paths and computes the feature identities. The training
cell must then pass those already-computed identities to `train()`:

```python
from rnnoise_mlx.training import TrainConfig, preflight_feature_identities, train

config = TrainConfig(
    features="/content/train.f32",
    output="/content/runs/notebook-run",
    eval_features=None,
    batch_size=2,
    sequence_length=2000,
    segmented_tbptt_length=250,
    segmented_tbptt_state="carry",
    max_updates=10,
    checkpoint_every=10,
    sync_eval=True,
    seed=141,
)

# Preflight cell: run once before the training cell.
feature_identity, evaluation_feature_identity = preflight_feature_identities(config)
```

The training cell must not hash feature files, inspect their contents for
identity, or perform other preflight work. It may only call `train()` with the
identities produced by the preflight cell:

```python
summary = train(
    config,
    progress_callback=on_progress,
    feature_identity=feature_identity,
    evaluation_feature_identity=evaluation_feature_identity,
)
```

This is an execution-safety rule, not merely a performance suggestion:
`train()` rejects omitted identities. If a notebook is restarted, rerun the
preflight cell before rerunning the training cell. Profiling runners follow the
same rule and must receive identities from the notebook or command line; they
must not calculate SHA-256 as part of the profiled training process.

The training command records local checkpoints, evaluation results, and
training history. Progress can be consumed directly by notebook callers with
`progress_callback`.

The same training core is available to Python callers without constructing a
CLI argument vector:

```python
from rnnoise_mlx.training import (
    TrainConfig,
    TrainingCheckpoint,
    TrainingProgress,
    preflight_feature_identities,
    train,
)

def on_progress(event):
    if isinstance(event, TrainingProgress) and event.update % 10 == 0:
        print(event)
    elif isinstance(event, TrainingCheckpoint):
        print(f"checkpoint: {event.path}")

config = TrainConfig(
    features="data/features/train.f32",
    output="runs/notebook-run",
    eval_features="data/features/eval.f32",
    max_updates=320,
    segmented_tbptt_length=500,
    segmented_tbptt_state="carry",
)
feature_identity, evaluation_feature_identity = preflight_feature_identities(config)
train(
    config,
    progress_callback=on_progress,
    feature_identity=feature_identity,
    evaluation_feature_identity=evaluation_feature_identity,
)
```

`parse_args()` and `main()` remain the CLI adapter; `train(TrainConfig(...))`
contains the reusable training operation for notebooks and other callers.

Training writes a complete checkpoint at every `--checkpoint-every` updates
(32 by default) under `OUTPUT/checkpoints/update-NNNNNNNN/`. Each checkpoint
contains model weights, AdamW state, MLX random state, the data cursor, elapsed
counters, and training history. Resume with:

```sh
python -m rnnoise_mlx.training.train ... \
  --resume-from OUTPUT/checkpoints/update-00000500
```

Resumed runs use the state stored in the checkpoint and do not require an
external tracking service.

Checkpoints are committed only at batch boundaries, so prefetched input cannot
move the saved data cursor past the next batch. The legacy `--stateful-tbptt`
mode finishes its current batch before checkpointing, so `--max-updates` can be
rounded up to that safe boundary.

## Generated artifacts

`data/`, `runs/`, `experiments/`, `.venv/`, and `.build/` are not tracked.
Store SafeTensors models, evaluation JSON, and generated WAV files there.

## Tests and builds

```sh
.venv/bin/python -m pytest -q
swift test

cmake -S . -B .build/cmake
cmake --build .build/cmake
```

## File denoising executable

The macOS executable uses AVFoundation for input decoding, so it can inspect
formats supported by the system, including WAV and MP4/AAC sources.

```sh
swift run rnnoise-mlx-denoise --probe input.wav
swift run rnnoise-mlx-denoise --model RNNoiseGraph.mlmodelc input.m4a output.wav
```

AVFoundation converts system-supported inputs to 48 kHz mono float32. The C
backend extracts RNNoise features, runs the BNNS model, applies 32 band gains,
and synthesizes a 48 kHz mono WAV. See
[the BNNS validation report](docs/bnns-validation.md).

## BNNSGraph export

The deployment model is a Core ML package compiled for BNNSGraph. Training
SafeTensors artifacts already use the canonical tensor layout and can be
converted directly; producing a separate intermediate file is optional.

```sh
.venv/bin/python -m rnnoise_mlx.tools.export_bnns_graph \
  runs/model/model.safetensors runs/model/RNNoiseGraph.mlpackage
xcrun coremlcompiler compile runs/model/RNNoiseGraph.mlpackage runs/model \
  --platform macOS --deployment-target 15.0
```

Official RNNoise PyTorch checkpoints can be converted without retraining. The
canonical SafeTensors artifact is bidirectional: it can also be exported back to a
checkpoint accepted by upstream's `dump_rnnoise_weights.py`.

```sh
.venv/bin/python -m rnnoise_mlx.tools.convert_official_weights import \
  rnnoise10Ga_12.pth official.canonical.safetensors
.venv/bin/python -m rnnoise_mlx.tools.export_bnns_graph \
  official.canonical.safetensors OfficialRNNoiseGraph.mlpackage
xcrun coremlcompiler compile OfficialRNNoiseGraph.mlpackage . \
  --platform macOS --deployment-target 15.0

.venv/bin/python -m rnnoise_mlx.tools.convert_official_weights export \
  official.canonical.safetensors official-roundtrip.pth
```

`rnnoise10Ga_12.pth` is the closest match for the generated quantized/sparse
weights in the validated upstream C checkout. See
[the official-weight validation report](docs/official-weights-bnns-validation.md)
for gain, VAD, and PCM comparisons.

## License

The project is BSD-3-Clause. Vendored upstream files retain their original
copyright and license notices. See [docs/licensing.md](docs/licensing.md).
