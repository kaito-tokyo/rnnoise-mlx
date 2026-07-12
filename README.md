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
  --eval-features data/features/eval.f32 \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-experiment "$MLFLOW_EXPERIMENT" \
  --mlflow-run-name smoke
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
  --max-updates 320 \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-experiment "$MLFLOW_EXPERIMENT" \
  --mlflow-run-name promoted-500
```

See [the segment-length experiment](docs/segment-length-100-250-500-experiment.md)
and [the reproducible test-training procedure](docs/test-training.md).

The training command requires an MLflow tracking URI, experiment, and run name.
It checks the server before evaluation or optimization, logs loss every ten
updates, records initial/trained/reloaded evaluation, and uploads the final
`model.safetensors`, `training.json`, and every complete checkpoint. Keep
machine-specific values outside the repository and pass them through the three
MLflow CLI options.

Training writes a complete checkpoint at every `--checkpoint-every` updates
(32 by default) under `OUTPUT/checkpoints/update-NNNNNNNN/`. Each checkpoint
contains model weights, AdamW state, MLX random state, the data cursor, elapsed
counters, and training history. Resume with:

```sh
python -m rnnoise_mlx.training.train ... \
  --resume-from OUTPUT/checkpoints/update-00000500
```

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

The deployment model is a Core ML package compiled for BNNSGraph. The compact
`RNMLXBN1` file is only an intermediate, portable weight container used by the
training-side converter.

```sh
.venv/bin/python -m rnnoise_mlx.tools.export_bnns_bundle \
  runs/model/model.safetensors runs/model/model.bnns
.venv/bin/python -m rnnoise_mlx.tools.export_bnns_graph \
  runs/model/model.bnns runs/model/RNNoiseGraph.mlpackage
xcrun coremlcompiler compile runs/model/RNNoiseGraph.mlpackage runs/model \
  --platform macOS --deployment-target 15.0
```

Official RNNoise PyTorch checkpoints can be converted without retraining. The
canonical `RNMLXBN1` bundle is bidirectional: it can also be exported back to a
checkpoint accepted by upstream's `dump_rnnoise_weights.py`.

```sh
.venv/bin/python -m rnnoise_mlx.tools.convert_official_weights import \
  rnnoise10Ga_12.pth official.bnns
.venv/bin/python -m rnnoise_mlx.tools.export_bnns_graph \
  official.bnns OfficialRNNoiseGraph.mlpackage
xcrun coremlcompiler compile OfficialRNNoiseGraph.mlpackage . \
  --platform macOS --deployment-target 15.0

.venv/bin/python -m rnnoise_mlx.tools.convert_official_weights export \
  official.bnns official-roundtrip.pth
```

`rnnoise10Ga_12.pth` is the closest match for the generated quantized/sparse
weights in the validated upstream C checkout. See
[the official-weight validation report](docs/official-weights-bnns-validation.md)
for gain, VAD, and PCM comparisons.

## License

The project is BSD-3-Clause. Vendored upstream files retain their original
copyright and license notices. See [docs/licensing.md](docs/licensing.md).
