# MLX and BNNS test-training procedure

This test verifies that RNNoise-format MLX training works; it does not produce
a finished audio-quality model. Training and evaluation use distinct utterance
sets: 256 training sequences, batch 8, ten epochs, and 320 updates.

## Data

- Train: 1,000 deterministic clips from LibriTTS-R `train-clean-100`
- Evaluation: 250 deterministic clips from LibriTTS-R `dev-clean`
- Noise: Xiph `background_noise_v2.sw` and `foreground_noise.sw`
- RIR: Xiph `measured_rirs-v2`

LibriTTS-R is CC BY 4.0. Downloads, PCM, features, and weights are not tracked.
`python -m rnnoise_mlx.tools.prepare_libritts_raw` converts clips to 48 kHz mono s16le, inserts
100 ms of silence, and records the selection in `.clips.txt`.

For the reproducible LibriTTS-R, MUSAN, and RIRS_NOISES base-model workflow,
see [base-model-preparation.md](base-model-preparation.md).

The vendored `dump_features` includes the `fstride[MAXFACTORS + 1]` fix required
for the 65536-point RIR FFT and moves large RIR work buffers to the heap.

```sh
.venv/bin/python -m rnnoise_mlx.tools.build_dump_features
```

Each frame contains 98 float32 values: 65 features, 32 gains, and one VAD
target. Each sequence has 2,000 frames. Expected sizes are 200,704,000 bytes for
training and 50,176,000 bytes for evaluation.

## Run

```sh
.venv/bin/python -m rnnoise_mlx.training.train data/features/train.f32 runs/test-training \
  --eval-features data/features/eval.f32 \
  --batch-size 8 --sequence-length 2000 \
  --segmented-tbptt-length 100 --segmented-tbptt-state carry \
  --epochs 10 --max-updates 320 \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-experiment "$MLFLOW_EXPERIMENT" --mlflow-run-name test-training
```

`training.json` records initial, trained, and reloaded losses; finiteness;
output ranges; elapsed time; processed frames; and throughput. Passing requires
320 completed updates, lower training and evaluation loss, no NaN or infinity,
outputs in `[0, 1]`, and matching evaluation after reload.

The 320-update run validates only the pipeline. WAV quality, C export, and BNNS
integration are outside this test.

MLX 0.32 expands `nn.GRU` over time in Python. Development runs process twenty
100-frame segments with continuous Conv/GRU state, stop gradients at boundaries,
and update once after all 2,000 frames. Promote candidates to length 500.

Forward, backward, and AdamW update are compiled together with model and
optimizer state. `--no-compile` is diagnostic only. Asynchronous evaluation is
the default, Python reads loss every ten updates, and one batch is prefetched.
Use `--sync-eval` and `--no-prefetch` for diagnostics.
