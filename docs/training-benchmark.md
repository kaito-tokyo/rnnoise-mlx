# MLX training benchmark — 2026-07-11

> Historical benchmark. Current policy uses length 100 for subjective
> screening and length 500 with state carry for promoted runs. See
> [the current experiment](segment-length-100-250-500-experiment.md).

The current `nn.GRU` trainer was measured on an Apple M2 Pro with 32 GB of
memory, MLX 0.32.0, Python 3.14, and Metal. All runs used batch 8, 320 updates,
seed 0, a compiled training step, and AdamW. Training had 256 sequences and
evaluation had 64 separate full-length sequences. Times cover only training.

## Throughput and loss

| chunk | training time | updates/s | processed frames | frames/s | audio seconds/s | eval total |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 11.11 s | 28.799 | 256,000 | 23,039 | 230.39 | 0.229661 |
| 200 | 29.32 s | 10.915 | 512,000 | 17,464 | 174.64 | 0.208147 |
| 400 | 93.13 s | 3.436 | 1,024,000 | 10,995 | 109.95 | 0.203946 |
| 800 | 311.14 s | 1.028 | 2,048,000 | 6,582 | 65.82 | 0.195729 |

`frames/s` includes the batch dimension. `audio seconds/s = frames/s × 0.01`.

## Loss details

| chunk | first 32 train | last 32 train | eval gain | eval VAD |
|---:|---:|---:|---:|---:|
| 100 | 0.436106 | 0.241134 | 0.229321 | 0.340618 |
| 200 | 0.443506 | 0.232190 | 0.207846 | 0.300382 |
| 400 | 0.426948 | 0.220189 | 0.203677 | 0.268257 |
| 800 | 0.418393 | 0.204904 | 0.195468 | 0.261367 |

Every run was finite, outputs remained in `[0, 1]`, and evaluation matched
after SafeTensors reload.

Length 100 had the highest throughput. Length 200 improved total loss by 9.4%
in about 29 seconds. Length 400 improved another 2.0% but took 3.18 times as
long. Length 800 had the best loss but took 3.34 times as long as 400 and 10.61
times as long as 200. Increasing length eightfold reduced throughput by 71.4%.

These runs processed different frame totals, so they are not a final quality-
efficiency comparison. Matched-frame and matched-wall-clock experiments are
required.

## Asynchronous evaluation and one-batch prefetch

| execution | time | updates/s | frames/s | eval total |
|---|---:|---:|---:|---:|
| per-step synchronization | 29.32 s | 10.915 | 17,464 | 0.208147 |
| async + prefetch 1 | 28.40 s | 11.267 | 18,027 | 0.208147 |

All 320 loss values and final evaluation results matched. The longer run was
3.1% faster; a short 32-update run improved by about 17%. Asynchronous
evaluation remains the default, loss is materialized every ten updates, and
prefetch is limited to one batch.

Artifacts are stored outside Git under
`runs/training-benchmark/chunk-{length}/`.
