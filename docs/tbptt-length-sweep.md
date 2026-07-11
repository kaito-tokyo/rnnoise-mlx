# TBPTT length sweep — 2026-07-11

> Historical sweep using the earlier single-random-chunk method. It is retained
> for reproducibility and superseded by
> [the current experiment](segment-length-100-250-500-experiment.md).

This 320-update sweep narrowed the candidate lengths rather than making a final
decision. Every condition used seed 0, batch 8, identical initial weights, the
same 256 training and 64 evaluation sequences, and a compiled training step.
Evaluation always used all 2,000 frames.

## Results

| chunk | duration | updates/s | frames/s | first 32 train | last 32 train | eval total | eval gain | eval VAD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 (1 s) | 11.53 s | 27.743 | 22,194 | 0.436106 | 0.241134 | 0.229661 | 0.229321 | 0.340618 |
| 200 (2 s) | 29.39 s | 10.887 | 17,418 | 0.443506 | 0.232190 | 0.208147 | 0.207846 | 0.300382 |
| 400 (4 s) | 93.28 s | 3.431 | 10,978 | 0.426948 | 0.220189 | 0.203946 | 0.203677 | 0.268257 |
| 800 (8 s) | 313.26 s | 1.022 | 6,538 | 0.418393 | 0.204904 | 0.195729 | 0.195468 | 0.261367 |

Initial total evaluation loss was `0.563211` in every condition. All values
were finite, outputs stayed in `[0, 1]`, and evaluation matched after reload.

## Interpretation

- 100→200 improved total loss by 9.4% at 2.55 times the duration, with a large VAD improvement.
- 200→400 improved total loss by 2.0% at 3.17 times the duration; VAD improved by 10.7%.
- 400→800 improved total loss by 4.0% at 3.36 times the duration, mostly through gain loss.
- Longer chunks reduced GPU throughput because Python-expanded GRU graph costs grew nonlinearly.

The runs processed different frame totals: 256,000, 512,000, 1,024,000, and
2,048,000. The results therefore do not separate longer temporal context from
additional training data. Matched-frame or matched-wall-clock comparisons are
required before a long training run.

## Candidates

- **100 frames:** fast regression testing only.
- **200 frames:** development and hyperparameter search; 320 updates take about 29 seconds.
- **400 frames:** an intermediate quality-speed candidate with better VAD loss than 200.
- **800 frames:** the provisional long-run candidate, with the best loss but 3.36 times the duration of 400.

Length 800 was promising but not final. Stateful TBPTT, fair-budget tests, and
WAV evaluation were still required.

## Compatibility contract

Training changes must not change inference structure. The contract remains:

- 65 input features
- Two Conv1d layers and three GRU layers
- GRU gate order `reset, update, new`
- 32 gain outputs and one VAD output
- Stable weight shapes and SafeTensors format
- BNNS and upstream-C conversion
- Correct chronological Conv/GRU state updates

Small MLX, BNNS, and C numerical differences are acceptable. Validation targets
semantic equivalence, frame alignment, finiteness, ranges, and audio quality,
not bit-identical output.

## Stateful TBPTT proposal

The earlier implementation selected one random chunk from 2,000 frames and
zero-initialized hidden state. The proposed comparison processes all chunks in
order, passes state forward, and stops gradients at boundaries.

```text
chunk 1 → state → chunk 2 → state → ... → chunk N
             ↑ stop_gradient at each boundary
```

| state | shape |
|---|---:|
| conv1 history | 2 × 65 |
| conv2 history | 2 × 128 |
| gru1 hidden | 256 |
| gru2 hidden | 256 |
| gru3 hidden | 256 |

Two kernel-size-3 convolutions require four historical input frames. Explicit
per-layer history yields K outputs from K new inputs without duplicate-loss
exclusion and closely matches streaming C inference. Per-chunk optimizer
updates and accumulated-gradient single updates must be separate conditions.

## Fair comparison budgets

| chunk | frames in 320 updates |
|---:|---:|
| 100 | 256,000 |
| 200 | 512,000 |
| 400 | 1,024,000 |
| 800 | 2,048,000 |

An equal-frame comparison at 2,048,000 frames requires:

| chunk | updates | total frames |
|---:|---:|---:|
| 200 | 1,280 | 2,048,000 |
| 400 | 640 | 2,048,000 |
| 800 | 320 | 2,048,000 |

An approximately equal 313-second wall-clock budget was estimated at about
3,400, 1,070, and 320 updates for lengths 200, 400, and 800 respectively.

```text
frames/s = updates/s × chunk length × batch size
audio seconds/s = frames/s × 0.01
```

## Scheduler and execution optimization

Frame-matched runs should compare a processed-frame-based learning-rate
schedule rather than update-based decay.

```text
processed_frames += chunk_length × batch_size
lr = initial_lr / (1 + decay_per_frame × processed_frames)
```

The full training step and array-based schedule are already compiled. Further
work should reduce CPU synchronization by using `mx.async_eval`, materializing
loss every 10–32 updates, accumulating segment loss on device, prefetching one
batch, keeping static shapes, masking padded tails correctly, and considering
Metal capture only after warm-up.

## Historical operating proposal

- Development validation: length 200, 320 updates
- Hyperparameter search: length 200
- Pre-production check: length 800, 320–1,000 updates
- Long-run candidate: length 800, thousands to 10,000+ updates
- Full 2,000-frame evaluation at every checkpoint
- Select the best checkpoint using both evaluation loss and WAV quality

At length 800, compare learning rates `1e-3` and `5e-4` with and without
gradient clipping.
