# State diagnostics — 2026-07-11

> Historical diagnostics for earlier random and stateful checkpoints. Current
> decisions are recorded in
> [the segment-length experiment](segment-length-100-250-500-experiment.md).

Five fixed models were inspected without ranking them by loss: untrained,
random-window lengths 200 and 800, and stateful lengths 200 and 800. Statistics
cover 64 evaluation sequences; reset-ablation traces were saved for the first
eight sequences at chunk lengths 100, 200, 400, and 800.

## Correctness

- Chunked and full forward passes matched exactly for gain and VAD.
- State comprised Conv1 history `2 × 65`, Conv2 history `2 × 128`, and three
  256-element GRU hidden vectors.
- Direct Conv2 effects from resetting convolution history were limited to
  offsets 0–3 after a boundary.
- All values were finite and outputs stayed in `[0, 1]`.
- Checkpoint SHA-256 values were unchanged by diagnostics.

## State distribution

Traces without resets do not depend on chunk length, so this table uses length
200. Saturation means `|h| > 0.9`.

| model | GRU1 RMS / sat | GRU2 RMS / sat | GRU3 RMS / sat |
|---|---:|---:|---:|
| untrained | 0.183 / 0.0% | 0.118 / 0.0% | 0.081 / 0.0% |
| random 200 | 0.673 / 21.9% | 0.612 / 14.9% | 0.649 / 36.7% |
| random 800 | 0.660 / 20.4% | 0.610 / 14.6% | 0.655 / 31.6% |
| stateful 200 | 0.744 / 35.4% | 0.699 / 29.3% | 0.623 / 25.2% |
| stateful 800 | 0.687 / 25.6% | 0.590 / 15.7% | 0.549 / 22.2% |

Training substantially increased state amplitude and saturation. Clipping was
not introduced immediately; these measurements should accompany later learning
rate and gradient-clipping comparisons. Boundary-to-normal-frame state-change
ratios were approximately 0.85–1.08, showing no boundary-only discontinuity.

## Reset memory horizon

At length 800, all state was reset and compared with continuous execution. Each
entry reports frames until the difference fell below 50%, 10%, and 1% of its
initial value. `>800` means the threshold was not reached before the next reset.

| model | GRU1 50/10/1% | GRU2 50/10/1% | GRU3 50/10/1% | gain 50/10/1% | VAD 50/10/1% |
|---|---:|---:|---:|---:|---:|
| untrained | 3/6/11 | 4/8/14 | 5/10/16 | 3/6/12 | 1/4/10 |
| random 200 | 10/179/>800 | 26/493/>800 | 21/>800/>800 | 4/67/733 | 5/5/>800 |
| random 800 | 14/250/>800 | 32/580/>800 | 47/696/>800 | 4/139/>800 | 7/240/>800 |
| stateful 200 | 14/197/>800 | 19/399/>800 | 13/387/>800 | 4/95/693 | 14/129/>800 |
| stateful 800 | 20/318/>800 | 45/326/>800 | 36/408/>800 | 5/140/555 | 29/251/>800 |

Trained-state effects commonly persisted for several seconds and often longer
than eight seconds at the 1% threshold. Carrying all three GRU states is
therefore preferable to carrying only one layer. Resetting convolution history
affected Conv2 directly for at most four frames, but recurrent propagation kept
gain/VAD differences visible for up to about 242 frames.

## Stale state after one optimizer update

State computed from a prefix before an update was compared with state recomputed
from the same prefix after one update. Values are relative L2 errors at length
200.

| model | GRU1 | GRU2 | GRU3 | next gain | next VAD |
|---|---:|---:|---:|---:|---:|
| untrained | 58.0% | 66.6% | 73.4% | 0.58% | 0.15% |
| random 200 | 39.1% | 45.7% | 23.6% | 4.68% | 2.68% |
| random 800 | 37.1% | 44.8% | 38.8% | 12.27% | 12.97% |
| stateful 200 | 26.7% | 32.7% | 31.8% | 6.28% | 1.50% |
| stateful 800 | 46.4% | 55.3% | 35.1% | 7.06% | 9.42% |

One update changed state substantially. Passing pre-update state into an
updated model is therefore not a negligible inconsistency.

## Candidate order

1. Process multiple chunks with unchanged weights, accumulate gradients, then update once.
2. Compare stateful training with a 200–400-frame warm-up excluded from loss.
3. Carry all three recurrent states rather than a partial state.
4. Keep random-window plus burn-in only as a comparison, not a full substitute.

Convolution reset behavior was correct, so alignment changes were unnecessary.
Machine-readable artifacts are stored outside Git under
`runs/state-diagnostics/`.
