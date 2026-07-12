# TBPTT segment length 100/250/500 experiment

## Design

All conditions use the same seed, 256 training sequences, 64 evaluation sequences, batch size 8, and 320 optimizer updates. Every update processes all 2,000 input frames before one optimizer update. Conv histories and all three GRU states are carried between segments while gradients are stopped at each segment boundary.

This isolates the effect of the gradient horizon from the amount of input data and the number of optimizer updates. The full 1,996 model output frames contribute to the length sweep loss.

## Segment-length sweep

| segment | segments/sequence | training time | frames/s | eval total | eval gain | eval VAD |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 20 | 261.79 s | 19,557.92 | 0.193669 | 0.193417 | 0.251488 |
| 250 | 8 | 366.87 s | 13,955.93 | 0.188889 | 0.188636 | 0.253328 |
| 500 | 4 | 593.64 s | 8,624.73 | 0.186005 | 0.185757 | 0.248103 |

Longer segments improve evaluation loss but reduce throughput. Segment 500 has the best evaluation result of these candidates and remains about twice as fast as the previous fair 1000/1000 carry run (1,207.64 s). It is the provisional production choice.

## Segment 500 state carry versus reset

Reset Conv processing cannot emit the first four outputs of each independent segment. For a fair comparison, the carry run also excludes those post-boundary outputs. Both variants therefore optimize the same 1,984 target frames per sequence.

| condition | training time | frames/s | eval total | eval gain | eval VAD |
|---|---:|---:|---:|---:|---:|
| state carry | 590.66 s | 8,668.32 | 0.187554 | 0.187308 | 0.245879 |
| state reset | 576.23 s | 8,885.33 | 0.191195 | 0.190933 | 0.261832 |

For each of the 64 evaluation sequences, `delta = reset loss - carry loss`; positive values favor carry. A 10,000-sample paired bootstrap produced:

| metric | mean delta | relative improvement | 95% bootstrap interval | carry better sequences |
|---|---:|---:|---:|---:|
| total | 0.003641 | 1.90% | [0.001739, 0.005493] | 64.1% |
| gain | 0.003625 | 1.90% | [0.001717, 0.005538] | 64.1% |
| VAD | 0.015953 | 6.09% | [0.007936, 0.024623] | 75.0% |

All intervals exclude zero. On this fixed training seed and evaluation set, preserving state at 500-frame boundaries gives a distinguishable improvement while costing only 2.5% more wall time. The intervals characterize evaluation-sequence variation, not training-seed variation.

## Resource snapshot

During the segment-500 reset run on the M2 Pro Mac mini:

- GPU device utilization: 96%
- renderer and tiler utilization: 82%
- GPU in-use unified memory: approximately 23.18 GB
- GPU allocated unified memory: approximately 25.13 GB
- Python process memory shown by `top`: approximately 22 GB
- Python CPU: 41% to 76% across samples
- system CPU idle: 73% to 88%
- memory free: 20%
- no new swap I/O during the observation
- GPU recovery count: 0

The run is GPU-bound and has high activation/graph memory use. CPU-side prefetch is unlikely to be the main remaining optimization. Reusing a separately compiled 500-frame forward/backward graph four times, accumulating gradients on device, and applying one optimizer update after all four calls is the next optimization candidate.

## Decision

- Provisional production TBPTT length: 500
- Preserve Conv and GRU state across all four segments
- Apply `stop_gradient` at each boundary
- Perform one optimizer update after all 2,000 frames
- In production, retain all valid carry outputs; the target exclusion above is only for the controlled carry/reset comparison
- Do not use 1000 based on the current speed/quality tradeoff

All runs were finite, produced gain/VAD in `[0, 1]`, and matched evaluation results after SafeTensors save/reload. Generated models and normalized comparison data are under `experiments/segment-length-sweep/` and excluded from Git.
