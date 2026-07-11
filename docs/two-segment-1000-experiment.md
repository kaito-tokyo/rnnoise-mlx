# 1000/1000 TBPTT state carry experiment

## Purpose

Compare two 1000/1000-frame TBPTT variants without running full-length BPTT:

- `carry`: preserve both Conv histories and all three GRU hidden states at the boundary, but apply `stop_gradient` to the boundary state.
- `reset`: process the second 1000-frame segment as an independent sequence.

Both variants process both segments with the same model weights and perform exactly one optimizer update after the complete 2,000-frame sequence.

## Fair comparison

Two valid kernel-size-3 convolutions remove four outputs when a segment is independently reset. Therefore, the final comparison excludes the first four outputs of the second segment from the `carry` loss as well. Both variants optimize the same 1,992 target frames per sequence.

Common conditions:

- seed: 0
- training sequences: 256
- evaluation sequences: 64
- batch size: 8
- epochs: 10
- optimizer updates: 320
- input frames per update: 16,000
- optimizer: AdamW
- initial learning rate: `1e-3`
- compiled MLX training step
- one optimizer update per two-segment sequence

## Results

| condition | training time | frames/s | eval total | eval gain | eval VAD |
|---|---:|---:|---:|---:|---:|
| state carry | 1,207.64 s | 4,239.67 | 0.188072 | 0.187825 | 0.246650 |
| state reset | 963.85 s | 5,312.00 | 0.187734 | 0.187486 | 0.247973 |

The initial evaluation was identical for both runs: total `0.563211`, gain `0.562516`, and VAD `0.694950`. Both runs were finite, produced outputs in `[0, 1]`, and reproduced the same evaluation after saving and reloading SafeTensors weights.

For each of the 64 evaluation sequences, define:

```text
delta = reset loss - carry loss
```

A positive value favors state carry. A 10,000-sample paired bootstrap produced:

| metric | mean delta | 95% bootstrap interval | carry better sequences |
|---|---:|---:|---:|
| total | -0.000338 | [-0.002535, 0.001588] | 57.8% |
| gain | -0.000339 | [-0.002520, 0.001580] | 56.2% |
| VAD | 0.001323 | [-0.002496, 0.005025] | 57.8% |

All intervals include zero. The experiment does not show a statistically distinguishable evaluation-loss improvement from preserving state at the 1000-frame boundary. The total-loss point estimate is 0.18% worse with carry, while carry takes about 25% more wall time in this implementation (or reset is about 20% faster relative to carry).

This is one training seed. The paired interval measures variation across the fixed 64 evaluation sequences, not variation across training seeds. It is sufficient to reject a claim of a large benefit at 320 updates, but not to prove exact equivalence between the methods.

## Decision

Do not select 1000-frame state carry as the production training configuration based on this result. Its measured benefit is indistinguishable from zero and it is slower. The next comparison should use fixed lengths 100, 250, and 500 under equal processed-frame and optimizer-update conditions. If 500 is the best speed/quality compromise, compare state carry against reset only for the `500 x 4` production candidate.

Artifacts are generated under `experiments/two-segment-1000/` and are excluded from Git. The normalized paired comparison is stored in `comparison-equal.json`.
