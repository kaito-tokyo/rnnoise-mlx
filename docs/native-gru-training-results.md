# MLX nn.GRU training result — 2026-07-11

All three recurrent layers were replaced with MLX 0.32 `nn.GRU`. Because that
implementation also expands the time dimension in Python, full 2,000-frame
BPTT did not become faster. Training therefore sampled deterministic
200-frame windows, while evaluation continued to use all 2,000 frames.

## Speed

| implementation | BPTT frames/update | updates/s | 320 updates |
|---|---:|---:|---:|
| explicit custom GRU | 2,000 | 0.292 | 1,094.1 s |
| MLX `nn.GRU` | 2,000 | 0.131 | 243.7 s / 32 updates |
| MLX `nn.GRU` + truncated BPTT | 200 | 8.214 | 39.0 s |

Compiling the entire 200-frame training step improved a 32-update comparison
from 7.520 to 8.746 updates/s, or 16.3%. The final compiled 320-update run took
29.10 seconds at 10.996 updates/s, compared with 38.96 seconds and 8.214
updates/s in eager mode. Compiled execution is therefore the default.

The final method is about 28.1 times faster than the original test-training
implementation, although it backpropagates through one tenth as many frames per
update. Full-length `nn.GRU` was slower than the former implementation and was
not adopted.

## 320-update validation

| metric | initial | trained | reloaded |
|---|---:|---:|---:|
| total loss | 0.564459 | 0.208413 | 0.208413 |
| gain loss | 0.563788 | 0.208099 | 0.208099 |
| VAD loss | 0.670966 | 0.313635 | 0.313635 |

- Mean of first 32 updates: 0.430909
- Mean of last 32 updates: 0.230247, a 46.6% decrease
- No NaN or infinity values
- Gain and VAD outputs remained in `[0, 1]`
- Evaluation matched exactly after SafeTensors reload

The result remains worse than the full-BPTT evaluation loss of `0.185812`.
Longer runs should compare segment lengths 100, 200, and 400 together with
listening quality before fixing the speed-versus-context tradeoff.
