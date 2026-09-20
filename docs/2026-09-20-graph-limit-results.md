# CUDA graph limit 20 versus 100

Executed on connected Colab, Tesla T4, MLX 0.32.2, application 023b523.
BS=4, sequence=2000, segmented TBPTT=250, three updates per subprocess.
Both conditions use MLX seed 141 and the existing NumPy batch seed 141.
Only MLX_MAX_OPS_PER_BUFFER differs (20 versus 100); other graph variables
are unset. The notebook/global environment and application source are unchanged.
The dated script in colab/ is a notebook launcher: FEATURE_IDENTITY must already
be defined by the notebook preflight. It writes an isolated wrapper and driver,
then runs both subprocesses sequentially with persistent logs and exit markers.

| Trial | Limit | Update 1 (s) | Update 2 (s) | Update 3 (s) | Mean updates 2-3 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 20 | 321.830 | 5.349 | 3.525 | 4.437 |
| First | 100 | 21.454 | 3.246 | 3.102 | 3.174 |
| Repeat | 20 | 21.453 | 3.673 | 3.627 | 3.650 |
| Repeat | 100 | 19.640 | 3.355 | 4.281 | 3.818 |

All four training subprocesses exited zero and produced training summaries.
An earlier launcher attempt failed before training due to import order exposing
a pre-existing circular import. Matching profile_train.py's import order fixed
the launcher; failed artifacts remain separate.

The first baseline's unusually long initialization coincided with increasing
PTX cache files (25 to 36), consistent with cold JIT work, though no native stack
sample was available to prove the complete cause. Do not count the initial
321-to-21-second difference as a graph-limit speedup. The warmed repeat does
not demonstrate a stable speedup: its mean is about 4.6% slower at 100, while
the third update is noisy. These short sequential runs do not establish a
statistically reliable slowdown either. Run order was always 20 then 100.

First-trial losses match exactly. Repeat losses differ by at most
5.960464477539063e-8; this is a loss check, not a full parameter/gradient check.
The sequence is approximately 0.5630614, 0.8003821, 0.6392139 for both settings.

Repeat cumulative MLX peak allocation: 6.010 GB at 20 versus 6.149 GB at 100.
GPU device memory during update 3: 8,419 MiB versus 9,617 MiB (sample medians).
CPU utilization during that update, mean/median/p95: 184.0/185.5/194.7% versus
144.0/139.2/193.0%; 100% represents one CPU core. GPU utilization:
26.57/34/47% versus 21.71/25/50%. Only seven samples per update in this
comparison; GPU statistics are device-wide and sampling is about 0.5 seconds.
All per-update resource statistics are preserved in the accompanying JSON.
MLX peak is cumulative since process initialization, not an isolated update peak.

Colab output roots (UTC timestamps):

- /content/runs/graph-limit-bs4-3-20260919-222243
- /content/runs/graph-limit-bs4-3-20260919-222911

Each contains config, PID, training log, training summary, timestamped update
events, resource samples, exit status, and completion files. Summarized results
are also saved locally in the first-run and repeat JSON files beside this note.

No new Nsight trace was collected, so launch/node counts after this change are
not measured. The experiment establishes successful execution at 100, near-equal
losses, increased memory, and no reproducible speed improvement in this short
test. The value 100 remains available in the experiment cell; it has not been
made the project's permanent default.
