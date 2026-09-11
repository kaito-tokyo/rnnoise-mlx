# MLflow checkpoint validation — 2026-09-10

## Implementation

Checkpoints are uploaded through the HTTP tracking server into unique attempt
directories. Each payload is downloaded and SHA-256 verified before publishing
`complete.json`. Recovery ignores attempts without that marker, verifies the
selected payload, and publishes it locally by a same-filesystem rename. Existing
destinations are rejected. Corrupt committed generations fail closed.

Upload is enabled by default; `--no-mlflow-log-checkpoints` retains local-only
checkpoint storage. Periodic checkpoints, normal final batches, and graceful
SIGINT/SIGTERM stops use the same mechanism. Forced runtime destruction can only
recover a previously committed generation. No retention deletion is implemented.

## Test environment

- Source base: `5deae8846da5d1fb2b1900ea841e8eda077a3d12`, with uncommitted changes.
- Colab: NVIDIA L4, 23,034 MiB, driver 580.82.07.
- Isolated environment: Python 3.13.15, MLX CUDA 0.32.2, MLflow 3.14.0.
- Windows artifact server: MLflow 3.14.0 at `http://localhost:5000` through WSL mirror mode.
- Synthetic input: 64 training and 16 evaluation sequences, 16 frames per
  sequence; batch size 2, seed 17, no compile, no prefetch, synchronous evaluation.
- The CUDA environment uses explicit dependencies and `PYTHONPATH`; this is not
  validation of a Linux install from the existing project's dependency bounds.

## Live experiment

[Restart validation run](https://prodesk-400-g4-dm.salmon-pollux.ts.net/#/experiments/3/runs/6d1c3c9aa58b4eaca2f0199b9ca22d97)

The first process received SIGTERM after update 10, saved a complete checkpoint,
uploaded it, verified it, and exited normally. That checkpoint was downloaded
from Windows. The original local checkpoint tree was moved out of the recovery
path, and features plus identity manifests were copied to another directory.

The first resume attempt exposed a real compatibility issue: MLX 0.32's
`random.state` is read-only. Restoration now reconstructs the uint64 seed from
the saved two-word PRNG key and calls `mx.random.seed`, without rebinding the
state attribute. The initial source snapshot, failed attempt log, corrected
resume snapshot, and their provenance are retained separately.

The corrected process resumed the same MLflow run from update 10 and reached
update 20. Native Windows `Get-FileHash` verified all payloads and manifest hashes
for both committed generations. Each payload is approximately 33 MiB.

The uninterrupted reference run is `102e8f87b2ac47e6926c69ca15eb444f`. Exact array
comparisons passed for every model, optimizer, and PRNG tensor. History, initial
evaluation, update count, processed-frame count, next epoch, and next batch also
matched. The resumed run finished successfully with metric steps `[10, 20]`.
The latest committed checkpoint was downloaded again and verified at update 20.

Evidence is stored in MLflow under `validation/verification.json` and locally
under `experiments/2026-09-10-rnnoise-l4-checkpoint-resume/`. The collected archive
SHA-256 is `82e2851d2fa8c37e25a6f932b45464e66e392ca7da3819de3be655b27bf8b58c`.
The local archive hash matched after download; the Colab L4 session was then
terminated successfully. Windows checkpoint artifacts remain available.

Windows update-20 checkpoint manifest SHA-256:
`6e763e5013beb5083072d833d88735669ac950a6ad1910d156a158b72897c92e`.

## Automated tests

Local macOS test suite: **185 passed** (`.venv/bin/python -m pytest -q`). Coverage
includes interrupted uploads, corruption, immutable attempts, destination
preservation including dangling symlinks, HTTP-only tracking preflight before
output creation, RNG continuation, final-epoch saves, and upload opt-out.

## Scope and operational implications

This is a checkpoint and restart test, not a denoising-quality evaluation.
It does not establish bitwise equivalence across Mac/Colab, GPU types, MLX
versions, or compiled/TBPTT modes. The live restart uses a new process on the same
L4 runtime with a server-downloaded checkpoint and relocated feature paths.

Transfers are synchronous and much slower than this tiny training workload.
Choose `--checkpoint-every` according to acceptable lost work and transfer cost;
the default 32 updates is not a recommended production Colab interval.
