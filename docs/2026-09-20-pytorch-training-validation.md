# PyTorch training implementation validation

This records the original implementation before the sequence-model refactor.
The current API uses `model.py` and `loss.py` instead of `graph.py`; see
[refinement validation](2026-09-20-pytorch-refinement-validation.md).

The optional backend lives in `rnnoise_mlx/training_pytorch/`: `graph.py`
contains the frame-step, chunk objective, and streaming wrapper; `loop.py`
implements segmented TBPTT; `train.py` provides the CLI, checkpoints and resume;
`weights.py` reuses the existing PyTorch/canonical conversion functions.
The model and loss follow the PyTorch implementation; a full sequence batch
performs one Adam step after averaged chunk gradients. Native PyTorch GRUs
replace the MLX GRU implementation. No torch.compile is required.

Framework-independent configuration and memory-mapped NumPy loading now live
under `training_common/`. Existing MLX import paths remain compatibility wrappers.
PyTorch imports do not load MLX, and importing the root package does not load
PyTorch. MLX remains the default installation dependency; `[torch]` adds PyTorch.

## Checks

- Ruff for new modules, shared code, MLX wrappers, and new tests: passed.
- Python compileall and git diff --check: passed.
- Focused Colab unittest suite: **11 passed** (6.79 s), covering import isolation,
  averaged gradients/one optimizer step, detached recurrent carry, invalid
  input rejection, canonical streaming output, raw/NumPy batch equivalence,
  MLX batch compatibility, unchanged fixed dimensions, exact CPU checkpoint
  resume across an epoch boundary, initial-weight loading, and MLX chunk
  loss/GRU-state/selected GRU-gradient agreement.
- Existing canonical SafeTensors dimension test included in those 11 tests.
- Wheel built without dependencies or build isolation. Inspected metadata:
  trainer included, MLX required by default, torch conditional on the torch
  or existing conversion extra.

The broader legacy `test_model.py` could not be collected in Colab because
and loader behavior were checked directly in the focused suite instead.
No claim is made that the entire repository suite was run.

## Full-sized CUDA smoke

Environment: connected Colab Tesla T4, PyTorch 2.11.0+cu128, actual model device
cuda:0. Source snapshot isolated at
`/content/runs/pytorch-implementation-20260919-225642`; original Colab checkout
and notebook environment settings were not replaced.

Invocation:

```sh
python -m tools.train_pytorch /content/train.npy NEW_OUTPUT \
  --device cuda --batch-size 4 --sequence-length 2000 \
  --segmented-tbptt-length 250 --max-updates 3 \
  --feature-identity 3dae648bb24075075cccfaa0b22020b5075859506f841bae52429520c0f1ccc4
```

| Update | Seconds | Loss | Cumulative CUDA peak allocated bytes |
| --- | ---: | ---: | ---: |
| 1 | 1.072436 | 0.5627091 | 113,767,936 |
| 2 | 0.299163 | 0.8697636 | 138,038,784 |
| 3 | 0.302736 | 0.3906989 | 138,038,784 |

Training exited zero and saved final canonical model.safetensors (11,531,644
bytes), checkpoint.ckpt (34,665,835 bytes), per-checkpoint canonical weights,
configuration, and training_summary.json. Artifacts are in the snapshot's
`cuda-smoke/training` directory. Training-only wall time includes input transfer
and synchronized updates; checkpoint writes are outside those timings.

CPU/GPU/memory samples were timestamp-filtered to run_update intervals and are
saved in the accompanying JSON. Updates 2 and 3 were so short that the sampler
captured only one observation each: CPU 80.1%/97.3%, GPU 49%/96%, device memory
493 MiB for both. These are observations, not representative utilization
distributions. The JSON includes mean/median/p95 and sample count so this
limitation is explicit. CPU percentages use 100% per core; device memory and
GPU utilization are device-wide. PyTorch's allocated-byte peak excludes CUDA
context/library allocations and is not directly equivalent to MLX's peak.

Mean updates 2-3: approximately 0.301 seconds. This is a smoke measurement,
not a controlled cross-framework benchmark: initial weights differ from the
earlier MLX experiment. Native PyTorch's separate GRU biases and optimizer
parameterization also mean identical converted forward functions do not imply
identical optimization trajectories.

## Scope

CLI defaults mirror the current MLX profiling path: float32, Adam at 1e-3,
seed 141, detached GRU carry between chunks and updates, four input overlap
frames, convolution history reset per update. `--no-carry-between-updates`
is available for independent batches. The CLI uses a deterministic per-epoch
permutation and restarts into a new output directory; resume validates the
configuration and feature file path/size/mtime plus optional supplied identity.
The supplied identity is recorded, not independently hashed by this CLI.

training modes are not duplicated. This implementation provides the requested
full-update model/chunk/loop path and a usable standalone training command.


