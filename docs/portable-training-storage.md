# Portable training storage

The registered active-training volume is `/Volumes/rnnoise-mlx-train`, APFS
volume UUID `A40C7B4F-BA11-4F0F-A307-308231058AC6`. It contains the source
checkout, source and prepared corpora, immutable feature generations, active
experiment checkpoints, and an HTTP-served local MLflow store. Do not treat
the volume as a backup; completed experiments still move to verified cold
archive storage after training stops.

Set the common root before using any data or training command:

```sh
export RNNOISE_MLX_STORAGE_ROOT=/Volumes/rnnoise-mlx-train
```

Initialize or validate the volume with the repository environment:

```sh
python -m rnnoise_mlx.tools.portable_storage init
python -m rnnoise_mlx.tools.portable_storage preflight
```

The command rejects a different UUID, an alternate mount such as
`/Volumes/rnnoise-mlx-train 1`, a read-only volume, and an uninitialized
layout. Paths in experiment configuration must be relative to
`RNNOISE_MLX_STORAGE_ROOT`; do not record `/Users/<name>` paths.

## Per-Mac environment

Create one environment per Mac below `runtime/<machine-id>/venv`. Environments
are disposable and must not be shared between machines. Install the pinned
environment from `requirements-lock.txt`, then install this checkout editable:

```sh
python3.13 -m venv "$RNNOISE_MLX_STORAGE_ROOT/runtime/$(hostname -s | tr '[:upper:]' '[:lower:]')/venv"
VENV="$RNNOISE_MLX_STORAGE_ROOT/runtime/$(hostname -s | tr '[:upper:]' '[:lower:]')/venv"
"$VENV/bin/pip" install -r requirements-lock.txt
"$VENV/bin/pip" install --no-deps -e '.[dev]'
```

## Local MLflow

Start the server before training and use only its HTTP endpoint:

```sh
python -m rnnoise_mlx.tools.portable_storage mlflow-start
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT=rnnoise-mlx
```

The backend is `mlflow/mlflow.db`, artifacts are under `mlflow/artifacts`, and
the server binds only to loopback. Direct `file:` and `sqlite:` tracking URIs
remain forbidden. Checkpoints remain solely in the active experiment output;
pass `--mlflow-log-checkpoints` only when deliberate duplication is required.

Stop the server and verify SQLite before moving the volume:

```sh
python -m rnnoise_mlx.tools.portable_storage mlflow-stop
python -m rnnoise_mlx.tools.portable_storage eject-check
diskutil eject /Volumes/rnnoise-mlx-train
```

Never move or eject the volume while training or MLflow is running. Resume a
stopped experiment with its complete checkpoint and original MLflow run ID.

## Verified migration

Use `copy-tree` only for a destination that does not yet exist. It copies to a
temporary sibling directory, verifies every file by SHA-256, atomically renames
the verified tree, and writes the requested inventory record:

```sh
python -m rnnoise_mlx.tools.portable_storage copy-tree SOURCE DESTINATION \
  --record "$RNNOISE_MLX_STORAGE_ROOT/inventory/checksums/NAME.json"
```

No source is deleted by this workflow. Keep at least 180 GiB free. Existing
`/Volumes/doc-*` volumes remain read-only inputs and cold archives, never active
output destinations.
