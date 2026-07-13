"""Transactional, resumable training checkpoints."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .model import ModelConfig, RNNoise


FORMAT_VERSION = 1


def _flat_state(state: Any) -> dict[str, mx.array]:
    return tree_flatten(state, destination={})


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    root: Path,
    model: RNNoise,
    optimizer: Any,
    config: ModelConfig,
    *,
    update: int,
    next_epoch: int,
    next_batch: int,
    processed_frames: int,
    elapsed_seconds: float,
    history: list[dict[str, Any]],
    training_config: dict[str, Any],
) -> Path:
    """Atomically write a complete checkpoint and return its final directory."""
    mx.eval(model.state, optimizer.state, mx.random.state)
    root.mkdir(parents=True, exist_ok=True)
    name = f"update-{update:08d}"
    destination = root / name
    temporary = root / f".{name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        model.save(str(temporary / "model.safetensors"))
        mx.save_safetensors(
            str(temporary / "optimizer.safetensors"), _flat_state(optimizer.state)
        )
        mx.save_safetensors(
            str(temporary / "mlx-random-state.safetensors"),
            _flat_state(mx.random.state),
        )
        trainer_state = {
            "format_version": FORMAT_VERSION,
            "update": update,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "processed_frames": processed_frames,
            "elapsed_seconds": elapsed_seconds,
            "history": history,
        }
        (temporary / "trainer-state.json").write_text(
            json.dumps(trainer_state, indent=2) + "\n"
        )
        files = {
            path.name: _sha256(path)
            for path in temporary.iterdir()
            if path.is_file()
        }
        manifest = {
            "format_version": FORMAT_VERSION,
            "model_config": asdict(config),
            "training_config": _json_value(training_config),
            "files": files,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint(
    checkpoint: Path,
    model: RNNoise,
    optimizer: Any,
    config: ModelConfig,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Restore model, optimizer, RNG, and trainer cursor from a checkpoint."""
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    if manifest.get("model_config") != asdict(config):
        raise ValueError("checkpoint model configuration does not match")
    files = manifest.get("files", {})
    required = {
        "mlx-random-state.safetensors",
        "model.safetensors",
        "optimizer.safetensors",
        "trainer-state.json",
    }
    if set(files) != required:
        raise ValueError("checkpoint file manifest is incomplete")
    corrupt = [name for name, digest in files.items() if _sha256(checkpoint / name) != digest]
    if corrupt:
        raise ValueError(f"checkpoint file checksum differs: {', '.join(corrupt)}")
    saved_config = manifest.get("training_config", {})
    current_config = _json_value(training_config)
    compatible_keys = (
        "features",
        "batch_size",
        "sequence_length",
        "learning_rate",
        "lr_decay",
        "seed",
        "training_chunk_length",
        "stateful_tbptt",
        "two_segment_tbptt",
        "segmented_tbptt_length",
        "segmented_tbptt_state",
        "equalize_reset_targets",
    )
    mismatches = [
        key for key in compatible_keys if saved_config.get(key) != current_config.get(key)
    ]
    if mismatches:
        raise ValueError(f"checkpoint training configuration differs: {', '.join(mismatches)}")
    model.load_weights(str(checkpoint / "model.safetensors"))
    optimizer.state = tree_unflatten(mx.load(str(checkpoint / "optimizer.safetensors")))
    # MLX keeps an internal reference to this list. Rebinding the public
    # attribute leaves the generator on its previous state, so replace the
    # list contents in place.
    mx.random.state[:] = tree_unflatten(
        mx.load(str(checkpoint / "mlx-random-state.safetensors"))
    )
    mx.eval(model.state, optimizer.state, mx.random.state)
    state = json.loads((checkpoint / "trainer-state.json").read_text())
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("trainer state format does not match")
    return state
