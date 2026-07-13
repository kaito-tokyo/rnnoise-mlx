"""Export an MLX SafeTensors checkpoint to the compact C/BNNS bundle format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .rnnoise_weights import TENSORS, write_bundle


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    """Export one official-profile SafeTensors checkpoint to a canonical bundle."""
    weights, metadata = mx.load(str(checkpoint), return_metadata=True)
    config = json.loads(metadata["config"])
    expected_config = {
        "input_dim": 65,
        "output_dim": 32,
        "cond_size": 128,
        "gru_size": 384,
    }
    if config != expected_config:
        raise ValueError(f"unsupported model config: {config}")
    if set(weights) != set(TENSORS):
        raise ValueError("checkpoint tensor set does not match BNNS format v1")

    canonical = {name: np.asarray(weights[name], dtype=np.float32) for name in TENSORS}
    write_bundle(output, canonical, config["cond_size"], config["gru_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    try:
        export_checkpoint(args.checkpoint, args.output)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
