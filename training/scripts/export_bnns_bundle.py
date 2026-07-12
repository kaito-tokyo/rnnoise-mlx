#!/usr/bin/env python3
"""Export an MLX SafeTensors checkpoint to the compact C/BNNS bundle format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from rnnoise_weights import TENSORS, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    weights, metadata = mx.load(str(args.checkpoint), return_metadata=True)
    config = json.loads(metadata["config"])
    if config["input_dim"] != 65 or config["output_dim"] != 32 or config["cond_size"] != 128:
        raise SystemExit(f"unsupported model config: {config}")
    if set(weights) != set(TENSORS):
        raise SystemExit("checkpoint tensor set does not match BNNS format v1")

    canonical = {name: np.asarray(weights[name], dtype=np.float32) for name in TENSORS}
    write_bundle(args.output, canonical, config["cond_size"], config["gru_size"])

    print(f"exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
