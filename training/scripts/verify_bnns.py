#!/usr/bin/env python3
"""Compare streaming C/BNNS inference with MLX on held-out feature frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import mlx.core as mx
import numpy as np

from rnnoise_mlx.data import FeatureDataset
from rnnoise_mlx.model import ModelConfig, RNNoise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True, help="compiled .mlmodelc directory")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    source = FeatureDataset(str(args.features), 2000)
    values = np.asarray(source.data[0, : args.frames, :65], dtype="<f4")
    model = RNNoise.load(str(args.checkpoint), ModelConfig())
    gains, vad, _ = model(mx.array(values[None]))
    mx.eval(gains, vad)
    reference = np.concatenate((np.asarray(gains[0]), np.asarray(vad[0])), axis=-1)

    with tempfile.TemporaryDirectory(prefix="rnnoise-bnns-") as directory:
        input_path = Path(directory) / "features.f32"
        output_path = Path(directory) / "output.f32"
        input_path.write_bytes(values.tobytes())
        subprocess.run(
            [str(args.runner), str(args.graph), str(input_path), str(args.frames), str(output_path)],
            check=True,
        )
        output = np.fromfile(output_path, dtype="<f4").reshape(args.frames, 33)[4:]

    difference = np.abs(output - reference)
    maximum = float(difference.max())
    print(f"frames={args.frames} max_abs={maximum:.9g} mean_abs={difference.mean():.9g}")
    if not np.isfinite(output).all() or maximum > args.tolerance:
        raise SystemExit("C/BNNS output did not match MLX")


if __name__ == "__main__":
    main()
