#!/usr/bin/env python3
"""Compare an official PyTorch RNNoise checkpoint with its BNNSGraph conversion."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from rnnoise_weights import infer_streaming, read_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--upstream-runner", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream / "torch/rnnoise"))
    from rnnoise import RNNoise

    raw = np.memmap(args.features, dtype="<f4", mode="r")
    values = np.asarray(raw[: args.frames * 98].reshape(args.frames, 98)[:, :65], dtype=np.float32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RNNoise(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # The streaming C/BNNS path starts both causal convolutions with zero
    # history. Four leading zero feature frames make the valid PyTorch
    # convolutions express the same initial condition and recurrent updates.
    padded = np.concatenate((np.zeros((4, 65), dtype=np.float32), values), axis=0)
    with torch.no_grad():
        gains, vad, _ = model(torch.from_numpy(padded[None].copy()))
    pytorch_reference = np.concatenate((gains[0].numpy(), vad[0].numpy()), axis=-1)
    canonical, config = read_bundle(args.bundle)
    canonical_gains, canonical_vad = infer_streaming(canonical, config, values)
    reference = np.concatenate((canonical_gains, canonical_vad), axis=-1)

    with tempfile.TemporaryDirectory(prefix="rnnoise-official-bnns-") as directory:
        input_path, output_path = Path(directory) / "features.f32", Path(directory) / "output.f32"
        input_path.write_bytes(values.astype("<f4").tobytes())
        subprocess.run([str(args.runner), str(args.graph), str(input_path), str(args.frames),
                        str(output_path)], check=True)
        output = np.fromfile(output_path, dtype="<f4").reshape(args.frames, 33)
        upstream_output = None
        if args.upstream_runner:
            upstream_path = Path(directory) / "upstream-output.f32"
            subprocess.run([str(args.upstream_runner), str(input_path), str(args.frames),
                            str(upstream_path)], check=True)
            upstream_output = np.fromfile(upstream_path, dtype="<f4").reshape(args.frames, 33)

    difference = np.abs(output - reference)
    maximum, mean = float(difference.max()), float(difference.mean())
    correlation = float(np.corrcoef(output.reshape(-1), reference.reshape(-1))[0, 1])
    print(f"frames={args.frames} max_abs={maximum:.9g} mean_abs={mean:.9g} correlation={correlation:.9g}")
    print(f"gain_max_abs={difference[:, :32].max():.9g} vad_max_abs={difference[:, 32].max():.9g} "
          f"worst_frame={np.unravel_index(np.argmax(difference), difference.shape)[0]}")
    pytorch_difference = np.abs(output - pytorch_reference)
    print(f"pytorch_max_abs={pytorch_difference.max():.9g} "
          f"pytorch_mean_abs={pytorch_difference.mean():.9g} "
          f"pytorch_correlation={np.corrcoef(output.reshape(-1), pytorch_reference.reshape(-1))[0, 1]:.9g}")
    if upstream_output is not None:
        upstream_difference = np.abs(output - upstream_output)
        print(f"upstream_c_max_abs={upstream_difference.max():.9g} "
              f"upstream_c_mean_abs={upstream_difference.mean():.9g} "
              f"upstream_c_correlation={np.corrcoef(output.reshape(-1), upstream_output.reshape(-1))[0, 1]:.9g} "
              f"upstream_c_gain_correlation={np.corrcoef(output[:, :32].reshape(-1), upstream_output[:, :32].reshape(-1))[0, 1]:.9g} "
              f"upstream_c_vad_correlation={np.corrcoef(output[:, 32], upstream_output[:, 32])[0, 1]:.9g}")
    if not np.isfinite(output).all() or maximum > args.tolerance:
        raise SystemExit("canonical float32 reference and BNNSGraph outputs did not match")


if __name__ == "__main__":
    main()
