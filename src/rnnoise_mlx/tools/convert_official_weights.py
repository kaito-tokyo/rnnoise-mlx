"""Convert official RNNoise PyTorch weights to/from the canonical bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .rnnoise_weights import read_bundle, write_bundle


def import_official(source: Path, output: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_kwargs", {})
    cond_size, gru_size = int(config.get("cond_size", 128)), int(config.get("gru_size", 256))
    state = checkpoint["state_dict"]
    array = lambda name: state[name].detach().cpu().numpy().astype(np.float32)
    weights = {
        "conv1.weight": array("conv1.weight").transpose(0, 2, 1),
        "conv1.bias": array("conv1.bias"),
        "conv2.weight": array("conv2.weight").transpose(0, 2, 1),
        "conv2.bias": array("conv2.bias"),
        "gain.weight": array("dense_out.weight"), "gain.bias": array("dense_out.bias"),
        "vad.weight": array("vad_dense.weight"), "vad.bias": array("vad_dense.bias"),
    }
    for prefix in ("gru1", "gru2", "gru3"):
        bias_ih, bias_hh = array(f"{prefix}.bias_ih_l0"), array(f"{prefix}.bias_hh_l0")
        weights[f"{prefix}.Wx"] = array(f"{prefix}.weight_ih_l0")
        weights[f"{prefix}.Wh"] = array(f"{prefix}.weight_hh_l0")
        weights[f"{prefix}.b"] = np.concatenate((
            bias_ih[:2 * gru_size] + bias_hh[:2 * gru_size],
            bias_ih[2 * gru_size:],
        ))
        weights[f"{prefix}.bhn"] = bias_hh[2 * gru_size:]
    write_bundle(output, weights, cond_size, gru_size)


def export_official(source: Path, output: Path) -> None:
    weights, config = read_bundle(source)
    gru_size = config["gru_size"]
    state = {
        "conv1.weight": torch.from_numpy(weights["conv1.weight"].transpose(0, 2, 1).copy()),
        "conv1.bias": torch.from_numpy(weights["conv1.bias"].copy()),
        "conv2.weight": torch.from_numpy(weights["conv2.weight"].transpose(0, 2, 1).copy()),
        "conv2.bias": torch.from_numpy(weights["conv2.bias"].copy()),
        "dense_out.weight": torch.from_numpy(weights["gain.weight"].copy()),
        "dense_out.bias": torch.from_numpy(weights["gain.bias"].copy()),
        "vad_dense.weight": torch.from_numpy(weights["vad.weight"].copy()),
        "vad_dense.bias": torch.from_numpy(weights["vad.bias"].copy()),
    }
    for prefix in ("gru1", "gru2", "gru3"):
        bias_ih = weights[f"{prefix}.b"].copy()
        bias_hh = np.zeros(3 * gru_size, dtype=np.float32)
        bias_hh[2 * gru_size:] = weights[f"{prefix}.bhn"]
        state[f"{prefix}.weight_ih_l0"] = torch.from_numpy(weights[f"{prefix}.Wx"].copy())
        state[f"{prefix}.weight_hh_l0"] = torch.from_numpy(weights[f"{prefix}.Wh"].copy())
        state[f"{prefix}.bias_ih_l0"] = torch.from_numpy(bias_ih)
        state[f"{prefix}.bias_hh_l0"] = torch.from_numpy(bias_hh)
    torch.save({"model_args": (), "model_kwargs": {"input_dim": 65, "output_dim": 32,
                "cond_size": config["cond_size"], "gru_size": gru_size},
                "state_dict": state}, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("import", "export"):
        sub = subparsers.add_parser(command)
        sub.add_argument("source", type=Path)
        sub.add_argument("output", type=Path)
    args = parser.parse_args()
    (import_official if args.command == "import" else export_official)(args.source, args.output)


if __name__ == "__main__":
    main()
