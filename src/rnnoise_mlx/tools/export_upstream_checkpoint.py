"""Bridge an MLX test model to upstream's temporary C weight exporter.

This is a validation-only bridge. The stable deployment format remains the
MLX/BNNS format documented under models/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with safe_open(args.input, framework="pt", device="cpu") as source:
        config = json.loads(source.metadata()["config"])
        tensors = {}
        for name in ("conv1", "conv2"):
            tensors[f"{name}.weight"] = source.get_tensor(f"{name}.weight").permute(0, 2, 1).contiguous()
            tensors[f"{name}.bias"] = source.get_tensor(f"{name}.bias")
        for name in ("gru1", "gru2", "gru3"):
            tensors[f"{name}.weight_ih_l0"] = source.get_tensor(f"{name}.Wx")
            tensors[f"{name}.weight_hh_l0"] = source.get_tensor(f"{name}.Wh")
            bias = source.get_tensor(f"{name}.b")
            candidate_recurrent_bias = source.get_tensor(f"{name}.bhn")
            hidden = candidate_recurrent_bias.shape[0]
            # Any split of the reset/update summed biases is equivalent. Keep
            # them on the input side and reserve recurrent bias for candidate.
            tensors[f"{name}.bias_ih_l0"] = bias
            tensors[f"{name}.bias_hh_l0"] = torch.cat(
                (torch.zeros(2 * hidden), candidate_recurrent_bias)
            )
        tensors["dense_out.weight"] = source.get_tensor("gain.weight")
        tensors["dense_out.bias"] = source.get_tensor("gain.bias")
        tensors["vad_dense.weight"] = source.get_tensor("vad.weight")
        tensors["vad_dense.bias"] = source.get_tensor("vad.bias")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"source": "rnnoise-mlx"})
    args.output.with_suffix(".json").write_text(
        json.dumps({"model_kwargs": {"cond_size": config["cond_size"], "gru_size": config["gru_size"]}}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
