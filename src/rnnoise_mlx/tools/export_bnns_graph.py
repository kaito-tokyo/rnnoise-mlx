"""Export RNNoise SafeTensors weights as a Core ML BNNSGraph package."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np

from .rnnoise_weights import read_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="training or canonical SafeTensors weights")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    w, config = read_weights(args.weights)
    cond_size, gru_size = config["cond_size"], config["gru_size"]

    def linear(x, prefix: str, name: str, bias: bool = True):
        weight = w[f"{prefix}.weight"] if f"{prefix}.weight" in w else w[f"{prefix}.Wx"]
        return mb.linear(
            x=x,
            weight=weight.reshape(weight.shape[0], -1),
            bias=w[f"{prefix}.bias"] if bias and f"{prefix}.bias" in w else None,
            name=name,
        )

    def gru(x, hidden, prefix: str):
        x_proj = mb.linear(x=x, weight=w[f"{prefix}.Wx"], bias=w[f"{prefix}.b"])
        h_proj = mb.linear(x=hidden, weight=w[f"{prefix}.Wh"])
        section = lambda value, begin, end: mb.slice_by_index(x=value, begin=[begin], end=[end])
        reset = mb.sigmoid(x=mb.add(x=section(x_proj, 0, gru_size), y=section(h_proj, 0, gru_size)))
        update = mb.sigmoid(x=mb.add(x=section(x_proj, gru_size, 2 * gru_size), y=section(h_proj, gru_size, 2 * gru_size)))
        candidate = mb.tanh(x=mb.add(
            x=section(x_proj, 2 * gru_size, 3 * gru_size),
            y=mb.mul(x=reset, y=mb.add(x=section(h_proj, 2 * gru_size, 3 * gru_size), y=w[f"{prefix}.bhn"])),
        ))
        return mb.add(
            x=mb.mul(x=mb.sub(x=np.float32(1), y=update), y=candidate),
            y=mb.mul(x=update, y=hidden),
        )

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(65,), dtype=types.fp32),
        mb.TensorSpec(shape=(130,), dtype=types.fp32),
        mb.TensorSpec(shape=(2 * cond_size,), dtype=types.fp32),
        mb.TensorSpec(shape=(gru_size,), dtype=types.fp32),
        mb.TensorSpec(shape=(gru_size,), dtype=types.fp32),
        mb.TensorSpec(shape=(gru_size,), dtype=types.fp32),
    ], opset_version=ct.target.macOS15)
    def frame(features, feature_history, conv1_history, gru1_state, gru2_state, gru3_state):
        conv1_input = mb.concat(values=[feature_history, features], axis=0)
        conv1 = mb.tanh(x=linear(conv1_input, "conv1", "conv1_linear"))
        conv2_input = mb.concat(values=[conv1_history, conv1], axis=0)
        conv2 = mb.tanh(x=linear(conv2_input, "conv2", "conv2_linear"))
        next_gru1 = gru(conv2, gru1_state, "gru1")
        next_gru2 = gru(next_gru1, gru2_state, "gru2")
        next_gru3 = gru(next_gru2, gru3_state, "gru3")
        joined = mb.concat(values=[conv2, next_gru1, next_gru2, next_gru3], axis=0)
        gains = mb.sigmoid(x=linear(joined, "gain", "gain_linear"), name="gains")
        vad = mb.sigmoid(x=linear(joined, "vad", "vad_linear"), name="vad")
        next_feature_history = mb.concat(values=[mb.slice_by_index(x=feature_history, begin=[65], end=[130]), features], axis=0,
                                         name="next_feature_history")
        next_conv1_history = mb.concat(values=[mb.slice_by_index(x=conv1_history, begin=[cond_size], end=[2 * cond_size]), conv1], axis=0,
                                       name="next_conv1_history")
        return (
            gains, vad, next_feature_history, next_conv1_history,
            mb.identity(x=next_gru1, name="next_gru1_state"),
            mb.identity(x=next_gru2, name="next_gru2_state"),
            mb.identity(x=next_gru3, name="next_gru3_state"),
        )

    model = ct.convert(frame, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()
