"""Canonical RNNoise weight schema shared by conversion and export commands."""

from __future__ import annotations

from pathlib import Path
import json
import struct

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

MAGIC = b"RNMLXBN1"
SCHEMA = "rnnoise-mlx-canonical"
SCHEMA_VERSION = 1
TENSORS = (
    "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
    "gru1.Wx", "gru1.Wh", "gru1.b", "gru1.bhn",
    "gru2.Wx", "gru2.Wh", "gru2.b", "gru2.bhn",
    "gru3.Wx", "gru3.Wh", "gru3.b", "gru3.bhn",
    "gain.weight", "gain.bias", "vad.weight", "vad.bias",
)


def shapes(cond_size: int, gru_size: int, input_dim: int = 65, output_dim: int = 32):
    result = {
        "conv1.weight": (cond_size, 3, input_dim), "conv1.bias": (cond_size,),
        "conv2.weight": (gru_size, 3, cond_size), "conv2.bias": (gru_size,),
        "gain.weight": (output_dim, 4 * gru_size), "gain.bias": (output_dim,),
        "vad.weight": (1, 4 * gru_size), "vad.bias": (1,),
    }
    for prefix in ("gru1", "gru2", "gru3"):
        result.update({
            f"{prefix}.Wx": (3 * gru_size, gru_size),
            f"{prefix}.Wh": (3 * gru_size, gru_size),
            f"{prefix}.b": (3 * gru_size,),
            f"{prefix}.bhn": (gru_size,),
        })
    return result


def _validate(weights: dict[str, np.ndarray], config: dict[str, int]) -> dict[str, np.ndarray]:
    if config["input_dim"] != 65 or config["output_dim"] != 32:
        raise ValueError("unsupported canonical RNNoise dimensions")
    expected = shapes(config["cond_size"], config["gru_size"])
    if set(weights) != set(TENSORS):
        raise ValueError("canonical tensor set does not match RNNoise format")
    canonical = {}
    for name in TENSORS:
        value = np.asarray(weights[name], dtype=np.float32)
        if value.shape != expected[name]:
            raise ValueError(f"{name}: expected {expected[name]}, got {value.shape}")
        canonical[name] = np.ascontiguousarray(value)
    return canonical


def write_weights(path: Path, weights: dict[str, np.ndarray], cond_size: int, gru_size: int) -> None:
    """Write the canonical deployment schema in a standard SafeTensors container."""
    config = {"input_dim": 65, "cond_size": cond_size, "gru_size": gru_size, "output_dim": 32}
    canonical = _validate(weights, config)
    metadata = {
        "schema": SCHEMA,
        "schema_version": str(SCHEMA_VERSION),
        "config": json.dumps(config, separators=(",", ":"), sort_keys=True),
        "gru_gate_order": "reset,update,new",
        "gru_bias_layout": "combined-reset-update,candidate-recurrent-bhn",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(canonical, str(path), metadata=metadata)


def _read_legacy_bundle(path: Path):
    """Read the retired RNMLXBN1 container for backwards compatibility."""
    with path.open("rb") as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise ValueError("truncated canonical bundle header")
        magic, version, input_dim, cond_size, gru_size, output_dim, count = struct.unpack("<8s6I", header)
        if magic != MAGIC or version != 1 or input_dim != 65 or output_dim != 32 or count != len(TENSORS):
            raise ValueError("unsupported canonical RNNoise bundle")
        expected = shapes(cond_size, gru_size, input_dim, output_dim)
        result = {}
        for expected_name in TENSORS:
            raw_header = stream.read(40)
            if len(raw_header) != 40:
                raise ValueError("truncated tensor header")
            encoded_name, element_count = struct.unpack("<32sQ", raw_header)
            name = encoded_name.split(b"\0", 1)[0].decode("ascii")
            if name != expected_name or element_count != np.prod(expected[name]):
                raise ValueError(f"invalid canonical tensor {name!r}")
            raw = stream.read(element_count * 4)
            if len(raw) != element_count * 4:
                raise ValueError(f"truncated tensor {name}")
            result[name] = np.frombuffer(raw, dtype="<f4").copy().reshape(expected[name])
        if stream.read(1):
            raise ValueError("trailing data in canonical bundle")
    config = {"input_dim": input_dim, "cond_size": cond_size,
              "gru_size": gru_size, "output_dim": output_dim}
    return _validate(result, config), config


def read_weights(path: Path):
    """Read a training/canonical SafeTensors file or a legacy RNMLXBN1 bundle."""
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) == MAGIC:
            return _read_legacy_bundle(path)

    with safe_open(str(path), framework="np") as source:
        metadata = source.metadata() or {}
        try:
            config = json.loads(metadata["config"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("SafeTensors weights are missing valid config metadata") from error
        try:
            config = {name: int(config[name]) for name in ("input_dim", "cond_size", "gru_size", "output_dim")}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("SafeTensors weights contain an invalid model config") from error
        if "schema" in metadata and (
            metadata["schema"] != SCHEMA or metadata.get("schema_version") != str(SCHEMA_VERSION)
        ):
            raise ValueError("unsupported canonical SafeTensors schema")
        weights = {name: source.get_tensor(name) for name in source.keys()}
    return _validate(weights, config), config


# Compatibility for callers using the old API. New files written here are SafeTensors.
read_bundle = read_weights
write_bundle = write_weights


def infer_streaming(weights: dict[str, np.ndarray], config: dict[str, int], features: np.ndarray):
    """Reference float32 inference with the same zero-history semantics as C."""
    cond_size, gru_size = config["cond_size"], config["gru_size"]
    feature_history = np.zeros(130, dtype=np.float32)
    conv1_history = np.zeros(2 * cond_size, dtype=np.float32)
    states = [np.zeros(gru_size, dtype=np.float32) for _ in range(3)]
    gains, vads = [], []
    sigmoid = lambda value: 1 / (1 + np.exp(-value))
    for feature in np.asarray(features, dtype=np.float32):
        conv1_input = np.concatenate((feature_history, feature))
        conv1 = np.tanh(weights["conv1.weight"].reshape(cond_size, -1) @ conv1_input + weights["conv1.bias"])
        conv2_input = np.concatenate((conv1_history, conv1))
        value = np.tanh(weights["conv2.weight"].reshape(gru_size, -1) @ conv2_input + weights["conv2.bias"])
        feature_history = np.concatenate((feature_history[65:], feature))
        conv1_history = np.concatenate((conv1_history[cond_size:], conv1))
        outputs = [value]
        for index, prefix in enumerate(("gru1", "gru2", "gru3")):
            x = weights[f"{prefix}.Wx"] @ value + weights[f"{prefix}.b"]
            h = weights[f"{prefix}.Wh"] @ states[index]
            reset = sigmoid(x[:gru_size] + h[:gru_size])
            update = sigmoid(x[gru_size:2 * gru_size] + h[gru_size:2 * gru_size])
            candidate = np.tanh(x[2 * gru_size:] + reset * (h[2 * gru_size:] + weights[f"{prefix}.bhn"]))
            states[index] = (1 - update) * candidate + update * states[index]
            value = states[index]
            outputs.append(value)
        joined = np.concatenate(outputs)
        gains.append(sigmoid(weights["gain.weight"] @ joined + weights["gain.bias"]))
        vads.append(sigmoid(weights["vad.weight"] @ joined + weights["vad.bias"]))
    return np.asarray(gains), np.asarray(vads)
