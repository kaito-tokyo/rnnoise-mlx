#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from rnnoise_mlx.data import FeatureDataset
from rnnoise_mlx.loss import rnnoise_loss
from rnnoise_mlx.model import ModelConfig, RNNoise


def sequence_losses(model: RNNoise, dataset: FeatureDataset, gamma: float) -> np.ndarray:
    values = []
    for features, gain, vad in dataset.batches(1, np.random.default_rng(0)):
        predicted_gain, predicted_vad, _ = model(mx.array(features))
        losses = rnnoise_loss(
            predicted_gain, predicted_vad, mx.array(gain), mx.array(vad), gamma
        )
        mx.eval(*losses)
        values.append([float(value.item()) for value in losses])
    return np.asarray(values, dtype=np.float64)


def paired_bootstrap(delta: np.ndarray, seed: int, samples: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(delta, size=delta.size, replace=True).mean()
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "carry_better_fraction": float(np.mean(delta > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("carry_model")
    parser.add_argument("reset_model")
    parser.add_argument("eval_features")
    parser.add_argument("output")
    parser.add_argument("--sequence-length", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = ModelConfig()
    dataset = FeatureDataset(args.eval_features, args.sequence_length)
    carry = sequence_losses(RNNoise.load(args.carry_model, config), dataset, args.gamma)
    reset = sequence_losses(RNNoise.load(args.reset_model, config), dataset, args.gamma)
    # Positive delta means state carry has lower loss.
    delta = reset - carry
    names = ("total", "gain", "vad")
    result = {
        "sequences": int(delta.shape[0]),
        "definition": "delta = reset loss - carry loss; positive favors carry",
        "carry_mean": dict(zip(names, carry.mean(axis=0).tolist(), strict=True)),
        "reset_mean": dict(zip(names, reset.mean(axis=0).tolist(), strict=True)),
        "paired_bootstrap": {
            name: paired_bootstrap(delta[:, index], args.seed + index, args.bootstrap_samples)
            for index, name in enumerate(names)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
