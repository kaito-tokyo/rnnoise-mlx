"""Training-time model evaluation helpers."""

from __future__ import annotations

import math
import mlx.core as mx
import numpy as np

from .data import FeatureDataset
from .loss import rnnoise_loss


def evaluate(model, dataset: FeatureDataset, batch_size: int, gamma: float = 0.25):
    totals = np.zeros(3, dtype=np.float64)
    batches = 0
    finite = True
    in_range = True
    # Evaluation order is fixed and does not update model state or weights.
    for features, gain, vad in dataset.batches(batch_size, np.random.default_rng(0)):
        predicted_gain, predicted_vad, _ = model(mx.array(features))
        losses = rnnoise_loss(predicted_gain, predicted_vad, mx.array(gain), mx.array(vad), gamma)
        mx.eval(predicted_gain, predicted_vad, *losses)
        values = np.array([value.item() for value in losses], dtype=np.float64)
        totals += values
        batches += 1
        finite &= bool(np.isfinite(values).all())
        in_range &= bool(
            mx.all(mx.isfinite(predicted_gain)).item()
            and mx.all(mx.isfinite(predicted_vad)).item()
            and mx.min(predicted_gain).item() >= 0
            and mx.max(predicted_gain).item() <= 1
            and mx.min(predicted_vad).item() >= 0
            and mx.max(predicted_vad).item() <= 1
        )
    if batches == 0:
        raise ValueError("evaluation dataset has no complete batch")
    means = totals / batches
    return {
        "total_loss": float(means[0]),
        "gain_loss": float(means[1]),
        "vad_loss": float(means[2]),
        "finite": finite and all(math.isfinite(value) for value in means),
        "outputs_in_unit_interval": in_range,
        "batches": batches,
    }
