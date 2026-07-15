"""Canary integration test for the deployable RNNoise artifact path."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from rnnoise_mlx.tools.export_canonical_weights import export_checkpoint
from rnnoise_mlx.tools.rnnoise_weights import infer_streaming, read_weights
from rnnoise_mlx.training.loss import rnnoise_loss
from rnnoise_mlx.training.model import ModelConfig, RNNoise


def test_training_safetensors_streaming_canary(tmp_path: Path):
    """Exercise the official model from MLX persistence through deployment inference."""
    mx.random.seed(17)
    model = RNNoise()
    features = mx.random.normal((1, 8, 65))
    target_gain = mx.random.uniform(shape=(1, 8, 32))
    target_vad = mx.random.uniform(shape=(1, 8, 1))

    def objective(candidate, inputs, gain, vad):
        predicted_gain, predicted_vad, _ = candidate(inputs)
        return rnnoise_loss(predicted_gain, predicted_vad, gain, vad)[0]

    loss_and_grad = nn.value_and_grad(model, objective)
    loss, gradients = loss_and_grad(model, features, target_gain, target_vad)
    optimizer = optim.AdamW(learning_rate=1e-3)
    optimizer.update(model, gradients)
    mx.eval(model.state, optimizer.state, loss)
    assert mx.isfinite(loss).item()

    state = (
        mx.zeros((1, 2, 65)),
        mx.zeros((1, 2, 128)),
        mx.zeros((1, 384)),
        mx.zeros((1, 384)),
        mx.zeros((1, 384)),
    )
    expected_gains, expected_vads = [], []
    for index in range(0, features.shape[1], 2):
        gain, vad, state = model.next_chunk(features[:, index : index + 2], state)
        expected_gains.append(gain)
        expected_vads.append(vad)
    expected_gain = mx.concatenate(expected_gains, axis=1)
    expected_vad = mx.concatenate(expected_vads, axis=1)
    mx.eval(expected_gain, expected_vad)

    checkpoint = tmp_path / "model.safetensors"
    canonical = tmp_path / "model.canonical.safetensors"
    model.save(str(checkpoint))
    reloaded = RNNoise.load(str(checkpoint), ModelConfig())
    state = (
        mx.zeros((1, 2, 65)),
        mx.zeros((1, 2, 128)),
        mx.zeros((1, 384)),
        mx.zeros((1, 384)),
        mx.zeros((1, 384)),
    )
    reloaded_gains, reloaded_vads = [], []
    for index in range(0, features.shape[1], 2):
        gain, vad, state = reloaded.next_chunk(features[:, index : index + 2], state)
        reloaded_gains.append(gain)
        reloaded_vads.append(vad)
    reloaded_gain = mx.concatenate(reloaded_gains, axis=1)
    reloaded_vad = mx.concatenate(reloaded_vads, axis=1)
    mx.eval(reloaded_gain, reloaded_vad)

    # Training artifacts can be consumed directly; canonical export is optional.
    weights, config = read_weights(checkpoint)
    actual_gain, actual_vad = infer_streaming(
        weights,
        config,
        np.asarray(features[0], dtype=np.float32),
    )

    np.testing.assert_array_equal(np.asarray(reloaded_gain), np.asarray(expected_gain))
    np.testing.assert_array_equal(np.asarray(reloaded_vad), np.asarray(expected_vad))
    np.testing.assert_allclose(actual_gain, np.asarray(expected_gain[0]), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual_vad, np.asarray(expected_vad[0]), atol=2e-6, rtol=2e-6)
    assert config == {
        "input_dim": 65,
        "cond_size": 128,
        "gru_size": 384,
        "output_dim": 32,
    }

    export_checkpoint(checkpoint, canonical)
    canonical_weights, canonical_config = read_weights(canonical)
    assert canonical_config == config
    for name in weights:
        np.testing.assert_array_equal(canonical_weights[name], weights[name])
