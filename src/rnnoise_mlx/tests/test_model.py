"""Tests for the MLX model, loss, and recurrent-state behavior."""

import mlx.core as mx
import numpy as np
import tempfile
from pathlib import Path

from rnnoise_mlx.training.loss import rnnoise_loss
from rnnoise_mlx.training.model import ModelConfig, RNNoise
from rnnoise_mlx.training.state_diagnostics import stale_state_diagnostic, trace_chunks
from rnnoise_mlx.training.tracking import _flatten_metrics


def test_shapes_and_finite_loss():
    model = RNNoise(ModelConfig(cond_size=8, gru_size=12))
    features = mx.random.normal((2, 10, 65))
    gain = mx.random.uniform(shape=(2, 10, 32))
    vad = mx.random.uniform(shape=(2, 10, 1))
    predicted_gain, predicted_vad, states = model(features)
    loss, _, _ = rnnoise_loss(predicted_gain, predicted_vad, gain, vad)
    mx.eval(loss)
    assert predicted_gain.shape == (2, 6, 32)
    assert predicted_vad.shape == (2, 6, 1)
    assert [state.shape for state in states] == [(2, 12)] * 3
    assert mx.isfinite(loss).item()


def test_stateful_chunks_match_full_sequence():
    model = RNNoise(ModelConfig(cond_size=8, gru_size=12))
    features = mx.random.normal((2, 600, 65))
    full_gain, full_vad, _ = model(features)
    gain1, vad1, state = model.first_chunk(features[:, :200, :])
    gain2, vad2, state = model.next_chunk(features[:, 200:400, :], state)
    gain3, vad3, _ = model.next_chunk(features[:, 400:, :], state)
    chunked_gain = mx.concatenate((gain1, gain2, gain3), axis=1)
    chunked_vad = mx.concatenate((vad1, vad2, vad3), axis=1)
    mx.eval(full_gain, full_vad, chunked_gain, chunked_vad)
    assert mx.max(mx.abs(full_gain - chunked_gain)).item() == 0
    assert mx.max(mx.abs(full_vad - chunked_vad)).item() == 0


def test_diagnostic_chunk_lengths_and_reset():
    model = RNNoise(ModelConfig(cond_size=8, gru_size=12))
    features = mx.random.normal((2, 600, 65))
    full_gain, full_vad, _ = model(features)
    for chunk_length in (100, 200, 400, 800):
        continuous, state, _ = trace_chunks(model, features, chunk_length)
        mx.eval(continuous["gain"], continuous["vad"], *state)
        assert mx.max(mx.abs(full_gain - continuous["gain"])).item() == 0
        assert mx.max(mx.abs(full_vad - continuous["vad"])).item() == 0
        assert state[0].shape == (2, 2, 65)
        assert state[1].shape == (2, 2, 8)
        assert [value.shape for value in state[2:]] == [(2, 12)] * 3
    reset, _, _ = trace_chunks(model, features, 200, "all")
    mx.eval(reset["gain"])
    assert mx.max(mx.abs(full_gain - reset["gain"])).item() > 0
    assert mx.all(mx.isfinite(reset["gain"])).item()


def test_stale_state_diagnostic_preserves_checkpoint():
    config = ModelConfig(cond_size=8, gru_size=12)
    model = RNNoise(config)
    rng = np.random.default_rng(0)
    batch = (
        rng.normal(size=(2, 400, 65)).astype(np.float32),
        rng.uniform(size=(2, 400, 32)).astype(np.float32),
        rng.uniform(size=(2, 400, 1)).astype(np.float32),
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "model.safetensors"
        model.save(str(checkpoint))
        result = stale_state_diagnostic(str(checkpoint), config, batch, 100)
        assert result["checkpoint_hash_unchanged"]
        assert result["next_gain_relative_error"] >= 0
        assert result["next_vad_relative_error"] >= 0


def test_mlflow_evaluation_metrics_are_flat_and_numeric():
    metrics = _flatten_metrics(
        "eval_initial",
        {"total_loss": 0.5, "finite": True, "batches": 4, "label": "ignored"},
    )
    assert metrics == {
        "eval_initial_total_loss": 0.5,
        "eval_initial_finite": 1.0,
        "eval_initial_batches": 4.0,
    }
