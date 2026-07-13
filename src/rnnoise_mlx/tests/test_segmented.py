"""Tests for reusable segmented-TBPTT training graphs."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from rnnoise_mlx.training.loss import rnnoise_loss_aligned
from rnnoise_mlx.training.model import RNNoise
from rnnoise_mlx.training.segmented import make_segmented_step


def _monolithic_step(
    model,
    optimizer,
    features,
    gain,
    vad,
    *,
    segment_length,
    segment_state,
    equalize_reset_targets,
    gamma=0.25,
):
    sequence_length = features.shape[1]

    def objective(candidate, inputs, target_gain, target_vad):
        weighted_loss = 0
        target_frames = 0
        state = None
        for start in range(0, sequence_length, segment_length):
            end = start + segment_length
            chunk = inputs[:, start:end, :]
            if start == 0 or segment_state == "reset":
                predicted_gain, predicted_vad, state = candidate.first_chunk(chunk)
                chunk_gain = target_gain[:, start + 3 : end - 1, :]
                chunk_vad = target_vad[:, start + 3 : end - 1, :]
            else:
                state = tuple(mx.stop_gradient(value) for value in state)
                predicted_gain, predicted_vad, state = candidate.next_chunk(chunk, state)
                if equalize_reset_targets:
                    predicted_gain = predicted_gain[:, 4:, :]
                    predicted_vad = predicted_vad[:, 4:, :]
                    chunk_gain = target_gain[:, start + 3 : end - 1, :]
                    chunk_vad = target_vad[:, start + 3 : end - 1, :]
                else:
                    chunk_gain = target_gain[:, start - 1 : end - 1, :]
                    chunk_vad = target_vad[:, start - 1 : end - 1, :]
            loss = rnnoise_loss_aligned(
                predicted_gain, predicted_vad, chunk_gain, chunk_vad, gamma
            )[0]
            frames = predicted_gain.shape[-2]
            weighted_loss = weighted_loss + frames * loss
            target_frames += frames
        return weighted_loss / target_frames

    loss_and_grad = nn.value_and_grad(model, objective)
    loss, gradients = loss_and_grad(model, features, gain, vad)
    optimizer.update(model, gradients)
    return loss


def _clone_model(source):
    clone = RNNoise()
    clone.update(source.parameters())
    return clone


def _assert_trees_close(left, right, *, atol=2e-6, rtol=2e-6):
    left_flat = tree_flatten(left)
    right_flat = tree_flatten(right)
    assert [name for name, _ in left_flat] == [name for name, _ in right_flat]
    for (name, left_value), (_, right_value) in zip(left_flat, right_flat):
        assert mx.allclose(left_value, right_value, atol=atol, rtol=rtol).item(), name


@pytest.mark.parametrize(
    ("segment_state", "equalize_reset_targets"),
    (("carry", False), ("carry", True), ("reset", False)),
)
def test_reusable_chunk_step_matches_monolithic_update(
    segment_state, equalize_reset_targets
):
    mx.random.seed(7)
    reference_model = RNNoise()
    reusable_model = _clone_model(reference_model)
    reference_optimizer = optim.AdamW(
        learning_rate=1e-3, betas=(0.8, 0.98), eps=1e-8
    )
    reusable_optimizer = optim.AdamW(
        learning_rate=1e-3, betas=(0.8, 0.98), eps=1e-8
    )
    features = mx.random.normal((1, 12, 65))
    gain = mx.random.uniform(shape=(1, 12, 32))
    vad = mx.random.uniform(shape=(1, 12, 1))

    reference_loss = _monolithic_step(
        reference_model,
        reference_optimizer,
        features,
        gain,
        vad,
        segment_length=6,
        segment_state=segment_state,
        equalize_reset_targets=equalize_reset_targets,
    )
    reusable_step = make_segmented_step(
        reusable_model,
        reusable_optimizer,
        sequence_length=12,
        segment_length=6,
        segment_state=segment_state,
        equalize_reset_targets=equalize_reset_targets,
        gamma=0.25,
        compile=False,
    )
    reusable_loss = reusable_step(features, gain, vad)
    mx.eval(
        reference_loss,
        reusable_loss,
        reference_model.state,
        reusable_model.state,
        reference_optimizer.state,
        reusable_optimizer.state,
    )

    assert mx.allclose(reference_loss, reusable_loss, atol=2e-6, rtol=2e-6).item()
    _assert_trees_close(reference_model.parameters(), reusable_model.parameters())
    _assert_trees_close(reference_optimizer.state, reusable_optimizer.state)


def test_compiled_reusable_chunk_step_matches_two_monolithic_updates():
    mx.random.seed(11)
    reference_model = RNNoise()
    model = _clone_model(reference_model)
    reference_optimizer = optim.AdamW(
        learning_rate=1e-3, betas=(0.8, 0.98), eps=1e-8
    )
    optimizer = optim.AdamW(
        learning_rate=1e-3, betas=(0.8, 0.98), eps=1e-8
    )
    step = make_segmented_step(
        model,
        optimizer,
        sequence_length=18,
        segment_length=6,
        segment_state="carry",
        equalize_reset_targets=False,
        gamma=0.25,
        compile=True,
    )
    features = mx.random.normal((1, 18, 65))
    gain = mx.random.uniform(shape=(1, 18, 32))
    vad = mx.random.uniform(shape=(1, 18, 1))

    first_loss = step(features, gain, vad)
    first_reference_loss = _monolithic_step(
        reference_model,
        reference_optimizer,
        features,
        gain,
        vad,
        segment_length=6,
        segment_state="carry",
        equalize_reset_targets=False,
    )
    second_loss = step(features, gain, vad)
    second_reference_loss = _monolithic_step(
        reference_model,
        reference_optimizer,
        features,
        gain,
        vad,
        segment_length=6,
        segment_state="carry",
        equalize_reset_targets=False,
    )
    mx.eval(
        first_loss,
        first_reference_loss,
        second_loss,
        second_reference_loss,
        model.state,
        optimizer.state,
        reference_model.state,
        reference_optimizer.state,
    )

    assert mx.allclose(first_loss, first_reference_loss, atol=2e-6, rtol=2e-6).item()
    assert mx.allclose(second_loss, second_reference_loss, atol=2e-6, rtol=2e-6).item()
    _assert_trees_close(reference_model.parameters(), model.parameters())
    _assert_trees_close(reference_optimizer.state, optimizer.state)
