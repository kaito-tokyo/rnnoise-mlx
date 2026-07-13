"""Reusable compiled steps for segmented truncated backpropagation."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from .loss import rnnoise_loss_aligned


def make_segmented_step(
    model: nn.Module,
    optimizer: Any,
    *,
    sequence_length: int,
    segment_length: int,
    segment_state: str,
    equalize_reset_targets: bool,
    gamma: float,
    compile: bool = True,
) -> Callable:
    """Build a sequence step from reusable first/next chunk graphs.

    Each chunk computes its forward pass and parameter gradients while model
    weights remain fixed. Gradients and loss are accumulated on device, then a
    single optimizer update is applied after the complete sequence.
    """
    if segment_state not in ("carry", "reset"):
        raise ValueError("segment_state must be 'carry' or 'reset'")
    if segment_length <= 4 or sequence_length % segment_length != 0:
        raise ValueError("segment_length must be >4 and divide sequence_length")

    segment_count = sequence_length // segment_length
    first_frames = segment_length - 4
    next_frames = first_frames if equalize_reset_targets else segment_length
    total_frames = (
        segment_count * first_frames
        if segment_state == "reset"
        else first_frames + (segment_count - 1) * next_frames
    )

    def first_objective(candidate, features, gain, vad):
        predicted_gain, predicted_vad, state = candidate.first_chunk(features)
        loss = rnnoise_loss_aligned(
            predicted_gain, predicted_vad, gain, vad, gamma
        )[0]
        return first_frames * loss, state

    def next_objective(candidate, features, gain, vad, state):
        state = tuple(mx.stop_gradient(value) for value in state)
        predicted_gain, predicted_vad, new_state = candidate.next_chunk(
            features, state
        )
        if equalize_reset_targets:
            predicted_gain = predicted_gain[:, 4:, :]
            predicted_vad = predicted_vad[:, 4:, :]
        loss = rnnoise_loss_aligned(
            predicted_gain, predicted_vad, gain, vad, gamma
        )[0]
        return next_frames * loss, new_state

    first_value_and_grad = nn.value_and_grad(model, first_objective)
    next_value_and_grad = nn.value_and_grad(model, next_objective)

    def first_chunk_step(features, gain, vad):
        (weighted_loss, state), gradients = first_value_and_grad(
            model, features, gain, vad
        )
        return weighted_loss, state, gradients

    def next_chunk_step(
        features, gain, vad, state, accumulated_loss, accumulated_gradients
    ):
        (weighted_loss, new_state), gradients = next_value_and_grad(
            model, features, gain, vad, state
        )
        accumulated_gradients = tree_map(
            lambda total, value: total + value,
            accumulated_gradients,
            gradients,
        )
        return (
            accumulated_loss + weighted_loss,
            new_state,
            accumulated_gradients,
        )

    def reset_chunk_step(
        features, gain, vad, accumulated_loss, accumulated_gradients
    ):
        (weighted_loss, state), gradients = first_value_and_grad(
            model, features, gain, vad
        )
        accumulated_gradients = tree_map(
            lambda total, value: total + value,
            accumulated_gradients,
            gradients,
        )
        return accumulated_loss + weighted_loss, state, accumulated_gradients

    def optimizer_step(accumulated_loss, accumulated_gradients):
        mean_gradients = tree_map(
            lambda value: value / total_frames, accumulated_gradients
        )
        optimizer.update(model, mean_gradients)
        return accumulated_loss / total_frames

    if compile:
        model_state = [model.state]
        training_state = [model.state, optimizer.state]
        first_chunk_step = partial(
            mx.compile, inputs=model_state, outputs=model_state
        )(first_chunk_step)
        next_chunk_step = partial(
            mx.compile, inputs=model_state, outputs=model_state
        )(next_chunk_step)
        reset_chunk_step = partial(
            mx.compile, inputs=model_state, outputs=model_state
        )(reset_chunk_step)
        optimizer_step = partial(
            mx.compile, inputs=training_state, outputs=training_state
        )(optimizer_step)

    def segmented_step(features, gain, vad):
        first_end = segment_length
        accumulated_loss, state, accumulated_gradients = first_chunk_step(
            features[:, :first_end, :],
            gain[:, 3 : first_end - 1, :],
            vad[:, 3 : first_end - 1, :],
        )
        mx.async_eval(accumulated_loss, state, accumulated_gradients)

        for start in range(segment_length, sequence_length, segment_length):
            end = start + segment_length
            chunk = features[:, start:end, :]
            if segment_state == "reset":
                accumulated_loss, state, accumulated_gradients = reset_chunk_step(
                    chunk,
                    gain[:, start + 3 : end - 1, :],
                    vad[:, start + 3 : end - 1, :],
                    accumulated_loss,
                    accumulated_gradients,
                )
            else:
                target_start = start + 3 if equalize_reset_targets else start - 1
                accumulated_loss, state, accumulated_gradients = next_chunk_step(
                    chunk,
                    gain[:, target_start : end - 1, :],
                    vad[:, target_start : end - 1, :],
                    state,
                    accumulated_loss,
                    accumulated_gradients,
                )
            # Bound the lazy graph and activation lifetime without waiting on
            # the host. State and accumulator dependencies serialize chunks.
            mx.async_eval(accumulated_loss, state, accumulated_gradients)

        return optimizer_step(accumulated_loss, accumulated_gradients)

    return segmented_step
