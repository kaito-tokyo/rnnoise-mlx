"""Execution policy for CUDA lazy graphs.

This module owns optional ``mx.compile`` and evaluation policy.  Graph
construction itself lives in :mod:`training_cuda.graph`.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map

class CompiledObjective:
    def __init__(self, model, gamma: float):
        self.model = model
        self.gamma = gamma

    def __call__(self, params, features, gain, vad, state, feature, first):
        self.model.update(params)
        if first:
            feature = mx.slice_update(
                feature, features, mx.array(4), axes=(1,)
            )
            zeros = mx.zeros(
                (features.shape[0], self.model.model_config.gru_size),
                dtype=features.dtype,
            )
            predicted_gain, predicted_vad, gru1_state, gru2_state, gru3_state = self.model(
                feature, zeros, zeros, zeros
            )
        else:
            gru1_state, gru2_state, gru3_state = state
            feature = mx.slice_update(
                feature, feature[:, -4:, :], mx.array(0), axes=(1,)
            )
            feature = mx.slice_update(
                feature, features, mx.array(4), axes=(1,)
            )
            predicted_gain, predicted_vad, gru1_state, gru2_state, gru3_state = self.model(
                feature, gru1_state, gru2_state, gru3_state
            )
        target = mx.maximum(gain, 0)
        target = target * mx.square(mx.tanh(8 * target))
        active = mx.minimum(gain + 1, 1)
        error = predicted_gain**self.gamma - target**self.gamma
        gain_loss = mx.mean((1 + 5 * vad) * active * mx.square(error))
        vad_weight = mx.abs(2 * vad - 1)
        vad_positive_loss = -vad * mx.log(0.01 + predicted_vad)
        vad_negative_loss = -(1 - vad) * mx.log(1.01 - predicted_vad)
        vad_loss = mx.mean(vad_weight * (vad_positive_loss + vad_negative_loss))
        return (
            gain_loss + 0.001 * vad_loss,
            gru1_state,
            gru2_state,
            gru3_state,
            feature,
        )


class CompiledFirstGrad:
    def __init__(self, value_and_grad, model):
        self.value_and_grad = value_and_grad
        self.model = model

    def __call__(self, features, gain, vad, feature):
        result, gradients = self.value_and_grad(
            self.model.trainable_parameters(), features, gain, vad, (),
            feature, True
        )
        loss, gru1_state, gru2_state, gru3_state, feature = result
        return loss, (gru1_state, gru2_state, gru3_state), gradients, feature


class CompiledNextGrad:
    def __init__(self, value_and_grad, model):
        self.value_and_grad = value_and_grad
        self.model = model

    def __call__(self, features, gain, vad, state, feature):
        result, gradients = self.value_and_grad(
            self.model.trainable_parameters(), features, gain, vad, state,
            feature, False
        )
        loss, gru1_state, gru2_state, gru3_state, feature = result
        return loss, (gru1_state, gru2_state, gru3_state), gradients, feature


class CompiledOptimizerUpdate:
    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model

    def __call__(self, gradients):
        self.optimizer.update(self.model, gradients)


class CompiledChunkRunner:
    """Apply CUDA execution policy to the fixed lazy chunk graphs."""

    def __init__(self, model, optimizer, gamma: float, captured_state, segment_length: int):
        self.model = model
        self.optimizer = optimizer
        self.segment_length = segment_length
        self.value_and_grad = mx.value_and_grad(CompiledObjective(model, gamma))
        self.compiled_update = mx.compile(
            self._update,
            inputs=captured_state,
            outputs=captured_state,
        )

        # Retain the old attributes for callers that still expose diagnostic
        # chunk steps.  The training path should use ``update`` instead.
        value_and_grad = self.value_and_grad
        self.first_grad = self.build_graph(
            True, CompiledFirstGrad(value_and_grad, model), captured_state
        )
        self.next_grad = self.build_graph(
            False, CompiledNextGrad(value_and_grad, model), captured_state
        )
        self.apply_optimizer = mx.compile(
            CompiledOptimizerUpdate(optimizer, model),
            inputs=captured_state, outputs=captured_state,
        )
        self.scale_gradients = mx.compile(
            lambda tree, scalar: tree_map(
                lambda value: value * scalar,
                tree,
            )
        )
        self.accumulate_gradients = mx.compile(
            lambda left, right: tree_map(
                lambda a, b: a + b,
                left,
                right,
            )
        )
        self.average_gradients = mx.compile(
            lambda tree, divisor: tree_map(
                lambda value: value / divisor,
                tree,
            )
        )

    def _update(self, features, gain, vad):
        """Compile one complete segmented-TBPTT update."""
        workspace = mx.zeros(
            (features.shape[0], self.segment_length + 4, features.shape[2]),
            dtype=features.dtype,
        )
        state = ()
        accumulated_loss = mx.zeros((), dtype=features.dtype)
        accumulated_gradients = tree_map(
            mx.zeros_like,
            self.model.trainable_parameters(),
        )
        target_frames = features.shape[1]

        for start in range(0, target_frames, self.segment_length):
            end = start + self.segment_length
            chunk_features = features[:, start:end, :]
            chunk_gain = gain[:, start:end, :]
            chunk_vad = vad[:, start:end, :]
            result, gradients = self.value_and_grad(
                self.model.trainable_parameters(),
                chunk_features,
                chunk_gain,
                chunk_vad,
                state,
                workspace,
                start == 0,
            )
            loss, h1, h2, h3, workspace = result
            state = (h1, h2, h3)
            accumulated_loss = accumulated_loss + loss * self.segment_length
            accumulated_gradients = tree_map(
                lambda total, value: total + value * self.segment_length,
                accumulated_gradients,
                gradients,
            )
            state = tuple(mx.stop_gradient(value) for value in state)

        gradients = tree_map(
            lambda value: value / target_frames,
            accumulated_gradients,
        )
        self.optimizer.update(self.model, gradients)
        return accumulated_loss / target_frames

    def update(self, features, gain, vad):
        """Run one compiled update and materialize its mutable state."""
        loss = self.compiled_update(features, gain, vad)
        mx.eval(self.model.state, self.optimizer.state, loss)
        return loss

    @staticmethod
    def build_graph(is_first, step, captured_state):
        """Compile one first- or next-chunk step with captured state."""
        del is_first  # The caller selects the step before compilation.
        return mx.compile(step, inputs=captured_state, outputs=captured_state)
