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
        loss, *_ = self.model._objective(
            features, gain, vad, gru1_state, gru2_state, gru3_state,
            feature,
        )
        return loss, gru1_state, gru2_state, gru3_state, feature


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

    def __init__(self, model, optimizer, gamma: float, captured_state):
        self.model = model
        self.optimizer = optimizer
        value_and_grad = mx.value_and_grad(CompiledObjective(model, gamma))
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

    @staticmethod
    def build_graph(is_first, step, captured_state):
        """Compile one first- or next-chunk step with captured state."""
        del is_first  # The caller selects the step before compilation.
        return mx.compile(step, inputs=captured_state, outputs=captured_state)
