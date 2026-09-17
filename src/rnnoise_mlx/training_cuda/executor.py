"""Execution policy for CUDA lazy graphs.

This module owns optional ``mx.compile`` and evaluation policy.  Graph
construction itself lives in :mod:`training_cuda.graph`.
"""

from __future__ import annotations

import mlx.core as mx

def _tree_map_binary(left, right, operation):
    if isinstance(left, dict):
        return {key: _tree_map_binary(left[key], right[key], operation) for key in left}
    if isinstance(left, list):
        return [_tree_map_binary(a, b, operation) for a, b in zip(left, right)]
    if isinstance(left, tuple):
        return tuple(_tree_map_binary(a, b, operation) for a, b in zip(left, right))
    return operation(left, right)


def _tree_map_scalar(tree, scalar, operation):
    if isinstance(tree, dict):
        return {key: _tree_map_scalar(value, scalar, operation) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_tree_map_scalar(value, scalar, operation) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_map_scalar(value, scalar, operation) for value in tree)
    return operation(tree, scalar)




class CompiledObjective:
    def __init__(self, model, gamma: float):
        self.model = model
        self.gamma = gamma

    def __call__(self, params, features, gain, vad, state, feature_window, first):
        self.model.update(params)
        if first:
            feature_window = mx.slice_update(
                feature_window, features, mx.array(4), axes=(1,)
            )
            zeros = mx.zeros(
                (features.shape[0], self.model.config.gru_size),
                dtype=features.dtype,
            )
            predicted_gain, predicted_vad, h1, h2, h3 = self.model.step_graph(
                feature_window, zeros, zeros, zeros
            )
        else:
            h1, h2, h3 = state
            feature_window = mx.slice_update(
                feature_window, feature_window[:, -4:, :], mx.array(0), axes=(1,)
            )
            feature_window = mx.slice_update(
                feature_window, features, mx.array(4), axes=(1,)
            )
            predicted_gain, predicted_vad, h1, h2, h3 = self.model.step_graph(
                feature_window, h1, h2, h3
            )
        loss = self.model.loss_graph(
            gain, vad, predicted_gain, predicted_vad, self.gamma
        )
        return loss, h1, h2, h3, feature_window


class CompiledFirstGrad:
    def __init__(self, value_and_grad, model):
        self.value_and_grad = value_and_grad
        self.model = model

    def __call__(self, features, gain, vad, feature_window):
        result, gradients = self.value_and_grad(
            self.model.trainable_parameters(), features, gain, vad, (),
            feature_window, True
        )
        loss, h1, h2, h3, feature_window = result
        return loss, (h1, h2, h3), gradients, feature_window


class CompiledNextGrad:
    def __init__(self, value_and_grad, model):
        self.value_and_grad = value_and_grad
        self.model = model

    def __call__(self, features, gain, vad, state, feature_window):
        result, gradients = self.value_and_grad(
            self.model.trainable_parameters(), features, gain, vad, state,
            feature_window, False
        )
        loss, h1, h2, h3, feature_window = result
        return loss, (h1, h2, h3), gradients, feature_window


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
            lambda tree, scalar: _tree_map_scalar(tree, scalar, lambda value, factor: value * factor)
        )
        self.accumulate_gradients = mx.compile(
            lambda left, right: _tree_map_binary(left, right, lambda a, b: a + b)
        )
        self.average_gradients = mx.compile(
            lambda tree, divisor: _tree_map_scalar(tree, divisor, lambda value, d: value / d)
        )

    @staticmethod
    def build_graph(is_first, step, captured_state):
        """Compile one first- or next-chunk step with captured state."""
        del is_first  # The caller selects the step before compilation.
        return mx.compile(step, inputs=captured_state, outputs=captured_state)
