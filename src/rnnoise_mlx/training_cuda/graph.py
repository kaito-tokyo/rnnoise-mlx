# SPDX-FileCopyrightText: 2026 Kaito Udagawa <umireon@kaito.tokyo>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional as Opt

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from ..training.config import ModelConfig, TrainConfig


class RNNoiseFrameStep(nn.Module):
    """RNNoise frame/window computation and recurrent-state transition."""

    def __init__(self, model_config: ModelConfig, train_config: Opt[TrainConfig]):
        super().__init__()

        assert model_config is not None

        self.model_config = model_config
        self.train_config = train_config

        if train_config is None:
            self.feature_shape = (1, 1, model_config.input_dim)
            self.gru_shape = (1, model_config.gru_size)
            self.conv1_state_shape = (1, 2, model_config.input_dim)
            self.conv2_state_shape = (1, 2, model_config.cond_size)
        else:
            self.feature_shape = (
                train_config.batch_size,
                train_config.tbptt_length + 4,
                model_config.input_dim,
            )
            self.gru_shape = (train_config.batch_size, model_config.gru_size)

        # Operators
        self.conv1 = nn.Conv1d(model_config.input_dim, model_config.cond_size, 3)
        self.conv2 = nn.Conv1d(model_config.cond_size, model_config.gru_size, 3)
        self.gru1 = nn.GRU(model_config.gru_size, model_config.gru_size)
        self.gru2 = nn.GRU(model_config.gru_size, model_config.gru_size)
        self.gru3 = nn.GRU(model_config.gru_size, model_config.gru_size)
        self.dense_out = nn.Linear(4 * model_config.gru_size, model_config.output_dim)
        self.vad = nn.Linear(4 * model_config.gru_size, 1)

    def __call__(
        self,
        feature,
        gru_states,
        conv_states=None,
    ):
        assert feature.shape == self.feature_shape
        assert gru_states is not None
        assert gru_states[0].shape == self.gru_shape
        assert gru_states[1].shape == self.gru_shape
        assert gru_states[2].shape == self.gru_shape

        if self.train_config is None:
            assert conv_states is not None
            assert conv_states[0].shape == self.conv1_state_shape
            assert conv_states[1].shape == self.conv2_state_shape
        else:
            assert conv_states is None

        if self.train_config is None:
            conv1_input = mx.concatenate((conv_states[0], feature), axis=1)
            conv1 = mx.tanh(self.conv1(conv1_input))
            conv2_input = mx.concatenate((conv_states[1], conv1), axis=1)
            conv2 = mx.tanh(self.conv2(conv2_input))
            next_conv_states = (conv1_input[:, -2:, :], conv2_input[:, -2:, :])
        else:
            conv1 = mx.tanh(self.conv1(feature))
            conv2 = mx.tanh(self.conv2(conv1))
            next_conv_states = None

        y1 = self.gru1(conv2, gru_states[0])
        y2 = self.gru2(y1, gru_states[1])
        y3 = self.gru3(y2, gru_states[2])
        next_gru_states = (y1[..., -1, :], y2[..., -1, :], y3[..., -1, :])

        joined = mx.concatenate((conv2, y1, y2, y3), axis=-1)
        dense_out = mx.sigmoid(self.dense_out(joined))
        vad = mx.sigmoid(self.vad(joined))

        return (dense_out, vad, next_gru_states, next_conv_states)


class RNNoiseChunk(nn.Module):
    """Compute one fixed-shape training chunk and its gradients."""

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        super().__init__()

        assert model_config is not None
        assert train_config is not None

        self.model_config = model_config
        self.train_config = train_config

        self.step = RNNoiseFrameStep(model_config, train_config)
        self.value_and_grad = nn.value_and_grad(self, self._objective)

    def __call__(self, feature, targets, state, accumulated_gradients):
        (loss, next_state), gradients = self.value_and_grad(feature, targets, state)
        weighted_gradients = tree_map(
            lambda x: x * self.train_config.tbptt_length, gradients
        )
        accumulated_gradients = tree_map(
            lambda x, y: x + y, accumulated_gradients, weighted_gradients
        )
        return (loss, next_state, accumulated_gradients)

    def _objective(self, feature, targets, state):
        target_dense_out, target_vad = targets
        model_out = self.step(feature, state)
        predicted_dense_out, predicted_vad = model_out[:2]
        target = mx.maximum(target_dense_out, 0)
        target = target * mx.square(mx.tanh(8 * target))
        active = mx.minimum(target_dense_out + 1, 1)
        gamma = self.train_config.gamma
        error = predicted_dense_out**gamma - target**gamma
        gain_loss = mx.mean((1 + 5 * target_vad) * active * mx.square(error))
        vad_weight = mx.abs(2 * target_vad - 1)
        vad_positive_loss = -target_vad * mx.log(0.01 + predicted_vad)
        vad_negative_loss = -(1 - target_vad) * mx.log(1.01 - predicted_vad)
        vad_loss = mx.mean(vad_weight * (vad_positive_loss + vad_negative_loss))
        return gain_loss + 0.001 * vad_loss, model_out[2]


class RNNoise(nn.Module):
    """Thin public composite model for RNNoise training and inference."""

    def __init__(
        self, model_config: ModelConfig, train_config: Opt[TrainConfig] = None
    ):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.chunk = (
            RNNoiseChunk(model_config, train_config)
            if train_config is not None
            else None
        )
        self.step = (
            RNNoiseFrameStep(model_config, None) if train_config is None else None
        )

    def __call__(self, *args, **kwargs):
        step = self.step if self.chunk is None else self.chunk.step
        return step(*args, **kwargs)

    def objective(self, *args, **kwargs):
        assert self.chunk is not None
        return self.chunk._objective(*args, **kwargs)

    def value_and_grad(self):
        assert self.chunk is not None
        return nn.value_and_grad(self, self.objective)
