# SPDX-FileCopyrightText: 2026 Kaito Udagawa <umireon@kaito.tokyo>
#
# SPDX-License-Identifier: BSD-3-Clause

import mlx.core as mx
import mlx.nn as nn

from ..training.config import ModelConfig

class RNNoise(nn.Module):
    def __init__(self, config: ModelConfig, batch_size: int = 8, tbptt_length: int = 250):
        """Initialize a fixed-shape CUDA RNNoise graph model.

        :param config: Network dimensions shared by the training backend.
        :param batch_size: Fixed batch size used by ``step_graph`` input and
            hidden-state shape checks.
        :param tbptt_length: Number of new frames processed by one TBPTT
            step.  The graph input window contains four additional causal
            context frames.
        :raises AssertionError: If ``batch_size`` or ``tbptt_length`` is not
            positive.
        """
        super().__init__()

        assert batch_size > 0
        assert tbptt_length > 0
        self.config = config
        self.batch_size = batch_size
        self.tbptt_length = tbptt_length

        # Registers the layers and modules used in the RNNoise model
        self.conv1 = nn.Conv1d(config.input_dim, config.cond_size, 3)
        self.conv2 = nn.Conv1d(config.cond_size, config.gru_size, 3)
        self.gru1 = nn.GRU(config.gru_size, config.gru_size)
        self.gru2 = nn.GRU(config.gru_size, config.gru_size)
        self.gru3 = nn.GRU(config.gru_size, config.gru_size)
        self.gain = nn.Linear(4 * config.gru_size, config.output_dim)
        self.vad = nn.Linear(4 * config.gru_size, 1)

    def step_graph(self, feature_window, h1, h2, h3):
        """Evaluate one fixed-shape RNNoise recurrent step.

        The input window contains four frames of causal convolution history
        followed by one ``tbptt_length``-frame chunk.  Both convolutions use
        valid kernels, so the four history frames are consumed and the
        outputs retain exactly ``tbptt_length`` frames.  This method constructs lazy MLX
        expressions; the caller controls compilation and evaluation.

        :param feature_window: MLX array with shape
            ``(batch_size, tbptt_length + 4, config.input_dim)``.  It is
            neither returned nor mutated; the caller owns its overlap-save
            update.
        :param h1: First GRU carry with shape
            ``(batch_size, config.gru_size)``.
        :param h2: Second GRU carry with shape
            ``(batch_size, config.gru_size)``.
        :param h3: Third GRU carry with shape
            ``(batch_size, config.gru_size)``.
        :returns: A flat tuple ``(gain, vad, h1_next, h2_next, h3_next)``.
            ``gain`` has shape ``(batch_size, tbptt_length, config.output_dim)``;
            ``vad`` has shape ``(batch_size, tbptt_length, 1)``; and each returned
            carry has shape ``(batch_size, config.gru_size)``.
        :raises AssertionError: If an input has an incompatible shape.

        Pass each returned carry to the corresponding hidden-state argument
        of the next chunk.
        """

        assert feature_window.shape == (
            self.batch_size,
            self.tbptt_length + 4,
            self.config.input_dim,
        )
        assert h1.shape == (self.batch_size, self.config.gru_size)
        assert h2.shape == (self.batch_size, self.config.gru_size)
        assert h3.shape == (self.batch_size, self.config.gru_size)

        conv1 = mx.tanh(self.conv1(feature_window))
        conv2 = mx.tanh(self.conv2(conv1))
        y1 = self.gru1(conv2, h1)
        h1 = y1[..., -1, :]
        y2 = self.gru2(y1, h2)
        h2 = y2[..., -1, :]
        y3 = self.gru3(y2, h3)
        h3 = y3[..., -1, :]
        joined = mx.concatenate((conv2, y1, y2, y3), axis=-1)
        predicted_gain = mx.sigmoid(self.gain(joined))
        predicted_vad = mx.sigmoid(self.vad(joined))
        return predicted_gain, predicted_vad, h1, h2, h3

    def loss_graph(self, target_gain, target_vad, predicted_gain, predicted_vad, gamma):
        """Compute the CUDA training loss for one fixed-shape chunk.

        :param target_gain: Gain targets with shape
            ``(batch_size, tbptt_length, config.output_dim)``.
        :param target_vad: VAD targets with shape
            ``(batch_size, tbptt_length, 1)``.
        :param predicted_gain: Predicted gains with the same shape as
            ``target_gain``.
        :param predicted_vad: Predicted VAD probabilities with the same shape
            as ``target_vad``.
        :param gamma: Positive exponent used by the gain loss.
        :returns: A scalar MLX loss expression.  The expression is lazy and
            is materialized by the caller.
        :raises AssertionError: If prediction and target shapes do not match,
            or if ``gamma`` is not positive.
        """

        assert target_gain.shape == (
            self.batch_size,
            self.tbptt_length,
            self.config.output_dim,
        )
        assert target_vad.shape == (
            self.batch_size,
            self.tbptt_length,
            1,
        )
        assert predicted_gain.shape == target_gain.shape
        assert predicted_vad.shape == target_vad.shape
        assert gamma > 0

        target = mx.maximum(target_gain, 0)
        target = target * mx.square(mx.tanh(8 * target))
        active = mx.minimum(target_gain + 1, 1)
        error = predicted_gain**gamma - target**gamma
        gain_loss = mx.mean((1 + 5 * target_vad) * active * mx.square(error))
        vad_loss = mx.mean(
            mx.abs(2 * target_vad - 1)
            * (
                -target_vad * mx.log(0.01 + predicted_vad)
                - (1 - target_vad) * mx.log(1.01 - predicted_vad)
            )
        )
        return gain_loss + 0.001 * vad_loss
