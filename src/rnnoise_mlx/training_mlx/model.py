"""RNNoise MLX inference model."""

from __future__ import annotations

from dataclasses import asdict
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils

from ..config import ModelConfig


class GRU(nn.GRU):
    """MLX GRU adapter exposing sequence output and final hidden state."""

    def __call__(self, x, hidden=None):
        outputs = super().__call__(x, hidden)
        return outputs, outputs[..., -1, :]


class RNNoiseFrameStep(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.input_dim, config.cond_size, 3)
        self.conv2 = nn.Conv1d(config.cond_size, config.gru_size, 3)
        self.gru1 = GRU(config.gru_size, config.gru_size)
        self.gru2 = GRU(config.gru_size, config.gru_size)
        self.gru3 = GRU(config.gru_size, config.gru_size)
        self.gain = nn.Linear(4 * config.gru_size, config.output_dim)
        self.vad = nn.Linear(4 * config.gru_size, 1)

    def _recurrent_outputs(self, x, states=None):
        states = states or (None, None, None)
        y1, h1 = self.gru1(x, states[0])
        y2, h2 = self.gru2(y1, states[1])
        y3, h3 = self.gru3(y2, states[2])
        joined = mx.concatenate((x, y1, y2, y3), axis=-1)
        return mx.sigmoid(self.gain(joined)), mx.sigmoid(self.vad(joined)), (h1, h2, h3)

    def __call__(self, features, states=None):
        x = mx.tanh(self.conv1(features))
        x = mx.tanh(self.conv2(x))
        return self._recurrent_outputs(x, states)

    def save(self, path: str):
        flat = dict(mlx_utils.tree_flatten(self.parameters()))
        mx.save_safetensors(path, flat, metadata={"config": json.dumps(asdict(self.config))})

    @classmethod
    def load(cls, path: str, config: ModelConfig):
        model = cls(config)
        model.load_weights(path)
        mx.eval(model.parameters())
        return model


# Compatibility alias for weight conversion and existing callers.
RNNoise = RNNoiseFrameStep
