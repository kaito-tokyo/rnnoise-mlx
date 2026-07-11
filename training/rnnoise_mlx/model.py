from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 65
    output_dim: int = 32
    cond_size: int = 128
    gru_size: int = 256


class GRU(nn.GRU):
    """MLX GRU with the sequence/final-state interface used by RNNoise."""

    def __call__(self, x, hidden=None):
        outputs = super().__call__(x, hidden)
        return outputs, outputs[..., -1, :]


class RNNoise(nn.Module):
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

    def first_chunk(self, features):
        """Process the first causal chunk and create convolution/RNN state."""
        conv1 = mx.tanh(self.conv1(features))
        conv2 = mx.tanh(self.conv2(conv1))
        gain, vad, recurrent = self._recurrent_outputs(conv2)
        state = (
            features[..., -2:, :],
            conv1[..., -2:, :],
            recurrent[0],
            recurrent[1],
            recurrent[2],
        )
        return gain, vad, state

    def next_chunk(self, features, state):
        """Process a following chunk while preserving causal Conv/GRU state."""
        feature_history, conv1_history, h1, h2, h3 = state
        conv1 = mx.tanh(self.conv1(mx.concatenate((feature_history, features), axis=-2)))
        conv2 = mx.tanh(self.conv2(mx.concatenate((conv1_history, conv1), axis=-2)))
        gain, vad, recurrent = self._recurrent_outputs(conv2, (h1, h2, h3))
        new_state = (
            features[..., -2:, :],
            conv1[..., -2:, :],
            recurrent[0],
            recurrent[1],
            recurrent[2],
        )
        return gain, vad, new_state

    def save(self, path: str):
        flat = dict(mlx_utils.tree_flatten(self.parameters()))
        mx.save_safetensors(path, flat, metadata={"config": json.dumps(asdict(self.config))})

    @classmethod
    def load(cls, path: str, config: ModelConfig):
        model = cls(config)
        model.load_weights(path)
        mx.eval(model.parameters())
        return model
