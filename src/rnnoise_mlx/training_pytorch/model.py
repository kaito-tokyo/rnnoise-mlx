"""Sequence model using native PyTorch convolution and GRU interfaces."""

import torch
from torch import nn

from ..config import ModelConfig

GRUState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class RNNoise(nn.Module):
    """Predict gain/VAD for a sequence, independently of training configuration.

    Input is (batch, frames + 4, input_dim); valid convolutions consume four
    context frames. Each optional GRU state uses PyTorch's (1, batch, hidden)
    layout. The caller owns context padding and recurrent-state truncation.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        c = model_config
        self.conv1 = nn.Conv1d(c.input_dim, c.cond_size, 3)
        self.conv2 = nn.Conv1d(c.cond_size, c.gru_size, 3)
        self.gru1 = nn.GRU(c.gru_size, c.gru_size, batch_first=True)
        self.gru2 = nn.GRU(c.gru_size, c.gru_size, batch_first=True)
        self.gru3 = nn.GRU(c.gru_size, c.gru_size, batch_first=True)
        self.dense_out = nn.Linear(4 * c.gru_size, c.output_dim)
        self.vad = nn.Linear(4 * c.gru_size, 1)

    def forward(self, features: torch.Tensor, state: GRUState | None = None):
        if features.ndim != 3 or features.shape[-1] != self.model_config.input_dim:
            raise ValueError("features must have shape (batch, frames + 4, input_dim)")
        if features.shape[1] < 5:
            raise ValueError("at least five input frames are required")
        h1, h2, h3 = (None, None, None) if state is None else state
        conv1 = self.conv1(features.transpose(1, 2)).tanh()
        conv2 = self.conv2(conv1).tanh().transpose(1, 2)
        y1, h1 = self.gru1(conv2, h1)
        y2, h2 = self.gru2(y1, h2)
        y3, h3 = self.gru3(y2, h3)
        joined = torch.cat((conv2, y1, y2, y3), dim=-1)
        return (
            self.dense_out(joined).sigmoid(),
            self.vad(joined).sigmoid(),
            (h1, h2, h3),
        )
