"""Configuration objects shared by the training implementations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Describe the RNNoise network dimensions shared by all backends."""

    input_dim: int = field(default=65, init=False)
    output_dim: int = field(default=32, init=False)
    cond_size: int = field(default=128, init=False)
    gru_size: int = field(default=384, init=False)

@dataclass(frozen=True)
class TrainConfig:
    """Describe the RNNoise training configuration shared by all backends."""

    batch_size: int = 8
    tbptt_length: int = 250
    gamma: float = 0.25
