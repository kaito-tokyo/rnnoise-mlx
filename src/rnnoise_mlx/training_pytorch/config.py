"""PyTorch-specific training configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for one PyTorch optimizer update."""

    batch_size: int = 8
    tbptt_length: int = 250
    gamma: float = 0.25
