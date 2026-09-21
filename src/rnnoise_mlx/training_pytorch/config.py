"""PyTorch-specific training configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for one PyTorch optimizer update."""

    batch_size: int = 8
    tbptt_length: int = 250
    gamma: float = 0.25


@dataclass(frozen=True)
class TrainingProgress:
    update: int
    epoch: int
    loss: float
    seconds: float
    processed_frames: int


@dataclass(frozen=True)
class TrainingCheckpoint:
    update: int
    path: Path


TrainingEvent = TrainingProgress | TrainingCheckpoint
