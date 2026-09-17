"""Configuration and callback event types shared by training backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class TrainConfig:
    """Configuration for one training run, independent of CLI parsing."""

    features: str
    output: str
    batch_size: int = 8
    sequence_length: int = 2000
    epochs: int = 200
    max_updates: int | None = None
    learning_rate: float = 1e-3
    lr_decay: float = 5e-5
    gamma: float = 0.25
    seed: int = 0
    eval_features: str | None = None
    training_chunk_length: int = 200
    no_compile: bool = True
    graph_mode: Literal["dynamic", "compiled_chunk"] = "dynamic"
    sync_eval: bool = True
    stateful_tbptt: bool = False
    two_segment_tbptt: str | None = None
    segmented_tbptt_length: int | None = None
    segmented_tbptt_state: str = "carry"
    equalize_reset_targets: bool = False
    checkpoint_every: int = 32
    resume_from: Path | None = None
    timing_path: Path | None = None


@dataclass(frozen=True)
class TrainingProgress:
    """Materialized progress for one completed optimizer update."""

    update: int
    epoch: int
    loss: float
    learning_rate: float
    processed_frames: int


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Notification emitted after a checkpoint is durably written."""

    update: int
    path: Path
    next_epoch: int
    next_batch: int
    processed_frames: int
    elapsed_seconds: float


TrainingEvent = TrainingProgress | TrainingCheckpoint
