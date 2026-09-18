"""CUDA-owned segmented TBPTT update loop."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim

from ..training.config import ModelConfig, TrainConfig
from .graph import RNNoise, RNNoiseChunk
from mlx.utils import tree_map


@dataclass
class CudaUpdateResult:
    """Materialized roots produced by one CUDA training update."""

    loss: mx.array
    state: tuple
    gradients: object
    target_frames: int


class CUDATrainingLoop:
    """Own one fixed-shape segmented TBPTT update on CUDA.

    The caller supplies already-loaded host batches and receives only the
    update result.  Chunk slicing, gradient accumulation, optimizer update,
    and evaluation boundaries remain in this backend.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig, optimizer):
        """Initialize a CUDA training loop around an RNNoise module."""
        assert train_config is not None
        self.model_config = model_config
        self.train_config = train_config
        self.model = RNNoise(model_config, train_config)
        self.optimizer = optimizer
        self.chunk = self.model.chunk
        assert self.chunk is not None

    @classmethod
    def create(
        cls,
        config: ModelConfig,
        *,
        batch_size: int = 8,
        tbptt_length: int = 250,
        learning_rate: float = 1e-3,
    ) -> "CUDATrainingLoop":
        """Create a CUDA training path backed by the CUDA RNNoise module."""
        train_config = TrainConfig(
            batch_size=batch_size,
            tbptt_length=tbptt_length,
        )
        optimizer = optim.Adam(learning_rate=learning_rate)
        loop = cls(config, train_config, optimizer)
        mx.eval(loop.model.parameters(), optimizer.state)
        return loop

    def run_update(
        self,
        features,
        target_gain,
        target_vad,
        *,
        segment_length: int,
        state,
    ):
        """Run one optimizer update using feature and target sequences.

        ``target_gain`` and ``target_vad`` are supervised labels aligned with
        ``features``; they are not predictions produced by the model.  The
        caller owns the initial GRU state and must provide it explicitly.
        """
        assert segment_length == self.train_config.tbptt_length
        assert features.shape[1] % segment_length == 0

        padding = mx.zeros(
            (features.shape[0], 4, features.shape[2]),
            dtype=features.dtype,
        )
        padded_features = mx.concatenate((padding, features), axis=1)
        accumulated_gradients = tree_map(
            mx.zeros_like,
            self.model.trainable_parameters(),
        )
        accumulated_loss = mx.zeros((), dtype=features.dtype)
        target_frames = 0

        for start in range(0, features.shape[1], segment_length):
            end = start + segment_length
            chunk_features = features[:, start:end, :]
            chunk_target_gain = target_gain[:, start:end, :]
            chunk_target_vad = target_vad[:, start:end, :]
            feature = padded_features[:, start : end + 4, :]
            accumulated_loss, accumulated_gradients, state = self.chunk(
                feature,
                (chunk_target_gain, chunk_target_vad),
                state,
                (
                    accumulated_gradients,
                    accumulated_loss,
                    chunk_target_gain.shape[1],
                ),
            )
            chunk_frames = chunk_target_gain.shape[1]
            target_frames += chunk_frames
            state = tuple(mx.stop_gradient(value) for value in state)
            # First milestone: materialize the compiled chunk outputs at the
            # end of each loop body.  This boundary can later be moved to the
            # update level once the fixed graph is validated.
            mx.eval(state, accumulated_gradients, accumulated_loss, feature)

        gradients = tree_map(
            lambda value: value / target_frames,
            accumulated_gradients,
        )
        self.optimizer.update(self.model, gradients)
        loss = accumulated_loss / target_frames
        mx.eval(self.model.state, self.optimizer.state, loss)
        return CudaUpdateResult(
            loss=loss,
            state=state,
            gradients=gradients,
            target_frames=target_frames,
        )
