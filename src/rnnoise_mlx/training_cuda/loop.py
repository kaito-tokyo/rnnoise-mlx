"""CUDA-owned segmented TBPTT update loop."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map

from ..training.config import ModelConfig, TrainConfig
from .graph import RNNoise


@dataclass
class CudaUpdateResult:
    """Materialized roots produced by one CUDA training update."""

    loss: mx.array
    state: tuple
    target_frames: int


class CUDATrainingLoop:
    """Own one fixed-shape, compiled segmented TBPTT update on CUDA.

    The compiled function contains all chunks, gradient accumulation, and
    the optimizer update. The Python caller only invokes it and materializes
    its roots with ``mx.eval``.
    """

    def __init__(self, model_config, train_config, optimizer, sequence_length):
        """Initialize a CUDA training loop around an RNNoise module."""
        assert train_config is not None
        assert sequence_length > 0
        assert sequence_length % train_config.tbptt_length == 0
        self.model_config = model_config
        self.train_config = train_config
        self.sequence_length = sequence_length
        self.chunk_count = sequence_length // train_config.tbptt_length
        self.model = RNNoise(model_config, train_config)
        self.optimizer = optimizer
        self.chunk = self.model.chunk
        assert self.chunk is not None
        self.compiled_state = None
        self.compiled_model_update = None

    @classmethod
    def create(
        cls,
        config: ModelConfig,
        *,
        batch_size: int = 8,
        tbptt_length: int = 250,
        learning_rate: float = 1e-3,
        sequence_length: int = 2000,
    ):
        """Create and compile a CUDA training path."""
        train_config = TrainConfig(
            batch_size=batch_size,
            tbptt_length=tbptt_length,
        )
        optimizer = optim.Adam(learning_rate=learning_rate)
        loop = cls(config, train_config, optimizer, sequence_length)
        mx.eval(loop.chunk.parameters())
        optimizer.init(loop.chunk.trainable_parameters())
        mx.eval(loop.chunk.state, optimizer.state)
        loop._build_compiled_model_update()
        return loop

    def _build_compiled_model_update(self):
        """Compile one complete fixed-shape optimizer update."""
        self.compiled_state = [self.chunk.state, self.optimizer.state]
        self.compiled_model_update = mx.compile(
            self._compiled_model_update,
            inputs=self.compiled_state,
            outputs=self.compiled_state,
        )

    def _compiled_model_update(self, features, target_gain, target_vad, state):
        """Build the complete update graph for one fixed-shape batch."""
        padding = mx.zeros(
            (features.shape[0], 4, features.shape[2]),
            dtype=features.dtype,
        )
        padded_features = mx.concatenate((padding, features), axis=1)
        accumulated_gradients = tree_map(
            mx.zeros_like,
            self.chunk.trainable_parameters(),
        )
        accumulated_loss = mx.zeros((), dtype=features.dtype)

        for chunk_index in range(self.chunk_count):
            start = chunk_index * self.train_config.tbptt_length
            end = start + self.train_config.tbptt_length
            chunk_target_gain = target_gain[:, start:end, :]
            chunk_target_vad = target_vad[:, start:end, :]
            feature = padded_features[:, start : end + 4, :]
            loss, state, accumulated_gradients = self.chunk(
                feature,
                (chunk_target_gain, chunk_target_vad),
                state,
                accumulated_gradients,
            )
            accumulated_loss = accumulated_loss + (
                loss * self.train_config.tbptt_length
            )
            state = tuple(mx.stop_gradient(value) for value in state)

        gradients = tree_map(
            lambda value: value / self.sequence_length,
            accumulated_gradients,
        )
        self.optimizer.update(self.chunk, gradients)
        return accumulated_loss / self.sequence_length, state

    def run_update(
        self,
        features,
        target_gain,
        target_vad,
        *,
        segment_length: int,
        state,
    ):
        """Run one compiled optimizer update and materialize its roots."""
        assert self.compiled_model_update is not None
        assert segment_length == self.train_config.tbptt_length
        assert features.shape[0] == self.train_config.batch_size
        assert features.shape[1] == self.sequence_length
        assert target_gain.shape[1] == self.sequence_length
        assert target_vad.shape[1] == self.sequence_length
        loss, state = self.compiled_model_update(
            features, target_gain, target_vad, state
        )
        mx.eval(self.compiled_state, loss, state)
        return CudaUpdateResult(
            loss=loss,
            state=state,
            target_frames=self.sequence_length,
        )
