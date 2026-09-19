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
    gradients: object
    target_frames: int


class CUDATrainingLoop:
    """Own one fixed-shape, compiled segmented TBPTT update on CUDA."""

    def __init__(self, model_config, train_config, optimizer):
        """Initialize a CUDA training loop around an RNNoise module."""
        assert train_config is not None
        self.model_config = model_config
        self.train_config = train_config
        self.model = RNNoise(model_config, train_config)
        self.optimizer = optimizer
        self.chunk = self.model.chunk
        assert self.chunk is not None
        self.compiled_chunk = None

    @classmethod
    def create(
        cls,
        config: ModelConfig,
        *,
        batch_size: int = 8,
        tbptt_length: int = 250,
        learning_rate: float = 1e-3,
    ):
        """Create a CUDA training path with one compiled chunk function."""
        train_config = TrainConfig(
            batch_size=batch_size,
            tbptt_length=tbptt_length,
        )
        optimizer = optim.Adam(learning_rate=learning_rate)
        loop = cls(config, train_config, optimizer)
        mx.eval(loop.chunk.parameters())
        optimizer.init(loop.chunk.trainable_parameters())
        mx.eval(loop.chunk.state, optimizer.state)
        loop.compiled_chunk = mx.compile(loop.chunk.value_and_grad)
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
        """Run one optimizer update from fixed-shape compiled chunks."""
        assert self.compiled_chunk is not None
        assert segment_length == self.train_config.tbptt_length
        assert features.shape[0] == self.train_config.batch_size
        assert target_gain.shape[1] == features.shape[1]
        assert target_vad.shape[1] == features.shape[1]
        assert features.shape[1] % segment_length == 0

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
        target_frames = features.shape[1]

        for start in range(0, target_frames, segment_length):
            end = start + segment_length
            feature = padded_features[:, start : end + 4, :]
            chunk_target_gain = target_gain[:, start:end, :]
            chunk_target_vad = target_vad[:, start:end, :]
            (loss, state), gradients = self.compiled_chunk(
                self.chunk.trainable_parameters(),
                feature,
                chunk_target_gain,
                chunk_target_vad,
                state,
            )
            accumulated_gradients = tree_map(
                lambda total, value: total + value * segment_length,
                accumulated_gradients,
                gradients,
            )
            accumulated_loss = accumulated_loss + loss * segment_length
            state = tuple(mx.stop_gradient(value) for value in state)

        gradients = tree_map(
            lambda value: value / target_frames,
            accumulated_gradients,
        )
        self.optimizer.update(self.chunk, gradients)
        loss = accumulated_loss / target_frames
        mx.eval(loss)
        return CudaUpdateResult(
            loss=loss,
            state=state,
            gradients=gradients,
            target_frames=target_frames,
        )
