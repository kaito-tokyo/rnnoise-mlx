"""CUDA-owned segmented TBPTT update loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_map

from ..training.config import TrainConfig
from .graph import RNNoiseChunk


@dataclass
class CudaUpdateResult:
    """Materialized roots produced by one CUDA training update."""

    loss: mx.array
    state: tuple
    gradients: object
    target_frames: int


class CUDATrainingLoop:
    """Own one fixed-shape, compiled segmented TBPTT update on CUDA."""

    def __init__(self, chunk: RNNoiseChunk, optimizer, *, compile_chunks=True):
        """Initialize a CUDA training loop around an RNNoise chunk."""
        assert chunk is not None
        self.model_config = chunk.model_config
        self.train_config = chunk.train_config
        assert self.train_config is not None
        self.optimizer = optimizer
        self.chunk = chunk

        mx.eval(self.chunk.parameters())
        self.optimizer.init(self.chunk.trainable_parameters())
        mx.eval(self.chunk.state, self.optimizer.state)

        self.compiled_chunk: Callable[..., object] = self._build_chunk_function(
            compile_chunks
        )

    def _build_chunk_function(self, compile_chunks: bool) -> Callable[..., object]:
        """Build the fixed-shape chunk function after state initialization."""

        return (
            mx.compile(
                self._chunk_value_and_grad,
                inputs=[self.chunk.state],
                outputs=[self.chunk.state],
            )
            if compile_chunks
            else self.chunk.value_and_grad
        )

    def _chunk_value_and_grad(self, feature, target_gain, target_vad, state):
        return self.chunk.value_and_grad(feature, target_gain, target_vad, state)

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
