"""CUDA-owned segmented TBPTT update loop."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .executor import CompiledChunkRunner
from .workspace import FixedChunkWorkspace


@dataclass
class CudaUpdateResult:
    """Materialized roots produced by one CUDA training update."""

    loss: mx.array
    state: tuple
    gradients: object
    workspace: FixedChunkWorkspace
    target_frames: int


class CUDATrainingLoop:
    """Own one fixed-shape segmented TBPTT update on CUDA.

    The caller supplies already-loaded host batches and receives only the
    update result.  Chunk slicing, workspace ownership, gradient accumulation,
    optimizer update, and evaluation boundaries remain in this backend.
    """

    def __init__(self, runner: CompiledChunkRunner, model):
        self.runner = runner
        self.model = model

    def run_update(self, features, gain, vad, *, segment_length: int, state=None):
        if segment_length != 250:
            raise ValueError("CUDATrainingLoop requires 250-frame chunks")
        if features.shape[1] % segment_length != 0:
            raise ValueError("sequence length must be divisible by 250")

        workspace = FixedChunkWorkspace.allocate(features)
        accumulated_gradients = None
        accumulated_loss = None
        target_frames = 0

        for start in range(0, features.shape[1], segment_length):
            end = start + segment_length
            chunk_features = features[:, start:end, :]
            if state is None:
                chunk_gain = gain[:, start:end, :]
                chunk_vad = vad[:, start:end, :]
                loss, h1, h2, h3, gradients = (
                    self.runner.first_grad(
                        chunk_features,
                        chunk_gain,
                        chunk_vad,
                        workspace.features,
                    )
                )
                state = (h1, h2, h3)
            else:
                chunk_gain = gain[:, start:end, :]
                chunk_vad = vad[:, start:end, :]
                loss, h1, h2, h3, gradients = (
                    self.runner.next_grad(
                        chunk_features,
                        chunk_gain,
                        chunk_vad,
                        state,
                        workspace.features,
                    )
                )
                state = (h1, h2, h3)

            chunk_frames = chunk_gain.shape[1]
            mx.eval(state, gradients)
            weighted = self.runner.scale_gradients(gradients, chunk_frames)
            accumulated_gradients = (
                weighted
                if accumulated_gradients is None
                else self.runner.accumulate_gradients(
                    accumulated_gradients, weighted
                )
            )
            accumulated_loss = (
                loss * chunk_frames
                if accumulated_loss is None
                else accumulated_loss + loss * chunk_frames
            )
            target_frames += chunk_frames
            state = tuple(mx.stop_gradient(value) for value in state)

        gradients = self.runner.average_gradients(
            accumulated_gradients, target_frames
        )
        self.runner.apply_optimizer(gradients)
        loss = accumulated_loss / target_frames
        mx.eval(self.model.state, self.runner.optimizer.state, loss)
        return CudaUpdateResult(
            loss=loss,
            state=state,
            gradients=gradients,
            workspace=workspace,
            target_frames=target_frames,
        )
