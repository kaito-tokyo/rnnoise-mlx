"""Fixed-shape workspaces used by the compiled MLX chunk path."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


def validate_compiled_chunk(args, segment_length: int | None) -> None:
    """Reject configurations that cannot use the fixed 250-frame graph."""
    if args.graph_mode != "compiled_chunk":
        return
    if segment_length != 250:
        raise ValueError("compiled_chunk requires segmented_tbptt_length=250")
    if args.sequence_length % 250 != 0:
        raise ValueError("compiled_chunk requires sequence_length divisible by 250")


@dataclass
class FixedChunkWorkspace:
    """Reusable temporal buffers for one batch.

    Four leading positions hold causal history and the remaining positions
    hold the current TBPTT chunk.  The buffers are intentionally owned by the update loop so
    they are not shared across batches or optimizer steps.
    """

    features: mx.array

    @classmethod
    def allocate(cls, features) -> "FixedChunkWorkspace":
        return cls(
            features=mx.zeros(
                (features.shape[0], features.shape[1] + 4, features.shape[2]), dtype=features.dtype
            ),
        )
