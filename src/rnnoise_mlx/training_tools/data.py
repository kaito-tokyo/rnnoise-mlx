"""Training feature dataset loading."""

from __future__ import annotations

import numpy as np

FRAME_DIM = 98
FEATURE_DIM = 65
GAIN_DIM = 32


class FeatureDataset:
    """Memory-mapped dataset emitted by upstream RNNoise dump_features."""

    def __init__(self, path: str, sequence_length: int = 2000):
        if sequence_length < 5:
            raise ValueError("sequence_length must be at least 5")
        self.sequence_length = sequence_length
        if path.endswith((".zst", ".gz", ".xz")):
            raise ValueError(
                "compressed feature files are not supported; decompress the file "
                "to an uncompressed .npy or .f32 file first"
            )
        if path.endswith(".npy"):
            # Keep the corpus on disk; copy only the selected minibatch.
            data = np.load(path, mmap_mode="r")
            if data.ndim != 3 or data.shape[1:] != (sequence_length, FRAME_DIM):
                raise ValueError(
                    "NumPy feature file must have shape "
                    f"(sequence_count, {sequence_length}, {FRAME_DIM})"
                )
            self.data = data
            self.sequence_count = data.shape[0]
        else:
            raw = np.memmap(path, dtype="<f4", mode="r")
            frames = raw.size // FRAME_DIM
            self.sequence_count = frames // sequence_length
            usable = self.sequence_count * sequence_length * FRAME_DIM
            self.data = raw[:usable].reshape(
                self.sequence_count, sequence_length, FRAME_DIM
            )
        if self.sequence_count == 0:
            raise ValueError("feature file contains no complete sequence")

    def batch_arrays(
        self, batch_size: int, rng: np.random.Generator, chunk_length: int | None = None
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        order = rng.permutation(self.sequence_count)
        complete = self.sequence_count - self.sequence_count % batch_size
        for start in range(0, complete, batch_size):
            # Advanced indexing copies only the selected minibatch from disk.
            batch = self.data[order[start : start + batch_size]]
            if chunk_length is not None:
                if chunk_length < 5 or chunk_length > self.sequence_length:
                    raise ValueError(
                        "chunk_length must be between 5 and sequence_length"
                    )
                offset = int(rng.integers(0, self.sequence_length - chunk_length + 1))
                batch = batch[:, offset : offset + chunk_length, :]
            yield batch

    def batches(
        self, batch_size: int, rng: np.random.Generator, chunk_length: int | None = None
    ):
        for batch in self.batch_arrays(batch_size, rng, chunk_length):
            yield batch[..., :FEATURE_DIM], batch[..., FEATURE_DIM:-1], batch[..., -1:]
