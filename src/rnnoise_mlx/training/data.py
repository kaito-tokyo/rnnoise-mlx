"""MLX minibatches from the shared memory-mapped feature dataset."""

import mlx.core as mx

from ..training_tools.data import FEATURE_DIM, FRAME_DIM, GAIN_DIM
from ..training_tools.data import FeatureDataset as HostFeatureDataset

__all__ = ["FEATURE_DIM", "FRAME_DIM", "GAIN_DIM", "FeatureDataset"]


class FeatureDataset(HostFeatureDataset):
    def batches(self, batch_size, rng, chunk_length=None):
        for host_batch in self.batch_arrays(batch_size, rng, chunk_length):
            batch = mx.array(host_batch)
            yield batch[..., :FEATURE_DIM], batch[..., FEATURE_DIM:-1], batch[..., -1:]
