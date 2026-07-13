"""Tests for official RNNoise weight conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rnnoise_mlx.tools.rnnoise_weights import read_bundle, shapes, write_bundle


def test_canonical_bundle_supports_official_dimensions(tmp_path: Path):
    cond_size, gru_size = 128, 384
    weights = {
        name: (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / max(1, np.prod(shape))).astype(np.float32)
        for name, shape in shapes(cond_size, gru_size).items()
    }
    path = tmp_path / "official.bnns"
    write_bundle(path, weights, cond_size, gru_size)
    restored, config = read_bundle(path)
    assert config == {"input_dim": 65, "cond_size": 128, "gru_size": 384, "output_dim": 32}
    for name in weights:
        np.testing.assert_array_equal(restored[name], weights[name])
