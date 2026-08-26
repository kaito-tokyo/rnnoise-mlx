"""MLX training and conversion tools for RNNoise-family models."""

from typing import Any

__all__ = ["ModelConfig", "RNNoise"]


def __getattr__(name: str) -> Any:
    """Load Metal-dependent model types only when callers request them."""
    if name in __all__:
        from .training import ModelConfig, RNNoise

        return {"ModelConfig": ModelConfig, "RNNoise": RNNoise}[name]
    raise AttributeError(name)
