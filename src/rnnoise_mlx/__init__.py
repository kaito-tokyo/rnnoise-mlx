"""RNNoise tools.

The audio-preparation tools intentionally do not import MLX.  Training
symbols remain available through a lazy compatibility export so a Mac-side
SpeexDSP cleanup can run in a CPU-only environment.
"""

__all__ = ["ModelConfig", "RNNoise"]


def __getattr__(name: str):
    if name in __all__:
        from .training import ModelConfig, RNNoise

        return {"ModelConfig": ModelConfig, "RNNoise": RNNoise}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
