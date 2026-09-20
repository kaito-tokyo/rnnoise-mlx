"""Use the established canonical conversion for PyTorch training models."""

from dataclasses import asdict
from pathlib import Path

from ..tools.convert_official_weights import (
    canonical_from_state_dict,
    state_dict_from_canonical,
)
from ..tools.rnnoise_weights import read_weights, write_weights
from .model import RNNoise


def save_weights(model: RNNoise, path: Path) -> None:
    state = {
        name.replace("vad.", "vad_dense."): tensor
        for name, tensor in model.state_dict().items()
    }
    c = model.model_config
    write_weights(
        path, canonical_from_state_dict(state, c.gru_size), c.cond_size, c.gru_size
    )


def load_weights(model: RNNoise, path: Path) -> None:
    weights, config = read_weights(path)
    if config != asdict(model.model_config):
        raise ValueError("weight dimensions do not match the model configuration")
    state = state_dict_from_canonical(weights, config["gru_size"])
    model.load_state_dict(
        {name.replace("vad_dense.", "vad."): value for name, value in state.items()}
    )
