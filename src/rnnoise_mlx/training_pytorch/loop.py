"""Segmented TBPTT with one optimizer step per complete sequence batch."""

import torch

from ..training_tools.model_config import TrainConfig
from .model import GRUState, RNNoise


def rnnoise_loss(predicted_gain, predicted_vad, target_gain, target_vad, *, gamma):
    target = target_gain.clamp_min(0)
    target = target * (8 * target).tanh().square()
    active = (target_gain + 1).clamp_max(1)
    error = predicted_gain.pow(gamma) - target.pow(gamma)
    gain_loss = ((1 + 5 * target_vad) * active * error.square()).mean()
    vad_weight = (2 * target_vad - 1).abs()
    vad_positive_loss = -target_vad * (0.01 + predicted_vad).log()
    vad_negative_loss = -(1 - target_vad) * (1.01 - predicted_vad).log()
    vad_loss = (vad_weight * (vad_positive_loss + vad_negative_loss)).mean()
    return gain_loss + 0.001 * vad_loss


class PyTorchTrainingLoop:
    def __init__(
        self,
        model: RNNoise,
        optimizer: torch.optim.Optimizer,
        train_config: TrainConfig,
    ):
        if train_config.batch_size <= 0 or train_config.tbptt_length <= 0:
            raise ValueError("batch_size and tbptt_length must be positive")
        if train_config.gamma <= 0:
            raise ValueError("gamma must be positive")
        self.model = model
        self.optimizer = optimizer
        self.model_config = model.model_config
        self.train_config = train_config

    def initial_state(self):
        parameter = next(self.model.parameters())
        return tuple(
            parameter.new_zeros(
                1, self.train_config.batch_size, self.model_config.gru_size
            )
            for _ in range(3)
        )

    def run_update(
        self, features, target_gain, target_vad, *, state: GRUState | None = None
    ):
        batch = self.train_config.batch_size
        segment_length = self.train_config.tbptt_length
        if (
            features.ndim != 3
            or features.shape[0] != batch
            or features.shape[2] != self.model_config.input_dim
        ):
            raise ValueError("features must have shape (batch_size, frames, input_dim)")
        frames = features.shape[1]
        if frames == 0 or frames % segment_length:
            raise ValueError(
                "frames must be a positive multiple of the configured TBPTT length"
            )
        if target_gain.shape != (
            batch,
            frames,
            self.model_config.output_dim,
        ) or target_vad.shape != (batch, frames, 1):
            raise ValueError(
                "target shapes must match the feature batch and frame count"
            )
        state = (
            self.initial_state() if state is None else tuple(s.detach() for s in state)
        )
        padded_features = torch.cat(
            (features.new_zeros(batch, 4, features.shape[2]), features), dim=1
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = features.new_zeros(())
        for start in range(0, frames, segment_length):
            end = start + segment_length
            gain, vad, state = self.model(padded_features[:, start : end + 4, :], state)
            loss = rnnoise_loss(
                gain,
                vad,
                target_gain[:, start:end, :],
                target_vad[:, start:end, :],
                gamma=self.train_config.gamma,
            )
            # Backward frees each chunk graph; GRU carry is truncated at its boundary.
            weighted_loss = loss * (segment_length / frames)
            weighted_loss.backward()
            total_loss += weighted_loss.detach()
            state = tuple(s.detach() for s in state)
        self.optimizer.step()
        if features.device.type == "cuda":
            torch.cuda.synchronize(features.device)
        return total_loss, state
