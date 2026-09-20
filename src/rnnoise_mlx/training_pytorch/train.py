"""Train RNNoise with PyTorch and export canonical weights for MLX/C."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..training_tools.data import FeatureDataset
from ..training_tools.model_config import ModelConfig, TrainConfig
from .model import GRUState, RNNoise
from .weights import load_weights, save_weights


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


def initial_state(model: RNNoise, train_config: TrainConfig) -> GRUState:
    parameter = next(model.parameters())
    return tuple(
        parameter.new_zeros(1, train_config.batch_size, model.model_config.gru_size)
        for _ in range(3)
    )


def train_update(model, optimizer, train_config, features, target_gain, target_vad, state=None):
    batch = train_config.batch_size
    segment_length = train_config.tbptt_length
    if batch <= 0 or segment_length <= 0 or train_config.gamma <= 0:
        raise ValueError("batch_size, tbptt_length, and gamma must be positive")
    if features.ndim != 3 or features.shape[0] != batch or features.shape[2] != model.model_config.input_dim:
        raise ValueError("features must have shape (batch_size, frames, input_dim)")
    frames = features.shape[1]
    if frames == 0 or frames % segment_length:
        raise ValueError("frames must be a positive multiple of the configured TBPTT length")
    if target_gain.shape != (batch, frames, model.model_config.output_dim) or target_vad.shape != (batch, frames, 1):
        raise ValueError("target shapes must match the feature batch and frame count")
    state = initial_state(model, train_config) if state is None else tuple(s.detach() for s in state)
    padded = torch.cat((features.new_zeros(batch, 4, features.shape[2]), features), dim=1)
    model.train(); optimizer.zero_grad(set_to_none=True); total_loss = features.new_zeros(())
    for start in range(0, frames, segment_length):
        end = start + segment_length
        gain, vad, state = model(padded[:, start:end + 4, :], state)
        loss = rnnoise_loss(gain, vad, target_gain[:, start:end], target_vad[:, start:end], gamma=train_config.gamma)
        weighted = loss * (segment_length / frames)
        weighted.backward(); total_loss = total_loss + weighted.detach()
        state = tuple(s.detach() for s in state)
    optimizer.step()
    if features.device.type == "cuda":
        torch.cuda.synchronize(features.device)
    return total_loss, state


def train_update_full(model, optimizer, train_config, features, target_gain, target_vad, state=None):
    """Run one update with full-sequence backpropagation and no TBPTT splits."""
    batch = train_config.batch_size
    if batch <= 0 or train_config.gamma <= 0:
        raise ValueError("batch_size and gamma must be positive")
    if features.ndim != 3 or features.shape[0] != batch or features.shape[2] != model.model_config.input_dim:
        raise ValueError("features must have shape (batch_size, frames, input_dim)")
    frames = features.shape[1]
    if frames == 0:
        raise ValueError("frames must be positive")
    if target_gain.shape != (batch, frames, model.model_config.output_dim) or target_vad.shape != (batch, frames, 1):
        raise ValueError("target shapes must match the feature batch and frame count")
    state = initial_state(model, train_config) if state is None else tuple(s.detach() for s in state)
    padded = torch.cat((features.new_zeros(batch, 4, features.shape[2]), features), dim=1)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    gain, vad, state = model(padded, state)
    loss = rnnoise_loss(gain, vad, target_gain, target_vad, gamma=train_config.gamma)
    loss.backward()
    optimizer.step()
    if features.device.type == "cuda":
        torch.cuda.synchronize(features.device)
    return loss.detach(), tuple(s.detach() for s in state)


class RNNoiseTrainer:
    """Small convenience wrapper around the functional training API."""
    def __init__(self, model, optimizer, train_config):
        self.model, self.optimizer, self.train_config = model, optimizer, train_config
    def initial_state(self):
        return initial_state(self.model, self.train_config)
    def run_update(self, features, target_gain, target_vad, *, state=None):
        return train_update(self.model, self.optimizer, self.train_config, features, target_gain, target_vad, state)



def train(args) -> dict:
    segment = args.segmented_tbptt_length
    if args.batch_size <= 0 or segment <= 0 or args.sequence_length % segment:
        raise ValueError(
            "positive batch/segment sizes and sequence divisibility are required"
        )
    if args.epochs <= 0 or (args.max_updates is not None and args.max_updates <= 0):
        raise ValueError("epochs and max_updates must be positive")
    if args.checkpoint_every <= 0 or args.learning_rate <= 0 or args.gamma <= 0:
        raise ValueError(
            "checkpoint interval, learning rate, and gamma must be positive"
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    dataset = FeatureDataset(str(args.features), args.sequence_length)
    batches_per_epoch = dataset.sequence_count // args.batch_size
    if batches_per_epoch == 0:
        raise ValueError("feature file has fewer sequences than batch_size")
    torch.manual_seed(args.seed)
    config = ModelConfig()
    train_config = TrainConfig(args.batch_size, segment, args.gamma)
    model = RNNoise(config).to(device)
    if args.init_weights:
        load_weights(model, args.init_weights)
    if getattr(args, "compile", False):
        model = torch.compile(model, mode=getattr(args, "compile_mode", "reduce-overhead"))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loop = RNNoiseTrainer(model, optimizer, train_config)
    signature = {
        "model": asdict(config),
        "training": asdict(train_config),
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "features": str(args.features.resolve()),
        "feature_identity": args.feature_identity,
        "feature_bytes": args.features.stat().st_size,
        "feature_mtime_ns": args.features.stat().st_mtime_ns,
        "carry_between_updates": args.carry_between_updates,
        "learning_rate": args.learning_rate,
    }
    update = 0
    state = None
    history = []
    if args.resume_from:
        saved = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        if saved.get("format_version") == 1:
            # Migrate checkpoints produced by the earlier chunk-wrapper layout.
            saved["signature"]["training"] = saved["signature"].pop("chunk")
            saved["model"] = {
                name.removeprefix("frame_step."): value
                for name, value in saved["model"].items()
            }
            if saved["carry"] is not None:
                saved["carry"] = tuple(s.unsqueeze(0) for s in saved["carry"])
            saved["format_version"] = 2
        if saved.get("format_version") != 2 or saved["signature"] != signature:
            raise ValueError("checkpoint configuration or feature identity differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        update, history = saved["update"], saved["history"]
        state = (
            tuple(s.to(device) for s in saved["carry"])
            if saved["carry"] is not None
            else None
        )
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    target_updates = min(
        args.epochs * batches_per_epoch,
        args.max_updates or args.epochs * batches_per_epoch,
    )
    if update >= target_updates:
        raise ValueError("checkpoint already reached the requested update/epoch limit")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "config.json").write_text(
        json.dumps(
            {
                **settings,
                "model": asdict(config),
                "device": str(device),
                "torch_version": str(torch.__version__),
            },
            indent=2,
        )
    )

    def checkpoint():
        path = args.output / f"update-{update:08d}"
        path.mkdir()
        save_weights(model, path / "model.safetensors")
        torch.save(
            {
                "format_version": 2,
                "signature": signature,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "history": history,
                "carry": tuple(s.cpu() for s in state) if state is not None else None,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None,
            },
            path / "checkpoint.pt.tmp",
        )
        (path / "checkpoint.pt.tmp").replace(path / "checkpoint.pt")

    print(
        json.dumps(
            {
                "device": str(device),
                "model_device": str(next(model.parameters()).device),
                "torch_version": str(torch.__version__),
                "resumed_update": update,
            }
        ),
        flush=True,
    )
    for epoch in range(update // batches_per_epoch, args.epochs):
        skip = update % batches_per_epoch if epoch == update // batches_per_epoch else 0
        # Per-epoch permutation makes the data cursor reproducible on resume.
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        for batch_index, host_batch in enumerate(dataset.batches(args.batch_size, rng)):
            if batch_index < skip:
                continue
            started = time.perf_counter()
            batch = tuple(
                torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
                for x in host_batch
            )
            loss_tensor, next_state = loop.run_update(*batch, state=state)
            loss = loss_tensor.item()
            if not np.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            state = next_state if args.carry_between_updates else None
            update += 1
            record = {
                "update": update,
                "epoch": epoch + 1,
                "seconds": time.perf_counter() - started,
                "loss": loss,
            }
            if device.type == "cuda":
                record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            history.append(record)
            print(json.dumps(record), flush=True)
            if update % args.checkpoint_every == 0 or update == target_updates:
                checkpoint()
            if update == target_updates:
                break
        if update == target_updates:
            break
    save_weights(model, args.output / "model.safetensors")
    summary = {
        "updates": update,
        "device": str(device),
        "feature_identity": args.feature_identity,
        "timings": history,
        "losses": [r["loss"] for r in history],
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

