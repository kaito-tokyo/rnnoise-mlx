"""RNNoise-MLX model training command."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
import json
import os
import signal
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import FeatureDataset
from .checkpoint import _feature_identity, load_checkpoint, save_checkpoint
from .evaluate import evaluate
from .loss import rnnoise_loss, rnnoise_loss_aligned
from .model import ModelConfig, RNNoise


_IDENTITY_NOT_PROVIDED = object()


@dataclass
class TrainConfig:
    """Configuration for one training run, independent of CLI parsing."""

    features: str
    output: str
    batch_size: int = 8
    sequence_length: int = 2000
    epochs: int = 200
    max_updates: int | None = None
    learning_rate: float = 1e-3
    lr_decay: float = 5e-5
    gamma: float = 0.25
    seed: int = 0
    eval_features: str | None = None
    training_chunk_length: int = 200
    no_compile: bool = False
    # Retained for checkpoint/config compatibility; training is always synchronous.
    sync_eval: bool = True
    stateful_tbptt: bool = False
    two_segment_tbptt: str | None = None
    segmented_tbptt_length: int | None = None
    segmented_tbptt_state: str = "carry"
    equalize_reset_targets: bool = False
    checkpoint_every: int = 32
    resume_from: Path | None = None


@dataclass(frozen=True)
class TrainingProgress:
    """Materialized progress for one completed optimizer update."""

    update: int
    epoch: int
    loss: float
    learning_rate: float
    processed_frames: int


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Notification emitted after a checkpoint is durably written."""

    update: int
    path: Path
    next_epoch: int
    next_batch: int
    processed_frames: int
    elapsed_seconds: float


TrainingEvent = TrainingProgress | TrainingCheckpoint


def parse_args(argv=None) -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("features")
    parser.add_argument("output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-features")
    parser.add_argument("--training-chunk-length", type=int, default=200)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--stateful-tbptt",
        action="store_true",
        help="legacy experiment: update after every stateful chunk",
    )
    parser.add_argument(
        "--two-segment-tbptt",
        choices=("carry", "reset"),
        help=(
            "compatibility interface for the historical 1000/1000 experiment; "
            "carry preserves Conv/GRU state and reset starts the second segment independently"
        ),
    )
    parser.add_argument(
        "--segmented-tbptt-length",
        type=int,
        help="fixed segment length; accumulate the full sequence before one update",
    )
    parser.add_argument(
        "--segmented-tbptt-state", choices=("carry", "reset"), default="carry"
    )
    parser.add_argument("--equalize-reset-targets", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="complete checkpoint directory to resume from",
    )
    args = parser.parse_args(argv)

    if args.two_segment_tbptt and args.segmented_tbptt_length:
        parser.error("select only one segmented TBPTT interface")
    if (args.two_segment_tbptt or args.segmented_tbptt_length) and args.stateful_tbptt:
        parser.error("segmented TBPTT and --stateful-tbptt are mutually exclusive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    return TrainConfig(**vars(args))


def preflight_feature_identities(
    args: TrainConfig,
) -> tuple[str | None, str | None]:
    """Compute input identities before training starts."""
    return (
        _feature_identity(vars(args)),
        _feature_identity(vars(args), "eval_features"),
    )


def train(
    args: TrainConfig,
    progress_callback: Callable[[TrainingEvent], None] | None = None,
    *,
    feature_identity: str | None | object = _IDENTITY_NOT_PROVIDED,
    evaluation_feature_identity: str | None | object = _IDENTITY_NOT_PROVIDED,
):
    """Run training and optionally report progress and checkpoint events.

    ``progress_callback`` receives a :class:`TrainingProgress` once per update
    after its loss is materialized by MLX and a :class:`TrainingCheckpoint`
    after each checkpoint is written. Exceptions raised by the callback are
    propagated so callers can stop a run when progress handling fails.
    """
    output = Path(args.output)
    if feature_identity is _IDENTITY_NOT_PROVIDED:
        raise ValueError("feature_identity must be computed by preflight_feature_identities")
    if evaluation_feature_identity is _IDENTITY_NOT_PROVIDED:
        raise ValueError(
            "evaluation_feature_identity must be computed by preflight_feature_identities"
        )
    verified_feature_identity = feature_identity
    verified_evaluation_feature_identity = evaluation_feature_identity
    dataset = FeatureDataset(args.features, args.sequence_length)
    if dataset.sequence_count < args.batch_size:
        raise ValueError("feature dataset must contain at least one complete batch")
    output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)
    config = ModelConfig()
    (output / "model-config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    )
    model = RNNoise(config)
    learning_rate = lambda step: args.learning_rate / (1 + args.lr_decay * step)
    optimizer = optim.AdamW(learning_rate=learning_rate, betas=(0.8, 0.98), eps=1e-8)
    history = []
    update = 0
    processed_frames = 0
    resume_epoch = 1
    resume_batch = 0
    elapsed_before_resume = 0.0
    initial_evaluation = None
    if args.resume_from:
        restored = load_checkpoint(
            args.resume_from.resolve(),
            model,
            optimizer,
            config,
            vars(args),
            feature_identity=verified_feature_identity,
            evaluation_feature_identity=verified_evaluation_feature_identity,
        )
        history = restored["history"]
        update = int(restored["update"])
        processed_frames = int(restored["processed_frames"])
        resume_epoch = int(restored["next_epoch"])
        resume_batch = int(restored["next_batch"])
        elapsed_before_resume = float(restored["elapsed_seconds"])
        if "initial_evaluation" in restored:
            initial_evaluation = restored["initial_evaluation"]
        print(
            json.dumps({"resumed_from": str(args.resume_from), "update": update}),
            flush=True,
        )
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    eval_dataset = FeatureDataset(args.eval_features, args.sequence_length) if args.eval_features else None
    if args.resume_from is None:
        initial_evaluation = (
            evaluate(model, eval_dataset, args.batch_size, args.gamma)
            if eval_dataset
            else None
        )
    if stop_requested:
        if args.resume_from is None:
            checkpoint = save_checkpoint(
                output / "checkpoints",
                model,
                optimizer,
                config,
                update=update,
                next_epoch=1,
                next_batch=0,
                processed_frames=processed_frames,
                elapsed_seconds=elapsed_before_resume,
                history=history,
                training_config=vars(args),
                initial_evaluation=initial_evaluation,
                feature_identity=verified_feature_identity,
                evaluation_feature_identity=verified_evaluation_feature_identity,
            )
        summary = {
            "updates": update,
            "stop_requested": True,
            "training_seconds": elapsed_before_resume,
            "processed_frames": processed_frames,
            "history": history,
        }
        (output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
        return

    def objective(model, features, gain, vad):
        pred_gain, pred_vad, _ = model(features)
        return rnnoise_loss(pred_gain, pred_vad, gain, vad, args.gamma)[0]

    value_and_grad = nn.value_and_grad(model, objective)

    def train_step(features, gain, vad):
        loss, gradients = value_and_grad(model, features, gain, vad)
        optimizer.update(model, gradients)
        return loss

    segment_length = (
        args.sequence_length // 2 if args.two_segment_tbptt else args.segmented_tbptt_length
    )
    segment_state = args.two_segment_tbptt or args.segmented_tbptt_state
    equalize_reset_targets = args.equalize_reset_targets or bool(args.two_segment_tbptt)
    if segment_length and (
        segment_length <= 4 or args.sequence_length % segment_length != 0
    ):
        raise ValueError("segmented TBPTT length must be >4 and divide --sequence-length")

    if not args.no_compile:
        captured_state = [model.state, optimizer.state]
        train_step = partial(mx.compile, inputs=captured_state, outputs=captured_state)(train_step)

    def stateful_objective(params, features, gain, vad, state, first):
        model.update(params)
        if first:
            predicted_gain, predicted_vad, new_state = model.first_chunk(features)
        else:
            predicted_gain, predicted_vad, new_state = model.next_chunk(features, state)
        loss = rnnoise_loss_aligned(predicted_gain, predicted_vad, gain, vad, args.gamma)[0]
        return loss, new_state

    stateful_value_and_grad = mx.value_and_grad(stateful_objective)

    def add_gradient_trees(left, right):
        if isinstance(left, dict):
            return {key: add_gradient_trees(left[key], right[key]) for key in left}
        if isinstance(left, list):
            return [add_gradient_trees(a, b) for a, b in zip(left, right)]
        if isinstance(left, tuple):
            return tuple(add_gradient_trees(a, b) for a, b in zip(left, right))
        return left + right

    def divide_gradient_tree(tree, divisor):
        if isinstance(tree, dict):
            return {key: divide_gradient_tree(value, divisor) for key, value in tree.items()}
        if isinstance(tree, list):
            return [divide_gradient_tree(value, divisor) for value in tree]
        if isinstance(tree, tuple):
            return tuple(divide_gradient_tree(value, divisor) for value in tree)
        return tree / divisor

    def multiply_gradient_tree(tree, multiplier):
        if isinstance(tree, dict):
            return {
                key: multiply_gradient_tree(value, multiplier)
                for key, value in tree.items()
            }
        if isinstance(tree, list):
            return [multiply_gradient_tree(value, multiplier) for value in tree]
        if isinstance(tree, tuple):
            return tuple(multiply_gradient_tree(value, multiplier) for value in tree)
        return tree * multiplier

    def first_chunk_grad(features, gain, vad):
        (loss, state), gradients = stateful_value_and_grad(
            model.trainable_parameters(), features, gain, vad, (), True
        )
        return loss, state, gradients

    def next_chunk_grad(features, gain, vad, state):
        (loss, new_state), gradients = stateful_value_and_grad(
            model.trainable_parameters(), features, gain, vad, state, False
        )
        return loss, new_state, gradients

    def first_chunk_step(features, gain, vad):
        loss, state, gradients = first_chunk_grad(features, gain, vad)
        optimizer.update(model, gradients)
        return loss, state

    def next_chunk_step(features, gain, vad, state):
        loss, new_state, gradients = next_chunk_grad(features, gain, vad, state)
        optimizer.update(model, gradients)
        return loss, new_state

    if not args.no_compile:
        first_chunk_grad = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(first_chunk_grad)
        next_chunk_grad = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(next_chunk_grad)
        first_chunk_step = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(first_chunk_step)
        next_chunk_step = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(next_chunk_step)
    pending_losses = []
    started = time.monotonic()

    def batches_for_epoch(epoch: int, skip_batches: int):
        # A per-epoch stream makes the next batch reproducible even when the
        # prefetch thread has already advanced its private generator.
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        batches = dataset.batches(
            args.batch_size,
            rng,
            chunk_length=None
            if args.stateful_tbptt or segment_length
            else args.training_chunk_length,
        )
        for _ in range(skip_batches):
            try:
                next(batches)
            except StopIteration as error:
                raise ValueError("checkpoint batch cursor is outside the dataset") from error
        yield from batches

    def collect_pending():
        if not pending_losses:
            return
        mx.eval(*(loss for _, _, _, loss in pending_losses))
        for pending_update, pending_epoch, pending_frames, pending_loss in pending_losses:
            record = {
                "update": pending_update,
                "epoch": pending_epoch,
                "loss": float(pending_loss.item()),
            }
            history.append(record)
            if progress_callback is not None:
                progress_callback(
                    TrainingProgress(
                        update=pending_update,
                        epoch=pending_epoch,
                        loss=record["loss"],
                        learning_rate=learning_rate(pending_update),
                        processed_frames=pending_frames,
                    )
                )
            if pending_update % 10 == 0:
                print(json.dumps(record), flush=True)
        pending_losses.clear()

    checkpoint_due = False

    def commit_checkpoint(next_epoch: int, next_batch: int) -> None:
        collect_pending()
        mx.eval(model.state, optimizer.state)
        checkpoint = save_checkpoint(
            output / "checkpoints",
            model,
            optimizer,
            config,
            update=update,
            next_epoch=next_epoch,
            next_batch=next_batch,
            processed_frames=processed_frames,
            elapsed_seconds=elapsed_before_resume + time.monotonic() - started,
            history=history,
            training_config=vars(args),
            initial_evaluation=initial_evaluation,
            feature_identity=verified_feature_identity,
            evaluation_feature_identity=verified_evaluation_feature_identity,
        )
        if progress_callback is not None:
            progress_callback(
                TrainingCheckpoint(
                    update=update,
                    path=checkpoint,
                    next_epoch=next_epoch,
                    next_batch=next_batch,
                    processed_frames=processed_frames,
                    elapsed_seconds=elapsed_before_resume + time.monotonic() - started,
                )
            )

    for epoch in range(resume_epoch, args.epochs + 1):
        stop_checkpoint_committed = False
        last_batch_checkpoint_committed = False
        first_batch = resume_batch if epoch == resume_epoch else 0
        batch_index = first_batch
        for features, gain, vad in batches_for_epoch(epoch, first_batch):
            if segment_length:
                state = None
                chunk_ranges = range(0, args.sequence_length, segment_length)
            elif args.stateful_tbptt:
                state = None
                chunk_ranges = range(0, args.sequence_length, args.training_chunk_length)
            else:
                state = None
                chunk_ranges = (0,)

            if segment_length:
                accumulated_gradients = None
                accumulated_loss = None
                target_frames = 0
                for start in chunk_ranges:
                    end = min(start + args.training_chunk_length, args.sequence_length)
                    end = start + segment_length
                    chunk_features = mx.asarray(features[:, start:end, :])
                    is_first_chunk = start == 0 or segment_state == "reset"
                    if is_first_chunk:
                        target_start = start + 3
                        chunk_gain = mx.asarray(gain[:, target_start : end - 1, :])
                        chunk_vad = mx.asarray(vad[:, target_start : end - 1, :])
                        loss, state, gradients = first_chunk_grad(
                            chunk_features, chunk_gain, chunk_vad
                        )
                    else:
                        chunk_gain = mx.asarray(gain[:, start - 1 : end - 1, :])
                        chunk_vad = mx.asarray(vad[:, start - 1 : end - 1, :])
                        loss, state, gradients = next_chunk_grad(
                            chunk_features, chunk_gain, chunk_vad, state
                        )
                    chunk_frames = chunk_gain.shape[1]
                    mx.eval(loss, state, gradients)
                    weighted_gradients = multiply_gradient_tree(gradients, chunk_frames)
                    accumulated_gradients = (
                        weighted_gradients
                        if accumulated_gradients is None
                        else add_gradient_trees(accumulated_gradients, weighted_gradients)
                    )
                    accumulated_loss = (
                        loss * chunk_frames
                        if accumulated_loss is None
                        else accumulated_loss + loss * chunk_frames
                    )
                    target_frames += chunk_frames
                    state = tuple(mx.stop_gradient(value) for value in state)
                gradients = divide_gradient_tree(accumulated_gradients, target_frames)
                optimizer.update(model, gradients)
                loss = accumulated_loss / target_frames
                update += 1
                processed_frames += args.batch_size * features.shape[1]
                pending_losses.append((update, epoch, processed_frames, loss))
                mx.eval(model.state, optimizer.state, loss)
                if len(pending_losses) >= 10:
                    collect_pending()
                if update % 32 == 0:
                    collect_pending()
                    mx.eval(model.state, optimizer.state)
                if update % args.checkpoint_every == 0:
                    checkpoint_due = True
            else:
                for start in chunk_ranges:
                    if args.stateful_tbptt:
                        end = min(start + args.training_chunk_length, args.sequence_length)
                        chunk_features = mx.asarray(features[:, start:end, :])
                        if start == 0 or segment_state == "reset":
                            target_start = start + 3
                            chunk_gain = mx.asarray(gain[:, target_start : end - 1, :])
                            chunk_vad = mx.asarray(vad[:, target_start : end - 1, :])
                            loss, state = first_chunk_step(chunk_features, chunk_gain, chunk_vad)
                        else:
                            chunk_gain = mx.asarray(gain[:, start - 1 : end - 1, :])
                            chunk_vad = mx.asarray(vad[:, start - 1 : end - 1, :])
                            loss, state = next_chunk_step(
                                chunk_features, chunk_gain, chunk_vad, state
                            )
                        state = tuple(mx.stop_gradient(value) for value in state)
                    else:
                        loss = train_step(mx.asarray(features), mx.asarray(gain), mx.asarray(vad))

                    update += 1
                    processed_frames += args.batch_size * (
                        chunk_features.shape[1] if args.stateful_tbptt else features.shape[1]
                    )
                    pending_losses.append((update, epoch, processed_frames, loss))
                    mx.eval(model.state, optimizer.state, loss)
                    if len(pending_losses) >= 10:
                        collect_pending()
                    if update % 32 == 0:
                        collect_pending()
                        mx.eval(model.state, optimizer.state)
                    if update % args.checkpoint_every == 0:
                        checkpoint_due = True
                    if args.max_updates is not None and update >= args.max_updates:
                        break
            batch_index += 1
            final_batch = (
                epoch == args.epochs
                and batch_index >= dataset.sequence_count // args.batch_size
            )
            checkpoint_committed = False
            if checkpoint_due or stop_requested or final_batch or (
                args.max_updates is not None and update >= args.max_updates
            ):
                next_epoch = epoch
                next_batch = batch_index
                if next_batch >= dataset.sequence_count // args.batch_size:
                    next_epoch += 1
                    next_batch = 0
                commit_checkpoint(next_epoch, next_batch)
                checkpoint_due = False
                checkpoint_committed = True
            last_batch_checkpoint_committed = checkpoint_committed
            if stop_requested:
                if not checkpoint_committed:
                    next_epoch = epoch
                    next_batch = batch_index
                    if next_batch >= dataset.sequence_count // args.batch_size:
                        next_epoch += 1
                        next_batch = 0
                    commit_checkpoint(next_epoch, next_batch)
                stop_checkpoint_committed = True
                break
            if args.max_updates is not None and update >= args.max_updates:
                break
        if stop_requested:
            if not stop_checkpoint_committed and not last_batch_checkpoint_committed:
                commit_checkpoint(epoch + 1, 0)
            break
        if args.max_updates is not None and update >= args.max_updates:
            break

    collect_pending()
    mx.eval(model.state, optimizer.state)

    training_elapsed = elapsed_before_resume + time.monotonic() - started
    if stop_requested:
        summary = {
            "updates": update,
            "stop_requested": True,
            "training_seconds": training_elapsed,
            "processed_frames": processed_frames,
            "history": history,
        }
        (output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
        return

    model.save(str(output / "model.safetensors"))
    trained_evaluation = evaluate(model, eval_dataset, args.batch_size, args.gamma) if eval_dataset else None
    reloaded = RNNoise.load(str(output / "model.safetensors"), config)
    reloaded_evaluation = evaluate(reloaded, eval_dataset, args.batch_size, args.gamma) if eval_dataset else None
    reload_matches = trained_evaluation == reloaded_evaluation
    if stop_requested:
        summary = {
            "updates": update,
            "stop_requested": True,
            "training_seconds": training_elapsed,
            "processed_frames": processed_frames,
            "history": history,
        }
        (output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
        return
    summary = {
        "updates": update,
        "stop_requested": stop_requested,
        "training_seconds": training_elapsed,
        "updates_per_second": update / training_elapsed,
        "processed_frames": processed_frames,
        "frames_per_second": processed_frames / training_elapsed,
        "audio_seconds_per_second": processed_frames * 0.01 / training_elapsed,
        "compiled": not args.no_compile,
        "async_eval": False,
        "training_chunk_length": args.training_chunk_length,
        "stateful_tbptt": args.stateful_tbptt,
        "two_segment_tbptt": args.two_segment_tbptt,
        "segmented_tbptt_length": segment_length,
        "segmented_tbptt_state": segment_state if segment_length else None,
        "equalize_reset_targets": equalize_reset_targets,
        "initial_evaluation": initial_evaluation,
        "trained_evaluation": trained_evaluation,
        "reloaded_evaluation": reloaded_evaluation,
        "reload_matches": reload_matches,
        "history": history,
    }
    (output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
def main(argv=None):
    parsed = parse_args(argv)
    training_identity, evaluation_identity = preflight_feature_identities(parsed)
    return train(
        parsed,
        feature_identity=training_identity,
        evaluation_feature_identity=evaluation_identity,
    )

if __name__ == "__main__":
    main()
