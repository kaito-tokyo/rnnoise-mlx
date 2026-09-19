"""RNNoise-MLX model training command."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from collections.abc import Callable
from dataclasses import asdict
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

try:
    import nvtx as _nvtx
except ImportError:  # Optional profiling-only dependency.
    _nvtx = None

from .data import FeatureDataset
from .checkpoint import (
    _feature_identity,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
)
from .evaluate import evaluate
from .old_impl import rnnoise_loss_aligned
from .config import ModelConfig
from .model import RNNoise
from ..training_tools import (
    TrainConfig,
    TrainingCheckpoint,
    TrainingEvent,
    TrainingProgress,
)
from ..training_cuda import FixedChunkWorkspace, validate_compiled_chunk
from ..training_cuda.executor import CompiledChunkRunner


_IDENTITY_NOT_PROVIDED = object()


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
    parser.add_argument(
        "--graph-mode",
        choices=("dynamic", "compiled_chunk"),
        default="dynamic",
        help="dynamic evaluation or fixed-shape compiled segmented chunks",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        dest="no_compile",
        action="store_false",
        help="enable MLX CUDA graph compilation (higher memory use)",
    )
    compile_group.add_argument(
        "--no-compile",
        dest="no_compile",
        action="store_true",
        help="disable MLX CUDA graph compilation (default)",
    )
    parser.set_defaults(no_compile=True)
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
    parser.add_argument("--timing-path", type=Path)
    args = parser.parse_args(argv)

    if args.two_segment_tbptt and args.segmented_tbptt_length:
        parser.error("select only one segmented TBPTT interface")
    if (args.two_segment_tbptt or args.segmented_tbptt_length) and args.stateful_tbptt:
        parser.error("segmented TBPTT and --stateful-tbptt are mutually exclusive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.graph_mode == "compiled_chunk" and not args.no_compile:
        parser.error("--graph-mode compiled_chunk cannot be combined with --compile")
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
    phase_timings: list[dict[str, object]] = []

    nvtx_enabled = _nvtx is not None and os.environ.get("RNNOISE_MLX_NVTX") == "1"

    def nvtx_range(name: str, **metadata):
        if not nvtx_enabled:
            return nullcontext()
        details = ",".join(f"{key}={value}" for key, value in sorted(metadata.items()))
        message = f"rnnoise:{name}" + (f" [{details}]" if details else "")
        return _nvtx.annotate(message)

    @contextmanager
    def measure_phase(name: str, **metadata):
        started = time.perf_counter()
        with nvtx_range(name, **metadata):
            try:
                yield
            finally:
                phase_timings.append({
                    "phase": name,
                    "seconds": time.perf_counter() - started,
                    **metadata,
                })

    def write_phase_timings() -> None:
        if args.timing_path is not None:
            args.timing_path.parent.mkdir(parents=True, exist_ok=True)
            args.timing_path.write_text(json.dumps(phase_timings, indent=2))

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
        return rnnoise_loss_aligned(pred_gain, pred_vad, gain[:, 3:-1, :], vad[:, 3:-1, :], args.gamma)[0]

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
    validate_compiled_chunk(args, segment_length)

    use_compiled_chunk = args.graph_mode == "compiled_chunk"
    if not args.no_compile or use_compiled_chunk:
        captured_state = [model.state, optimizer.state]
        if not use_compiled_chunk:
            train_step = partial(
                mx.compile, inputs=captured_state, outputs=captured_state
            )(train_step)

    def stateful_objective(
        params, features, gain, vad, state, feature_workspace, conv1_workspace, first
    ):
        model.update(params)
        if not use_compiled_chunk and first:
            predicted_gain, predicted_vad, new_state = model.first_chunk(features)
            return rnnoise_loss_aligned(
                predicted_gain, predicted_vad, gain, vad, args.gamma
            )[0], new_state
        if not use_compiled_chunk:
            predicted_gain, predicted_vad, new_state = model.next_chunk(features, state)
            return rnnoise_loss_aligned(
                predicted_gain, predicted_vad, gain, vad, args.gamma
            )[0], new_state
        if first:
            (
                predicted_gain,
                predicted_vad,
                new_state,
                feature_workspace,
                conv1_workspace,
            ) = model.first_chunk_fixed(feature_workspace, features, conv1_workspace)
        else:
            (
                predicted_gain,
                predicted_vad,
                new_state,
                feature_workspace,
                conv1_workspace,
            ) = model.next_chunk_fixed(feature_workspace, features, conv1_workspace, state)
        loss = rnnoise_loss_aligned(predicted_gain, predicted_vad, gain, vad, args.gamma)[0]
        return loss, new_state, feature_workspace, conv1_workspace

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

    # In compiled_chunk mode these tree operations are part of the same
    # fixed-structure MLX graph as the chunk outputs.  The Python recursion
    # still describes the tree once, but the per-chunk arithmetic is emitted
    # as MLX operations and can be reused for every 250-frame chunk.
    scale_gradients = multiply_gradient_tree
    accumulate_gradients = add_gradient_trees
    average_gradients = divide_gradient_tree
    compiled_runner = None
    if use_compiled_chunk:
        compiled_runner = CompiledChunkRunner(
            model, optimizer, args.gamma, captured_state, segment_length
        )
        scale_gradients = compiled_runner.scale_gradients
        accumulate_gradients = compiled_runner.accumulate_gradients
        average_gradients = compiled_runner.average_gradients

    def first_chunk_grad(features, gain, vad, feature_workspace=(), conv1_workspace=()):
        (loss, state, feature_workspace, conv1_workspace), gradients = stateful_value_and_grad(
            model.trainable_parameters(),
            features,
            gain,
            vad,
            (),
            feature_workspace,
            conv1_workspace,
            True,
        )
        if use_compiled_chunk:
            return loss, state, gradients, feature_workspace, conv1_workspace
        return loss, state, gradients

    def next_chunk_grad(
        features, gain, vad, state, feature_workspace=(), conv1_workspace=()
    ):
        (loss, new_state, feature_workspace, conv1_workspace), gradients = stateful_value_and_grad(
            model.trainable_parameters(),
            features,
            gain,
            vad,
            state,
            feature_workspace,
            conv1_workspace,
            False,
        )
        if use_compiled_chunk:
            return loss, new_state, gradients, feature_workspace, conv1_workspace
        return loss, new_state, gradients

    def first_chunk_step(features, gain, vad, feature_workspace=(), conv1_workspace=()):
        result = first_chunk_grad(features, gain, vad, feature_workspace, conv1_workspace)
        if use_compiled_chunk:
            loss, state, gradients, feature_workspace, conv1_workspace = result
        else:
            loss, state, gradients = result
        optimizer.update(model, gradients)
        if use_compiled_chunk:
            return loss, state, feature_workspace, conv1_workspace
        return loss, state

    def next_chunk_step(
        features, gain, vad, state, feature_workspace=(), conv1_workspace=()
    ):
        result = next_chunk_grad(
            features, gain, vad, state, feature_workspace, conv1_workspace
        )
        if use_compiled_chunk:
            loss, new_state, gradients, feature_workspace, conv1_workspace = result
        else:
            loss, new_state, gradients = result
        optimizer.update(model, gradients)
        if use_compiled_chunk:
            return loss, new_state, feature_workspace, conv1_workspace
        return loss, new_state

    def apply_optimizer(gradients):
        optimizer.update(model, gradients)

    if not args.no_compile and not use_compiled_chunk:
        first_chunk_grad = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(first_chunk_grad)
        next_chunk_grad = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(next_chunk_grad)
        if not use_compiled_chunk:
            first_chunk_step = partial(
                mx.compile, inputs=captured_state, outputs=captured_state
            )(first_chunk_step)
            next_chunk_step = partial(
                mx.compile, inputs=captured_state, outputs=captured_state
            )(next_chunk_step)
    if use_compiled_chunk:
        first_chunk_grad = compiled_runner.first_grad
        next_chunk_grad = compiled_runner.next_grad
        apply_optimizer = compiled_runner.apply_optimizer
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
            batch_started = time.perf_counter()
            if use_compiled_chunk:
                # The compiled runner owns the complete update boundary:
                # segmented forward/backward, gradient accumulation, and
                # optimizer state mutation all happen in one transformation.
                loss = compiled_runner.update(features, gain, vad)
                update += 1
                processed_frames += args.batch_size * features.shape[1]
                pending_losses.append((update, epoch, processed_frames, loss))
                if len(pending_losses) >= 10:
                    collect_pending()
            elif segment_length:
                fixed_workspace = None
                if use_compiled_chunk:
                    fixed_workspace = FixedChunkWorkspace.allocate(features)
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
                    chunk_features = features[:, start:end, :]
                    is_first_chunk = start == 0 or segment_state == "reset"
                    gradient_started = time.perf_counter()
                    with nvtx_range(
                        "gradient_graph",
                        update=update + 1,
                        chunk_start=start,
                        chunk_length=end - start,
                    ):
                        if is_first_chunk:
                            if use_compiled_chunk:
                                chunk_gain = gain[:, start:end, :]
                                chunk_vad = vad[:, start:end, :]
                                loss, state, gradients, fixed_workspace.features = first_chunk_grad(
                                    chunk_features,
                                    chunk_gain,
                                    chunk_vad,
                                    fixed_workspace.features,
                                )
                            else:
                                target_start = start + 3
                                chunk_gain = gain[:, target_start : end - 1, :]
                                chunk_vad = vad[:, target_start : end - 1, :]
                                loss, state, gradients = first_chunk_grad(
                                    chunk_features, chunk_gain, chunk_vad
                                )
                        else:
                            if use_compiled_chunk:
                                chunk_gain = gain[:, start:end, :]
                                chunk_vad = vad[:, start:end, :]
                                loss, state, gradients, fixed_workspace.features = next_chunk_grad(
                                    chunk_features,
                                    chunk_gain,
                                    chunk_vad,
                                    state,
                                    fixed_workspace.features,
                                )
                            else:
                                chunk_gain = gain[:, start - 1 : end - 1, :]
                                chunk_vad = vad[:, start - 1 : end - 1, :]
                                loss, state, gradients = next_chunk_grad(
                                    chunk_features, chunk_gain, chunk_vad, state
                                )
                    phase_timings.append({
                        "phase": "gradient_graph",
                        "seconds": time.perf_counter() - gradient_started,
                        "update": update + 1,
                        "chunk_start": start,
                    })
                    chunk_frames = chunk_gain.shape[1]
                    with measure_phase("mx_eval", update=update + 1, chunk_start=start):
                        # ``gradients`` depend on ``loss`` already.  Evaluating
                        # loss as a separate root here adds a scalar output to
                        # every chunk boundary without making state carry or
                        # gradient accumulation more concrete.  Keep loss
                        # lazy until the update-level evaluation below.
                        mx.eval(state, gradients)
                    weighted_gradients = scale_gradients(gradients, chunk_frames)
                    accumulated_gradients = (
                        weighted_gradients
                        if accumulated_gradients is None
                        else accumulate_gradients(accumulated_gradients, weighted_gradients)
                    )
                    accumulated_loss = (
                        loss * chunk_frames
                        if accumulated_loss is None
                        else accumulated_loss + loss * chunk_frames
                    )
                    target_frames += chunk_frames
                    state = tuple(mx.stop_gradient(value) for value in state)
                gradients = average_gradients(accumulated_gradients, target_frames)
                with measure_phase("optimizer_update", update=update + 1):
                    apply_optimizer(gradients)
                loss = accumulated_loss / target_frames
                update += 1
                processed_frames += args.batch_size * features.shape[1]
                pending_losses.append((update, epoch, processed_frames, loss))
                with measure_phase("mx_eval_update", update=update):
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
                        chunk_features = features[:, start:end, :]
                        if start == 0 or segment_state == "reset":
                            target_start = start + 3
                            chunk_gain = gain[:, target_start : end - 1, :]
                            chunk_vad = vad[:, target_start : end - 1, :]
                            loss, state = first_chunk_step(chunk_features, chunk_gain, chunk_vad)
                        else:
                            chunk_gain = gain[:, start - 1 : end - 1, :]
                            chunk_vad = vad[:, start - 1 : end - 1, :]
                            loss, state = next_chunk_step(
                                chunk_features, chunk_gain, chunk_vad, state
                            )
                        state = tuple(mx.stop_gradient(value) for value in state)
                    else:
                        loss = train_step(features, gain, vad)

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
                with measure_phase("checkpoint", update=update):
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
    write_phase_timings()

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

    save_model(model, str(output / "model.safetensors"))
    trained_evaluation = evaluate(model, eval_dataset, args.batch_size, args.gamma) if eval_dataset else None
    reloaded = load_model(
        RNNoise,
        str(output / "model.safetensors"),
        config,
        batch_size=args.batch_size,
    )
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
