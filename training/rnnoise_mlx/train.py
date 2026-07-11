from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import FeatureDataset
from .evaluate import evaluate
from .loss import rnnoise_loss, rnnoise_loss_aligned
from .model import ModelConfig, RNNoise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features")
    parser.add_argument("output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=5e-5)
    parser.add_argument("--cond-size", type=int, default=128)
    parser.add_argument("--gru-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-features")
    parser.add_argument("--training-chunk-length", type=int, default=200)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--sync-eval", action="store_true")
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
    args = parser.parse_args()

    if args.two_segment_tbptt and args.segmented_tbptt_length:
        parser.error("select only one segmented TBPTT interface")
    if (args.two_segment_tbptt or args.segmented_tbptt_length) and args.stateful_tbptt:
        parser.error("segmented TBPTT and --stateful-tbptt are mutually exclusive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = FeatureDataset(args.features, args.sequence_length)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = ModelConfig(cond_size=args.cond_size, gru_size=args.gru_size)
    model = RNNoise(config)
    learning_rate = lambda step: args.learning_rate / (1 + args.lr_decay * step)
    optimizer = optim.AdamW(learning_rate=learning_rate, betas=(0.8, 0.98), eps=1e-8)
    eval_dataset = FeatureDataset(args.eval_features, args.sequence_length) if args.eval_features else None
    initial_evaluation = evaluate(model, eval_dataset, args.batch_size, args.gamma) if eval_dataset else None

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
        parser.error("segmented TBPTT length must be >4 and divide --sequence-length")

    def segmented_objective(model, features, gain, vad):
        weighted_loss = 0
        target_frames = 0
        state = None
        for start in range(0, args.sequence_length, segment_length):
            end = start + segment_length
            chunk = features[:, start:end, :]
            if start == 0 or segment_state == "reset":
                pred_gain, pred_vad, state = model.first_chunk(chunk)
                target_gain = gain[:, start + 3 : end - 1, :]
                target_vad = vad[:, start + 3 : end - 1, :]
            else:
                state = tuple(mx.stop_gradient(value) for value in state)
                pred_gain, pred_vad, state = model.next_chunk(chunk, state)
                if equalize_reset_targets:
                    pred_gain = pred_gain[:, 4:, :]
                    pred_vad = pred_vad[:, 4:, :]
                    target_gain = gain[:, start + 3 : end - 1, :]
                    target_vad = vad[:, start + 3 : end - 1, :]
                else:
                    target_gain = gain[:, start - 1 : end - 1, :]
                    target_vad = vad[:, start - 1 : end - 1, :]
            loss = rnnoise_loss_aligned(
                pred_gain, pred_vad, target_gain, target_vad, args.gamma
            )[0]
            frames = pred_gain.shape[-2]
            weighted_loss = weighted_loss + frames * loss
            target_frames += frames
        return weighted_loss / target_frames

    segmented_value_and_grad = nn.value_and_grad(model, segmented_objective)

    def segmented_step(features, gain, vad):
        loss, gradients = segmented_value_and_grad(model, features, gain, vad)
        optimizer.update(model, gradients)
        return loss

    if not args.no_compile:
        captured_state = [model.state, optimizer.state]
        train_step = partial(mx.compile, inputs=captured_state, outputs=captured_state)(train_step)
        segmented_step = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(segmented_step)

    def stateful_objective(params, features, gain, vad, state, first):
        model.update(params)
        if first:
            predicted_gain, predicted_vad, new_state = model.first_chunk(features)
        else:
            predicted_gain, predicted_vad, new_state = model.next_chunk(features, state)
        loss = rnnoise_loss_aligned(predicted_gain, predicted_vad, gain, vad, args.gamma)[0]
        return loss, new_state

    stateful_value_and_grad = mx.value_and_grad(stateful_objective)

    def first_chunk_step(features, gain, vad):
        (loss, state), gradients = stateful_value_and_grad(
            model.trainable_parameters(), features, gain, vad, (), True
        )
        optimizer.update(model, gradients)
        return loss, state

    def next_chunk_step(features, gain, vad, state):
        (loss, new_state), gradients = stateful_value_and_grad(
            model.trainable_parameters(), features, gain, vad, state, False
        )
        optimizer.update(model, gradients)
        return loss, new_state

    if not args.no_compile:
        first_chunk_step = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(first_chunk_step)
        next_chunk_step = partial(
            mx.compile, inputs=captured_state, outputs=captured_state
        )(next_chunk_step)
    history = []
    pending_losses = []
    update = 0
    processed_frames = 0
    started = time.monotonic()

    def batches_for_epoch():
        batches = dataset.batches(
            args.batch_size,
            rng,
            chunk_length=None
            if args.stateful_tbptt or segment_length
            else args.training_chunk_length,
        )
        if args.no_prefetch:
            yield from batches
            return
        iterator = iter(batches)
        sentinel = object()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rnnoise-data") as executor:
            future = executor.submit(next, iterator, sentinel)
            while True:
                batch = future.result()
                if batch is sentinel:
                    break
                future = executor.submit(next, iterator, sentinel)
                yield batch

    def collect_pending():
        if not pending_losses:
            return
        mx.eval(*(loss for _, _, loss in pending_losses))
        for pending_update, pending_epoch, pending_loss in pending_losses:
            record = {
                "update": pending_update,
                "epoch": pending_epoch,
                "loss": float(pending_loss.item()),
            }
            history.append(record)
            if pending_update % 10 == 0:
                print(json.dumps(record), flush=True)
        pending_losses.clear()

    for epoch in range(1, args.epochs + 1):
        for features, gain, vad in batches_for_epoch():
            if segment_length:
                state = None
                chunk_ranges = (0,)
            elif args.stateful_tbptt:
                state = None
                chunk_ranges = range(0, args.sequence_length, args.training_chunk_length)
            else:
                state = None
                chunk_ranges = (0,)

            for start in chunk_ranges:
                if segment_length:
                    loss = segmented_step(
                        mx.array(features), mx.array(gain), mx.array(vad)
                    )
                elif args.stateful_tbptt:
                    end = min(start + args.training_chunk_length, args.sequence_length)
                    chunk_features = mx.array(features[:, start:end, :])
                    if start == 0:
                        chunk_gain = mx.array(gain[:, 3 : end - 1, :])
                        chunk_vad = mx.array(vad[:, 3 : end - 1, :])
                        loss, state = first_chunk_step(chunk_features, chunk_gain, chunk_vad)
                    else:
                        chunk_gain = mx.array(gain[:, start - 1 : end - 1, :])
                        chunk_vad = mx.array(vad[:, start - 1 : end - 1, :])
                        loss, state = next_chunk_step(chunk_features, chunk_gain, chunk_vad, state)
                    state = tuple(mx.stop_gradient(value) for value in state)
                else:
                    loss = train_step(mx.array(features), mx.array(gain), mx.array(vad))

                update += 1
                processed_frames += args.batch_size * (
                    chunk_features.shape[1]
                    if args.stateful_tbptt and not segment_length
                    else features.shape[1]
                )
                pending_losses.append((update, epoch, loss))
                if args.sync_eval:
                    mx.eval(model.state, optimizer.state, loss)
                else:
                    mx.async_eval(model.state, optimizer.state, loss)
                if len(pending_losses) >= 10:
                    collect_pending()
                if update % 32 == 0:
                    collect_pending()
                    mx.eval(model.state, optimizer.state)
                    model.save(str(output / "checkpoint.safetensors"))
                if args.max_updates is not None and update >= args.max_updates:
                    break
            if args.max_updates is not None and update >= args.max_updates:
                break
        if args.max_updates is not None and update >= args.max_updates:
            break

    collect_pending()
    mx.eval(model.state, optimizer.state)

    training_elapsed = time.monotonic() - started
    model.save(str(output / "model.safetensors"))
    trained_evaluation = evaluate(model, eval_dataset, args.batch_size, args.gamma) if eval_dataset else None
    reloaded = RNNoise.load(str(output / "model.safetensors"), config)
    reloaded_evaluation = evaluate(reloaded, eval_dataset, args.batch_size, args.gamma) if eval_dataset else None
    reload_matches = trained_evaluation == reloaded_evaluation
    summary = {
        "updates": update,
        "training_seconds": training_elapsed,
        "updates_per_second": update / training_elapsed,
        "processed_frames": processed_frames,
        "frames_per_second": processed_frames / training_elapsed,
        "audio_seconds_per_second": processed_frames * 0.01 / training_elapsed,
        "compiled": not args.no_compile,
        "async_eval": not args.sync_eval,
        "prefetch_batches": 0 if args.no_prefetch else 1,
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


if __name__ == "__main__":
    main()
