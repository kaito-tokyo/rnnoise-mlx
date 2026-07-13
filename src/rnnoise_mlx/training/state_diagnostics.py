"""Training diagnostics for recurrent model state handling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import FeatureDataset
from .loss import rnnoise_loss_aligned
from .model import ModelConfig, RNNoise

LAYERS = ("conv1", "conv2", "gru1", "gru2", "gru3", "gain", "vad")
RESET_MODES = ("all", "conv", "gru1", "gru2", "gru3", "gru12", "gru123")


def _recurrent_trace(model, x, states=None):
    states = states or (None, None, None)
    y1, h1 = model.gru1(x, states[0])
    y2, h2 = model.gru2(y1, states[1])
    y3, h3 = model.gru3(y2, states[2])
    joined = mx.concatenate((x, y1, y2, y3), axis=-1)
    return {
        "gru1": y1,
        "gru2": y2,
        "gru3": y3,
        "gain": mx.sigmoid(model.gain(joined)),
        "vad": mx.sigmoid(model.vad(joined)),
    }, (h1, h2, h3)


def _reset_state(state, mode):
    feature_history, conv1_history, h1, h2, h3 = state
    if mode in (None, "continuous"):
        return state
    if mode in ("all", "conv"):
        feature_history = mx.zeros_like(feature_history)
        conv1_history = mx.zeros_like(conv1_history)
    if mode in ("all", "gru1", "gru12", "gru123"):
        h1 = mx.zeros_like(h1)
    if mode in ("all", "gru2", "gru12", "gru123"):
        h2 = mx.zeros_like(h2)
    if mode in ("all", "gru3", "gru123"):
        h3 = mx.zeros_like(h3)
    return feature_history, conv1_history, h1, h2, h3


def trace_chunks(model, features, chunk_length, reset_mode=None):
    traces = {name: [] for name in LAYERS}
    state = None
    boundaries = []
    output_count = 0
    for start in range(0, features.shape[1], chunk_length):
        end = min(start + chunk_length, features.shape[1])
        chunk = features[:, start:end, :]
        if start == 0:
            conv1 = mx.tanh(model.conv1(chunk))
            conv2 = mx.tanh(model.conv2(conv1))
            recurrent, hidden = _recurrent_trace(model, conv2)
            state = (chunk[..., -2:, :], conv1[..., -2:, :], *hidden)
            traces["conv1"].append(conv1)
        else:
            boundaries.append(output_count)
            state = _reset_state(state, reset_mode)
            feature_history, conv1_history, h1, h2, h3 = state
            conv1 = mx.tanh(model.conv1(mx.concatenate((feature_history, chunk), axis=-2)))
            conv2 = mx.tanh(model.conv2(mx.concatenate((conv1_history, conv1), axis=-2)))
            recurrent, hidden = _recurrent_trace(model, conv2, (h1, h2, h3))
            state = (chunk[..., -2:, :], conv1[..., -2:, :], *hidden)
            # The first two Conv1 values correspond to history-only positions.
            traces["conv1"].append(conv1)
        traces["conv2"].append(conv2)
        for name, value in recurrent.items():
            traces[name].append(value)
        output_count += conv2.shape[1]
    result = {name: mx.concatenate(values, axis=1) for name, values in traces.items()}
    mx.eval(*result.values(), *state)
    return result, state, boundaries


def _distribution_metrics(values, boundaries):
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(-1)
    vectors = values.reshape(-1, values.shape[-2], values.shape[-1])
    delta = np.linalg.norm(np.diff(vectors, axis=1), axis=-1)
    left = vectors[:, :-1, :]
    right = vectors[:, 1:, :]
    cosine = np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-12
    )
    boundary_indices = [index - 1 for index in boundaries if 0 < index < values.shape[-2]]
    mask = np.ones(delta.shape[1], dtype=bool)
    mask[boundary_indices] = False
    boundary_delta = delta[:, boundary_indices].mean() if boundary_indices else 0.0
    normal_delta = delta[:, mask].mean() if mask.any() else 0.0
    norms = np.linalg.norm(vectors, axis=-1).mean(axis=0)
    steady = float(np.median(norms[max(0, len(norms) * 3 // 4) :]))
    rolling = np.convolve(norms, np.ones(20) / 20, mode="valid") if len(norms) >= 20 else norms
    tolerance = max(abs(steady) * 0.1, 1e-6)
    candidates = np.flatnonzero(np.abs(rolling - steady) <= tolerance)
    steady_frame = int(candidates[0]) if candidates.size else None
    absolute = np.abs(flat)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "max_abs": float(absolute.max()),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "saturation_0_9": float(np.mean(absolute > 0.9)),
        "saturation_0_99": float(np.mean(absolute > 0.99)),
        "mean_delta_l2": float(delta.mean()),
        "mean_cosine": float(cosine.mean()),
        "boundary_delta_l2": float(boundary_delta),
        "normal_delta_l2": float(normal_delta),
        "boundary_delta_ratio": float(boundary_delta / (normal_delta + 1e-12)),
        "steady_state_frame": steady_frame,
    }


def _memory_horizon(reference, reset, boundaries, chunk_length):
    reference = np.asarray(reference, dtype=np.float32)
    reset = np.asarray(reset, dtype=np.float32)
    distance = np.linalg.norm(reference - reset, axis=-1).mean(axis=0)
    horizons = {"50": [], "10": [], "1": []}
    for boundary in boundaries:
        segment = distance[boundary : min(boundary + chunk_length, len(distance))]
        if not len(segment) or segment[0] <= 1e-12:
            for values in horizons.values():
                values.append(0)
            continue
        ratio = segment / segment[0]
        for label, threshold in (("50", 0.5), ("10", 0.1), ("1", 0.01)):
            hits = np.flatnonzero(ratio <= threshold)
            horizons[label].append(int(hits[0]) if hits.size else None)
    result = {}
    for label, values in horizons.items():
        finite = [value for value in values if value is not None]
        result[f"frames_to_{label}_percent"] = int(np.median(finite)) if finite else None
        result[f"persisted_to_boundary_{label}_percent"] = len(finite) != len(values)
    result["initial_distance_mean"] = float(
        np.mean([distance[index] for index in boundaries]) if boundaries else 0
    )
    return result


def _relative_error(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stale_state_diagnostic(checkpoint, config, batch, chunk_length, learning_rate=1e-3):
    if checkpoint is None:
        mx.random.seed(0)
        model = RNNoise(config)
        original_hash = None
    else:
        original_hash = _file_hash(checkpoint)
        model = RNNoise.load(checkpoint, config)
    features, gain, vad = batch
    prefix = mx.array(features[:, :chunk_length, :])
    next_features = mx.array(features[:, chunk_length : 2 * chunk_length, :])
    target_gain = mx.array(gain[:, 3 : chunk_length - 1, :])
    target_vad = mx.array(vad[:, 3 : chunk_length - 1, :])
    _, _, stale_state = model.first_chunk(prefix)
    optimizer = optim.AdamW(learning_rate=learning_rate, betas=(0.8, 0.98), eps=1e-8)

    def objective(current_model, x, g, v):
        predicted_gain, predicted_vad, _ = current_model.first_chunk(x)
        return rnnoise_loss_aligned(predicted_gain, predicted_vad, g, v)[0]

    loss_and_grad = nn.value_and_grad(model, objective)
    loss, gradients = loss_and_grad(model, prefix, target_gain, target_vad)
    optimizer.update(model, gradients)
    mx.eval(model.state, optimizer.state, loss)
    _, _, recomputed_state = model.first_chunk(prefix)
    stale_gain, stale_vad, _ = model.next_chunk(next_features, stale_state)
    fresh_gain, fresh_vad, _ = model.next_chunk(next_features, recomputed_state)
    mx.eval(*stale_state, *recomputed_state, stale_gain, stale_vad, fresh_gain, fresh_vad)
    names = ("conv1_history", "conv2_history", "gru1", "gru2", "gru3")
    result = {
        "state_relative_error": {
            name: _relative_error(old, new)
            for name, old, new in zip(names, stale_state, recomputed_state)
        },
        "next_gain_relative_error": _relative_error(stale_gain, fresh_gain),
        "next_vad_relative_error": _relative_error(stale_vad, fresh_vad),
        "diagnostic_loss": float(loss.item()),
    }
    if checkpoint is not None:
        result["checkpoint_hash_unchanged"] = original_hash == _file_hash(checkpoint)
    return result


def _load_model(checkpoint, config):
    if checkpoint is None:
        mx.random.seed(0)
        return RNNoise(config)
    return RNNoise.load(checkpoint, config)


def _parse_checkpoints(values):
    parsed = [("untrained", None)]
    for value in values:
        label, separator, path = value.partition("=")
        if not separator:
            raise ValueError("checkpoint must be LABEL=PATH")
        parsed.append((label, path))
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_features")
    parser.add_argument("output")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--chunk-lengths", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trace-sequences", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2000)
    args = parser.parse_args()

    output = Path(args.output)
    trace_dir = output / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    dataset = FeatureDataset(args.eval_features, args.sequence_length)
    config = ModelConfig()
    checkpoints = _parse_checkpoints(args.checkpoint)
    all_batches = list(dataset.batches(args.batch_size, np.random.default_rng(0)))
    trace_features = mx.array(all_batches[0][0][: args.trace_sequences])
    summary = {"config": vars(args), "checkpoints": {}}
    csv_rows = []

    for label, checkpoint in checkpoints:
        print(f"diagnosing {label}", flush=True)
        model = _load_model(checkpoint, config)
        checkpoint_result = {"chunks": {}, "stale_state": {}}
        for chunk_length in args.chunk_lengths:
            layer_values = {name: [] for name in LAYERS}
            equivalence_errors = []
            boundaries = None
            for features, _, _ in all_batches:
                feature_array = mx.array(features)
                chunked, _, current_boundaries = trace_chunks(model, feature_array, chunk_length)
                full_gain, full_vad, _ = model(feature_array)
                mx.eval(full_gain, full_vad)
                equivalence_errors.append(
                    max(
                        float(mx.max(mx.abs(chunked["gain"] - full_gain)).item()),
                        float(mx.max(mx.abs(chunked["vad"] - full_vad)).item()),
                    )
                )
                for name in LAYERS:
                    layer_values[name].append(np.asarray(chunked[name]))
                boundaries = current_boundaries
            metrics = {
                name: _distribution_metrics(np.concatenate(values, axis=0), boundaries)
                for name, values in layer_values.items()
            }
            reference, _, trace_boundaries = trace_chunks(model, trace_features, chunk_length)
            reset_results = {}
            trace_payload = {f"continuous_{name}": np.asarray(value) for name, value in reference.items()}
            for mode in RESET_MODES:
                reset, _, _ = trace_chunks(model, trace_features, chunk_length, mode)
                reset_results[mode] = {
                    name: _memory_horizon(reference[name], reset[name], trace_boundaries, chunk_length)
                    for name in ("gru1", "gru2", "gru3", "gain", "vad")
                }
                if mode in ("all", "conv"):
                    trace_payload.update({f"{mode}_{name}": np.asarray(value) for name, value in reset.items()})
            conv_reset, _, _ = trace_chunks(model, trace_features, chunk_length, "conv")
            conv_difference = np.linalg.norm(
                np.asarray(reference["conv2"]) - np.asarray(conv_reset["conv2"]), axis=-1
            ).mean(axis=0)
            spans = []
            for boundary_index, boundary in enumerate(trace_boundaries):
                segment_end = (
                    trace_boundaries[boundary_index + 1]
                    if boundary_index + 1 < len(trace_boundaries)
                    else len(conv_difference)
                )
                affected = np.flatnonzero(conv_difference[boundary:segment_end] > 1e-6)
                spans.append(int(affected.max()) if affected.size else 0)
            conv_reset_span = max(spans, default=0)
            np.savez_compressed(trace_dir / f"{label}-chunk-{chunk_length}.npz", **trace_payload)
            checkpoint_result["chunks"][str(chunk_length)] = {
                "max_full_chunk_error": max(equivalence_errors),
                "metrics": metrics,
                "reset_memory_horizon": reset_results,
                "conv_reset_max_affected_offset": conv_reset_span,
            }
            for name, values in metrics.items():
                csv_rows.append({"checkpoint": label, "chunk": chunk_length, "layer": name, **values})
        for chunk_length in (200, 800):
            checkpoint_result["stale_state"][str(chunk_length)] = stale_state_diagnostic(
                checkpoint, config, all_batches[0], chunk_length
            )
        summary["checkpoints"][label] = checkpoint_result

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fieldnames = list(csv_rows[0])
    with (output / "state_metrics.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
