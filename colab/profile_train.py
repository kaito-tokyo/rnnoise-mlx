"""Run a small synchronous training job for external profilers such as py-spy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from rnnoise_mlx.training.config import ModelConfig
from rnnoise_mlx.training.data import FeatureDataset
from rnnoise_mlx.training_cuda import CUDATrainingLoop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="/content/train.f32")
    parser.add_argument("--output", default="/content/runs/profile")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=20)
    parser.add_argument("--segmented-tbptt-length", type=int, default=250)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--feature-identity", required=True)
    parser.add_argument("--evaluation-feature-identity")
    parser.add_argument("--timing-path", type=Path)
    args = parser.parse_args()

    features = Path(args.features)
    if not features.is_file():
        raise SystemExit(f"missing features: {features}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = FeatureDataset(str(features), sequence_length=2000)
    loop = CUDATrainingLoop.create(
        ModelConfig(),
        batch_size=args.batch_size,
        tbptt_length=args.segmented_tbptt_length,
        compile_chunks=not args.no_compile,
    )
    rng = np.random.default_rng(141)
    state = tuple(
        mx.zeros((args.batch_size, ModelConfig.gru_size), dtype=mx.float32)
        for _ in range(3)
    )
    timings = []
    losses = []
    batches = dataset.batches(args.batch_size, rng)
    for update in range(args.max_updates):
        features_batch, target_gain, target_vad = next(batches)
        started = time.perf_counter()
        result = loop.run_update(
            features_batch,
            target_gain,
            target_vad,
            segment_length=args.segmented_tbptt_length,
            state=state,
        )
        elapsed = time.perf_counter() - started
        state = result.state
        loss = float(result.loss.item())
        losses.append(loss)
        timings.append({"update": update + 1, "seconds": elapsed, "loss": loss})
        print(
            f"update={update + 1}/{args.max_updates} loss={loss:.6f} "
            f"seconds={elapsed:.3f}",
            flush=True,
        )

    summary = {
        "feature_identity": args.feature_identity,
        "batch_size": args.batch_size,
        "sequence_length": 2000,
        "segmented_tbptt_length": args.segmented_tbptt_length,
        "updates": args.max_updates,
        "losses": losses,
        "timings": timings,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2))
    if args.timing_path:
        args.timing_path.parent.mkdir(parents=True, exist_ok=True)
        args.timing_path.write_text(json.dumps(timings, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
