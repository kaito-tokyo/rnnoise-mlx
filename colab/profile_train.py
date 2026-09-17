"""Run a small synchronous training job for external profilers such as py-spy."""

from __future__ import annotations

import argparse
from pathlib import Path

from rnnoise_mlx.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="/content/train.f32")
    parser.add_argument("--output", default="/content/runs/profile")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=20)
    parser.add_argument("--segmented-tbptt-length", type=int, default=250)
    parser.add_argument(
        "--graph-mode", choices=("dynamic", "compiled_chunk"), default="dynamic"
    )
    parser.add_argument("--feature-identity", required=True)
    parser.add_argument("--evaluation-feature-identity")
    parser.add_argument("--timing-path", type=Path)
    args = parser.parse_args()

    features = Path(args.features)
    if not features.is_file():
        raise SystemExit(f"missing features: {features}")

    config = TrainConfig(
        features=str(features),
        output=args.output,
        batch_size=args.batch_size,
        sequence_length=2000,
        segmented_tbptt_length=args.segmented_tbptt_length,
        segmented_tbptt_state="carry",
        graph_mode=args.graph_mode,
        max_updates=args.max_updates,
        checkpoint_every=args.max_updates,
        sync_eval=True,
        seed=141,
        timing_path=args.timing_path,
    )
    summary = train(
        config,
        feature_identity=args.feature_identity,
        evaluation_feature_identity=args.evaluation_feature_identity,
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
