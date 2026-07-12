"""Generate and size-check RNNoise train/evaluation feature files."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


BYTES_PER_SEQUENCE = 2000 * 98 * 4


def generate(
    dump_features: Path,
    prepared: Path,
    output: Path,
    split: str,
    count: int,
) -> None:
    destination = output / f"{split}.f32"
    subprocess.run(
        [
            str(dump_features),
            "-rir_list",
            str(prepared / f"{split}_rir_list.txt"),
            str(prepared / f"{split}_speech.pcm"),
            str(prepared / f"{split}_background.pcm"),
            str(prepared / f"{split}_foreground.pcm"),
            str(destination),
            str(count),
        ],
        check=True,
    )
    actual = destination.stat().st_size
    expected = count * BYTES_PER_SEQUENCE
    if actual != expected:
        raise SystemExit(
            f"unexpected size for {destination}: {actual}, expected {expected}"
        )
    print(f"verified {destination}: {actual} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_features", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-count", type=int, default=10_000)
    parser.add_argument("--eval-count", type=int, default=500)
    args = parser.parse_args()

    dump_features = args.dump_features.resolve()
    if not dump_features.is_file():
        parser.error(f"dump_features does not exist: {dump_features}")
    prepared = args.prepared.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generate(dump_features, prepared, output, "train", args.train_count)
    generate(dump_features, prepared, output, "eval", args.eval_count)


if __name__ == "__main__":
    main()
