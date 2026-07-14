"""Generate and size-check RNNoise train/evaluation feature files."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time


BYTES_PER_SEQUENCE = 2000 * 98 * 4


def generate(
    dump_features: Path,
    prepared: Path,
    output: Path,
    split: str,
    count: int,
    speech_offsets: Path | None = None,
    progress_interval: float = 10.0,
    disable_foreground: bool = False,
) -> None:
    destination = output / f"{split}.f32"
    options = []
    if speech_offsets is not None:
        options = ["-speech_offsets", str(speech_offsets)]
    if disable_foreground:
        options.append("-disable_foreground")
    command = [
        str(dump_features),
        "-rir_list",
        str(prepared / f"{split}_rir_list.txt"),
        *options,
        str(prepared / f"{split}_speech.pcm"),
        str(prepared / f"{split}_background.pcm"),
        str(prepared / f"{split}_foreground.pcm"),
        str(destination),
        str(count),
    ]
    started = time.monotonic()
    print(f"generating {split}: 0/{count} sequences", flush=True)
    process = subprocess.Popen(command)
    while True:
        try:
            return_code = process.wait(timeout=progress_interval)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            completed = min(
                destination.stat().st_size // BYTES_PER_SEQUENCE
                if destination.exists()
                else 0,
                count,
            )
            rate = completed / elapsed if elapsed else 0.0
            eta = (count - completed) / rate if rate else None
            eta_text = f"{eta:.0f}s" if eta is not None else "unknown"
            print(
                f"generating {split}: {completed}/{count} sequences "
                f"({100 * completed / count:.1f}%), elapsed {elapsed:.0f}s, "
                f"{rate:.2f} sequences/s, ETA {eta_text}",
                flush=True,
            )
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
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
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="seconds between progress reports (default: 10)",
    )
    parser.add_argument(
        "--speech-offsets",
        type=Path,
        help="directory containing train.txt and eval.txt sample-offset manifests",
    )
    parser.add_argument(
        "--disable-foreground",
        action="store_true",
        help=(
            "disable foreground-speech augmentation while preserving background-noise "
            "and RIR augmentation; intended for conservative corpus-cleaner training"
        ),
    )
    args = parser.parse_args()
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")

    dump_features = args.dump_features.resolve()
    if not dump_features.is_file():
        parser.error(f"dump_features does not exist: {dump_features}")
    prepared = args.prepared.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    offsets = args.speech_offsets.resolve() if args.speech_offsets else None
    generate(
        dump_features, prepared, output, "train", args.train_count,
        offsets / "train.txt" if offsets else None,
        args.progress_interval,
        args.disable_foreground,
    )
    generate(
        dump_features, prepared, output, "eval", args.eval_count,
        offsets / "eval.txt" if offsets else None,
        args.progress_interval,
        args.disable_foreground,
    )


if __name__ == "__main__":
    main()
