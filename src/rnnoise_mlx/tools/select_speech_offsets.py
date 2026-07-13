"""Select reproducible fixed-length windows from one concatenated s16le PCM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .splitmix64 import MASK64, SplitMix64, uniform_below


ALGORITHM = "splitmix64-multiply-high-offset-v1"
DEFAULT_SEQUENCE_SAMPLES = 2000 * 480


def select_offsets(
    total_samples: int,
    sequence_samples: int,
    count: int,
    seed: int,
) -> list[int]:
    if sequence_samples < 1:
        raise ValueError("sequence samples must be at least 1")
    if total_samples < sequence_samples:
        raise ValueError("PCM is shorter than one sequence")
    if count < 1:
        raise ValueError("count must be at least 1")
    start_count = total_samples - sequence_samples + 1
    rng = SplitMix64(seed)
    return [uniform_below(rng, start_count) for _ in range(count)]


def generate(pcm: Path, output: Path, count: int, seed: int, sequence_samples: int) -> None:
    size = pcm.stat().st_size
    if size % 2:
        raise ValueError(f"PCM byte size must be even: {pcm}")
    total_samples = size // 2
    offsets = select_offsets(total_samples, sequence_samples, count, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{offset}\n" for offset in offsets), encoding="ascii")
    metadata = {
        "format_version": 1,
        "algorithm": ALGORITHM,
        "pcm": str(pcm.resolve()),
        "pcm_bytes": size,
        "total_samples": total_samples,
        "sequence_samples": sequence_samples,
        "count": count,
        "seed": seed & MASK64,
        "rng_calls": count,
        "offset_unit": "int16_sample",
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcm", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sequence-samples", type=int, default=DEFAULT_SEQUENCE_SAMPLES)
    args = parser.parse_args()
    try:
        generate(args.pcm, args.output, args.count, args.seed, args.sequence_samples)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
