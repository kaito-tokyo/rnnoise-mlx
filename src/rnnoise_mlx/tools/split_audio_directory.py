"""Create a deterministic train/eval symlink split with a provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rnnoise_mlx.tools.prepare_speech_mix import AUDIO_SUFFIXES


def assignment(relative_path: str, eval_fraction: float, seed: int) -> str:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")
    digest = hashlib.sha256(f"{seed}:{relative_path}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "eval" if value < eval_fraction else "train"


def split(source: Path, output: Path, eval_fraction: float, seed: int) -> dict[str, object]:
    source = source.resolve()
    paths = sorted(
        path for path in source.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no audio files in {source}")
    records = []
    counts = {"train": 0, "eval": 0}
    for path in paths:
        relative = path.relative_to(source).as_posix()
        target_split = assignment(relative, eval_fraction, seed)
        destination = output / target_split / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(path)
        counts[target_split] += 1
        records.append({"path": relative, "split": target_split})
    manifest = {
        "format_version": 1,
        "algorithm": "sha256-seeded-threshold-v1",
        "source": str(source),
        "seed": seed,
        "eval_fraction": eval_fraction,
        "counts": counts,
        "files": records,
    }
    (output / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(split(args.source, args.output, args.eval_fraction, args.seed), indent=2))


if __name__ == "__main__":
    main()
