#!/usr/bin/env python3
"""Build split/role PCM files from a classified noise manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def ffconcat_quote(path: Path) -> str:
    return "file '" + str(path).replace("'", "'\\''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    with args.manifest.open(newline="") as source:
        rows = list(csv.DictReader(source))
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval"):
        for role in ("background", "foreground"):
            selected = sorted(
                (args.audio_root / row["path"]).resolve()
                for row in rows
                if row["split"] == split and row["label"] == role
            )
            if not selected:
                raise ValueError(f"no files for {split}/{role}")
            concat = args.output / f"{split}-{role}.ffconcat"
            concat.write_text("ffconcat version 1.0\n" + "\n".join(map(ffconcat_quote, selected)) + "\n")
            pcm = args.output / f"{split}-{role}.pcm"
            subprocess.run(
                [args.ffmpeg, "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                 "-ar", "48000", "-ac", "1", "-f", "s16le", "-y", str(pcm)],
                check=True,
            )
            print(f"{split}/{role}: {len(selected)} files -> {pcm}")


if __name__ == "__main__":
    main()
