"""Convert a deterministic LibriTTS-R subset to RNNoise input PCM."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


def selection_score(path: Path) -> bytes:
    return hashlib.sha256(f"141:{path.as_posix()}".encode()).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("libritts_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clip-count", type=int, default=1000)
    args = parser.parse_args()

    root = args.libritts_root.resolve()
    output = args.output.resolve()
    clips = sorted(root.rglob("*.wav"), key=selection_score)[: args.clip_count]
    if not clips:
        parser.error(f"no WAV files found below {root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".clips.txt")
    silence = bytes(9600)
    with output.open("wb") as destination:
        for clip in clips:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(clip),
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    "-f",
                    "s16le",
                    "-",
                ],
                check=True,
                stdout=destination,
            )
            destination.write(silence)
    manifest.write_text("".join(f"{clip}\n" for clip in clips))
    print(f"converted {len(clips)} clips to {output}")


if __name__ == "__main__":
    main()
