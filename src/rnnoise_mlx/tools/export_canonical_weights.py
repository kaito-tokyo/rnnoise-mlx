"""Normalize RNNoise weights into the canonical deployment SafeTensors schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from .rnnoise_weights import read_weights, write_weights


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    """Export a validated, backend-independent deployment artifact."""
    weights, config = read_weights(checkpoint)
    write_weights(output, weights, config["cond_size"], config["gru_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    try:
        export_checkpoint(args.checkpoint, args.output)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
