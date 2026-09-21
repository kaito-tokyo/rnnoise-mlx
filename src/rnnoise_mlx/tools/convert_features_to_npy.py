"""Convert RNNoise raw float32 features to a sequence-shaped NumPy file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FRAME_DIM = 98


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, destination: Path, sequence_length: int) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if sequence_length < 5:
        raise ValueError("sequence_length must be at least 5")
    source_bytes = source.stat().st_size
    bytes_per_sequence = sequence_length * FRAME_DIM * np.dtype("<f4").itemsize
    if source_bytes % bytes_per_sequence:
        raise ValueError(
            f"{source} is not an integral number of {sequence_length}-frame sequences"
        )
    sequence_count = source_bytes // bytes_per_sequence
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)

    source_data = np.memmap(source, dtype="<f4", mode="r")
    source_data = source_data.reshape(sequence_count, sequence_length, FRAME_DIM)
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype="<f4",
        shape=source_data.shape,
    )
    # Copy in bounded chunks so conversion does not require a second full-size
    # RAM allocation on the Mac mini.
    for start in range(0, sequence_count, 64):
        output[start : start + 64] = source_data[start : start + 64]
    output.flush()
    del output
    del source_data

    metadata = {
        "format_version": 1,
        "kind": "rnnoise-training-features-npy",
        "source": {
            "filename": source.name,
            "bytes": source_bytes,
            "sha256": _sha256(source),
        },
        "output": {
            "filename": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        },
        "dtype": "<f4",
        "shape": [sequence_count, sequence_length, FRAME_DIM],
    }
    destination.with_name(destination.name + ".manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sequence-length", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.destination, args.sequence_length), indent=2))


if __name__ == "__main__":
    main()
