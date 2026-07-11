#!/usr/bin/env python3
"""Create deterministic, file-disjoint train/eval noise manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def split_for(path: str, eval_fraction: float, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}\0{path}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "eval" if value < eval_fraction else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("classification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--salt", default="rnnoise-mlx-musan-v1")
    parser.add_argument("--include", nargs="+", default=["background", "foreground"])
    args = parser.parse_args()
    if not 0 < args.eval_fraction < 1:
        parser.error("--eval-fraction must be between zero and one")

    with args.classification.open(newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["label"] in args.include]
        fields = list(rows[0]) + ["split"] if rows else ["path", "label", "split"]
    for row in rows:
        row["split"] = split_for(row["path"], args.eval_fraction, args.salt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
