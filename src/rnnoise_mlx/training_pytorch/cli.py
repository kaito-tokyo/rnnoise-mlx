"""Command-line interface for PyTorch RNNoise training."""

import argparse
from pathlib import Path

from .train import train

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("features", type=Path)
    p.add_argument("output", type=Path, help="new output directory")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sequence-length", type=int, default=2000)
    p.add_argument("--segmented-tbptt-length", type=int, default=250)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=141)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--checkpoint-every", type=int, default=32)
    p.add_argument("--feature-identity")
    p.add_argument("--carry-between-updates", action=argparse.BooleanOptionalAction, default=True)
    initial = p.add_mutually_exclusive_group()
    initial.add_argument("--init-weights", type=Path)
    initial.add_argument("--resume-from", type=Path)
    return p

def main():
    train(parser().parse_args())

if __name__ == "__main__":
    main()
