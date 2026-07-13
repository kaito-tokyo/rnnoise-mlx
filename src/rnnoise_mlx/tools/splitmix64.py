"""Stable SplitMix64 primitives used by reproducible data selection."""

from __future__ import annotations

from dataclasses import dataclass


MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


@dataclass
class SplitMix64:
    """Small, fixed 64-bit SplitMix64 generator."""

    state: int
    calls: int = 0

    def __post_init__(self) -> None:
        self.state &= MASK64

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        self.calls += 1
        return (value ^ (value >> 31)) & MASK64


def uniform_below(rng: SplitMix64, upper: int) -> int:
    """Map one u64 draw into ``range(upper)`` without rejection."""

    if not 1 <= upper <= 1 << 64:
        raise ValueError("upper must be between 1 and 2^64")
    return (rng.next_u64() * upper) >> 64
