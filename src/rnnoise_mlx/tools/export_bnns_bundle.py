"""Compatibility wrapper for :mod:`export_canonical_weights`."""

from __future__ import annotations

from .export_canonical_weights import export_checkpoint, main

__all__ = ["export_checkpoint", "main"]

if __name__ == "__main__":
    main()
