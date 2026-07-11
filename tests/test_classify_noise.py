import importlib.util
from pathlib import Path
import sys

import numpy as np

SPEC = importlib.util.spec_from_file_location("classify_noise", Path(__file__).parents[1] / "scripts/classify_noise.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stationary_noise_scores_more_stationary_than_impulses():
    rng = np.random.default_rng(7)
    stationary = rng.normal(0, 0.05, 48000 * 4).astype(np.float32)
    impulses = np.zeros(48000 * 4, dtype=np.float32)
    impulses[np.arange(12000, impulses.size, 24000)] = 0.9
    impulses += rng.normal(0, 0.0001, impulses.size).astype(np.float32)
    bg = MODULE.analyze(stationary, 48000)
    fg = MODULE.analyze(impulses, 48000)
    assert bg.stationarity_score > fg.stationarity_score
    assert fg.transient_score > bg.transient_score


def test_short_stationary_clip_is_not_background():
    metrics = MODULE.analyze(np.full(48000, 0.1, dtype=np.float32), 48000)
    assert MODULE.label(metrics, 0.23, 45.0, 10.0) != "background"
