import numpy as np

from rnnoise_mlx.tools.analyze_speech_onsets import onsets


def test_onset_requires_three_consecutive_active_frames():
    rate = 16_000
    samples = np.zeros(rate, dtype=np.float32)
    samples[round(.25 * rate):round(.35 * rate)] = .1
    assert onsets(samples, rate, [-40])["-40"] == .24


def test_onset_is_none_without_active_audio():
    assert onsets(np.zeros(16_000, dtype=np.float32), 16_000, [-40])["-40"] is None
