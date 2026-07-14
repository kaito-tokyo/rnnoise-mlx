import numpy as np

from rnnoise_mlx.tools.audit_corpus_audio_quality import diagnostic_flags, edge_low_rms_seconds, measure_pcm16


def test_measure_pcm16_detects_clipping_and_dc():
    samples = np.array([32767, 32767, 1000, 1000], dtype=np.int16)
    result = measure_pcm16(samples, 2)
    assert result["duration_seconds"] == 2
    assert result["hard_clipped_sample_count"] == 2
    assert result["hard_clipped_fraction"] == 0.5
    assert result["dc_offset_fraction"] > 0
    assert "hard_clipping" in diagnostic_flags(result)


def test_edge_low_rms_seconds_counts_complete_frames():
    rate = 1000
    samples = np.concatenate(
        [
            np.zeros(20, dtype=np.int16),
            np.full(20, 10000, dtype=np.int16),
            np.zeros(30, dtype=np.int16),
        ]
    )
    assert edge_low_rms_seconds(samples, rate) == (0.02, 0.03)


def test_measure_retains_edge_padding():
    samples = np.concatenate(
        [
            np.zeros(500, dtype=np.int16),
            np.full(1000, 10000, dtype=np.int16),
            np.zeros(500, dtype=np.int16),
        ]
    )
    result = measure_pcm16(samples, 1000)
    assert result["duration_seconds"] == 2
    assert result["edge_trimmed_duration_seconds"] == 1.6
