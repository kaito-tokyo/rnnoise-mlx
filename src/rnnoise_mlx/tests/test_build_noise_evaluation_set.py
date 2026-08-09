import hashlib

import numpy as np

from rnnoise_mlx.tools.build_noise_evaluation_set import (
    evaluation_groups,
    mix_at_snr,
    output_record,
)


def test_mix_at_snr_is_exact_and_avoids_clipping():
    clean = np.linspace(-0.2, 0.2, 48_000)
    noise = np.sin(np.linspace(0, 100, 48_000))
    clean_scaled, noise_scaled, mixture = mix_at_snr(clean, noise, 10)
    measured = 20 * np.log10(
        np.sqrt(np.mean(clean_scaled * clean_scaled))
        / np.sqrt(np.mean(noise_scaled * noise_scaled))
    )
    np.testing.assert_allclose(measured, 10, atol=1e-9)
    np.testing.assert_allclose(mixture, clean_scaled + noise_scaled)
    assert np.max(np.abs(mixture)) <= 0.99 + 1e-12


def test_mix_at_snr_scales_large_component_even_when_mixture_cancels():
    clean = np.full(100, 0.9)
    noise = np.full(100, -0.9)
    clean_scaled, noise_scaled, mixture = mix_at_snr(clean, noise, 0)
    assert max(
        np.max(np.abs(clean_scaled)),
        np.max(np.abs(noise_scaled)),
        np.max(np.abs(mixture)),
    ) <= 0.99 + 1e-12
    np.testing.assert_allclose(mixture, clean_scaled + noise_scaled)


def test_evaluation_groups_excludes_undecodable_rejected_musan():
    records = []
    categories = (
        "dns_typing", "dns_door", "dns_squeak", "dns_dragging",
        "dns_copy-machine", "dns_human", "dns_fan",
    )
    for category in categories:
        records.append({
            "accepted": True,
            "split": "eval",
            "category": category,
            "source": "dns",
            "relative_path": f"{category}.wav",
        })
    for source in (
        "mka-lenovo", "mka-msi", "mka-mac", "mka-messenger", "mka-zoom",
        "mka-hp", "multi-pressure-highp", "multi-pressure-mediump",
        "multi-pressure-lowp",
    ):
        records.append({
            "accepted": True,
            "split": "eval",
            "category": "mka" if source.startswith("mka-") else "multi_pressure",
            "source": source,
            "relative_path": f"{source}.wav",
        })
    records.extend([
        {
            "accepted": False,
            "category": "musan_curated",
            "relative_path": "broken.wav",
            "exclusion_reasons": ["decode_error"],
        },
        {
            "accepted": False,
            "category": "musan_curated",
            "relative_path": "quiet.wav",
            "exclusion_reasons": ["activity_below_90pct"],
        },
    ])
    selected = evaluation_groups({"records": records})
    assert selected["musan-rejected"]["relative_path"] == "quiet.wav"


def test_output_record_binds_evaluation_wav_bytes(tmp_path):
    path = tmp_path / "case.wav"
    path.write_bytes(b"wav-bytes")
    record = output_record(path)
    assert record == {
        "path": str(path.resolve()),
        "bytes": 9,
        "sha256": hashlib.sha256(b"wav-bytes").hexdigest(),
    }
