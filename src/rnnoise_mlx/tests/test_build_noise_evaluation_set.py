import numpy as np

from rnnoise_mlx.tools.build_noise_evaluation_set import mix_at_snr


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
