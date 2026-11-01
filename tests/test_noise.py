"""Unit tests for synthetic Gaussian white-noise generation.

Stage 1.1 Test Suite: Noise Generator and Statistical Properties.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference.conventions import (
    expected_white_noise_psd,
    one_sided_periodogram,
)
from gw_inference.noise import generate_white_noise


class TestWhiteNoiseGenerator:
    """Test suite for generate_white_noise function."""

    def test_sample_count_and_dtype(self):
        n_samples = 32768
        sigma = 1.0
        data = generate_white_noise(n_samples=n_samples, sigma=sigma, seed=42)

        assert isinstance(data, np.ndarray)
        assert data.ndim == 1
        assert len(data) == n_samples
        assert data.dtype == np.float64

    def test_output_finite(self):
        data = generate_white_noise(n_samples=10000, sigma=2.5, seed=123)
        assert np.all(np.isfinite(data))

    def test_reproducibility_identical_seeds(self):
        """Identical seeds must yield bitwise identical output arrays."""
        seed = 987654321
        run1 = generate_white_noise(n_samples=4096, sigma=1.2, seed=seed)
        run2 = generate_white_noise(n_samples=4096, sigma=1.2, seed=seed)

        np.testing.assert_array_equal(run1, run2)

    def test_different_seeds_produce_different_realizations(self):
        run1 = generate_white_noise(n_samples=4096, sigma=1.0, seed=100)
        run2 = generate_white_noise(n_samples=4096, sigma=1.0, seed=200)

        assert not np.array_equal(run1, run2)

    def test_generator_instance_support(self):
        """Passing an explicit Generator instance operates correctly."""
        rng = np.random.default_rng(555)
        data = generate_white_noise(n_samples=1024, sigma=1.0, generator=rng)
        assert len(data) == 1024

    def test_zero_sigma(self):
        data = generate_white_noise(n_samples=100, sigma=0.0, seed=42)
        np.testing.assert_array_equal(data, np.zeros(100))

    def test_sample_statistics_convergence(self):
        """For large sample size, sample mean and sample variance must approach population values."""
        n_samples = 200000
        sigma = 2.0
        seed = 2026
        data = generate_white_noise(n_samples=n_samples, sigma=sigma, seed=seed)

        sample_mean = float(np.mean(data))
        sample_var = float(np.var(data, ddof=1))

        # Standard error of mean: sigma / sqrt(N) = 2.0 / sqrt(200000) approx 0.00447
        se_mean = sigma / math.sqrt(n_samples)
        assert abs(sample_mean) < 4.0 * se_mean, f"Sample mean {sample_mean} too far from 0"

        # Standard error of sample variance for normal: sigma^2 * sqrt(2 / (N - 1))
        se_var = (sigma ** 2) * math.sqrt(2.0 / (n_samples - 1))
        assert (
            abs(sample_var - sigma ** 2) < 4.0 * se_var
        ), f"Sample variance {sample_var} too far from {sigma**2}"

    def test_input_validation(self):
        with pytest.raises(ValueError, match="positive integer"):
            generate_white_noise(n_samples=0, sigma=1.0)
        with pytest.raises(ValueError, match="positive integer"):
            generate_white_noise(n_samples=-5, sigma=1.0)
        with pytest.raises(ValueError, match="positive integer"):
            generate_white_noise(n_samples="1000", sigma=1.0)  # type: ignore
        with pytest.raises(ValueError, match="non-negative finite float"):
            generate_white_noise(n_samples=100, sigma=-1.0)
        with pytest.raises(ValueError, match="non-negative finite float"):
            generate_white_noise(n_samples=100, sigma=float("nan"))
        with pytest.raises(ValueError, match="non-negative finite float"):
            generate_white_noise(n_samples=100, sigma=float("inf"))
        with pytest.raises(TypeError, match="instance of np.random.Generator"):
            generate_white_noise(n_samples=100, sigma=1.0, generator="not_a_generator")  # type: ignore


class TestEnsemblePeriodogramMini:
    """Fast automated ensemble test of periodogram normalization."""

    def test_ensemble_periodogram_expectation(self):
        """Test ensemble mean over 200 realizations matches S_0 within 3 standard errors."""
        n_realizations = 200
        n_samples = 1024
        f_s = 4096.0
        delta_t = 1.0 / f_s
        sigma = 1.5

        s_interior_expected = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
        s_dc_expected = expected_white_noise_psd(sigma, delta_t, bin_type="dc")
        s_nyquist_expected = expected_white_noise_psd(sigma, delta_t, bin_type="nyquist")

        rng = np.random.default_rng(888)
        psd_accum = np.zeros(n_samples // 2 + 1, dtype=np.float64)

        for _ in range(n_realizations):
            noise = generate_white_noise(n_samples, sigma, generator=rng)
            psd = one_sided_periodogram(noise, delta_t)
            psd_accum += psd

        psd_mean = psd_accum / n_realizations

        # Interior bins average:
        interior_mean = np.mean(psd_mean[1:-1])
        rel_diff_interior = abs(interior_mean - s_interior_expected) / s_interior_expected
        # For band-averaged across ~500 bins and 200 realizations, relative error is << 1%
        assert rel_diff_interior < 0.02, f"Interior PSD relative error {rel_diff_interior:.4f}"

        # DC and Nyquist are single bins: standard error is S_expected / sqrt(n_realizations) = S / sqrt(200) ~ 7%
        rel_diff_dc = abs(psd_mean[0] - s_dc_expected) / s_dc_expected
        rel_diff_nyq = abs(psd_mean[-1] - s_nyquist_expected) / s_nyquist_expected

        # Allow 4 standard errors (~ 4 * 0.0707 = 0.28)
        assert rel_diff_dc < 0.28, f"DC PSD relative error {rel_diff_dc:.4f}"
        assert rel_diff_nyq < 0.28, f"Nyquist PSD relative error {rel_diff_nyq:.4f}"
