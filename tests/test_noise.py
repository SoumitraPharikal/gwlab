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
from scripts.validate_white_noise_psd import run_ensemble_validation


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
        """Test ensemble mean over 200 realizations matches expectations within calibrated standard errors.

        Statistical Justification:
        --------------------------
        For discrete Gaussian white noise n_j ~ N(0, sigma^2):
        1. Interior bins (1 <= k < N/2) are complex Gaussian:
           |X_k|^2 ~ (N sigma^2 / 2) * chi^2_2 (exponential distribution).
           Var(S_hat(f_k)) = (E[S_hat(f_k)])^2 = S_interior^2.
           Over R independent realizations:
           SE(S_bar) = S_interior / sqrt(R).

        2. DC (k = 0) and Nyquist (k = N/2) are purely real Gaussian:
           X_0, X_{N/2} ~ N(0, N sigma^2) => |X|^2 ~ N sigma^2 * chi^2_1.
           Var(S_hat) = 2 * (E[S_hat])^2 = 2 * S_DC^2.
           Over R independent realizations:
           SE(S_bar) = sqrt(2) * S_DC / sqrt(R) = sqrt(2/R) * S_DC.

        For R = 200 realizations:
           - Interior relative SE: 1 / sqrt(200) approx 7.07%
           - DC / Nyquist relative SE: sqrt(2 / 200) = sqrt(1 / 100) = 10.00%
        """
        n_realizations = 200
        n_samples = 1024
        f_s = 4096.0
        delta_t = 1.0 / f_s
        sigma = 1.5

        s_interior_expected = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
        s_dc_expected = expected_white_noise_psd(sigma, delta_t, bin_type="dc")
        s_nyquist_expected = expected_white_noise_psd(sigma, delta_t, bin_type="nyquist")

        # Theoretical relative standard errors for R=200
        se_rel_dc_theoretical = math.sqrt(2.0 / n_realizations)  # exactly 0.10 (10%)
        se_rel_nyq_theoretical = math.sqrt(2.0 / n_realizations)  # exactly 0.10 (10%)

        rng = np.random.default_rng(888)
        psd_accum = np.zeros(n_samples // 2 + 1, dtype=np.float64)
        psd_sq_accum = np.zeros(n_samples // 2 + 1, dtype=np.float64)

        for _ in range(n_realizations):
            noise = generate_white_noise(n_samples, sigma, generator=rng)
            psd = one_sided_periodogram(noise, delta_t)
            psd_accum += psd
            psd_sq_accum += psd ** 2

        psd_mean = psd_accum / n_realizations
        psd_var = (psd_sq_accum - n_realizations * (psd_mean ** 2)) / (n_realizations - 1)
        psd_sem = np.sqrt(psd_var / n_realizations)

        # 1. Interior bins average across ~500 bins and 200 realizations:
        interior_mean = float(np.mean(psd_mean[1:-1]))
        rel_diff_interior = abs(interior_mean - s_interior_expected) / s_interior_expected
        assert rel_diff_interior < 0.02, f"Interior PSD relative error {rel_diff_interior:.4f}"

        # 2. DC bin check against theoretical and empirical standard error:
        rel_diff_dc = abs(psd_mean[0] - s_dc_expected) / s_dc_expected
        z_dc_theoretical = rel_diff_dc / se_rel_dc_theoretical
        z_dc_empirical = abs(psd_mean[0] - s_dc_expected) / psd_sem[0]

        # 3.5-sigma acceptance bound (p approx 0.000465)
        assert z_dc_theoretical < 3.5, f"DC z-score (theoretical) too large: {z_dc_theoretical:.2f}"
        assert z_dc_empirical < 3.5, f"DC z-score (empirical) too large: {z_dc_empirical:.2f}"
        assert rel_diff_dc < 3.5 * se_rel_dc_theoretical  # < 0.35

        # 3. Nyquist bin check against theoretical and empirical standard error:
        rel_diff_nyq = abs(psd_mean[-1] - s_nyquist_expected) / s_nyquist_expected
        z_nyq_theoretical = rel_diff_nyq / se_rel_nyq_theoretical
        z_nyq_empirical = abs(psd_mean[-1] - s_nyquist_expected) / psd_sem[-1]

        assert z_nyq_theoretical < 3.5, f"Nyquist z-score (theoretical) too large: {z_nyq_theoretical:.2f}"
        assert z_nyq_empirical < 3.5, f"Nyquist z-score (empirical) too large: {z_nyq_empirical:.2f}"
        assert rel_diff_nyq < 3.5 * se_rel_nyq_theoretical  # < 0.35


class TestValidationScriptEdgeCases:
    """Validate input parameter bounds in the Monte Carlo validation script."""

    def test_invalid_n_realizations(self):
        with pytest.raises(ValueError, match="n_realizations must be an integer > 1"):
            run_ensemble_validation(n_realizations=1)
        with pytest.raises(ValueError, match="n_realizations must be an integer > 1"):
            run_ensemble_validation(n_realizations=0)
        with pytest.raises(ValueError, match="n_realizations must be an integer > 1"):
            run_ensemble_validation(n_realizations=-10)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            run_ensemble_validation(n_samples=0)
        with pytest.raises(ValueError, match="n_samples must be an even integer"):
            run_ensemble_validation(n_samples=1023)

    def test_invalid_f_s(self):
        with pytest.raises(ValueError, match="f_s must be a positive finite float"):
            run_ensemble_validation(f_s=0.0)
        with pytest.raises(ValueError, match="f_s must be a positive finite float"):
            run_ensemble_validation(f_s=-4096.0)
        with pytest.raises(ValueError, match="f_s must be a positive finite float"):
            run_ensemble_validation(f_s=float("nan"))

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma must be a non-negative finite float"):
            run_ensemble_validation(sigma=-1.0)
        with pytest.raises(ValueError, match="sigma must be a non-negative finite float"):
            run_ensemble_validation(sigma=float("nan"))
