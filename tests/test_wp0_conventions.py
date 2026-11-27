"""WP0 Conventions and Normalization Verification Suite.

Validates:
1. Inverse-transform consistency: irfft(x_tilde / dt) == x.
2. Raw DFT covariance vs time-scaled DFT covariance (Group B/E normalization).
3. One-sided vs two-sided PSD relations across interior and endpoint bins.
4. Gaussian white noise normalization and strict RNG reproducibility.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference.conventions import (
    build_frequency_grid,
    continuous_rfft,
    expected_white_noise_psd,
    one_sided_periodogram,
)
from gw_inference.noise import generate_white_noise


class TestWP0Conventions:
    """Rigorous verification of Fourier transforms, covariances, and scaling."""

    def test_inverse_transform_consistency(self):
        """Verify that irfft(continuous_rfft(x, dt) / dt) reconstructs x exactly."""
        N = 1024
        fs = 2048.0
        dt = 1.0 / fs
        rng = np.random.default_rng(12345)
        x = rng.normal(size=N)

        # Forward continuous transform approximation
        x_tilde = continuous_rfft(x, dt)

        # Invert: raw unscaled rfft is x_tilde / dt
        x_rec = np.fft.irfft(x_tilde / dt, n=N)

        np.testing.assert_allclose(x_rec, x, rtol=1e-14, atol=1e-14)

    def test_raw_vs_time_scaled_covariance(self):
        """Verify separation between raw DFT covariance and time-scaled DFT covariance.

        For n_j ~ N(0, sigma^2) i.i.d.:
        Raw DFT:
            X_k = sum_{j=0}^{N-1} n_j e^{-i 2pi j k / N}
            E[|X_k|^2] = N * sigma^2   (for all k)
        Time-scaled continuous transform:
            n_tilde_k = dt * X_k
            E[|n_tilde_k|^2] = dt^2 * E[|X_k|^2] = dt^2 * N * sigma^2 = (N / fs^2) * sigma^2
        Since S_n(f) = 2 * sigma^2 * dt = 2 * sigma^2 / fs for interior bins:
            E[|n_tilde_k|^2] = (N / (2 * fs)) * S_n(f) = (T / 2) * S_n(f)
        """
        N = 512
        fs = 1024.0
        dt = 1.0 / fs
        sigma = 2.5e-21
        n_realizations = 10000

        rng = np.random.default_rng(777)
        raw_sq_sum = np.zeros(N // 2 + 1, dtype=np.float64)
        scaled_sq_sum = np.zeros(N // 2 + 1, dtype=np.float64)

        for _ in range(n_realizations):
            n = generate_white_noise(N, sigma=sigma, generator=rng)
            X = np.fft.rfft(n)
            n_tilde = continuous_rfft(n, dt)

            raw_sq_sum += np.abs(X) ** 2
            scaled_sq_sum += np.abs(n_tilde) ** 2

        emp_raw_cov = raw_sq_sum / n_realizations
        emp_scaled_cov = scaled_sq_sum / n_realizations

        # Theoretical values
        expected_raw_cov = N * (sigma**2)
        expected_scaled_cov = (dt**2) * N * (sigma**2)
        expected_psd = expected_white_noise_psd(sigma, dt, bin_type="interior")
        expected_scaled_from_psd = (N / (2.0 * fs)) * expected_psd

        # 1. Interior bins of raw covariance should match N * sigma^2
        interior_raw = emp_raw_cov[1:-1]
        np.testing.assert_allclose(
            np.mean(interior_raw), expected_raw_cov, rtol=0.03
        )

        # 2. Time-scaled covariance must equal dt^2 * raw covariance
        np.testing.assert_allclose(
            emp_scaled_cov, (dt**2) * emp_raw_cov, rtol=1e-15
        )

        # 3. Time-scaled covariance must match (T/2) * S_n
        np.testing.assert_allclose(
            np.mean(emp_scaled_cov[1:-1]), expected_scaled_from_psd, rtol=0.03
        )

    def test_one_sided_vs_two_sided_psd_relations(self):
        """Verify one-sided S_n(f) and two-sided S_n^(2)(f) relations.

        Two-sided discrete PSD for white noise: S_n^(2)(f) = sigma^2 * dt = sigma^2 / fs
        One-sided discrete PSD:
            S_n(f_k) = 2 * S_n^(2)(f_k) = 2 * sigma^2 / fs   (interior: 1 <= k < N/2)
            S_n(0) = S_n^(2)(0) = sigma^2 / fs               (DC: k = 0)
            S_n(f_Nyq) = S_n^(2)(f_Nyq) = sigma^2 / fs       (Nyquist: k = N/2)
        """
        sigma = 1.8e-21
        dt = 1.0 / 4096.0

        psd_interior = expected_white_noise_psd(sigma, dt, bin_type="interior")
        psd_dc = expected_white_noise_psd(sigma, dt, bin_type="dc")
        psd_nyquist = expected_white_noise_psd(sigma, dt, bin_type="nyquist")

        s2_theoretical = (sigma**2) * dt

        assert psd_interior == 2.0 * s2_theoretical
        assert psd_dc == s2_theoretical
        assert psd_nyquist == s2_theoretical

    def test_rng_reproducibility(self):
        """Verify bitwise RNG reproducibility with identical seeds."""
        N = 2048
        sigma = 1e-21
        seed = 424242

        n1 = generate_white_noise(N, sigma=sigma, seed=seed)
        n2 = generate_white_noise(N, sigma=sigma, seed=seed)
        np.testing.assert_array_equal(n1, n2)

        # Different seed produces different noise
        n3 = generate_white_noise(N, sigma=sigma, seed=seed + 1)
        assert not np.array_equal(n1, n3)
