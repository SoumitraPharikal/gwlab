"""Unit tests for discrete Fourier transform conventions, scaling, and Parseval identity.

Stage 1.1 Test Suite: Conventions and Mathematical Validation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference.conventions import (
    build_frequency_grid,
    continuous_rfft,
    discrete_parseval_freq_energy,
    discrete_parseval_time_energy,
    expected_white_noise_psd,
    one_sided_periodogram,
    validate_parseval_identity,
)
from gw_inference.noise import generate_white_noise


class TestFrequencyGrid:
    """Test frequency grid construction against Stage 0 conventions."""

    def test_baseline_grid_properties(self):
        """Verify baseline parameters: N=32768, f_s=4096 Hz, df=0.125 Hz."""
        n_samples = 32768
        f_s = 4096.0
        delta_t = 1.0 / f_s
        expected_df = 0.125
        expected_nyquist = 2048.0
        expected_len = n_samples // 2 + 1  # 16385

        grid = build_frequency_grid(n_samples, delta_t)

        assert len(grid) == expected_len
        assert grid[0] == 0.0  # DC
        assert math.isclose(grid[-1], expected_nyquist, rel_tol=1e-14)  # Nyquist
        assert math.isclose(grid[1] - grid[0], expected_df, rel_tol=1e-14)  # df
        # Check uniform spacing
        diffs = np.diff(grid)
        np.testing.assert_allclose(diffs, expected_df, rtol=1e-14)

    @pytest.mark.parametrize("n_samples", [10, 64, 1024, 8192])
    def test_grid_length_and_endpoints(self, n_samples):
        delta_t = 0.001
        grid = build_frequency_grid(n_samples, delta_t)
        assert len(grid) == n_samples // 2 + 1
        assert grid[0] == 0.0
        assert math.isclose(grid[-1], 0.5 / delta_t, rel_tol=1e-14)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="positive integer"):
            build_frequency_grid(0, 0.001)
        with pytest.raises(ValueError, match="positive integer"):
            build_frequency_grid(-100, 0.001)
        with pytest.raises(ValueError, match="even integer"):
            build_frequency_grid(101, 0.001)
        with pytest.raises(ValueError, match="positive finite float"):
            build_frequency_grid(100, -0.001)
        with pytest.raises(ValueError, match="positive finite float"):
            build_frequency_grid(100, float("nan"))


class TestContinuousRFFTScaling:
    """Test continuous Fourier transform approximation scaling: x_tilde(f_k) = dt * X_k."""

    def test_impulse_response(self):
        """Discrete delta at origin x = [1, 0, ..., 0] has continuous FT x_tilde(f) = dt."""
        n_samples = 1024
        delta_t = 1.0 / 4096.0
        x = np.zeros(n_samples, dtype=np.float64)
        x[0] = 1.0

        x_tilde = continuous_rfft(x, delta_t)
        expected = np.full(n_samples // 2 + 1, delta_t, dtype=np.complex128)
        np.testing.assert_allclose(x_tilde, expected, rtol=1e-14)

    def test_unnormalised_rfft_scaling_factor(self):
        """Verify that continuous_rfft strictly applies delta_t and no extra 1/N factor."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=512)
        delta_t = 0.0025

        raw_rfft = np.fft.rfft(x)
        scaled_rfft = continuous_rfft(x, delta_t)

        np.testing.assert_allclose(scaled_rfft, delta_t * raw_rfft, rtol=1e-15)

    def test_input_validation(self):
        with pytest.raises(ValueError, match="1D array"):
            continuous_rfft(np.zeros((10, 10)), 0.001)
        with pytest.raises(ValueError, match="even length"):
            continuous_rfft(np.zeros(11), 0.001)
        with pytest.raises(ValueError, match="real-valued"):
            continuous_rfft(np.zeros(10, dtype=complex), 0.001)
        with pytest.raises(ValueError, match="finite numbers"):
            continuous_rfft(np.array([1.0, float("nan")]), 0.001)


class TestParsevalIdentity:
    """Validate discrete Parseval energy balance across deterministic and random signals.

    Identity:
        dt * sum |x_j|^2 = df * dt^2 * [|X_0|^2 + 2 * sum_{k=1}^{N/2-1} |X_k|^2 + |X_{N/2}|^2]
    Relative numerical error must be below 10^{-12}.
    """

    @pytest.mark.parametrize(
        "signal_type",
        [
            "unit_impulse_origin",
            "shifted_impulse",
            "pure_dc",
            "pure_nyquist",
            "single_tone",
            "multitone",
        ],
    )
    def test_deterministic_signals(self, signal_type):
        n_samples = 2048
        f_s = 4096.0
        delta_t = 1.0 / f_s
        t = np.arange(n_samples) * delta_t
        delta_f = 1.0 / (n_samples * delta_t)

        if signal_type == "unit_impulse_origin":
            x = np.zeros(n_samples)
            x[0] = 1.0
        elif signal_type == "shifted_impulse":
            x = np.zeros(n_samples)
            x[123] = 2.5
        elif signal_type == "pure_dc":
            x = np.full(n_samples, 3.7)
        elif signal_type == "pure_nyquist":
            x = np.array([(-1.0) ** j * 1.8 for j in range(n_samples)])
        elif signal_type == "single_tone":
            f0 = 50 * delta_f
            x = 2.0 * np.cos(2.0 * np.pi * f0 * t + 0.3)
        elif signal_type == "multitone":
            x = (
                1.5 * np.cos(2.0 * np.pi * 10 * delta_f * t)
                + 0.8 * np.sin(2.0 * np.pi * 100 * delta_f * t)
                + 2.1 * np.cos(2.0 * np.pi * 500 * delta_f * t + 1.1)
            )

        e_time, e_freq, rel_err = validate_parseval_identity(x, delta_t)
        assert rel_err < 1e-12, f"Failed for {signal_type}: rel_err={rel_err:.2e}"
        assert math.isclose(e_time, e_freq, rel_tol=1e-12)

    @pytest.mark.parametrize("n_samples", [128, 1024, 8192, 32768])
    def test_random_gaussian_signals(self, n_samples):
        """Test on baseline-scale and varied-scale random Gaussian sequences."""
        rng = np.random.default_rng(12345 + n_samples)
        delta_t = 1.0 / 4096.0
        x = rng.normal(loc=0.0, scale=1.5, size=n_samples)

        e_time, e_freq, rel_err = validate_parseval_identity(x, delta_t)
        assert rel_err < 1e-12, f"Failed for N={n_samples}: rel_err={rel_err:.2e}"

    def test_dc_bin_weighting(self):
        """Verify DC bin (k=0) must NOT be doubled."""
        n_samples = 512
        delta_t = 0.01
        delta_f = 1.0 / (n_samples * delta_t)
        c_val = 4.2
        x = np.full(n_samples, c_val)

        X = np.fft.rfft(x)
        assert math.isclose(np.abs(X[0]), n_samples * c_val, rel_tol=1e-14)
        np.testing.assert_allclose(X[1:], 0.0, atol=1e-10)

        e_time = discrete_parseval_time_energy(x, delta_t)
        e_freq_correct = discrete_parseval_freq_energy(X, delta_t, delta_f)
        assert abs(e_time - e_freq_correct) / e_time < 1e-12

        # If DC were doubled, error would be 100%
        e_freq_erroneous = delta_f * (delta_t ** 2) * (2.0 * np.abs(X[0]) ** 2)
        assert math.isclose(e_freq_erroneous, 2.0 * e_time, rel_tol=1e-12)

    def test_nyquist_bin_weighting(self):
        """Verify Nyquist bin (k=N/2) must NOT be doubled."""
        n_samples = 512
        delta_t = 0.01
        delta_f = 1.0 / (n_samples * delta_t)
        c_val = 3.5
        x = np.array([(-1.0) ** j * c_val for j in range(n_samples)])

        X = np.fft.rfft(x)
        assert math.isclose(np.abs(X[-1]), n_samples * c_val, rel_tol=1e-14)
        np.testing.assert_allclose(X[:-1], 0.0, atol=1e-10)

        e_time = discrete_parseval_time_energy(x, delta_t)
        e_freq_correct = discrete_parseval_freq_energy(X, delta_t, delta_f)
        assert abs(e_time - e_freq_correct) / e_time < 1e-12

        # If Nyquist were doubled, error would be 100%
        e_freq_erroneous = delta_f * (delta_t ** 2) * (2.0 * np.abs(X[-1]) ** 2)
        assert math.isclose(e_freq_erroneous, 2.0 * e_time, rel_tol=1e-12)

    def test_interior_bin_weighting(self):
        """Verify interior bins (1 <= k < N/2) MUST be doubled due to real-FFT folding."""
        n_samples = 512
        delta_t = 0.01
        delta_f = 1.0 / (n_samples * delta_t)
        k_target = 15
        t = np.arange(n_samples) * delta_t
        x = 2.0 * np.cos(2.0 * np.pi * k_target * delta_f * t)

        X = np.fft.rfft(x)
        e_time = discrete_parseval_time_energy(x, delta_t)
        e_freq_correct = discrete_parseval_freq_energy(X, delta_t, delta_f)
        assert abs(e_time - e_freq_correct) / e_time < 1e-12

        # If interior were undoubled, error would be 50%
        e_freq_undoubled = delta_f * (delta_t ** 2) * np.sum(np.abs(X) ** 2)
        assert math.isclose(e_freq_undoubled, 0.5 * e_time, rel_tol=1e-4)


class TestOneSidedPeriodogram:
    """Test one-sided periodogram normalization and total energy integral."""

    def test_periodogram_integral_equals_time_mean_square(self):
        """Verify integral property: sum_{k=0}^{N/2} S_n(f_k) df = (1/N) sum |x_j|^2."""
        rng = np.random.default_rng(999)
        n_samples = 4096
        delta_t = 1.0 / 4096.0
        delta_f = 1.0 / (n_samples * delta_t)
        x = rng.normal(loc=0.0, scale=2.5, size=n_samples)

        psd = one_sided_periodogram(x, delta_t)
        integral_psd = np.sum(psd) * delta_f
        mean_square_time = np.mean(x ** 2)

        rel_err = abs(integral_psd - mean_square_time) / mean_square_time
        assert rel_err < 1e-12, f"Integral property error: {rel_err:.2e}"

    def test_expected_white_noise_psd_formula(self):
        sigma = 1.5
        f_s = 4096.0
        delta_t = 1.0 / f_s

        interior = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
        dc = expected_white_noise_psd(sigma, delta_t, bin_type="dc")
        nyquist = expected_white_noise_psd(sigma, delta_t, bin_type="nyquist")

        assert math.isclose(interior, 2.0 * (sigma ** 2) / f_s, rel_tol=1e-15)
        assert math.isclose(dc, (sigma ** 2) / f_s, rel_tol=1e-15)
        assert math.isclose(nyquist, (sigma ** 2) / f_s, rel_tol=1e-15)
        assert math.isclose(dc, 0.5 * interior, rel_tol=1e-15)


class TestStage11MandatorySummary:
    """Direct 1-to-1 mapping with the mandatory test matrix from prompt Section 7."""

    def test_frequency_grid_construction(self):
        """Correct length, spacing, DC, and Nyquist."""
        grid = build_frequency_grid(32768, 1.0 / 4096.0)
        assert len(grid) == 16385
        assert grid[0] == 0.0
        assert math.isclose(grid[-1], 2048.0, rel_tol=1e-14)
        assert math.isclose(grid[1] - grid[0], 0.125, rel_tol=1e-14)

    def test_fft_scaling(self):
        """x_tilde_k = dt * X_k."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        dt = 0.5
        X = np.fft.rfft(x)
        x_tilde = continuous_rfft(x, dt)
        np.testing.assert_allclose(x_tilde, dt * X, rtol=1e-15)

    def test_parseval_identity_tolerance(self):
        """Relative error below 10^{-12}."""
        rng = np.random.default_rng(2026)
        x = rng.normal(size=32768)
        _, _, rel_err = validate_parseval_identity(x, 1.0 / 4096.0)
        assert rel_err < 1e-12

    def test_dc_weighting_not_doubled(self):
        """DC weighting: Not doubled in one-sided spectrum."""
        x = np.ones(1024)
        X = np.fft.rfft(x)
        dt = 0.01
        df = 1.0 / (1024 * dt)
        e_time = discrete_parseval_time_energy(x, dt)
        e_freq = discrete_parseval_freq_energy(X, dt, df)
        assert abs(e_time - e_freq) / e_time < 1e-12

    def test_nyquist_weighting_not_doubled(self):
        """Nyquist weighting: Not doubled in one-sided spectrum."""
        x = np.array([(-1.0) ** j for j in range(1024)])
        X = np.fft.rfft(x)
        dt = 0.01
        df = 1.0 / (1024 * dt)
        e_time = discrete_parseval_time_energy(x, dt)
        e_freq = discrete_parseval_freq_energy(X, dt, df)
        assert abs(e_time - e_freq) / e_time < 1e-12

    def test_interior_bin_weighting_doubled(self):
        """Interior-bin weighting: Doubled in one-sided spectrum."""
        n = 1024
        dt = 0.001
        df = 1.0 / (n * dt)
        t = np.arange(n) * dt
        x = np.cos(2 * np.pi * 10 * df * t)
        X = np.fft.rfft(x)
        e_time = discrete_parseval_time_energy(x, dt)
        e_freq = discrete_parseval_freq_energy(X, dt, df)
        assert abs(e_time - e_freq) / e_time < 1e-12

    def test_periodogram_units_and_scaling(self):
        """Matches stated convention and preserves Parseval integral."""
        x = np.array([1.0, -2.0, 3.0, -1.0])
        dt = 0.25
        df = 1.0 / (4 * dt)
        psd = one_sided_periodogram(x, dt)
        assert math.isclose(np.sum(psd) * df, np.mean(x ** 2), rel_tol=1e-14)

    def test_noise_reproducibility(self):
        """Identical seed gives identical samples."""
        n1 = generate_white_noise(1024, 1.0, seed=777)
        n2 = generate_white_noise(1024, 1.0, seed=777)
        np.testing.assert_array_equal(n1, n2)

    def test_noise_input_validation(self):
        """Invalid inputs raise appropriate errors."""
        with pytest.raises(ValueError):
            generate_white_noise(-1, 1.0)
        with pytest.raises(ValueError):
            generate_white_noise(100, -0.5)

    def test_white_noise_ensemble_psd_interior(self):
        """Interior-bin mean agrees statistically with theoretical PSD."""
        dt = 1.0 / 4096.0
        sigma = 1.0
        s0 = expected_white_noise_psd(sigma, dt, "interior")
        rng = np.random.default_rng(123)
        accum = np.zeros(513)
        for _ in range(100):
            n = generate_white_noise(1024, sigma, generator=rng)
            accum += one_sided_periodogram(n, dt)
        mean_psd = accum / 100
        interior_mean = np.mean(mean_psd[1:-1])
        assert abs(interior_mean - s0) / s0 < 0.03

    def test_dc_nyquist_ensemble_psd_half_level(self):
        """DC/Nyquist ensemble PSD: Correct half-level expectation."""
        dt = 1.0 / 4096.0
        sigma = 1.0
        s_half = expected_white_noise_psd(sigma, dt, "dc")
        rng = np.random.default_rng(456)
        accum_dc = 0.0
        accum_nyq = 0.0
        n_trials = 200
        for _ in range(n_trials):
            n = generate_white_noise(1024, sigma, generator=rng)
            p = one_sided_periodogram(n, dt)
            accum_dc += p[0]
            accum_nyq += p[-1]
        mean_dc = accum_dc / n_trials
        mean_nyq = accum_nyq / n_trials
        # 1/sqrt(200) ~ 7% standard error; 4 sigma ~ 28%
        assert abs(mean_dc - s_half) / s_half < 0.28
        assert abs(mean_nyq - s_half) / s_half < 0.28
