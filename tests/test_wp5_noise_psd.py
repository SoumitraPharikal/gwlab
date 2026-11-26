"""Unit tests for Work Package 5 (WP5): Noise Generation and PSD Characterization.

Tests verify:
1. `generate_colored_noise` API, input validation, reproducibility, and real-valued output.
2. Odd and even grid handling, endpoint variance scaling (c_q factor), and interior bins.
3. Parseval discrete energy conservation to float64 machine precision (< 1e-14).
4. Monte Carlo verification of Fourier coefficient variances and endpoint statistics.
5. Zero-PSD bin handling (DC, Nyquist, or interior zeros producing exact zeros without NaN/Inf).
6. `analytical_detector_psd` proxy models ('aligo_representative', 'flat'), finiteness,
   positivity, and input validation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference import (
    analytical_detector_psd,
    build_frequency_grid,
    generate_colored_noise,
)


class TestAnalyticalDetectorPSD:
    """Tests for `analytical_detector_psd`."""

    def test_aligo_representative_values_and_endpoints(self):
        """Verify analytical values at DC, knee, high, and asymptotic regions."""
        freqs = np.array([0.0, 15.0, 35.0, 100.0, 300.0, 1000.0], dtype=np.float64)
        psd = analytical_detector_psd(freqs, model="aligo_representative")

        assert psd.shape == freqs.shape
        assert psd.dtype == np.float64
        assert np.all(np.isfinite(psd))
        assert np.all(psd > 0.0)

        # Exact DC check: S_0 * [ (35/15)^4 + 1.0 ]
        s0 = 1.0e-46
        expected_dc = s0 * ((35.0 / 15.0) ** 4 + 1.0)
        assert math.isclose(psd[0], expected_dc, rel_tol=1e-14)

        # Exact 300 Hz check: S_0 * [ (35^2 / (300^2 + 15^2))^2 + 1.0 + (300/300)^2 ]
        expected_300 = s0 * (((35.0**2) / (300.0**2 + 15.0**2)) ** 2 + 1.0 + 1.0)
        assert math.isclose(psd[4], expected_300, rel_tol=1e-14)

    def test_flat_model(self):
        """Verify constant flat spectrum."""
        freqs = np.linspace(0.0, 1000.0, 501)
        psd = analytical_detector_psd(freqs, model="flat")
        assert np.all(psd == 1.0e-46)

    def test_invalid_inputs(self):
        """Verify proper error handling for invalid frequencies and model strings."""
        # Non-1D
        with pytest.raises(ValueError, match="1D array"):
            analytical_detector_psd(np.ones((5, 5)))

        # Negative frequency
        with pytest.raises(ValueError, match="negative"):
            analytical_detector_psd(np.array([10.0, -5.0, 20.0]))

        # NaN / Inf
        with pytest.raises(ValueError, match="finite"):
            analytical_detector_psd(np.array([10.0, np.nan, 20.0]))
        with pytest.raises(ValueError, match="finite"):
            analytical_detector_psd(np.array([10.0, np.inf, 20.0]))

        # Unknown model
        with pytest.raises(ValueError, match="Unknown PSD model"):
            analytical_detector_psd(np.array([10.0, 20.0]), model="nonexistent_model")

        # Non-string model
        with pytest.raises(ValueError, match="model must be a string"):
            analytical_detector_psd(np.array([10.0, 20.0]), model=123)  # type: ignore


class TestGenerateColoredNoiseValidation:
    """Tests for input validation and error handling in `generate_colored_noise`."""

    def test_invalid_n_samples(self):
        psd = np.ones(5)
        with pytest.raises(ValueError, match="n_samples must be an integer >= 2"):
            generate_colored_noise(1, delta_t=0.001, psd=psd)
        with pytest.raises(ValueError, match="n_samples must be an integer >= 2"):
            generate_colored_noise(-10, delta_t=0.001, psd=psd)
        with pytest.raises(ValueError, match="n_samples must be an integer >= 2"):
            generate_colored_noise(8.5, delta_t=0.001, psd=psd)  # type: ignore

    def test_invalid_delta_t(self):
        psd = np.ones(5)
        with pytest.raises(ValueError, match="delta_t must be a positive finite float"):
            generate_colored_noise(8, delta_t=0.0, psd=psd)
        with pytest.raises(ValueError, match="delta_t must be a positive finite float"):
            generate_colored_noise(8, delta_t=-0.001, psd=psd)
        with pytest.raises(ValueError, match="delta_t must be a positive finite float"):
            generate_colored_noise(8, delta_t=float("nan"), psd=psd)
        with pytest.raises(ValueError, match="delta_t must be a positive finite float"):
            generate_colored_noise(8, delta_t=float("inf"), psd=psd)

    def test_invalid_psd_shape_and_length(self):
        # 8 samples requires psd of length 8 // 2 + 1 = 5
        with pytest.raises(ValueError, match="psd length .* must match"):
            generate_colored_noise(8, delta_t=0.001, psd=np.ones(4))
        with pytest.raises(ValueError, match="psd length .* must match"):
            generate_colored_noise(8, delta_t=0.001, psd=np.ones(6))
        with pytest.raises(ValueError, match="1D array"):
            generate_colored_noise(8, delta_t=0.001, psd=np.ones((5, 1)))

    def test_invalid_psd_values(self):
        # Negative value
        psd = np.ones(5)
        psd[2] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            generate_colored_noise(8, delta_t=0.001, psd=psd)

        # NaN / Inf
        psd = np.ones(5)
        psd[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            generate_colored_noise(8, delta_t=0.001, psd=psd)

        psd[3] = np.inf
        with pytest.raises(ValueError, match="finite"):
            generate_colored_noise(8, delta_t=0.001, psd=psd)

    def test_all_zero_psd(self):
        # All zeros in interior
        psd = np.zeros(5)
        with pytest.raises(ValueError, match="identically zero"):
            generate_colored_noise(8, delta_t=0.001, psd=psd)

    def test_invalid_generator_type(self):
        psd = np.ones(5)
        with pytest.raises(TypeError, match="generator must be an instance"):
            generate_colored_noise(8, delta_t=0.001, psd=psd, generator="bad_generator")  # type: ignore


class TestGenerateColoredNoiseProperties:
    """Tests for physical, mathematical, and algorithmic properties of `generate_colored_noise`."""

    def test_reproducibility_and_generator(self):
        """Test seed reproducibility and independent generator instances."""
        psd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n1 = generate_colored_noise(8, delta_t=0.1, psd=psd, seed=42)
        n2 = generate_colored_noise(8, delta_t=0.1, psd=psd, seed=42)
        n3 = generate_colored_noise(8, delta_t=0.1, psd=psd, seed=43)

        np.testing.assert_array_equal(n1, n2)
        assert not np.array_equal(n1, n3)

        # Generator object
        rng = np.random.default_rng(12345)
        n4 = generate_colored_noise(8, delta_t=0.1, psd=psd, generator=rng)
        assert n4.shape == (8,)

    @pytest.mark.parametrize("n_samples", [16, 17, 64, 65, 128, 129])
    def test_real_valued_finite_output(self, n_samples):
        """Output must always be 1D, real-valued float64, and strictly finite."""
        n_freqs = n_samples // 2 + 1
        psd = np.linspace(1.0, 10.0, n_freqs)
        noise = generate_colored_noise(n_samples, delta_t=0.01, psd=psd, seed=101)

        assert noise.shape == (n_samples,)
        assert noise.dtype == np.float64
        assert np.isrealobj(noise)
        assert np.all(np.isfinite(noise))

    @pytest.mark.parametrize("n_samples", [32, 33, 128, 129, 256])
    def test_parseval_energy_conservation(self, n_samples):
        """Verify exact discrete Parseval energy balance to float64 machine precision (< 1e-14)."""
        dt = 1.0 / 2048.0
        n_freqs = n_samples // 2 + 1
        freqs = np.arange(n_freqs) / (n_samples * dt)
        psd = analytical_detector_psd(freqs, model="aligo_representative")

        noise = generate_colored_noise(n_samples, delta_t=dt, psd=psd, seed=777)

        # Time energy: E_t = dt * sum(n_j^2)
        e_time = dt * float(np.sum(noise**2))

        # Fourier transform: n_tilde = dt * rfft(n)
        df = 1.0 / (n_samples * dt)
        n_tilde = dt * np.fft.rfft(noise)
        mag_sq = np.abs(n_tilde) ** 2

        if n_samples % 2 == 0:
            # Even N: DC (endpoint=1), interior (factor=2), Nyquist (endpoint=1)
            e_freq = df * (mag_sq[0] + 2.0 * np.sum(mag_sq[1 : n_samples // 2]) + mag_sq[n_samples // 2])
        else:
            # Odd N: DC (endpoint=1), interior (factor=2)
            e_freq = df * (mag_sq[0] + 2.0 * np.sum(mag_sq[1:]))

        rel_diff = abs(e_time - e_freq) / e_time
        assert rel_diff < 1.0e-14, f"Parseval relative error {rel_diff} exceeded 1e-14"

    def test_zero_psd_bin_handling(self):
        """Bins with zero PSD must yield exact zero Fourier coefficients without NaN/Inf."""
        N = 16
        dt = 0.1
        n_freqs = N // 2 + 1  # 9 bins
        psd = np.ones(n_freqs)
        psd[0] = 0.0  # DC is zero
        psd[2] = 0.0  # interior bin is zero
        psd[8] = 0.0  # Nyquist bin is zero

        noise = generate_colored_noise(N, delta_t=dt, psd=psd, seed=999)
        dft = np.fft.rfft(noise)

        # Inversion followed by forward rfft roundtrips zero bins to floating-point machine precision (< 1e-14)
        assert abs(dft[0]) < 1.0e-14
        assert abs(dft[2]) < 1.0e-14
        assert abs(dft[8]) < 1.0e-14
        assert np.all(np.isfinite(noise))

    def test_fourier_coefficient_variances_ensemble(self):
        """Monte Carlo verification of interior and endpoint Fourier coefficient variances.

        For interior bins: Var(Re X_k) = Var(Im X_k) = N * S_n(f_k) / (4 * dt).
        For DC (k=0): Var(X_0) = N * S_n(0) / (2 * dt).
        For Nyquist (k=N/2): Var(X_{N/2}) = N * S_n(f_s/2) / (2 * dt).
        """
        N = 32
        dt = 0.05
        n_freqs = N // 2 + 1
        psd = np.linspace(2.0, 10.0, n_freqs)

        R = 10000
        rng = np.random.default_rng(2026)

        # Pre-allocate array for Fourier transforms of generated noise
        dfts = np.empty((R, n_freqs), dtype=np.complex128)
        for r in range(R):
            n = generate_colored_noise(N, delta_t=dt, psd=psd, generator=rng)
            dfts[r] = np.fft.rfft(n)

        # Verify DC variance: E[X_0^2] = N * S_n[0] / (2 * dt)
        expected_var_dc = N * psd[0] / (2.0 * dt)
        measured_var_dc = float(np.var(dfts[:, 0].real))
        # 10k samples -> relative SE approx sqrt(2 / 10000) approx 1.4%
        assert math.isclose(measured_var_dc, expected_var_dc, rel_tol=0.06)
        assert np.all(dfts[:, 0].imag == 0.0)

        # Verify Nyquist variance: E[X_{N/2}^2] = N * S_n[N/2] / (2 * dt)
        expected_var_nyq = N * psd[N // 2] / (2.0 * dt)
        measured_var_nyq = float(np.var(dfts[:, N // 2].real))
        assert math.isclose(measured_var_nyq, expected_var_nyq, rel_tol=0.06)
        assert np.all(dfts[:, N // 2].imag == 0.0)

        # Verify an interior bin (e.g. k=4)
        k = 4
        expected_var_int_part = N * psd[k] / (4.0 * dt)
        measured_var_re = float(np.var(dfts[:, k].real))
        measured_var_im = float(np.var(dfts[:, k].imag))
        assert math.isclose(measured_var_re, expected_var_int_part, rel_tol=0.06)
        assert math.isclose(measured_var_im, expected_var_int_part, rel_tol=0.06)
