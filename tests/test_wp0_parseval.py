"""WP0 Parseval validation suite covering diverse signal regimes.

Tests:
1. Smooth deterministic signals (sinusoids, windowed chirps, polynomials).
2. Stochastic signals (Gaussian white noise, colored AR(1) noise).
3. High-dynamic-range signals spanning ~10 orders of magnitude.
4. Adversarial signals (delta impulses, alternating +1/-1, sparse arrays).
5. Zero-energy and near-zero-energy signals.
6. Independent direct DFT reference calculation vs FFT.
7. Odd-length handling without Nyquist misapplication.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference.conventions import (
    build_frequency_grid,
    compute_parseval_error,
    continuous_rfft,
    direct_dft,
    discrete_parseval_freq_energy,
    discrete_parseval_time_energy,
    validate_parseval_identity,
)
from gw_inference.noise import generate_white_noise
from gw_inference.psd import tukey_window


class TestParsevalSuite:
    """Rigorous empirical Parseval energy balance validation across signal classes."""

    @pytest.fixture
    def dt(self) -> float:
        return 1.0 / 4096.0  # 4096 Hz sampling

    def test_smooth_deterministic_signals(self, dt: float):
        """Test smooth sinusoidal, polynomial, and windowed chirp signals."""
        N = 4096
        t = np.arange(N) * dt
        rng = np.random.default_rng(12345)

        # 1. Single sine wave
        x_sine = np.sin(2.0 * np.pi * 100.0 * t)
        _, _, err_sine = validate_parseval_identity(x_sine, dt)
        assert err_sine < 1e-14, f"Sine Parseval error {err_sine} exceeds 1e-14"

        # 2. Multi-tone smooth signal
        x_multi = (
            np.sin(2.0 * np.pi * 50.0 * t)
            + 0.5 * np.cos(2.0 * np.pi * 120.0 * t)
            + 0.25 * np.sin(2.0 * np.pi * 300.0 * t)
        )
        _, _, err_multi = validate_parseval_identity(x_multi, dt)
        assert err_multi < 1e-14, f"Multi-tone Parseval error {err_multi} exceeds 1e-14"

        # 3. Windowed linear chirp
        f0, f1 = 20.0, 500.0
        phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * (t**2) / (N * dt))
        w = tukey_window(N, alpha=0.2)
        x_chirp = w * np.sin(phase)
        _, _, err_chirp = validate_parseval_identity(x_chirp, dt)
        assert err_chirp < 1e-14, f"Windowed chirp Parseval error {err_chirp} exceeds 1e-14"

        # 4. Smooth polynomial
        t_norm = (t - t[N // 2]) / (N * dt)
        x_poly = 1.0 - 2.0 * (t_norm**2) + 3.0 * (t_norm**4)
        _, _, err_poly = validate_parseval_identity(x_poly, dt)
        assert err_poly < 1e-14, f"Polynomial Parseval error {err_poly} exceeds 1e-14"

    def test_stochastic_signals(self, dt: float):
        """Test Gaussian white noise and autoregressive AR(1) colored noise."""
        N = 8192
        rng = np.random.default_rng(42)

        # 1. Gaussian white noise realizations
        for seed in [101, 202, 303]:
            x_white = generate_white_noise(N, sigma=1e-21, seed=seed)
            _, _, err_white = validate_parseval_identity(x_white, dt)
            assert err_white < 5e-14, f"White noise Parseval error {err_white} exceeds 5e-14"

        # 2. AR(1) colored noise: x[j] = phi * x[j-1] + w[j]
        phi = 0.85
        w = rng.normal(scale=1e-21, size=N)
        x_ar1 = np.empty(N, dtype=np.float64)
        x_ar1[0] = w[0] / math.sqrt(1.0 - phi**2)
        for j in range(1, N):
            x_ar1[j] = phi * x_ar1[j - 1] + w[j]

        _, _, err_ar1 = validate_parseval_identity(x_ar1, dt)
        assert err_ar1 < 5e-14, f"AR(1) Parseval error {err_ar1} exceeds 5e-14"

    def test_high_dynamic_range_signals(self, dt: float):
        """Test signals spanning ~10 orders of magnitude in sample amplitudes."""
        N = 2048
        t = np.arange(N) * dt

        # Signal with components ranging from 1.0 down to 1e-10
        x_hdr = (
            1.0 * np.sin(2.0 * np.pi * 30.0 * t)
            + 1e-3 * np.sin(2.0 * np.pi * 70.0 * t)
            + 1e-6 * np.sin(2.0 * np.pi * 150.0 * t)
            + 1e-10 * np.sin(2.0 * np.pi * 300.0 * t)
        )
        e_time, e_freq, err_hdr = validate_parseval_identity(x_hdr, dt)
        # In double precision, 10 orders of magnitude dynamic range still retains ~6 digits on smallest
        # and total energy dominated by largest component achieves < 1e-14 relative energy error.
        assert err_hdr < 1e-13, f"HDR Parseval error {err_hdr} exceeds 1e-13"

        # Signal with geometric progression over time
        x_geom = np.geomspace(1e-10, 1.0, num=N)
        _, _, err_geom = validate_parseval_identity(x_geom, dt)
        assert err_geom < 1e-13, f"Geometric scale Parseval error {err_geom} exceeds 1e-13"

    def test_adversarial_signals(self, dt: float):
        """Test delta impulses, alternating +1/-1 sequences, and sparse arrays."""
        N = 1024

        # 1. Kronecker delta impulse at origin
        x_impulse_0 = np.zeros(N, dtype=np.float64)
        x_impulse_0[0] = 1.0
        _, _, err_imp0 = validate_parseval_identity(x_impulse_0, dt)
        assert err_imp0 < 1e-14, f"Impulse(0) Parseval error {err_imp0} exceeds 1e-14"

        # 2. Kronecker delta impulse in interior
        x_impulse_mid = np.zeros(N, dtype=np.float64)
        x_impulse_mid[N // 3] = 42.0
        _, _, err_impmid = validate_parseval_identity(x_impulse_mid, dt)
        assert err_impmid < 1e-14, f"Impulse(mid) Parseval error {err_impmid} exceeds 1e-14"

        # 3. Alternating signs (+1, -1, +1, -1, ...) which lands on Nyquist frequency
        x_alt = np.empty(N, dtype=np.float64)
        x_alt[0::2] = 1.0
        x_alt[1::2] = -1.0
        _, _, err_alt = validate_parseval_identity(x_alt, dt)
        assert err_alt < 1e-14, f"Alternating signs Parseval error {err_alt} exceeds 1e-14"

        # 4. Extremely sparse array (only 3 non-zero samples)
        x_sparse = np.zeros(N, dtype=np.float64)
        x_sparse[10] = 3.5
        x_sparse[100] = -2.1
        x_sparse[500] = 7.8
        _, _, err_sparse = validate_parseval_identity(x_sparse, dt)
        assert err_sparse < 1e-14, f"Sparse Parseval error {err_sparse} exceeds 1e-14"

    def test_zero_and_near_zero_energy(self, dt: float):
        """Test zero-energy signal and sub-underflow signals with fixed reference scale."""
        N = 512

        # 1. Exactly zero signal
        x_zero = np.zeros(N, dtype=np.float64)
        e_time, e_freq, err_zero = validate_parseval_identity(x_zero, dt, ref_scale=1.0)
        assert e_time == 0.0
        assert e_freq == 0.0
        assert err_zero == 0.0

        # 2. Near-zero signal (e.g. 1e-250)
        x_tiny = np.full(N, 1e-150, dtype=np.float64)
        e_time_tiny = discrete_parseval_time_energy(x_tiny, dt)
        X_tiny = np.fft.rfft(x_tiny)
        delta_f = 1.0 / (N * dt)
        e_freq_tiny = discrete_parseval_freq_energy(X_tiny, dt, delta_f)
        # Relative error should be small even at small scale
        err_rel_tiny = compute_parseval_error(e_time_tiny, e_freq_tiny)
        assert err_rel_tiny < 1e-14

        # Fixed reference error using ref_scale = 1.0
        err_ref_tiny = compute_parseval_error(0.0, e_freq_tiny, ref_scale=1.0)
        assert err_ref_tiny == abs(e_freq_tiny)

    def test_direct_dft_reference_comparison(self, dt: float):
        """Validate FFT output against independent direct matrix-vector O(N^2) DFT."""
        N = 256  # Small array for O(N^2) direct computation
        rng = np.random.default_rng(999)
        x = rng.normal(size=N)

        # FFT continuous transform
        x_fft = continuous_rfft(x, dt)

        # Independent direct DFT
        x_dft = direct_dft(x, dt)

        # Compare coefficient by coefficient
        abs_diff = np.abs(x_fft - x_dft)
        norm_ref = np.abs(x_dft)
        # Avoid division by zero for near-zero bins
        mask = norm_ref > 1e-14
        rel_diff = abs_diff[mask] / norm_ref[mask]

        max_rel_diff = float(np.max(rel_diff))
        max_abs_diff = float(np.max(abs_diff))

        # In double precision, the direct matrix-vector DFT accumulates O(N * eps_mach) roundoff
        # (N=256 -> ~5.7e-14), whereas radix-2 FFT accumulates O(log2(N) * eps_mach) (~1.8e-15).
        # When comparing individual spectral bins (especially bins with moderate amplitude),
        # the observed max relative discrepancy is ~2.7e-13. An empirical tolerance of 1e-12
        # is justified and verified.
        assert max_rel_diff < 1e-12, f"Max relative difference FFT vs Direct DFT: {max_rel_diff}"
        assert max_abs_diff < 1e-12, f"Max absolute difference FFT vs Direct DFT: {max_abs_diff}"

    def test_odd_length_parseval_handling(self, dt: float):
        """Verify that odd-N frequency energy formula handles absence of Nyquist bin correctly."""
        # For odd N, discrete_parseval_freq_energy should apply folding factor 2 to all k >= 1
        N_odd = 127
        t_odd = np.arange(N_odd) * dt
        x_odd = np.sin(2.0 * np.pi * 50.0 * t_odd)

        e_time_odd = float(dt * np.sum(x_odd**2))
        X_odd = np.fft.rfft(x_odd)
        delta_f_odd = 1.0 / (N_odd * dt)

        # Verify with explicit n_samples=N_odd
        e_freq_odd = discrete_parseval_freq_energy(
            X_odd, dt, delta_f_odd, n_samples=N_odd
        )
        err_odd = abs(e_time_odd - e_freq_odd) / e_time_odd

        assert err_odd < 1e-14, f"Odd-length Parseval error {err_odd} exceeds 1e-14"

        # Contrast: if misapplying even-N formula (which would assign weight 1 to last bin),
        # the error would be large whenever energy is present in the highest bin.
        x_high = np.cos(np.pi * (N_odd - 1) * np.arange(N_odd) / N_odd)
        e_time_high = float(dt * np.sum(x_high**2))
        X_high = np.fft.rfft(x_high)
        e_freq_correct = discrete_parseval_freq_energy(
            X_high, dt, delta_f_odd, n_samples=N_odd
        )
        err_correct = abs(e_time_high - e_freq_correct) / e_time_high
        assert err_correct < 1e-14

        # If we mistakenly assumed even N without passing n_samples:
        e_freq_wrong = discrete_parseval_freq_energy(
            X_high, dt, delta_f_odd, n_samples=None
        )
        err_wrong = abs(e_time_high - e_freq_wrong) / e_time_high
        # The misapplication would drop half the energy of the last bin
        assert err_wrong > 1e-3, "Expected even-N formula to fail on odd-N array with high-frequency energy"

    def test_parseval_input_length_validation(self, dt: float):
        """Verify strict validation of X length against supplied n_samples."""
        delta_f = 1.0

        # 1. Correct even N: N=100 -> expected len = 100//2 + 1 = 51
        X_even = np.ones(51, dtype=np.complex128)
        e_even = discrete_parseval_freq_energy(X_even, dt, delta_f, n_samples=100)
        assert e_even > 0.0

        # 2. Correct odd N: N=99 -> expected len = 99//2 + 1 = 50
        X_odd = np.ones(50, dtype=np.complex128)
        e_odd = discrete_parseval_freq_energy(X_odd, dt, delta_f, n_samples=99)
        assert e_odd > 0.0

        # 3. Too short input: pass 50 bins when 51 expected for N=100
        with pytest.raises(ValueError, match="does not match expected length"):
            discrete_parseval_freq_energy(X_odd, dt, delta_f, n_samples=100)

        # 4. Too long input: pass 51 bins when 50 expected for N=99
        with pytest.raises(ValueError, match="does not match expected length"):
            discrete_parseval_freq_energy(X_even, dt, delta_f, n_samples=99)

        # 5. Invalid n_samples values
        with pytest.raises(ValueError, match="positive integer"):
            discrete_parseval_freq_energy(X_even, dt, delta_f, n_samples=0)
        with pytest.raises(ValueError, match="positive integer"):
            discrete_parseval_freq_energy(X_even, dt, delta_f, n_samples=-10)
        with pytest.raises(ValueError, match="positive integer"):
            discrete_parseval_freq_energy(X_even, dt, delta_f, n_samples=51.5)  # float

        # 6. Backward compatibility: when n_samples is None, length matching is skipped
        e_compat = discrete_parseval_freq_energy(X_even, dt, delta_f, n_samples=None)
        assert e_compat > 0.0
