"""WP6 Stage 5: Stage A Deterministic Welch PSD Validation & Grid Ingestion Boundary Tests.

This test module implements strictly deterministic tests for Work Package 6 Stage 5:
1. Exact Parseval Energy Conservation:
       sum_{k=0}^{M/2} S_hat(f_k) * delta_f = (sum_{j=0}^{M-1} w_j^2 * x_j^2) / (sum_{j=0}^{M-1} w_j^2)
   verified across impulses, sinusoids on exact DFT bins, constant signals, and window variations.
2. Direct Endpoint Normalization:
   Verification of exact one-sided scaling coefficients:
       DC (k=0) and Nyquist (k=M/2):      delta_t / (M * S_2)
       Interior positive bins (1 <= k < M/2): 2 * delta_t / (M * S_2)
3. Frequency-Grid Validation Boundary:
   Verification that welch_psd matches FrequencyGrid.frequencies bit-for-bit for M=8192,
   and strict validation in FrequencyDomainGaussianLikelihood and validate_psd_frequency_alignment
   rejecting same-length shifted or differently-spaced frequency grids, non-positive or
   non-finite active bins, and mismatched shapes, while preserving backward compatibility.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference.grid import FrequencyGrid
from gw_inference.likelihood import (
    FrequencyDomainGaussianLikelihood,
    compute_log_likelihood,
    validate_psd_frequency_alignment,
)
from gw_inference.psd import (
    single_segment_psd,
    tukey_window,
    welch_psd,
    window_power,
)


# =====================================================================
# SECTION 1: Exact Parseval Energy Conservation Identity
# =====================================================================


class TestParsevalIdentity:
    """Verify the exact one-sided Welch periodogram Parseval identity.

    For single-segment one-sided PSD with window w_j:
        sum_{k=0}^{M/2} S_hat(f_k) * delta_f = sum_{j=0}^{M-1} (w_j * x_j)^2 / sum_{j=0}^{M-1} w_j^2
    """

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
    @pytest.mark.parametrize("M", [256, 1024, 8192])
    def test_parseval_impulse_at_window_center(self, alpha: float, M: int):
        """Test Parseval identity for discrete impulse at window center (w_j = 1.0)."""
        dt = 1.0 / 2048.0
        df = 1.0 / (M * dt)
        w = tukey_window(M, alpha=alpha) if alpha > 0.0 else np.ones(M, dtype=np.float64)

        x = np.zeros(M, dtype=np.float64)
        x[M // 2] = 2.75

        psd = single_segment_psd(x, delta_t=dt, window=w)
        lhs = float(np.sum(psd) * df)
        rhs = float(np.sum((w * x) ** 2) / np.sum(w ** 2))

        assert math.isclose(lhs, rhs, rel_tol=1e-13, abs_tol=1e-14)

    @pytest.mark.parametrize("alpha", [0.25, 0.5])
    def test_parseval_impulse_in_taper_region(self, alpha: float):
        """Test Parseval identity for discrete impulse located in the window taper region."""
        M = 8192
        dt = 1.0 / 2048.0
        df = 1.0 / (M * dt)
        w = tukey_window(M, alpha=alpha)

        # Place impulse in the taper region where 0 < w_j < 1.0
        n_taper = int(alpha * (M - 1) / 2.0)
        idx_taper = n_taper // 2
        assert 0.0 < w[idx_taper] < 1.0

        x = np.zeros(M, dtype=np.float64)
        x[idx_taper] = 4.2

        psd = single_segment_psd(x, delta_t=dt, window=w)
        lhs = float(np.sum(psd) * df)
        rhs = float(np.sum((w * x) ** 2) / np.sum(w ** 2))

        assert math.isclose(lhs, rhs, rel_tol=1e-13, abs_tol=1e-14)

    @pytest.mark.parametrize("k0", [1, 10, 50, 200, 1024])
    def test_parseval_sinusoid_on_exact_dft_bin(self, k0: int):
        """Test Parseval identity for a pure sinusoid exactly centered on an integer DFT bin."""
        M = 8192
        dt = 1.0 / 2048.0
        df = 1.0 / (M * dt)
        w = tukey_window(M, alpha=0.25)

        j = np.arange(M, dtype=np.float64)
        A = 3.5
        x = A * np.cos(2.0 * math.pi * k0 * j / float(M))

        psd = single_segment_psd(x, delta_t=dt, window=w)
        lhs = float(np.sum(psd) * df)
        rhs = float(np.sum((w * x) ** 2) / np.sum(w ** 2))

        assert math.isclose(lhs, rhs, rel_tol=1e-13, abs_tol=1e-14)

    def test_parseval_dc_constant_signal(self):
        """Test Parseval identity for a DC constant signal x_j = c."""
        M = 8192
        dt = 1.0 / 2048.0
        df = 1.0 / (M * dt)
        w = tukey_window(M, alpha=0.25)

        c = 1.618
        x = np.full(M, c, dtype=np.float64)

        psd = single_segment_psd(x, delta_t=dt, window=w)
        lhs = float(np.sum(psd) * df)
        rhs = float(np.sum((w * x) ** 2) / np.sum(w ** 2))

        assert math.isclose(lhs, rhs, rel_tol=1e-13, abs_tol=1e-14)

    def test_parseval_linear_combination_of_tones(self):
        """Test Parseval identity for a multi-component harmonic signal."""
        M = 8192
        dt = 1.0 / 2048.0
        df = 1.0 / (M * dt)
        w = tukey_window(M, alpha=0.25)

        j = np.arange(M, dtype=np.float64)
        x = (
            1.5 * np.cos(2.0 * math.pi * 25.0 * j / float(M))
            + 0.8 * np.sin(2.0 * math.pi * 80.0 * j / float(M))
            + 2.2 * np.cos(2.0 * math.pi * 350.0 * j / float(M))
            + 0.5
        )

        psd = single_segment_psd(x, delta_t=dt, window=w)
        lhs = float(np.sum(psd) * df)
        rhs = float(np.sum((w * x) ** 2) / np.sum(w ** 2))

        assert math.isclose(lhs, rhs, rel_tol=1e-13, abs_tol=1e-14)

    def test_parseval_unweighted_rectangular_equivalence(self):
        """Verify that with a rectangular window (w_j = 1), the RHS reduces to sample mean square."""
        M = 4096
        dt = 1.0 / 1024.0
        df = 1.0 / (M * dt)

        rng = np.random.default_rng(9999)
        x = rng.normal(loc=0.5, scale=2.0, size=M)

        psd = single_segment_psd(x, delta_t=dt, window=None)
        lhs = float(np.sum(psd) * df)
        rhs_sample_variance = float(np.mean(x ** 2))

        assert math.isclose(lhs, rhs_sample_variance, rel_tol=1e-13, abs_tol=1e-14)


# =====================================================================
# SECTION 2: Direct Endpoint Normalization Tests
# =====================================================================


class TestEndpointNormalization:
    """Verify exact one-sided PSD scaling coefficients at DC, Nyquist, and interior bins.

    Scaling coefficients:
        C_DC = delta_t / (M * S_2)
        C_Nyquist = delta_t / (M * S_2)
        C_interior = 2 * delta_t / (M * S_2)
    """

    def test_direct_scaling_coefficients_via_flat_fourier_impulse(self):
        """Test exact scaling coefficients using an impulse with flat Fourier amplitude.

        An impulse x_j = delta_{j, j0} at j0 where w_{j0} = 1.0 has discrete Fourier
        transform X_k = 1.0 for all k in {0, ..., M/2}. Therefore, S_hat(f_k) measures
        the exact scaling coefficients directly.
        """
        M = 8192
        dt = 1.0 / 2048.0
        w = tukey_window(M, alpha=0.25)
        s2 = window_power(w)

        # Place impulse at center where w_j = 1.0
        j0 = M // 2
        assert w[j0] == 1.0

        x = np.zeros(M, dtype=np.float64)
        x[j0] = 1.0

        psd = single_segment_psd(x, delta_t=dt, window=w)

        expected_endpoint = dt / (float(M) * s2)
        expected_interior = 2.0 * dt / (float(M) * s2)

        # 1. DC bin (k = 0)
        assert psd[0] == expected_endpoint

        # 2. Nyquist bin (k = M // 2)
        assert psd[-1] == expected_endpoint

        # 3. All interior positive-frequency bins (1 <= k < M // 2)
        interior = psd[1:-1]
        assert np.all(interior == expected_interior)

        # 4. Exact factor of 2.0 ratio
        assert np.all(interior / psd[0] == 2.0)
        assert np.all(interior / psd[-1] == 2.0)

    def test_pure_dc_tone_known_analytical_psd(self):
        """Verify exact PSD value for a pure DC signal x_j = A with rectangular window."""
        M = 4096
        dt = 1.0 / 1024.0
        T = M * dt
        A = 2.5

        x = np.full(M, A, dtype=np.float64)
        psd = single_segment_psd(x, delta_t=dt, window=None)

        # DC bin energy = A^2 * T
        expected_dc_psd = (A ** 2) * T
        assert math.isclose(psd[0], expected_dc_psd, rel_tol=1e-13)

        # All AC bins must be strictly zero (orthogonal DFT bins)
        assert np.max(np.abs(psd[1:])) == 0.0

    def test_pure_nyquist_tone_known_analytical_psd(self):
        """Verify exact PSD value for a pure Nyquist signal x_j = A * (-1)^j with rectangular window."""
        M = 4096
        dt = 1.0 / 1024.0
        T = M * dt
        A = 3.0

        j = np.arange(M, dtype=np.float64)
        x = A * ((-1.0) ** j)
        psd = single_segment_psd(x, delta_t=dt, window=None)

        # Nyquist bin energy = A^2 * T
        expected_nyquist_psd = (A ** 2) * T
        assert math.isclose(psd[-1], expected_nyquist_psd, rel_tol=1e-13)

        # All other bins must be strictly zero
        assert np.max(np.abs(psd[:-1])) == 0.0

    def test_pure_interior_tone_known_analytical_psd(self):
        """Verify exact PSD value for an interior tone x_j = A * cos(2*pi*k0*j/M) with rectangular window."""
        M = 4096
        dt = 1.0 / 1024.0
        T = M * dt
        A = 2.0
        k0 = 120

        j = np.arange(M, dtype=np.float64)
        x = A * np.cos(2.0 * math.pi * k0 * j / float(M))
        psd = single_segment_psd(x, delta_t=dt, window=None)

        # Positive interior bin folded energy = 0.5 * A^2 * T
        expected_interior_psd = 0.5 * (A ** 2) * T
        assert math.isclose(psd[k0], expected_interior_psd, rel_tol=1e-13)

        # Orthogonal off-bin energy must be zero to machine precision
        mask_other = np.ones(psd.size, dtype=bool)
        mask_other[k0] = False
        assert np.max(np.abs(psd[mask_other])) < 1e-24


# =====================================================================
# SECTION 3: Frequency-Grid Validation Boundary & Likelihood Ingestion
# =====================================================================


class TestFrequencyGridValidationBoundary:
    """Verify frequency-grid alignment between welch_psd and FrequencyGrid, and test ingestion boundary."""

    def test_welch_psd_matches_frequency_grid_coordinates(self):
        """Verify welch_psd frequency vector matches FrequencyGrid.frequencies via exact equality and tolerance."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)

        # Dummy data for Welch estimator (2 segments, 50% overlap)
        x = np.zeros(M * 2, dtype=np.float64)
        welch_freqs, _ = welch_psd(x, delta_t=dt, segment_length=M)

        assert welch_freqs.shape == grid.frequencies.shape
        # Exact array equality
        assert np.array_equal(welch_freqs, grid.frequencies)
        # Numerical agreement within tolerance
        assert np.allclose(welch_freqs, grid.frequencies, atol=1e-12, rtol=1e-12)
        assert float(np.max(np.abs(welch_freqs - grid.frequencies))) == 0.0

    def test_empirical_psd_route_rejects_omitted_frequencies(self):
        """Verify the Stage 5 empirical-PSD ingestion route strictly rejects omitted frequency coordinates."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)

        data_fd = np.ones(grid.n_frequencies, dtype=np.complex128)
        psd = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)

        # 1. Omitted keyword (default None)
        with pytest.raises(ValueError, match="strictly required for empirical PSD ingestion"):
            FrequencyDomainGaussianLikelihood.from_empirical_psd(
                data_fd=data_fd,
                psd=psd,
                grid=grid,
                mask=mask,
            )

        # 2. Explicit None
        with pytest.raises(ValueError, match="strictly required for empirical PSD ingestion"):
            FrequencyDomainGaussianLikelihood.from_empirical_psd(
                data_fd=data_fd,
                psd=psd,
                grid=grid,
                mask=mask,
                psd_frequencies=None,
            )

    def test_empirical_psd_route_accepts_correctly_aligned_coordinates(self):
        """Verify the Stage 5 empirical-PSD route accepts correctly aligned coordinates for full and active grids."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)
        idx = grid.get_active_indices(mask)

        data_fd = np.ones(grid.n_frequencies, dtype=np.complex128)
        psd_full = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)

        # Full-grid empirical ingestion
        like_full = FrequencyDomainGaussianLikelihood.from_empirical_psd(
            data_fd=data_fd,
            psd=psd_full,
            grid=grid,
            mask=mask,
            psd_frequencies=grid.frequencies,
        )
        assert like_full.n_active_bins == 1969
        assert like_full.psd_frequencies is not None
        assert np.allclose(like_full.psd_frequencies, grid.frequencies, atol=1e-12, rtol=1e-12)

        # Active-sliced empirical ingestion
        psd_act = np.full(idx.size, 1e-46, dtype=np.float64)
        freqs_act = grid.frequencies[idx]
        like_act = FrequencyDomainGaussianLikelihood.from_empirical_psd(
            data_fd=data_fd,
            psd=psd_act,
            grid=grid,
            mask=mask,
            psd_frequencies=freqs_act,
        )
        assert like_act.n_active_bins == idx.size
        assert like_act.psd_frequencies is not None
        assert np.allclose(like_act.psd_frequencies, freqs_act, atol=1e-12, rtol=1e-12)

    def test_empirical_psd_route_rejects_same_length_shifted_grid(self):
        """Verify the Stage 5 empirical route rejects identical-shape PSD with shifted frequency coordinates."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)

        data_fd = np.ones(grid.n_frequencies, dtype=np.complex128)
        psd = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)
        shifted_freqs = grid.frequencies + 0.25

        with pytest.raises(ValueError, match="do not match expected likelihood grid frequencies within tolerance"):
            FrequencyDomainGaussianLikelihood.from_empirical_psd(
                data_fd=data_fd,
                psd=psd,
                grid=grid,
                mask=mask,
                psd_frequencies=shifted_freqs,
            )

    def test_empirical_psd_route_rejects_same_length_differently_spaced_grid(self):
        """Verify the Stage 5 empirical route rejects identical-shape PSD with different frequency spacing."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)

        data_fd = np.ones(grid.n_frequencies, dtype=np.complex128)
        psd = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)
        wrong_spacing_freqs = np.linspace(0.0, 512.0, grid.n_frequencies)

        with pytest.raises(ValueError, match="do not match expected likelihood grid frequencies within tolerance"):
            FrequencyDomainGaussianLikelihood.from_empirical_psd(
                data_fd=data_fd,
                psd=psd,
                grid=grid,
                mask=mask,
                psd_frequencies=wrong_spacing_freqs,
            )

    def test_legacy_calls_omitting_coordinates_retain_existing_behavior(self):
        """Verify backward compatibility: legacy callers intentionally omitting coordinates behave as before."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)

        data_fd = np.ones(grid.n_frequencies, dtype=np.complex128)
        template_fd = 0.5 * data_fd
        psd = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)

        # 1. FrequencyDomainGaussianLikelihood direct construction with omitted psd_frequencies
        like_legacy = FrequencyDomainGaussianLikelihood(
            data_fd=data_fd,
            psd=psd,
            grid=grid,
            mask=mask,
        )
        assert like_legacy.n_active_bins == 1969
        assert like_legacy.psd_frequencies is None
        ln_L_class = like_legacy.evaluate(template_fd)

        # 2. compute_log_likelihood direct functional call with omitted psd_frequencies
        ln_L_func = compute_log_likelihood(
            data_fd=data_fd,
            template_fd=template_fd,
            psd=psd,
            grid=grid,
            mask=mask,
        )
        assert np.isclose(ln_L_class, ln_L_func, atol=1e-14, rtol=1e-14)

    def test_full_grid_and_active_sliced_map_to_correct_psd_values_and_active_indices(self):
        """Verify full-grid and active-sliced inputs compare against correct coordinates and map to identical active PSD."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)
        idx = grid.get_active_indices(mask)

        # Distinct varying PSD profile across frequencies
        full_psd = 1e-46 * (1.0 + np.sin(2.0 * np.pi * grid.frequencies / 500.0) ** 2)
        act_psd = full_psd[idx]

        data_fd = np.ones(grid.n_frequencies, dtype=np.complex128)
        template_fd = 0.8 * data_fd

        like_full = FrequencyDomainGaussianLikelihood.from_empirical_psd(
            data_fd=data_fd,
            psd=full_psd,
            grid=grid,
            mask=mask,
            psd_frequencies=grid.frequencies,
        )

        like_act = FrequencyDomainGaussianLikelihood.from_empirical_psd(
            data_fd=data_fd,
            psd=act_psd,
            grid=grid,
            mask=mask,
            psd_frequencies=grid.frequencies[idx],
        )

        # 1. Exact equality of internal active PSD values and inverse PSD
        assert np.array_equal(like_full._psd_active, like_act._psd_active)
        assert np.array_equal(like_full._inv_psd_active, like_act._inv_psd_active)
        assert np.array_equal(like_full._data_active, like_act._data_active)

        # 2. Normalization term exact equality
        assert like_full.log_likelihood_norm == like_act.log_likelihood_norm

        # 3. Identical likelihood evaluation
        ln_L_full = like_full.evaluate(template_fd)
        ln_L_act = like_act.evaluate(template_fd)
        assert np.isclose(ln_L_full, ln_L_act, atol=1e-14, rtol=1e-14)

        # 4. Validated frequency shapes and correspondence
        assert like_full.psd_frequencies.shape == (grid.n_frequencies,)
        assert like_act.psd_frequencies.shape == (idx.size,)
        assert np.allclose(like_full.psd_frequencies[idx], like_act.psd_frequencies, atol=1e-12, rtol=1e-12)

    def test_likelihood_rejects_incompatible_array_shapes(self):
        """Verify rejection of a PSD array whose length matches neither full grid nor active bins."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)

        data_fd = np.ones(grid.n_frequencies, dtype=np.complex128)
        incompatible_psd = np.ones(1025, dtype=np.float64)

        with pytest.raises(ValueError, match="does not match full grid"):
            FrequencyDomainGaussianLikelihood(
                data_fd=data_fd,
                psd=incompatible_psd,
                grid=grid,
                mask=mask,
            )

    def test_likelihood_rejects_non_finite_and_non_positive_active_psd(self):
        """Verify rejection of non-finite (NaN, inf) and non-positive (<= 0) PSD values in active bins."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)
        idx = grid.get_active_indices(mask)

        data_fd = np.ones(grid.n_frequencies, dtype=np.complex128)

        # 1. Non-positive in active band
        psd_zero = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)
        psd_zero[idx[50]] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            FrequencyDomainGaussianLikelihood(data_fd=data_fd, psd=psd_zero, grid=grid, mask=mask)

        psd_neg = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)
        psd_neg[idx[50]] = -1e-47
        with pytest.raises(ValueError, match="strictly positive"):
            FrequencyDomainGaussianLikelihood(data_fd=data_fd, psd=psd_neg, grid=grid, mask=mask)

        # 2. Non-finite in active band
        psd_nan = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)
        psd_nan[idx[50]] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            FrequencyDomainGaussianLikelihood(data_fd=data_fd, psd=psd_nan, grid=grid, mask=mask)

        psd_inf = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)
        psd_inf[idx[50]] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            FrequencyDomainGaussianLikelihood(data_fd=data_fd, psd=psd_inf, grid=grid, mask=mask)

    def test_standalone_validate_psd_frequency_alignment(self):
        """Test standalone helper validate_psd_frequency_alignment directly."""
        M = 8192
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=M, delta_t=dt)
        idx = np.array([10, 20, 30])
        psd = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)

        # Valid full grid
        p_act, v_freqs = validate_psd_frequency_alignment(
            psd=psd,
            psd_frequencies=grid.frequencies,
            grid=grid,
            active_indices=idx,
        )
        assert p_act.size == 3
        assert np.array_equal(v_freqs, grid.frequencies)

        # Shape mismatch in frequencies
        with pytest.raises(ValueError, match="psd_frequencies shape"):
            validate_psd_frequency_alignment(
                psd=psd,
                psd_frequencies=grid.frequencies[:100],
                grid=grid,
                active_indices=idx,
            )

        # Non-finite in frequencies
        bad_f = np.array(grid.frequencies, copy=True)
        bad_f[5] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            validate_psd_frequency_alignment(
                psd=psd,
                psd_frequencies=bad_f,
                grid=grid,
                active_indices=idx,
            )
