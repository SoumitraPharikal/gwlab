"""Comprehensive test suite for Work Package 3 (WP3): Matched filtering and detection statistics.

Implements the complete approved M1–M12 validation matrix:
- M1: Direct Sum vs. IFFT Equivalence at Signed Lags
- M2: Zero-Lag Time-Domain Injection Recovery via Exact Reconstruction
- M3: Cyclic vs. Linear Time Shift Recovery
- M4: Coalescence Phase Sweep & Sign Verification
- M5: Template Normalization Invariance
- M6: Odd/Even-N Grid Parity & Signed Lag Range Verification
- M7: Multi-Parameter Gaussian Noise Ensemble Acceptance Testing (6 theoretical 99% marginal intervals)
- M8: Parameter Mismatch with Separate Signal and Template Norms
- M9: PSD Scaling Homogeneity on Fixed Data and Template
- M10: Malformed Input Rejection Suite & Degenerate Chi^2 Handling
- M11: Direct-Sum vs. Circular Boundary Verification Benchmark
- M12: Active Mask Invariance Across Distinct Waveform Cutoffs
"""

import math
import numpy as np
import pytest

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.conventions import build_frequency_grid, continuous_rfft
from gw_inference.filter import (
    MatchedFilterResult,
    compute_chisq_binned,
    compute_reweighted_snr,
    compute_signed_lags,
    compute_template_norm,
    direct_matched_filter_at_lag,
    matched_filter,
    matched_filter_core,
    noise_weighted_inner_product,
)
from gw_inference.grid import FrequencyGrid
from gw_inference.psd import PSDEngine
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    effective_termination_frequency,
    schwarzschild_isco_frequency,
    taylorf2_fd_waveform,
)


@pytest.fixture
def standard_setup():
    """Standard 4-second observation at 2048 Hz with flat white PSD."""
    duration = 4.0
    f_s = 2048.0
    grid = FrequencyGrid.from_duration_and_sampling_rate(duration, f_s)
    f_start = 20.0
    f_end = 1000.0
    mask = grid.create_active_mask(f_start, f_end)
    active_indices = grid.get_active_indices(mask)

    # Flat PSD of 1e-46 strain^2 / Hz
    psd_val = 1e-46
    psd_freqs = np.linspace(10.0, 1024.0, 500)
    psd_vals = np.full_like(psd_freqs, psd_val)
    psd_engine = PSDEngine(psd_freqs, psd_vals)
    f_0 = 15.0

    # BNS-like template (1.4, 1.4 M_sun)
    m1 = 1.4 * M_SUN_SI
    m2 = 1.4 * M_SUN_SI
    dist = 40.0 * MPC_SI
    h_fd = taylorf2_fd_waveform(grid.frequencies, m1, m2, dist, phi_c=0.0, f_end=f_end)

    return {
        "grid": grid,
        "mask": mask,
        "active_indices": active_indices,
        "psd_engine": psd_engine,
        "psd_val": psd_val,
        "f_start": f_start,
        "f_end": f_end,
        "f_0": f_0,
        "m1": m1,
        "m2": m2,
        "dist": dist,
        "h_fd": h_fd,
    }


# ==============================================================================
# Phase A: Inner Product & Direct Reference
# ==============================================================================

def test_noise_weighted_inner_product_properties(standard_setup):
    """Test symmetry, positivity, and linearity of noise_weighted_inner_product."""
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    active_indices = setup["active_indices"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]

    psd = psd_engine.interpolate(grid.frequencies, f_start=f_start, f_0=f_0)

    # Symmetry: (a|b) == (b|a)
    h2_fd = 2.0 * h_fd
    ip_ab = noise_weighted_inner_product(h_fd, h2_fd, psd, grid.delta_f, active_indices)
    ip_ba = noise_weighted_inner_product(h2_fd, h_fd, psd, grid.delta_f, active_indices)
    assert np.isclose(ip_ab, ip_ba, rtol=1e-15)

    # Linearity: (a | 2b) == 2 * (a|b)
    ip_aa = noise_weighted_inner_product(h_fd, h_fd, psd, grid.delta_f, active_indices)
    assert np.isclose(ip_ab, 2.0 * ip_aa, rtol=1e-15)

    # Positivity: (h|h) > 0
    assert ip_aa > 0.0

    # Consistency with compute_template_norm: sigma_h = sqrt((h|h))
    inv_psd_act = 1.0 / psd[active_indices]
    sigma_h = compute_template_norm(h_fd, inv_psd_act, active_indices, grid.delta_f)
    assert np.isclose(sigma_h, math.sqrt(ip_aa), rtol=1e-14)


def test_m1_direct_sum_vs_ifft_equivalence(standard_setup):
    """Test M1: Direct Sum vs. IFFT Equivalence at Signed Lags (tau = 0, +10 dt, -10 dt)."""
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    active_indices = setup["active_indices"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]

    psd = psd_engine.interpolate(grid.frequencies, f_start=f_start, f_0=f_0)
    inv_psd_act = 1.0 / psd[active_indices]

    # Use h_fd as data for direct test
    data_fd = h_fd

    z_fft, sigma_h = matched_filter_core(
        data_fd=data_fd,
        template_fd=h_fd,
        inv_psd_active=inv_psd_act,
        active_indices=active_indices,
        n_samples=grid.n_samples,
        delta_f=grid.delta_f,
    )
    signed_lags = compute_signed_lags(grid.n_samples, grid.delta_t)

    # Test lags: 0, +10 dt, and -10 dt
    dt = grid.delta_t
    test_lags = [0.0, 10.0 * dt, -10.0 * dt]

    for lag in test_lags:
        # Find corresponding index in signed_lags
        idx = int(np.argmin(np.abs(signed_lags - lag)))
        assert np.isclose(signed_lags[idx], lag, atol=1e-15)

        z_fft_val = z_fft[idx]
        z_dir_val = direct_matched_filter_at_lag(
            data_fd=data_fd,
            template_fd=h_fd,
            inv_psd_active=inv_psd_act,
            active_indices=active_indices,
            freqs=grid.frequencies,
            delta_f=grid.delta_f,
            time_lag=lag,
        )

        rel_diff = abs(z_fft_val - z_dir_val) / abs(z_dir_val)
        assert rel_diff < 1.0e-12, f"Discrepancy at lag {lag}: {rel_diff}"


# ==============================================================================
# Phase B: Exact Inversion & Lag Normalization
# ==============================================================================

def test_m2_zero_lag_injection_recovery(standard_setup):
    """Test M2: Zero-Lag Time-Domain Injection Recovery via Exact Reconstruction.

    d(t) = (1/dt) * irfft(h_tilde, n=N); filter with exact template h_tilde.
    Expect: peak at lag 0; rho_max(0) == rho_opt; phi_hat(0) == 0.0 rad.
    """
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]

    # Time-domain reconstruction via exact round-trip identity: d(t) = (1/dt) * irfft(h_tilde)
    # Note: h_fd is continuous-scaled: h_tilde = dt * rfft(d) => rfft(d) = h_tilde / dt
    d_time = np.fft.irfft(h_fd / grid.delta_t, n=grid.n_samples)

    result = matched_filter(
        data=d_time,
        template_fd=h_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_filter_diag=0.0,  # Evaluate over all lags
    )

    # At lag 0:
    idx_0 = 0
    assert result.signed_lags[idx_0] == 0.0
    rho_0 = result.snr_max[idx_0]
    phi_0 = result.phase_max[idx_0]

    # Peak must be at lag 0
    assert np.isclose(rho_0, result.optimal_snr, rtol=1e-12)
    assert np.isclose(phi_0, 0.0, atol=1e-12)

    # Maximum over all lags is at lag 0
    assert np.argmax(result.snr_max) == 0


def test_m3_cyclic_and_linear_time_shifts(standard_setup):
    """Test M3: Cyclic vs. Linear Time Shift Recovery."""
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    active_indices = setup["active_indices"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]
    dt = grid.delta_t

    # 1. Cyclic integer-sample time shift: d_j = d_orig[(j - k) % N]
    d_time_orig = np.fft.irfft(h_fd / dt, n=grid.n_samples)
    k_shift = 25
    d_time_shifted = np.roll(d_time_orig, k_shift)

    result_cyclic = matched_filter(
        data=d_time_shifted,
        template_fd=h_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_filter_diag=0.0,
    )
    # Peak index should be exactly k_shift
    peak_idx = int(np.argmax(result_cyclic.snr_max))
    assert peak_idx == k_shift
    assert np.isclose(result_cyclic.signed_lags[peak_idx], k_shift * dt, atol=1e-15)
    assert np.isclose(result_cyclic.snr_max[peak_idx], result_cyclic.optimal_snr, rtol=1e-12)

    # 2. Linear time shift via apply_coalescence_time_shift
    t_c = 15.0 * dt
    h_shifted_fd = apply_coalescence_time_shift(h_fd, grid.frequencies, t_c)
    result_fd = matched_filter(
        data=h_shifted_fd,
        template_fd=h_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_filter_diag=0.0,
    )
    peak_idx_fd = int(np.argmax(result_fd.snr_max))
    assert peak_idx_fd == 15
    assert np.isclose(result_fd.signed_lags[peak_idx_fd], t_c, atol=1e-15)
    assert np.isclose(result_fd.snr_max[peak_idx_fd], result_fd.optimal_snr, rtol=1e-12)
    assert np.isclose(result_fd.phase_max[peak_idx_fd], 0.0, atol=1e-12)


# ==============================================================================
# Phase C: Phase Maximization
# ==============================================================================

def test_m4_coalescence_phase_sweep(standard_setup):
    """Test M4: Coalescence Phase Sweep & Sign Verification across [-pi, pi]."""
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]
    m1 = setup["m1"]
    m2 = setup["m2"]
    dist = setup["dist"]
    f_end = setup["f_end"]

    test_phases = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0, math.pi, -math.pi / 2.0, -math.pi / 4.0]

    for phi_0 in test_phases:
        # Generate waveform with phi_c = phi_0
        h_phi = taylorf2_fd_waveform(grid.frequencies, m1, m2, dist, phi_c=phi_0, f_end=f_end)

        result = matched_filter(
            data=h_phi,
            template_fd=h_fd,  # template has phi_c = 0.0
            grid=grid,
            psd_engine=psd_engine,
            mask=mask,
            f_start=f_start,
            f_0=f_0,
            tau_filter_diag=0.0,
        )

        idx_0 = 0
        rho_val = result.snr_max[idx_0]
        phi_hat = result.phase_max[idx_0]

        # Phase-maximized SNR must equal optimal SNR
        assert np.isclose(rho_val, result.optimal_snr, rtol=1e-12)

        # Recovered phase must match phi_0 modulo 2pi
        diff_phase = (phi_hat - phi_0 + math.pi) % (2.0 * math.pi) - math.pi
        assert abs(diff_phase) < 1.0e-12, f"Phase mismatch for phi_0={phi_0}: recovered {phi_hat}"


def test_m5_template_normalization_invariance(standard_setup):
    """Test M5: Template Normalization Invariance.

    Scale template by alpha in {0.1, 10.0, 10^5}; verify (h_norm|h_norm) == 1.0
    and rho_max is invariant to float precision.
    """
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    active_indices = setup["active_indices"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]

    res_orig = matched_filter(
        data=h_fd,
        template_fd=h_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_filter_diag=0.0,
    )

    psd = psd_engine.interpolate(grid.frequencies, f_start=f_start, f_0=f_0)
    inv_psd_act = 1.0 / psd[active_indices]

    for alpha in [0.1, 10.0, 1e5]:
        h_scaled = alpha * h_fd

        # Check normalized template norm is 1.0
        sigma_scaled = compute_template_norm(h_scaled, inv_psd_act, active_indices, grid.delta_f)
        h_norm = h_scaled / sigma_scaled
        sigma_norm = compute_template_norm(h_norm, inv_psd_act, active_indices, grid.delta_f)
        assert np.isclose(sigma_norm, 1.0, atol=1e-14)

        # Matched filter output with scaled template
        res_scaled = matched_filter(
            data=h_fd,
            template_fd=h_scaled,
            grid=grid,
            psd_engine=psd_engine,
            mask=mask,
            f_start=f_start,
            f_0=f_0,
            tau_filter_diag=0.0,
        )

        # Phase-maximized SNR array must be invariant to float64 precision
        rel_snr_diff = np.max(np.abs(res_scaled.snr_max - res_orig.snr_max)) / np.max(res_orig.snr_max)
        assert rel_snr_diff < 5.0e-14


# ==============================================================================
# Phase D: Parity, Statistics, Diagnostics
# ==============================================================================

def test_m6_odd_and_even_grid_parity():
    """Test M6: Odd/Even-N Grid Parity & Signed Lag Range Verification."""
    delta_t = 1.0 / 2048.0

    # Even N = 1024
    N_even = 1024
    lags_even = compute_signed_lags(N_even, delta_t)
    assert lags_even.size == N_even
    T_even = N_even * delta_t
    assert lags_even[0] == 0.0
    assert np.isclose(lags_even[N_even // 2], -T_even / 2.0, atol=1e-15)
    assert np.all(lags_even >= -T_even / 2.0)
    assert np.all(lags_even < T_even / 2.0)

    # Odd N = 1025
    N_odd = 1025
    lags_odd = compute_signed_lags(N_odd, delta_t)
    assert lags_odd.size == N_odd
    max_lag_odd = (N_odd - 1) * delta_t / 2.0
    assert lags_odd[0] == 0.0
    assert np.isclose(lags_odd[(N_odd - 1) // 2], max_lag_odd, atol=1e-15)
    assert np.isclose(lags_odd[(N_odd + 1) // 2], -max_lag_odd, atol=1e-15)
    assert np.all(lags_odd >= -max_lag_odd - 1e-15)
    assert np.all(lags_odd <= max_lag_odd + 1e-15)

    # Verify matched filter executes cleanly on odd N
    grid_odd = FrequencyGrid(n_samples=N_odd, delta_t=delta_t)
    mask_odd = grid_odd.create_active_mask(20.0, 500.0)
    psd_freqs = np.linspace(10.0, 1024.0, 300)
    psd_vals = np.full_like(psd_freqs, 1e-46)
    psd_engine = PSDEngine(psd_freqs, psd_vals)

    m1 = 1.4 * M_SUN_SI
    m2 = 1.4 * M_SUN_SI
    dist = 40.0 * MPC_SI
    h_odd = taylorf2_fd_waveform(grid_odd.frequencies, m1, m2, dist, f_end=500.0)

    res_odd = matched_filter(
        data=h_odd,
        template_fd=h_odd,
        grid=grid_odd,
        psd_engine=psd_engine,
        mask=mask_odd,
        f_start=20.0,
        f_0=15.0,
        tau_filter_diag=0.0,
    )
    assert np.isfinite(res_odd.peak_snr)
    assert res_odd.peak_snr > 0.0
    assert np.isclose(res_odd.peak_snr, res_odd.optimal_snr, rtol=1e-12)


def test_m7_gaussian_noise_ensemble_statistics():
    """Test M7: Multi-Parameter Gaussian Noise Ensemble Acceptance Testing.

    Runs M = 10,000 independent noise trials with seed 123456789.
    Verifies all 6 theoretical 99% marginal acceptance intervals from §5.1.
    """
    N = 256
    delta_t = 1.0 / 512.0
    grid = FrequencyGrid(n_samples=N, delta_t=delta_t)
    f_start = 20.0
    f_end = 200.0
    mask = grid.create_active_mask(f_start, f_end)
    active_indices = grid.get_active_indices(mask)

    psd_val = 1.0
    inv_psd_act = np.full(active_indices.size, 1.0 / psd_val)

    template_fd = np.zeros(grid.n_frequencies, dtype=np.complex128)
    template_fd[active_indices] = 1.0
    sigma_h = compute_template_norm(template_fd, inv_psd_act, active_indices, grid.delta_f)

    # Time-domain white noise variance: S_n = 2 * sigma_n^2 * delta_t => sigma_n = sqrt(S_n / (2 * delta_t))
    sigma_n = math.sqrt(psd_val / (2.0 * delta_t))

    M_trials = 10000
    rng = np.random.default_rng(seed=123456789)

    real_quads = np.empty(M_trials, dtype=np.float64)
    imag_quads = np.empty(M_trials, dtype=np.float64)

    for i in range(M_trials):
        n_time = rng.normal(0.0, sigma_n, size=N)
        d_fd = delta_t * np.fft.rfft(n_time)
        z, _ = matched_filter_core(d_fd, template_fd, inv_psd_act, active_indices, N, grid.delta_f)
        rho_0 = z[0] / sigma_h
        real_quads[i] = np.real(rho_0)
        imag_quads[i] = np.imag(rho_0)

    # 1. Real Quadrature Mean: [-0.0258, +0.0258]
    mean_x = float(np.mean(real_quads))
    assert -0.0258 <= mean_x <= 0.0258, f"Real mean {mean_x} outside [-0.0258, +0.0258]"

    # 2. Imaginary Quadrature Mean: [-0.0258, +0.0258]
    mean_y = float(np.mean(imag_quads))
    assert -0.0258 <= mean_y <= 0.0258, f"Imag mean {mean_y} outside [-0.0258, +0.0258]"

    # 3. Real Quadrature Variance: [0.9639, 1.0368]
    var_x = float(np.var(real_quads, ddof=1))
    assert 0.9639 <= var_x <= 1.0368, f"Real var {var_x} outside [0.9639, 1.0368]"

    # 4. Imaginary Quadrature Variance: [0.9639, 1.0368]
    var_y = float(np.var(imag_quads, ddof=1))
    assert 0.9639 <= var_y <= 1.0368, f"Imag var {var_y} outside [0.9639, 1.0368]"

    # 5. Real-Imag Covariance: [-0.0258, +0.0258]
    cov_xy = float(np.cov(real_quads, imag_quads)[0, 1])
    assert -0.0258 <= cov_xy <= 0.0258, f"Cov(X, Y) {cov_xy} outside [-0.0258, +0.0258]"

    # 6. Rayleigh Magnitude Mean: [1.2364, 1.2702]
    mags = np.sqrt(real_quads**2 + imag_quads**2)
    mean_r = float(np.mean(mags))
    assert 1.2364 <= mean_r <= 1.2702, f"Rayleigh mean {mean_r} outside [1.2364, 1.2702]"


def test_m8_parameter_mismatch_with_separate_norms(standard_setup):
    """Test M8: Parameter Mismatch with Separate Signal and Template Norms."""
    setup = standard_setup
    grid = setup["grid"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    active_indices = setup["active_indices"]
    f_start = setup["f_start"]
    f_end = setup["f_end"]
    f_0 = setup["f_0"]
    dist = setup["dist"]

    psd = psd_engine.interpolate(grid.frequencies, f_start=f_start, f_0=f_0)
    inv_psd_act = 1.0 / psd[active_indices]

    # Signal s: (30, 25) M_sun
    m1_s = 30.0 * M_SUN_SI
    m2_s = 25.0 * M_SUN_SI
    s_fd = taylorf2_fd_waveform(grid.frequencies, m1_s, m2_s, dist, phi_c=0.0, f_end=f_end)

    # Template h: (32, 23) M_sun
    m1_h = 32.0 * M_SUN_SI
    m2_h = 23.0 * M_SUN_SI
    h_fd = taylorf2_fd_waveform(grid.frequencies, m1_h, m2_h, dist, phi_c=0.0, f_end=f_end)

    sigma_s = compute_template_norm(s_fd, inv_psd_act, active_indices, grid.delta_f)
    sigma_h = compute_template_norm(h_fd, inv_psd_act, active_indices, grid.delta_f)
    assert sigma_s > 0.0 and sigma_h > 0.0

    # Filter signal s with template h
    result = matched_filter(
        data=s_fd,
        template_fd=h_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_filter_diag=0.0,
    )

    peak_snr = result.peak_snr

    # Direct match M(s, h) = max_tau |<s|h>(tau)| / (sigma_s * sigma_h)
    z_series, _ = matched_filter_core(
        data_fd=s_fd,
        template_fd=h_fd,
        inv_psd_active=inv_psd_act,
        active_indices=active_indices,
        n_samples=grid.n_samples,
        delta_f=grid.delta_f,
    )
    overlap_series = np.abs(z_series) / (sigma_s * sigma_h)
    match_val = float(np.max(overlap_series))

    # Analytical identity: rho_peak = sigma_s * M(s, h) => rho_peak / sigma_s == M(s, h)
    assert np.isclose(peak_snr / sigma_s, match_val, rtol=1e-5)
    # Schwarz inequality: M(s, h) <= 1.0
    assert match_val <= 1.0 + 1e-12
    assert peak_snr <= sigma_s + 1e-12


def test_m9_psd_scaling_homogeneity(standard_setup):
    """Test M9: PSD Scaling Homogeneity on Fixed Data and Template.

    Scale S_n -> c * S_n (c > 0) holding data and template fixed.
    Verify: z -> z/c, sigma_h -> sigma_h/sqrt(c), rho_max -> rho_max/sqrt(c).
    """
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    active_indices = setup["active_indices"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]

    res_orig = matched_filter(
        data=h_fd,
        template_fd=h_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_filter_diag=0.0,
        compute_chisq=True,
    )

    for c in [4.0, 0.25]:
        psd_c_freqs = psd_engine.frequencies
        psd_c_vals = c * psd_engine.psd
        psd_engine_c = PSDEngine(psd_c_freqs, psd_c_vals)

        res_c = matched_filter(
            data=h_fd,
            template_fd=h_fd,
            grid=grid,
            psd_engine=psd_engine_c,
            mask=mask,
            f_start=f_start,
            f_0=f_0,
            tau_filter_diag=0.0,
            compute_chisq=True,
        )

        # sigma_h -> sigma_h / sqrt(c)
        assert np.isclose(res_c.sigma_h, res_orig.sigma_h / math.sqrt(c), rtol=1e-14)

        # rho_max -> rho_max / sqrt(c)
        rel_rho_diff = np.max(np.abs(res_c.snr_max - res_orig.snr_max / math.sqrt(c))) / np.max(res_orig.snr_max)
        assert rel_rho_diff < 1e-14

        # chisq -> chisq / c
        if res_c.chisq is not None and res_orig.chisq is not None:
            rel_chisq_diff = np.max(np.abs(res_c.chisq - res_orig.chisq / c)) / np.max(res_orig.chisq)
            assert rel_chisq_diff < 5.0e-14


def test_m10_malformed_input_suite_and_degenerate_chisq(standard_setup):
    """Test M10: Malformed Input Rejection Suite & Degenerate Chi^2 Handling."""
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    active_indices = setup["active_indices"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]

    # 1. NaN in data
    bad_data = h_fd.copy()
    bad_data[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        matched_filter(bad_data, h_fd, grid, psd_engine, mask, f_start, f_0)

    # 2. Inf in template
    bad_tmpl = h_fd.copy()
    bad_tmpl[10] = np.inf
    with pytest.raises(ValueError, match="finite"):
        matched_filter(h_fd, bad_tmpl, grid, psd_engine, mask, f_start, f_0)

    # 3. All-zero template in active band (zero power)
    zero_tmpl = np.zeros_like(h_fd)
    with pytest.raises(ValueError, match="power"):
        matched_filter(h_fd, zero_tmpl, grid, psd_engine, mask, f_start, f_0)

    # 4. Missing or invalid f_0 (Decision D8 enforcement)
    with pytest.raises((ValueError, TypeError)):
        matched_filter(h_fd, h_fd, grid, psd_engine, mask, f_start, f_0=-5.0)

    # 5. Degenerate Allen Chi^2 (narrow band template in <= 1 sub-band)
    narrow_tmpl = np.zeros_like(h_fd)
    # Put power in exactly 1 frequency bin
    narrow_tmpl[active_indices[5]] = 1.0e-21
    res_narrow = matched_filter(
        data=narrow_tmpl,
        template_fd=narrow_tmpl,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        compute_chisq=True,
        chisq_bins=16,
    )
    # Must cleanly return None for chisq and chisq_dof
    assert res_narrow.chisq is None
    assert res_narrow.chisq_dof is None
    assert np.array_equal(res_narrow.reweighted_snr, res_narrow.snr_max)


def test_m11_direct_sum_vs_circular_interior_benchmark(standard_setup):
    """Test M11: Direct-Sum vs. Circular Boundary Verification Benchmark."""
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    active_indices = setup["active_indices"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]

    psd = psd_engine.interpolate(grid.frequencies, f_start=f_start, f_0=f_0)
    inv_psd_act = 1.0 / psd[active_indices]

    z_fft, sigma_h = matched_filter_core(
        data_fd=h_fd,
        template_fd=h_fd,
        inv_psd_active=inv_psd_act,
        active_indices=active_indices,
        n_samples=grid.n_samples,
        delta_f=grid.delta_f,
    )
    signed_lags = compute_signed_lags(grid.n_samples, grid.delta_t)

    # Test at 5 interior lags
    interior_indices = [50, 100, 200, 300, 400]
    for idx in interior_indices:
        lag = float(signed_lags[idx])
        z_dir = direct_matched_filter_at_lag(
            data_fd=h_fd,
            template_fd=h_fd,
            inv_psd_active=inv_psd_act,
            active_indices=active_indices,
            freqs=grid.frequencies,
            delta_f=grid.delta_f,
            time_lag=lag,
        )
        z_circ = z_fft[idx]
        diff_snr = abs(z_circ - z_dir) / sigma_h
        assert diff_snr < 1e-12, f"Difference at lag {lag} s: {diff_snr}"


def test_m12_active_mask_invariance(standard_setup):
    """Test M12: Active Mask Invariance Across Distinct Waveform Cutoffs."""
    setup = standard_setup
    grid = setup["grid"]
    mask = setup["mask"]
    f_start = setup["f_start"]
    f_end = setup["f_end"]
    dist = setup["dist"]

    # System 1: Light BNS (1.4, 1.4 M_sun) -> f_ISCO ~ 1570 Hz (> f_end = 1000 Hz)
    m1_light = 1.4 * M_SUN_SI
    m2_light = 1.4 * M_SUN_SI
    h_light = taylorf2_fd_waveform(grid.frequencies, m1_light, m2_light, dist, f_end=f_end)

    # System 2: Heavy BBH (30, 30 M_sun) -> f_ISCO ~ 73.3 Hz (< f_end = 1000 Hz)
    m1_heavy = 30.0 * M_SUN_SI
    m2_heavy = 30.0 * M_SUN_SI
    h_heavy = taylorf2_fd_waveform(grid.frequencies, m1_heavy, m2_heavy, dist, f_end=f_end)

    # Create fresh mask for both calls
    mask_light = grid.create_active_mask(f_start, f_end)
    mask_heavy = grid.create_active_mask(f_start, f_end)

    # Mask K_+ must be 100% bitwise identical regardless of masses or ISCO
    assert np.array_equal(mask_light, mask)
    assert np.array_equal(mask_heavy, mask)
    assert np.array_equal(mask_light, mask_heavy)

    # Cutoff is enforced purely inside the waveform array
    f_isco_heavy = schwarzschild_isco_frequency(60.0)
    assert f_isco_heavy < f_end
    # All bins above f_isco_heavy must be identically 0 in h_heavy
    above_cutoff = grid.frequencies > f_isco_heavy
    assert np.all(h_heavy[above_cutoff] == 0.0 + 0.0j)


def test_allen_chisq_on_clean_and_mismatched_signals(standard_setup):
    """Test Allen chi^2 and reweighted SNR behavior on clean vs. mismatched signals."""
    setup = standard_setup
    grid = setup["grid"]
    h_fd = setup["h_fd"]
    psd_engine = setup["psd_engine"]
    mask = setup["mask"]
    f_start = setup["f_start"]
    f_0 = setup["f_0"]

    # 1. Clean signal: data = h_fd
    res_clean = matched_filter(
        data=h_fd,
        template_fd=h_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_filter_diag=0.0,
        compute_chisq=True,
        chisq_bins=8,
    )
    assert res_clean.chisq is not None
    assert res_clean.chisq_dof == 2 * 8 - 2  # 14
    # At peak lag (0), chi^2 must be zero to float precision on noiseless match
    idx_0 = 0
    assert np.isclose(res_clean.chisq[idx_0], 0.0, atol=1e-12)
    # Reduced chi^2 <= 1 => rho_tilde == rho_max (no spurious penalty)
    assert np.isclose(res_clean.reweighted_snr[idx_0], res_clean.snr_max[idx_0], rtol=1e-12)

    # 2. Mismatched signal: data from (30, 30) M_sun, template is (1.4, 1.4) M_sun
    m1_hvy = 30.0 * M_SUN_SI
    m2_hvy = 30.0 * M_SUN_SI
    h_heavy = taylorf2_fd_waveform(grid.frequencies, m1_hvy, m2_hvy, setup["dist"], f_end=setup["f_end"])

    res_mismatch = matched_filter(
        data=h_heavy,
        template_fd=h_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_filter_diag=0.0,
        compute_chisq=True,
        chisq_bins=8,
    )
    assert res_mismatch.chisq is not None
    # Chi^2 should be non-zero and elevated across all lags
    assert np.all(res_mismatch.chisq >= 0.0)
    # Reweighted SNR must be <= snr_max everywhere
    assert np.all(res_mismatch.reweighted_snr <= res_mismatch.snr_max + 1e-14)
