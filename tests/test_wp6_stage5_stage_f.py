"""WP6 Stage 5F: Controlled CBC Likelihood Validation Unit Tests.

This test module verifies the core mathematical foundations, grid conventions,
parameter point coordinates, and numerical stability for WP6 Stage 5F:
1. Exact normalization integral of the per-bin complex Student-t predictive density.
2. Limiting convergence of Student-t likelihood to Gaussian Whittle as nu_eff -> inf.
3. Strict active-band frequency masking and exhaustive 8-block partition conformance.
4. Canonical injection waveform parameters, optimal SNR recovery (approx 45.3153),
   and evaluation point coordinates (Points A through F).
5. Seed schedule disjointness and single-realization deterministic reproducibility.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pytest
import scipy.integrate as integrate

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.conventions import continuous_rfft
from gw_inference.grid import FrequencyGrid
from gw_inference.noise import generate_colored_noise
from gw_inference.psd import analytical_detector_psd, welch_psd
from gw_inference.waveform import apply_coalescence_time_shift, taylorf2_fd_waveform


# =====================================================================
# SECTION 1: Mathematical Foundations of Candidate Student-t Density
# =====================================================================


class TestCandidateLikelihoodNormalization:
    """Verify mathematical properties of the candidate complex Student-t density."""

    def test_analytical_normalization_integral_substitution(self):
        """Verify the analytical substitution integral:

        integral_0^inf (nu / 2) * (1 + w)^(-(nu / 2 + 1)) dw = 1.0 identically.
        """
        for nu in [2.0, 5.0, 10.0, 44.8522039758, 100.0, 1000.0]:
            # Antiderivative is -(1 + w)^(-nu / 2). At w=0 it is -1, at inf it is 0.
            # Value = 0 - (-1) = 1.0 identically for all nu > 0.
            def integrand(w):
                return 0.5 * nu * ((1.0 + w) ** (-(0.5 * nu + 1.0)))

            val, abserr = integrate.quad(integrand, 0.0, np.inf)
            assert math.isclose(val, 1.0, rel_tol=1e-12, abs_tol=1e-12)
            assert abserr < 1e-8

    def test_complex_plane_radial_quadrature(self):
        """Verify 2D complex plane normalization:

        integral_C p(r_tilde) d^2 r_tilde = integral_0^inf 2 pi r p(r) dr = 1.0
        using standardized radial coordinate u = r / sigma where sigma^2 = S / (2 df).
        """
        nu_eff = 51894.0 / 1157.0  # 44.8522039758
        for nu in [4.0, 10.0, nu_eff, 80.0]:
            # In standardized coordinates u = |r| / sigma:
            # integrand is 2 * u * (1 + 2 * u^2 / nu)^(-(nu / 2 + 1))
            def radial_density(u):
                return 2.0 * u * ((1.0 + (2.0 * (u**2) / nu)) ** (-(0.5 * nu + 1.0)))

            integral_val, abserr = integrate.quad(radial_density, 0.0, np.inf)
            assert math.isclose(integral_val, 1.0, rel_tol=1e-12, abs_tol=1e-12)

    def test_limiting_gaussian_whittle_convergence(self):
        """Verify that as nu -> inf, the Student-t density converges to Gaussian Whittle.

        ln p_t(r) - ln p_Gauss(r) -> 0 as nu -> inf.
        """
        df = 0.25
        psd_val = 2.5e-46
        r_power = 1.2e-46  # arbitrary realistic residual power |r|^2

        # Gaussian Whittle term
        q_plug = (2.0 * df * r_power) / psd_val
        ln_norm = -math.log((math.pi * psd_val) / (2.0 * df))
        ln_L_gauss = ln_norm - q_plug

        # Candidate Student-t terms for large nu
        for nu in [1e4, 1e6, 1e8]:
            u = (2.0 / nu) * q_plug
            ln_fit_t = -(0.5 * nu + 1.0) * math.log(1.0 + u)
            ln_L_t = ln_norm + ln_fit_t
            diff = abs(ln_L_t - ln_L_gauss)
            # Taylor expansion of ln(1 + u) = u - u^2/2 + ...
            # -(nu/2 + 1) * (2/nu * q - 2/nu^2 * q^2) = -q + q^2/nu - 2/nu * q + ...
            # Discrepancy scales as O(1 / nu)
            assert diff < (10.0 * (q_plug**2 + q_plug) / nu)


# =====================================================================
# SECTION 2: Grid and 8-Block Exhaustive Partition Conformance
# =====================================================================


class TestGridAndFrequencyBlocks:
    """Verify grid definition and the 8-block frequency partition."""

    @pytest.fixture
    def setup_grid(self):
        dt = 1.0 / 2048.0
        n_on = 8192
        grid = FrequencyGrid(n_samples=n_on, delta_t=dt)
        mask = (grid.frequencies >= 20.0) & (grid.frequencies <= 512.0)
        return grid, mask

    def test_active_bins_and_mask_conformance(self, setup_grid):
        """Verify exactly 1969 active bins spanning 20.0 to 512.0 Hz."""
        grid, mask = setup_grid
        k_act = np.where(mask)[0]

        assert len(k_act) == 1969
        assert k_act[0] == 80  # 80 * 0.25 Hz = 20.0 Hz
        assert k_act[-1] == 2048  # 2048 * 0.25 Hz = 512.0 Hz
        assert grid.frequencies[k_act[0]] == 20.0
        assert grid.frequencies[k_act[-1]] == 512.0
        # Exclude DC and Nyquist
        assert 0 not in k_act
        assert (grid.n_samples // 2) not in k_act

    def test_exhaustive_8_blocks_structure(self, setup_grid):
        """Verify the 8-block partition is exhaustive and mutually exclusive."""
        grid, mask = setup_grid
        k_act = np.where(mask)[0]

        block_ranges = [
            (80, 325),
            (326, 571),
            (572, 817),
            (818, 1063),
            (1064, 1309),
            (1310, 1555),
            (1556, 1801),
            (1802, 2048),
        ]

        all_block_bins = []
        for b_idx, (k_start, k_end) in enumerate(block_ranges):
            bins = list(range(k_start, k_end + 1))
            all_block_bins.extend(bins)
            if b_idx < 7:
                assert len(bins) == 246, f"Block {b_idx + 1} has {len(bins)} bins, expected 246"
            else:
                assert len(bins) == 247, f"Block {b_idx + 1} has {len(bins)} bins, expected 247"

        # Check total count and uniqueness
        assert len(all_block_bins) == 1969
        assert len(set(all_block_bins)) == 1969
        assert np.array_equal(np.array(all_block_bins), k_act)


# =====================================================================
# SECTION 3: Parameter Evaluation Points & Canonical Injection
# =====================================================================


def generate_test_waveform(
    grid: FrequencyGrid,
    mc_msun: float,
    eta: float,
    tc: float,
    phic: float,
    dist_mpc: float = 410.0,
    f_end: float = 512.0,
) -> np.ndarray:
    """Helper to generate time-shifted TaylorF2 template."""
    delta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    m1_msun = 0.5 * mc_msun * (eta ** (-0.6)) * (1.0 + delta)
    m2_msun = 0.5 * mc_msun * (eta ** (-0.6)) * (1.0 - delta)
    m1_kg = m1_msun * M_SUN_SI
    m2_kg = m2_msun * M_SUN_SI
    dist_m = dist_mpc * MPC_SI

    h_fd = taylorf2_fd_waveform(
        grid.frequencies,
        m1_kg,
        m2_kg,
        dist_m,
        phi_c=phic,
        f_end=f_end,
    )
    return apply_coalescence_time_shift(h_fd, grid.frequencies, tc)


class TestCanonicalWaveformsAndPoints:
    """Verify canonical injection, evaluation points, and optimal SNR."""

    @pytest.fixture
    def setup_env(self):
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=8192, delta_t=dt)
        psd = analytical_detector_psd(grid.frequencies, model="aligo_representative")
        mask = (grid.frequencies >= 20.0) & (grid.frequencies <= 512.0)
        return grid, psd, mask

    def test_canonical_injection_optimal_snr(self, setup_env):
        """Verify Point A (canonical injection) yields optimal SNR approx 45.3153."""
        grid, psd, mask = setup_env
        k_act = np.where(mask)[0]

        # Canonical injection parameters
        mc_inj = 28.095555795460438
        eta_inj = 0.24710059171597634
        tc_inj = 2.0
        phic_inj = 0.0

        h_A = generate_test_waveform(grid, mc_inj, eta_inj, tc_inj, phic_inj)
        snr_sq = float(4.0 * grid.delta_f * np.sum(np.abs(h_A[k_act]) ** 2 / psd[k_act]))
        snr_opt = math.sqrt(snr_sq)

        expected_snr = 45.31529969318099
        assert math.isclose(snr_opt, expected_snr, rel_tol=1e-10)

    def test_all_six_evaluation_points_finiteness_and_offsets(self, setup_env):
        """Verify all 6 evaluation points generate finite, non-zero templates."""
        grid, psd, mask = setup_env
        k_act = np.where(mask)[0]

        points = {
            "A": (28.095555795460438, 0.24710059171597634, 2.0, 0.0, 410.0),
            "B": (28.04322948443798, 0.2447929986739996, 2.000405919519245, 5.388460406503973, 410.0),
            "C": (28.171683991360438, 0.24710059171597634, 2.0, 0.0, 410.0),
            "D": (28.095555795460438, 0.24710059171597634, 2.00060, 0.0, 410.0),
            "E": (28.095555795460438, 0.24710059171597634, 2.0, 0.0, 450.0),
            "F": (28.15000, 0.24000, 2.00200, 1.5707963267948966, 410.0),
        }

        wfs = {}
        for name, coords in points.items():
            h = generate_test_waveform(grid, *coords)
            assert np.all(np.isfinite(h))
            assert np.any(np.abs(h[k_act]) > 0.0)
            wfs[name] = h

        h_A = wfs["A"]
        # Check zero-noise Delta ln L = -0.5 * (h_A - h | h_A - h)
        offsets = {}
        for name, h in wfs.items():
            diff = h_A[k_act] - h[k_act]
            q_diff = float(4.0 * grid.delta_f * np.sum(np.abs(diff) ** 2 / psd[k_act]))
            offsets[name] = -0.5 * q_diff

        assert offsets["A"] == 0.0
        # Point B is near-peak (-3 to -4 nats)
        assert -5.0 < offsets["B"] < -1.0
        # Point C is approx -25 nats
        assert -30.0 < offsets["C"] < -20.0
        # Point D is approx -24 nats
        assert -30.0 < offsets["D"] < -20.0
        # Point E is approx -8 nats
        assert -12.0 < offsets["E"] < -5.0
        # Point F is far shoulder (<-1000 nats)
        assert offsets["F"] < -1000.0


# =====================================================================
# SECTION 4: Reproducibility and Seed Disjointness
# =====================================================================


class TestReproducibilityAndSeedSchedule:
    """Verify seed schedule integrity and deterministic execution."""

    def test_seed_schedule_disjointness(self):
        """Verify Stage 5F seeds (off 6701-6750, on 6751-6800) are disjoint from all prior stages."""
        off_seeds = set(range(6701, 6751))
        on_seeds = set(range(6751, 6801))
        stage_5f_seeds = off_seeds.union(on_seeds)

        assert len(stage_5f_seeds) == 100
        assert len(off_seeds.intersection(on_seeds)) == 0

        # Prior stages:
        # Stage 5B: 6101-6150
        stage_5b = set(range(6101, 6151))
        # Stage 5C/5D: 6201-6250
        stage_5c = set(range(6201, 6251))
        # Stage 5E.1: 6301-6500
        stage_5e1 = set(range(6301, 6501))
        # Stage 5E.2: 6501-6700
        stage_5e2 = set(range(6501, 6701))

        prior_seeds = stage_5b | stage_5c | stage_5e1 | stage_5e2
        assert len(stage_5f_seeds.intersection(prior_seeds)) == 0

    def test_deterministic_single_realization(self):
        """Verify off seed 6701 and on seed 6751 produce identical results across two runs."""
        dt = 1.0 / 2048.0
        grid_on = FrequencyGrid(n_samples=8192, delta_t=dt)
        grid_off = FrequencyGrid(n_samples=131072, delta_t=dt)
        psd_on = analytical_detector_psd(grid_on.frequencies, model="aligo_representative")
        psd_off = analytical_detector_psd(grid_off.frequencies, model="aligo_representative")

        def run_one_trial():
            # Off-source Welch PSD
            rng_off = np.random.default_rng(6701)
            noise_off = generate_colored_noise(
                n_samples=grid_off.n_samples, delta_t=dt, psd=psd_off, generator=rng_off
            )
            _, s_welch = welch_psd(
                noise_off, delta_t=dt, segment_length=8192, overlap_samples=4096, window="tukey"
            )

            # On-source data
            rng_on = np.random.default_rng(6751)
            noise_on = generate_colored_noise(
                n_samples=grid_on.n_samples, delta_t=dt, psd=psd_on, generator=rng_on
            )
            noise_fd = continuous_rfft(noise_on, delta_t=dt)
            return s_welch, noise_fd

        s1, n1 = run_one_trial()
        s2, n2 = run_one_trial()

        assert np.array_equal(s1, s2)
        assert np.array_equal(n1, n2)
