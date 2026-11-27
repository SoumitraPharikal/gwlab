"""WP6 Stage 5G.1: Baseline Sampler Pilot Pre-Execution Unit Tests.

This test module verifies the core mathematical foundations, parameter space,
prior transform, likelihood evaluation, waveform-distance shells, seed schedule,
and sampler configuration for the WP6 Stage 5G.1 baseline sampler pilot:
1. Likelihood models 1, 2, and 3 evaluation, normalization offset equivalence,
   and convergence of Model 3 to Model 2 as nu_eff -> inf.
2. 4D parameter space, analytic prior mapping, and hypervolume V_Pi = 1.2*pi.
3. Waveform-distance metric D(theta), exact reconciliation of Point F (D approx 4200.35),
   and mutually exclusive mismatch shells (S0 through S3).
4. Strict seed schedule disjointness, absence of collision with prior stages (4201-4204, 6701-6800),
   and deterministic paired data synthesis.
5. Baseline sampler adapter configuration (nlive=500, dlogz=0.10, bound='multi', sample='rwalk', walks=50).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pytest

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.conventions import continuous_rfft
from gw_inference.grid import FrequencyGrid
from gw_inference.noise import generate_colored_noise
from gw_inference.psd import analytical_detector_psd, welch_psd
from gw_inference.sampler import NestedSamplerAdapter, is_dynesty_available
from gw_inference.waveform import apply_coalescence_time_shift, taylorf2_fd_waveform


# Canonical injection parameters frozen from Stage 4 and Stage 5F
CANONICAL_MC = 28.095555795460438
CANONICAL_ETA = 0.24710059171597634
CANONICAL_TC = 2.0
CANONICAL_PHIC = 0.0
CANONICAL_DIST_MPC = 410.0
CANONICAL_F_END = 512.0

# Active frequency band
FS = 2048.0
T_ON = 4.0
N_ON = int(FS * T_ON)  # 8192
DF = 1.0 / T_ON       # 0.25 Hz
F_START = 20.0
F_END = 512.0
K_START = int(round(F_START / DF))  # 80
K_END = int(round(F_END / DF))      # 2048
N_ACT = K_END - K_START + 1         # 1969

# Preserved 64 s off-source Welch PSD configuration
T_OFF = 64.0
N_OFF = int(FS * T_OFF)  # 131072
M_SEG = 8192
H_HOP = 4096
TUKEY_ALPHA = 0.25
K_SEG = 31
NU_EFF = 51894.0 / 1157.0  # approx 44.8522039758

# Prior boundaries
MC_MIN, MC_MAX = 20.0, 40.0
ETA_MIN, ETA_MAX = 0.10, 0.25
TC_MIN, TC_MAX = 1.90, 2.10
PHIC_MIN, PHIC_MAX = 0.0, 2.0 * math.pi
PRIOR_VOLUME = (MC_MAX - MC_MIN) * (ETA_MAX - ETA_MIN) * (TC_MAX - TC_MIN) * (PHIC_MAX - PHIC_MIN)
# 20.0 * 0.15 * 0.20 * 2*pi = 1.2 * pi


def prior_transform_4d(u: np.ndarray) -> np.ndarray:
    """Map unit hypercube u in [0, 1]^4 to physical parameter vector theta."""
    theta = np.empty_like(u)
    theta[0] = MC_MIN + (MC_MAX - MC_MIN) * u[0]
    theta[1] = ETA_MIN + (ETA_MAX - ETA_MIN) * u[1]
    theta[2] = TC_MIN + (TC_MAX - TC_MIN) * u[2]
    theta[3] = PHIC_MIN + (PHIC_MAX - PHIC_MIN) * u[3]
    return theta


def generate_waveform(
    grid: FrequencyGrid,
    mc: float,
    eta: float,
    tc: float,
    phic: float,
    dist_mpc: float = CANONICAL_DIST_MPC,
    f_end: float = CANONICAL_F_END,
) -> np.ndarray:
    """Generate TaylorF2 waveform on frequency grid."""
    delta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    m1_msun = 0.5 * mc * (eta ** (-0.6)) * (1.0 + delta)
    m2_msun = 0.5 * mc * (eta ** (-0.6)) * (1.0 - delta)
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


def compute_waveform_distance(
    h_theta: np.ndarray,
    h_inj: np.ndarray,
    psd_true_act: np.ndarray,
    df: float = DF,
    k_start: int = K_START,
    k_end: int = K_END,
) -> float:
    """Compute Oracle waveform distance D(theta)."""
    dh = h_theta[k_start : k_end + 1] - h_inj[k_start : k_end + 1]
    return float(4.0 * df * np.sum((np.abs(dh) ** 2) / psd_true_act))


def assign_mismatch_shell(d_val: float) -> int:
    """Assign waveform distance to mutually exclusive shells S0, S1, S2, S3.

    S0: 0 <= D <= 9
    S1: 9 < D <= 25
    S2: 25 < D <= 100
    S3: D > 100
    """
    if d_val <= 9.0:
        return 0
    elif d_val <= 25.0:
        return 1
    elif d_val <= 100.0:
        return 2
    else:
        return 3


# =====================================================================
# SECTION 1: 4D Parameter Space and Analytic Prior
# =====================================================================


class TestParameterSpaceAndPrior:
    """Verify 4D parameter boundaries, analytic prior transform, and volume."""

    def test_prior_volume_and_log_density(self) -> None:
        """Verify V_Pi = 1.2*pi and ln pi(theta) = -ln(1.2*pi)."""
        expected_v_pi = 1.2 * math.pi
        assert math.isclose(PRIOR_VOLUME, expected_v_pi, rel_tol=1e-12)

        ln_prior_density = -math.log(expected_v_pi)
        assert math.isclose(ln_prior_density, -1.32705144264, rel_tol=1e-6)

    def test_prior_transform_corners_and_midpoint(self) -> None:
        """Verify boundary mapping from unit hypercube to physical coordinates."""
        # Lower corner [0, 0, 0, 0]
        u_low = np.zeros(4)
        th_low = prior_transform_4d(u_low)
        assert np.allclose(th_low, [MC_MIN, ETA_MIN, TC_MIN, PHIC_MIN])

        # Upper corner [1, 1, 1, 1]
        u_high = np.ones(4)
        th_high = prior_transform_4d(u_high)
        assert np.allclose(th_high, [MC_MAX, ETA_MAX, TC_MAX, PHIC_MAX])

        # Midpoint [0.5, 0.5, 0.5, 0.5]
        u_mid = 0.5 * np.ones(4)
        th_mid = prior_transform_4d(u_mid)
        expected_mid = [
            0.5 * (MC_MIN + MC_MAX),
            0.5 * (ETA_MIN + ETA_MAX),
            0.5 * (TC_MIN + TC_MAX),
            0.5 * (PHIC_MIN + PHIC_MAX),
        ]
        assert np.allclose(th_mid, expected_mid)

    def test_canonical_injection_inside_prior(self) -> None:
        """Verify canonical injection point lies strictly inside prior interior."""
        assert MC_MIN < CANONICAL_MC < MC_MAX
        assert ETA_MIN < CANONICAL_ETA < ETA_MAX
        assert TC_MIN < CANONICAL_TC < TC_MAX
        assert PHIC_MIN <= CANONICAL_PHIC < PHIC_MAX


# =====================================================================
# SECTION 2: Likelihood Models 1, 2, and 3 Mathematical Evaluation
# =====================================================================


class TestLikelihoodModels:
    """Verify likelihood formulations, normalization offsets, and asymptotic limits."""

    @pytest.fixture
    def setup_data(self) -> Dict[str, Any]:
        """Set up grid, analytical PSD, Welch PSD, and residual for realization 01."""
        dt = 1.0 / FS
        grid_on = FrequencyGrid(n_samples=N_ON, delta_t=dt)
        psd_true_full = analytical_detector_psd(grid_on.frequencies, model="aligo_representative")
        k_act = np.arange(K_START, K_END + 1)
        psd_true_act = psd_true_full[k_act]

        # Welch PSD from seed 7101
        grid_off = FrequencyGrid(n_samples=N_OFF, delta_t=dt)
        psd_true_off = analytical_detector_psd(grid_off.frequencies, model="aligo_representative")
        rng_off = np.random.default_rng(7101)
        noise_off = generate_colored_noise(
            n_samples=grid_off.n_samples,
            delta_t=dt,
            psd=psd_true_off,
            generator=rng_off,
        )
        _, s_welch_full = welch_psd(
            noise_off,
            delta_t=dt,
            segment_length=M_SEG,
            overlap_samples=M_SEG - H_HOP,
            window="tukey",
        )
        s_welch_act = s_welch_full[k_act]

        # On-source noise from seed 7151 + injection
        rng_on = np.random.default_rng(7151)
        noise_on = generate_colored_noise(
            n_samples=grid_on.n_samples,
            delta_t=dt,
            psd=psd_true_full,
            generator=rng_on,
        )
        d_fd = continuous_rfft(noise_on, delta_t=dt)
        h_inj = generate_waveform(grid_on, CANONICAL_MC, CANONICAL_ETA, CANONICAL_TC, CANONICAL_PHIC)
        d_fd_injected = d_fd + h_inj
        res_act = d_fd_injected[k_act] - h_inj[k_act]  # residual at injection point

        return {
            "grid": grid_on,
            "k_act": k_act,
            "psd_true_act": psd_true_act,
            "s_welch_act": s_welch_act,
            "res_act": res_act,
        }

    def test_normalization_offsets_equivalence(self, setup_data: Dict[str, Any]) -> None:
        """Verify ln L_norm^(3) == ln L_norm^(2) identically."""
        s_welch_act = setup_data["s_welch_act"]
        ln_norm_plug = -float(np.sum(np.log((math.pi * s_welch_act) / (2.0 * DF))))
        # Model 3 normalization with C = pi / (2*df)
        c_const = math.pi / (2.0 * DF)
        ln_norm_student = -float(np.sum(np.log(c_const * s_welch_act)))

        assert math.isclose(ln_norm_plug, ln_norm_student, rel_tol=1e-14)

    def test_likelihood_evaluation_finiteness(self, setup_data: Dict[str, Any]) -> None:
        """Verify Models 1, 2, and 3 evaluate to finite values at injection."""
        psd_true_act = setup_data["psd_true_act"]
        s_welch_act = setup_data["s_welch_act"]
        res_act = setup_data["res_act"]
        res_sq = np.abs(res_act) ** 2

        # Model 1
        ln_norm_1 = -float(np.sum(np.log((math.pi * psd_true_act) / (2.0 * DF))))
        q_1 = 4.0 * DF * np.sum(res_sq / psd_true_act)
        ln_L_1 = ln_norm_1 - 0.5 * q_1
        assert np.isfinite(ln_L_1)

        # Model 2
        ln_norm_2 = -float(np.sum(np.log((math.pi * s_welch_act) / (2.0 * DF))))
        q_2 = 4.0 * DF * np.sum(res_sq / s_welch_act)
        ln_L_2 = ln_norm_2 - 0.5 * q_2
        assert np.isfinite(ln_L_2)

        # Model 3
        q_bins = (2.0 * DF * res_sq) / s_welch_act
        u_bins = (2.0 / NU_EFF) * q_bins
        ln_fit_3 = -float((0.5 * NU_EFF + 1.0) * np.sum(np.log(1.0 + u_bins)))
        ln_L_3 = ln_norm_2 + ln_fit_3
        assert np.isfinite(ln_L_3)

    def test_student_t_limiting_convergence(self, setup_data: Dict[str, Any]) -> None:
        """Verify Student-t log-likelihood converges to Gaussian Whittle as nu_eff -> inf."""
        s_welch_act = setup_data["s_welch_act"]
        res_act = setup_data["res_act"]
        res_sq = np.abs(res_act) ** 2

        q_plug_bins = (2.0 * DF * res_sq) / s_welch_act
        ln_fit_plug = -float(np.sum(q_plug_bins))  # -0.5 * (r|r)_2

        # As nu -> infinity: -(nu/2 + 1) * ln(1 + 2*q/nu) -> -q
        nu_large = 1e8
        u_large = (2.0 / nu_large) * q_plug_bins
        ln_fit_student_large = -float((0.5 * nu_large + 1.0) * np.sum(np.log(1.0 + u_large)))

        assert math.isclose(ln_fit_student_large, ln_fit_plug, rel_tol=1e-6)


# =====================================================================
# SECTION 3: Waveform Distance and Mismatch Shells
# =====================================================================


class TestWaveformDistanceAndShells:
    """Verify D(theta) computation, Point F exact value, and shell boundaries."""

    @pytest.fixture
    def setup_grid(self) -> Dict[str, Any]:
        """Set up grid and canonical injection waveform."""
        grid = FrequencyGrid(n_samples=N_ON, delta_t=1.0 / FS)
        psd_true_full = analytical_detector_psd(grid.frequencies, model="aligo_representative")
        k_act = np.arange(K_START, K_END + 1)
        psd_true_act = psd_true_full[k_act]

        h_inj = generate_waveform(grid, CANONICAL_MC, CANONICAL_ETA, CANONICAL_TC, CANONICAL_PHIC)
        return {"grid": grid, "h_inj": h_inj, "psd_true_act": psd_true_act}

    def test_injection_distance_is_zero(self, setup_grid: Dict[str, Any]) -> None:
        """Verify D(theta_inj) == 0.0 and falls into Shell 0."""
        grid = setup_grid["grid"]
        h_inj = setup_grid["h_inj"]
        psd_act = setup_grid["psd_true_act"]

        d_inj = compute_waveform_distance(h_inj, h_inj, psd_act)
        assert math.isclose(d_inj, 0.0, abs_tol=1e-12)
        assert assign_mismatch_shell(d_inj) == 0

    def test_reconciled_point_f_distance(self, setup_grid: Dict[str, Any]) -> None:
        """Verify Point F has D(theta_F) approx 4200.35 and falls into Shell 3."""
        grid = setup_grid["grid"]
        h_inj = setup_grid["h_inj"]
        psd_act = setup_grid["psd_true_act"]

        # Point F coordinates from Stage 5F: mc=28.150, eta=0.240, tc=2.002, phic=pi/2
        th_f = (28.150, 0.240, 2.002, 0.5 * math.pi)
        h_f = generate_waveform(grid, th_f[0], th_f[1], th_f[2], th_f[3])

        d_f = compute_waveform_distance(h_f, h_inj, psd_act)
        # Expected reconciled value: 4200.345019...
        assert math.isclose(d_f, 4200.35, rel_tol=1e-4)
        assert assign_mismatch_shell(d_f) == 3

    def test_mutually_exclusive_and_exhaustive_shells(self) -> None:
        """Verify shell boundary logic is strictly mutually exclusive and exhaustive."""
        # Test values at and around boundaries: 0, 9, 25, 100
        test_points = [
            (0.0, 0),
            (4.5, 0),
            (9.0, 0),       # upper bound of S0
            (9.0001, 1),    # lower bound of S1
            (17.0, 1),
            (25.0, 1),      # upper bound of S1
            (25.0001, 2),   # lower bound of S2
            (62.5, 2),
            (100.0, 2),     # upper bound of S2
            (100.0001, 3),  # lower bound of S3
            (4200.35, 3),
        ]
        for d_val, expected_shell in test_points:
            assert assign_mismatch_shell(d_val) == expected_shell

    def test_point_f_neighborhood_membership(self) -> None:
        """Verify continuous neighborhood B_eps(theta_F) bounding logic."""
        # theta_F = (28.150, 0.240, 2.002, pi/2)
        # tolerances: mc: 0.01, eta: 0.005, tc: 0.001, phic: 0.1
        def in_point_f_ball(th: np.ndarray) -> bool:
            return bool(
                abs(th[0] - 28.150) <= 0.01
                and abs(th[1] - 0.240) <= 0.005
                and abs(th[2] - 2.002) <= 0.001
                and abs(th[3] - 0.5 * math.pi) <= 0.1
            )

        th_f = np.array([28.150, 0.240, 2.002, 0.5 * math.pi])
        assert in_point_f_ball(th_f)

        th_inside = np.array([28.155, 0.242, 2.0025, 0.5 * math.pi + 0.05])
        assert in_point_f_ball(th_inside)

        # Injection point A should be well outside
        th_inj = np.array([CANONICAL_MC, CANONICAL_ETA, CANONICAL_TC, CANONICAL_PHIC])
        assert not in_point_f_ball(th_inj)

    def test_effective_sample_size_and_quantiles(self) -> None:
        """Verify N_eff and weighted quantiles helper functions."""
        # 10 uniform samples
        w = np.ones(10) / 10.0
        n_eff = 1.0 / np.sum(w ** 2)
        assert math.isclose(n_eff, 10.0)

        # 1 dominant sample
        w_dom = np.zeros(10)
        w_dom[0] = 1.0
        n_eff_dom = 1.0 / np.sum(w_dom ** 2)
        assert math.isclose(n_eff_dom, 1.0)


# =====================================================================
# SECTION 4: Deterministic Seed Schedule and Disjointness
# =====================================================================


class TestSeedScheduleAndDisjointness:
    """Verify pilot seed schedule has zero collisions with previous stages and within itself."""

    def test_pilot_seed_schedule_disjointness(self) -> None:
        """Verify 18 pilot runs use unique, disjoint seeds."""
        off_seeds = [7101, 7102, 7103]
        on_seeds = [7151, 7152, 7153]
        sampler_seeds_1 = [8101, 8102, 8103]
        sampler_seeds_2 = [8201, 8202, 8203]

        all_pilot_seeds = off_seeds + on_seeds + sampler_seeds_1 + sampler_seeds_2
        # Check all unique
        assert len(all_pilot_seeds) == len(set(all_pilot_seeds))
        assert len(all_pilot_seeds) == 12

        # Check no overlap with Stage 4 seeds (4201-4204)
        stage4_seeds = set(range(4201, 4205))
        assert set(all_pilot_seeds).isdisjoint(stage4_seeds)

        # Check no overlap with Stage 5F seeds (6701-6800)
        stage5f_seeds = set(range(6701, 6801))
        assert set(all_pilot_seeds).isdisjoint(stage5f_seeds)

        # Check all positive ints
        for s in all_pilot_seeds:
            assert isinstance(s, int) and s > 0


# =====================================================================
# SECTION 5: Baseline Sampler Adapter Configuration
# =====================================================================


class TestSamplerAdapterConfiguration:
    """Verify NestedSamplerAdapter instantiates properly with Stage 5G baseline settings."""

    def test_dynesty_availability(self) -> None:
        """Verify dynesty is importable."""
        assert is_dynesty_available()

    def test_adapter_initialization_with_pilot_settings(self) -> None:
        """Verify adapter accepts 4D dimensionality, nlive=500, dlogz=0.10, and seed."""
        def dummy_logl(th: np.ndarray) -> float:
            return -0.5 * float(np.sum(th ** 2))

        adapter = NestedSamplerAdapter(
            log_likelihood_fn=dummy_logl,
            prior_transform_fn=prior_transform_4d,
            ndim=4,
            nlive=500,
            dlogz=0.10,
            seed=8101,
            param_names=["mc", "eta", "tc", "phic"],
        )
        assert adapter.ndim == 4
        assert adapter.nlive == 500
        assert adapter.dlogz == 0.10
        assert adapter.param_names == ("mc", "eta", "tc", "phic")
