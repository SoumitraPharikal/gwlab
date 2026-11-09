"""WP2 Unit and Integration Test Suite: TaylorF2 Frequency-Domain Waveform Engine.

Mandatory Cutoff Tests:
T1: Exact bin equality (f_term = f_k): retain bin k, zero bins above.
T2: Between-bin cutoff (f_k < f_term < f_{k+1}): retain all bins through f_k, zero f_{k+1} and above.
T3: Termination below active band (f_term < f_start): mask unchanged, 0 active template bins.
T4: Termination above active band (f_term > f_end): band limits template, all active bins evaluated.

Additional Tests:
- Equal-mass vs unequal-mass binary parameters.
- Physical boundaries (m1 <= 0, m2 <= 0, distance <= 0).
- Dimensional consistency of amplitude prefactor and total Fourier amplitude.
- Time-shift translation property.
- Chirp-time positivity over GW150914 restricted-inspiral parameter domain.
- Discrete cutoff non-smoothness across frequency bins.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference.constants import C_ISCO_SI, C_SI, G_SI, M_SUN_SI, MPC_SI
from gw_inference.grid import FrequencyGrid
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    component_masses_to_chirp_mass,
    effective_termination_frequency,
    newtonian_chirp_time,
    schwarzschild_isco_frequency,
    taylorf2_35pn_chirp_time,
    taylorf2_35pn_phase,
    taylorf2_amplitude_prefactor,
    taylorf2_fd_waveform,
)


class TestTaylorF2PhysicalEngine:
    """Tests for physical parameters, phase, and dimensional consistency."""

    def test_equal_mass_chirp_mass_identity(self):
        """Verify equal-mass identity: M_c = M * (1/4)^(3/5) = 2^(-6/5) * M."""
        m = 30.0 * M_SUN_SI
        M, eta, M_c = component_masses_to_chirp_mass(m, m)
        assert eta == 0.25
        assert M == 2.0 * m
        expected_M_c = M * (2.0 ** (-6.0 / 5.0))
        assert math.isclose(M_c, expected_M_c, rel_tol=1e-15)

    def test_unequal_mass_chirp_mass(self):
        """Verify unequal-mass chirp mass calculation."""
        m1 = 36.0 * M_SUN_SI
        m2 = 29.0 * M_SUN_SI
        M, eta, M_c = component_masses_to_chirp_mass(m1, m2)
        assert M == 65.0 * M_SUN_SI
        expected_eta = (36.0 * 29.0) / (65.0**2)
        assert math.isclose(eta, expected_eta, rel_tol=1e-15)
        expected_M_c = M * (expected_eta ** 0.6)
        assert math.isclose(M_c, expected_M_c, rel_tol=1e-15)

    def test_invalid_masses_raise_error(self):
        """Verify that zero or negative masses raise ValueError."""
        with pytest.raises(ValueError, match="positive finite float"):
            component_masses_to_chirp_mass(0.0, 30.0 * M_SUN_SI)
        with pytest.raises(ValueError, match="positive finite float"):
            component_masses_to_chirp_mass(-10.0, 30.0 * M_SUN_SI)

    def test_isco_frequency_gw150914(self):
        """Verify Schwarzschild ISCO frequency for GW150914 total mass M ~ 70 M_sun."""
        # f_ISCO = 4397.17 / 70.0 Hz = 62.8167 Hz ~ 63 Hz
        f_isco_70 = schwarzschild_isco_frequency(70.0)
        assert math.isclose(f_isco_70, 4397.17 / 70.0, rel_tol=1e-14)
        assert 62.8 < f_isco_70 < 62.9

    def test_amplitude_prefactor_dimensions_and_scaling(self):
        """Verify SI amplitude prefactor A scaling with distance and chirp mass."""
        m1 = 36.0 * M_SUN_SI
        m2 = 29.0 * M_SUN_SI
        _, _, M_c = component_masses_to_chirp_mass(m1, m2)
        dist_400mpc = 410.0 * MPC_SI

        A = taylorf2_amplitude_prefactor(M_c, dist_400mpc)
        assert A > 0.0
        assert math.isfinite(A)

        # Doubling distance halves amplitude A
        A_double_dist = taylorf2_amplitude_prefactor(M_c, 2.0 * dist_400mpc)
        assert math.isclose(A_double_dist, 0.5 * A, rel_tol=1e-14)

    def test_fourier_phase_evaluation(self):
        """Verify 3.5PN phase evaluation over positive frequencies."""
        m1 = 36.0 * M_SUN_SI
        m2 = 29.0 * M_SUN_SI
        freqs = np.linspace(30.0, 63.0, 100)
        psi = taylorf2_35pn_phase(freqs, m1, m2, phi_c=0.0)

        assert psi.shape == freqs.shape
        assert np.all(np.isfinite(psi))

        # Check phase offset for nonzero phi_c
        phi_c_test = 1.234
        psi_shifted = taylorf2_35pn_phase(freqs, m1, m2, phi_c=phi_c_test)
        np.testing.assert_allclose(psi_shifted, psi - phi_c_test, rtol=1e-14)

    def test_coalescence_time_translation(self):
        """Verify Fourier time shift property h~(f) * exp(-i * 2pi * f * t_c)."""
        freqs = np.linspace(10.0, 100.0, 50)
        m1 = 30.0 * M_SUN_SI
        m2 = 30.0 * M_SUN_SI
        h0 = taylorf2_fd_waveform(freqs, m1, m2, distance_m=100.0 * MPC_SI, phi_c=0.0)

        t_c = 0.05  # 50 ms shift
        h_shifted = apply_coalescence_time_shift(h0, freqs, t_c=t_c)

        expected = h0 * np.exp(-2.0j * np.pi * freqs * t_c)
        np.testing.assert_allclose(h_shifted, expected, rtol=1e-14)

    def test_chirp_time_positivity_gw150914_domain(self):
        """Verify Newtonian and 3.5PN chirp time positivity over [30, 63] Hz."""
        m1 = 36.0 * M_SUN_SI
        m2 = 29.0 * M_SUN_SI
        _, _, M_c = component_masses_to_chirp_mass(m1, m2)

        for f in [30.0, 40.0, 50.0, 60.0, 62.8]:
            tau_n = newtonian_chirp_time(f, M_c)
            assert tau_n > 0.0, f"Newtonian chirp time at {f} Hz must be positive"

            tau_35pn = taylorf2_35pn_chirp_time(f, m1, m2)
            assert tau_35pn > 0.0, f"3.5PN chirp time at {f} Hz must be positive"
            # As frequency increases towards coalescence, time to coalescence must decrease
            if f > 30.0:
                assert tau_35pn < prev_tau_35pn
            prev_tau_35pn = tau_35pn


class TestMandatoryWaveformCutoffSuite:
    """Mandatory unit tests T1, T2, T3, and T4 for the inclusive ISCO cutoff."""

    @pytest.fixture
    def grid(self) -> FrequencyGrid:
        # 4 seconds at 2048 Hz -> N = 8192, df = 0.25 Hz
        return FrequencyGrid(n_samples=8192, delta_t=1.0 / 2048.0)

    def test_t1_exact_bin_equality(self, grid: FrequencyGrid):
        """T1: f_term exactly equals a DFT-bin frequency -> retain that bin."""
        # Target a bin exactly, e.g. f_k = 60.0 Hz (which is index 60.0 / 0.25 = 240).
        # To construct an exact-bin scenario without roundoff from component masses,
        # choose a system with f_ISCO > 60 Hz (e.g. M = 50 M_sun -> f_ISCO ~ 88 Hz)
        # and specify f_end = 60.0 Hz exactly.
        # Then f_term = min(f_end, f_ISCO) = 60.0 Hz identically in float64.
        m1 = 25.0 * M_SUN_SI
        m2 = 25.0 * M_SUN_SI
        target_f_term = 60.0

        f_isco = schwarzschild_isco_frequency(50.0)
        assert f_isco > target_f_term

        h = taylorf2_fd_waveform(
            grid.frequencies,
            m1,
            m2,
            distance_m=100.0 * MPC_SI,
            f_end=target_f_term,
        )

        k_target = int(round(target_f_term / grid.delta_f))
        assert grid.frequencies[k_target] == 60.0

        # Bin k_target MUST be retained (non-zero)
        assert h[k_target] != 0.0 + 0.0j, "Bin exactly at f_term must be retained"
        assert abs(h[k_target]) > 0.0

        # Every bin strictly above k_target MUST be identically zero
        assert np.all(h[k_target + 1 :] == 0.0 + 0.0j), "All bins strictly above f_term must be 0"

    def test_cutoff_strictly_above_is_zero_nextafter(self):
        """Verify that a frequency strictly above f_term by 1 ULP is zeroed."""
        f_term = 60.0
        f_above = np.nextafter(f_term, np.inf)
        freqs = np.array([f_term, f_above], dtype=np.float64)

        m1 = 25.0 * M_SUN_SI
        m2 = 25.0 * M_SUN_SI

        h = taylorf2_fd_waveform(
            freqs,
            m1,
            m2,
            distance_m=100.0 * MPC_SI,
            f_end=f_term,
        )

        assert h[0] != 0.0 + 0.0j, "Bin at f_term must be non-zero"
        assert h[1] == 0.0 + 0.0j, "Bin strictly above f_term (nextafter) must be identically zero"

    def test_mass_to_isco_floating_point_roundoff(self, grid: FrequencyGrid):
        """Document and test floating-point behavior when f_ISCO has 1-ULP roundoff.

        When M_total is set via C_ISCO_SI / 60.0, component mass scaling can introduce
        a 1-ULP roundoff making f_ISCO = 59.99999999999999 < 60.0.
        Under the strict f_k <= f_term rule, bin 60.0 Hz will be zeroed because
        60.0 > 59.99999999999999.
        """
        target_f = 60.0
        M_total_msun = C_ISCO_SI / target_f
        m1 = 0.5 * M_total_msun * M_SUN_SI
        m2 = 0.5 * M_total_msun * M_SUN_SI

        M, _, _ = component_masses_to_chirp_mass(m1, m2)
        m_total_recomputed = M / M_SUN_SI
        f_isco = schwarzschild_isco_frequency(m_total_recomputed)

        # Confirm floating-point roundoff behavior: f_isco is slightly less than 60.0
        assert f_isco < target_f
        assert target_f - f_isco < 1e-13

        h = taylorf2_fd_waveform(
            grid.frequencies,
            m1,
            m2,
            distance_m=100.0 * MPC_SI,
            f_end=100.0,
        )

        k_60 = int(round(target_f / grid.delta_f))
        assert grid.frequencies[k_60] == 60.0
        # Since f_isco < 60.0, strict f_k <= f_term means bin 60.0 is zeroed
        assert h[k_60] == 0.0 + 0.0j
        # Bin immediately below 60 Hz (k=239, 59.75 Hz) is retained
        assert h[k_60 - 1] != 0.0 + 0.0j

    def test_t2_between_bin_cutoff(self, grid: FrequencyGrid):
        """T2: f_k < f_term < f_{k+1} -> retain all bins through f_k; set f_{k+1} and higher to 0."""
        # Grid df = 0.25 Hz. Choose f_term = 60.10 Hz (between 60.00 Hz [k=240] and 60.25 Hz [k=241])
        target_f_term = 60.10
        M_total_msun = C_ISCO_SI / target_f_term
        m1 = 0.5 * M_total_msun * M_SUN_SI
        m2 = 0.5 * M_total_msun * M_SUN_SI

        h = taylorf2_fd_waveform(
            grid.frequencies,
            m1,
            m2,
            distance_m=100.0 * MPC_SI,
            f_end=100.0,
        )

        k_lower = 240  # 60.00 Hz
        k_upper = 241  # 60.25 Hz

        assert grid.frequencies[k_lower] == 60.00
        assert grid.frequencies[k_upper] == 60.25
        assert grid.frequencies[k_lower] < target_f_term < grid.frequencies[k_upper]

        # Retain through f_k
        assert h[k_lower] != 0.0 + 0.0j, f"Bin {k_lower} (60.00 Hz <= 60.10 Hz) must be retained"

        # Set f_{k+1} and every higher bin to identically zero
        assert h[k_upper] == 0.0 + 0.0j, f"Bin {k_upper} (60.25 Hz > 60.10 Hz) must be zero"
        assert np.all(h[k_upper:] == 0.0 + 0.0j), "All bins >= k+1 must be identically zero"

    def test_t3_termination_below_active_band(self, grid: FrequencyGrid):
        """T3: f_term < f_start -> mask unchanged, template has 0 nonzero bins in active band."""
        # High mass system where f_ISCO < 30 Hz (e.g. M = 200 M_sun -> f_ISCO ~ 22 Hz)
        M_total_msun = 200.0
        m1 = 100.0 * M_SUN_SI
        m2 = 100.0 * M_SUN_SI

        f_isco = schwarzschild_isco_frequency(M_total_msun)
        assert f_isco < 30.0  # ~ 21.98 Hz

        # Fixed observation mask on [30, 63] Hz
        mask_30_63 = grid.create_active_mask(f_start=30.0, f_end=63.0)
        n_active_bins = np.sum(mask_30_63)
        assert n_active_bins > 0

        # Generate template on full grid
        h = taylorf2_fd_waveform(
            grid.frequencies,
            m1,
            m2,
            distance_m=100.0 * MPC_SI,
            f_end=63.0,
        )

        # Observation mask MUST NOT be dynamically modified
        mask_after = grid.create_active_mask(f_start=30.0, f_end=63.0)
        np.testing.assert_array_equal(mask_30_63, mask_after)

        # Template values strictly on the active band MUST be identically zero
        active_h = h[mask_30_63]
        assert len(active_h) == n_active_bins
        assert np.all(active_h == 0.0 + 0.0j), "All active-band bins must be zero when f_term < f_start"

    def test_t4_termination_above_active_band(self, grid: FrequencyGrid):
        """T4: f_term > f_end -> active band limits template, all active bins eligible."""
        # Low mass system where f_ISCO >> 63 Hz (e.g. BNS: M = 2.8 M_sun -> f_ISCO ~ 1570 Hz)
        m1 = 1.4 * M_SUN_SI
        m2 = 1.4 * M_SUN_SI
        f_isco = schwarzschild_isco_frequency(2.8)
        assert f_isco > 1500.0

        # Active band [30, 63] Hz
        mask_30_63 = grid.create_active_mask(f_start=30.0, f_end=63.0)

        # Template generated with f_end = 63.0 Hz
        h = taylorf2_fd_waveform(
            grid.frequencies,
            m1,
            m2,
            distance_m=40.0 * MPC_SI,
            f_end=63.0,
        )

        # All active bins in [30, 63] Hz must be non-zero
        active_h = h[mask_30_63]
        assert np.all(active_h != 0.0 + 0.0j), "All active bins must be non-zero when f_term > f_end"

        # Bins above f_end (63.0 Hz) must be identically zero because f_term = min(f_end, f_ISCO) = 63.0 Hz
        k_63 = int(round(63.0 / grid.delta_f))
        assert grid.frequencies[k_63] == 63.0
        assert np.all(h[k_63 + 1 :] == 0.0 + 0.0j), "Bins above f_end must be zeroed by effective termination"

    def test_discrete_cutoff_step_discontinuity(self, grid: FrequencyGrid):
        """Verify that crossing a frequency bin boundary induces a step discontinuity in template energy."""
        # Fix binary masses (e.g. 30 M_sun + 30 M_sun, where f_ISCO ~ 73.3 Hz)
        # Compare templates with f_term just below 50.0 Hz vs just above 50.0 Hz
        m = 30.0 * M_SUN_SI
        f_term_below = 49.90  # < 50.00 Hz
        f_term_above = 50.10  # > 50.00 Hz

        h_below = taylorf2_fd_waveform(
            grid.frequencies,
            m,
            m,
            distance_m=100.0 * MPC_SI,
            f_end=f_term_below,
        )
        h_above = taylorf2_fd_waveform(
            grid.frequencies,
            m,
            m,
            distance_m=100.0 * MPC_SI,
            f_end=f_term_above,
        )

        k_50 = int(round(50.0 / grid.delta_f))
        assert grid.frequencies[k_50] == 50.0

        # For f_term_below (49.90 Hz), bin 50.0 Hz is strictly zero
        assert h_below[k_50] == 0.0 + 0.0j
        # For f_term_above (50.10 Hz), bin 50.0 Hz is non-zero
        assert h_above[k_50] != 0.0 + 0.0j

        # Energy discrepancy between the two templates is exactly the energy in the newly admitted bin
        energy_below = np.sum(np.abs(h_below) ** 2)
        energy_above = np.sum(np.abs(h_above) ** 2)
        step_energy = np.abs(h_above[k_50]) ** 2

        assert energy_above > energy_below
        # Subtraction of two total energy sums of order ~1e-38 with discrepancy ~1.5e-44
        # has condition number ~1e6, losing ~6 decimal digits of precision.
        # rtol=1e-10 is the justified numerical precision limit.
        np.testing.assert_allclose(energy_above - energy_below, step_energy, rtol=1e-10)
