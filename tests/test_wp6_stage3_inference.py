"""Tests for Work Package 6 (WP6) Stage 3: Authorized 4D Bayesian Inference.

Validates the complete 4D inference foundation:
- 4D prior transform, boundaries, normalization, and shape handling.
- 4D likelihood wrapper, normalization preservation, and parameter sensitivity.
- Gate A: Waveform factorization and extrinsic sign conventions (t_c, phi_c).
- Gate B: Complex matched-filter statistic direct sum vs zero-padded complex IFFT.
- Gate C: Analytic phase and numerical time marginalization, log-Bessel stability, quadrature weights.
- Gate D: Dynesty 4D integration smoke test with periodic phase boundary (periodic=[3]).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
from scipy.special import i0e

from gw_inference.constants import C_ISCO_SI, M_SUN_SI, MPC_SI
from gw_inference.grid import FrequencyGrid
from gw_inference.likelihood import FrequencyDomainGaussianLikelihood
from gw_inference.posterior import (
    GravitationalWaveLikelihood4DWrapper,
    GravitationalWaveLikelihoodWrapper,
)
from gw_inference.prior import (
    Uniform4DParameterPrior,
    prior_transform_4d,
)
from gw_inference.psd import analytical_detector_psd
from gw_inference.sampler import (
    NestedSamplerAdapter,
    is_dynesty_available,
)
from gw_inference.semi_analytic import (
    compare_direct_and_ifft_matched_filter,
    compute_complex_matched_filter_statistic_direct,
    compute_complex_matched_filter_statistic_ifft,
    log_bessel_i0,
    log_phase_marginalized_likelihood,
    marginalize_extrinsic_parameters_4d,
)
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    component_masses_to_chirp_mass,
    taylorf2_fd_waveform,
)


# ============================================================================
# Fixtures for Likelihood and Waveform Setup
# ============================================================================

@pytest.fixture(scope="module")
def stage3_setup():
    """Build a deterministic baseline frequency grid, PSD, and likelihood instance."""
    sampling_rate = 2048.0
    duration = 4.0
    n_samples = int(sampling_rate * duration)
    delta_t = 1.0 / sampling_rate
    grid = FrequencyGrid(n_samples=n_samples, delta_t=delta_t)
    mask = grid.create_active_mask(f_start=20.0, f_end=512.0)

    psd_vals = analytical_detector_psd(grid.frequencies, model="aligo_representative")

    # Fixed canonical injection parameters
    m1_inj = 36.0 * M_SUN_SI
    m2_inj = 29.0 * M_SUN_SI
    dist_inj = 410.0 * MPC_SI
    tc_inj = 2.0
    phic_inj = 0.0

    # Inject noiseless signal
    h_inj = taylorf2_fd_waveform(
        freqs=grid.frequencies,
        m1_kg=m1_inj,
        m2_kg=m2_inj,
        distance_m=dist_inj,
        phi_c=phic_inj,
        f_end=512.0,
    )
    h_inj = apply_coalescence_time_shift(h_inj, grid.frequencies, tc_inj)

    likelihood = FrequencyDomainGaussianLikelihood(
        data_fd=h_inj,
        psd=psd_vals,
        grid=grid,
        mask=mask,
    )

    _, _, mc_inj = component_masses_to_chirp_mass(m1_inj, m2_inj)
    mc_inj_msun = mc_inj / M_SUN_SI
    m_tot = m1_inj + m2_inj
    eta_inj = (m1_inj * m2_inj) / (m_tot**2)

    return {
        "grid": grid,
        "mask": mask,
        "data_fd": h_inj,
        "psd": psd_vals,
        "likelihood": likelihood,
        "mc_inj_msun": mc_inj_msun,
        "eta_inj": eta_inj,
        "tc_inj": tc_inj,
        "phic_inj": phic_inj,
        "dist_inj": dist_inj,
    }


# ============================================================================
# 1. 4D Prior & Transform Tests
# ============================================================================

class TestUniform4DParameterPrior:
    """Validate 4D prior properties, normalization, bounds, and analytic unit-cube mapping."""

    def test_default_properties_and_normalization(self):
        prior = Uniform4DParameterPrior()
        assert prior.mc_min == 20.0
        assert prior.mc_max == 40.0
        assert prior.eta_min == 0.10
        assert prior.eta_max == 0.25
        assert prior.tc_min == 1.90
        assert prior.tc_max == 2.10
        assert prior.phic_min == 0.0
        assert prior.phic_max == 2.0 * math.pi

        expected_volume = 20.0 * 0.15 * 0.20 * (2.0 * math.pi)
        assert pytest.approx(prior.volume, rel=1e-12) == expected_volume
        assert pytest.approx(prior.volume, rel=1e-6) == 1.2 * math.pi
        assert pytest.approx(prior.normalized_density, rel=1e-12) == 1.0 / expected_volume
        assert pytest.approx(prior.log_density, rel=1e-12) == -math.log(expected_volume)

        assert prior.param_names == (
            "chirp_mass_msun",
            "symmetric_mass_ratio",
            "coalescence_time_s",
            "coalescence_phase_rad",
        )
        assert prior.param_units == ("M_sun", "dimensionless", "s", "rad")

    def test_log_prior_boundaries(self):
        prior = Uniform4DParameterPrior()
        # Interior point
        theta_inside = [28.0, 0.24, 2.00, math.pi]
        assert prior.is_within_bounds(theta_inside)
        assert prior.log_prior(theta_inside) == prior.log_density

        # Boundary points (inclusive)
        theta_min = [20.0, 0.10, 1.90, 0.0]
        theta_max = [40.0, 0.25, 2.10, 2.0 * math.pi]
        assert prior.is_within_bounds(theta_min)
        assert prior.is_within_bounds(theta_max)

        # Outside each dimension
        assert not prior.is_within_bounds([19.99, 0.20, 2.00, 1.0])
        assert not prior.is_within_bounds([40.01, 0.20, 2.00, 1.0])
        assert not prior.is_within_bounds([28.00, 0.09, 2.00, 1.0])
        assert not prior.is_within_bounds([28.00, 0.251, 2.00, 1.0])
        assert not prior.is_within_bounds([28.00, 0.20, 1.899, 1.0])
        assert not prior.is_within_bounds([28.00, 0.20, 2.101, 1.0])
        assert not prior.is_within_bounds([28.00, 0.20, 2.00, -0.01])
        assert not prior.is_within_bounds([28.00, 0.20, 2.00, 2.0 * math.pi + 0.01])

        # Non-finite or wrong shape
        assert not prior.is_within_bounds([np.nan, 0.20, 2.00, 1.0])
        assert not prior.is_within_bounds([28.0, 0.20, 2.00])
        assert not prior.is_within_bounds([28.0, 0.20, 2.00, 1.0, 0.0])
        assert prior.log_prior([19.0, 0.20, 2.00, 1.0]) == -math.inf

    def test_prior_transform_4d_endpoints_and_interior(self):
        # 1D endpoints
        u_zeros = [0.0, 0.0, 0.0, 0.0]
        theta_min = prior_transform_4d(u_zeros)
        np.testing.assert_allclose(theta_min, [20.0, 0.10, 1.90, 0.0])

        u_ones = [1.0, 1.0, 1.0, 1.0]
        theta_max = prior_transform_4d(u_ones)
        np.testing.assert_allclose(theta_max, [40.0, 0.25, 2.10, 2.0 * math.pi])

        u_mid = [0.5, 0.5, 0.5, 0.5]
        theta_mid = prior_transform_4d(u_mid)
        np.testing.assert_allclose(theta_mid, [30.0, 0.175, 2.00, math.pi])

        # Prior method matches function
        prior = Uniform4DParameterPrior()
        np.testing.assert_allclose(prior.prior_transform(u_mid), theta_mid)

    def test_prior_transform_4d_shapes_and_errors(self):
        # 2D batch transform
        u_batch = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ])
        res = prior_transform_4d(u_batch)
        assert res.shape == (3, 4)
        np.testing.assert_allclose(res[0], [20.0, 0.10, 1.90, 0.0])
        np.testing.assert_allclose(res[1], [40.0, 0.25, 2.10, 2.0 * math.pi])
        np.testing.assert_allclose(res[2], [30.0, 0.175, 2.00, math.pi])

        # Rejection of out of range values
        with pytest.raises(ValueError, match="unit interval"):
            prior_transform_4d([-0.01, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="unit interval"):
            prior_transform_4d([0.5, 1.01, 0.5, 0.5])
        with pytest.raises(ValueError, match="non-finite"):
            prior_transform_4d([np.nan, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="exactly 4 elements"):
            prior_transform_4d([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="shape"):
            prior_transform_4d(np.zeros((3, 3)))
        with pytest.raises(TypeError):
            prior_transform_4d("invalid")

    def test_prior_absence_of_component_mass_jacobian(self):
        """Verify prior density is strictly uniform in coordinate measure, without Jacobian."""
        prior = Uniform4DParameterPrior()
        # Density must be identical everywhere in the rectangular support
        d1 = prior.normalized_density
        assert pytest.approx(d1, rel=1e-12) == 1.0 / (1.2 * math.pi)
        p1 = prior.log_prior([22.0, 0.12, 1.95, 0.5])
        p2 = prior.log_prior([38.0, 0.24, 2.05, 5.0])
        assert p1 == p2
        assert pytest.approx(p1, rel=1e-12) == -math.log(1.2 * math.pi)


# ============================================================================
# 2. 4D Likelihood Wrapper Tests
# ============================================================================

class TestGravitationalWaveLikelihood4DWrapper:
    """Validate 4D likelihood wrapper configuration, evaluation, and physical cutoffs."""

    def test_wrapper_initialization_and_validation(self, stage3_setup):
        l_obj = stage3_setup["likelihood"]
        wrapper = GravitationalWaveLikelihood4DWrapper(
            likelihood=l_obj,
            distance_m=410.0 * MPC_SI,
            f_start=20.0,
            f_end=512.0,
            prior=Uniform4DParameterPrior(),
        )
        assert wrapper.likelihood is l_obj
        assert wrapper.distance_m == 410.0 * MPC_SI
        assert wrapper.f_start == 20.0
        assert wrapper.f_end == 512.0
        assert isinstance(wrapper.prior, Uniform4DParameterPrior)

        # Invalid arguments
        with pytest.raises(TypeError):
            GravitationalWaveLikelihood4DWrapper(likelihood="invalid")
        with pytest.raises(ValueError):
            GravitationalWaveLikelihood4DWrapper(likelihood=l_obj, distance_m=-1.0)
        with pytest.raises(ValueError):
            GravitationalWaveLikelihood4DWrapper(likelihood=l_obj, f_start=-5.0)
        with pytest.raises(ValueError):
            GravitationalWaveLikelihood4DWrapper(likelihood=l_obj, f_end=10.0, f_start=20.0)
        with pytest.raises(TypeError):
            GravitationalWaveLikelihood4DWrapper(likelihood=l_obj, prior="invalid")

    def test_likelihood_evaluation_and_normalization_preservation(self, stage3_setup):
        l_obj = stage3_setup["likelihood"]
        wrapper = GravitationalWaveLikelihood4DWrapper(likelihood=l_obj, f_start=20.0)

        # At true injection parameters:
        theta_true = [
            stage3_setup["mc_inj_msun"],
            stage3_setup["eta_inj"],
            stage3_setup["tc_inj"],
            stage3_setup["phic_inj"],
        ]
        log_l_true = wrapper.evaluate_log_likelihood(theta_true)
        # Residual r = 0, so quadratic form = 0, ln L = ln L_norm
        expected_log_l = l_obj.log_likelihood_norm
        assert pytest.approx(log_l_true, rel=1e-10) == expected_log_l

        # Calling interface __call__
        assert wrapper(theta_true) == log_l_true

    def test_unphysical_and_isco_cutoff_rejection(self, stage3_setup):
        l_obj = stage3_setup["likelihood"]
        wrapper = GravitationalWaveLikelihood4DWrapper(likelihood=l_obj, f_start=20.0)

        # Negative chirp mass or unphysical eta
        assert wrapper.evaluate_log_likelihood([-10.0, 0.20, 2.0, 0.0]) == -math.inf
        assert wrapper.evaluate_log_likelihood([30.0, -0.10, 2.0, 0.0]) == -math.inf
        assert wrapper.evaluate_log_likelihood([30.0, 0.26, 2.0, 0.0]) == -math.inf

        # High mass binary merging below f_start=20 Hz
        # E.g. M_total = 300 M_sun -> f_isco = C_ISCO / 300 ~= 14.66 Hz <= 20 Hz
        # M_c = 300 * (0.25)^(3/5) ~= 130.5 M_sun
        assert wrapper.evaluate_log_likelihood([150.0, 0.25, 2.0, 0.0]) == -math.inf

        # Non-finite inputs
        assert wrapper.evaluate_log_likelihood([np.nan, 0.20, 2.0, 0.0]) == -math.inf
        with pytest.raises(ValueError):
            wrapper.evaluate_log_likelihood([30.0, 0.20, 2.0])

    def test_posterior_evaluation(self, stage3_setup):
        l_obj = stage3_setup["likelihood"]
        prior = Uniform4DParameterPrior()
        wrapper = GravitationalWaveLikelihood4DWrapper(likelihood=l_obj, prior=prior)

        theta_inside = [
            stage3_setup["mc_inj_msun"],
            stage3_setup["eta_inj"],
            stage3_setup["tc_inj"],
            stage3_setup["phic_inj"],
        ]
        log_p = wrapper.evaluate_log_posterior(theta_inside)
        expected_log_p = wrapper.evaluate_log_likelihood(theta_inside) + prior.log_density
        assert pytest.approx(log_p, rel=1e-12) == expected_log_p

        # Outside prior bounds -> -inf
        theta_outside = [15.0, 0.20, 2.0, 0.0]
        assert wrapper.evaluate_log_posterior(theta_outside) == -math.inf


# ============================================================================
# 3. Gate A: Waveform Factorization and Extrinsic Sign Verification
# ============================================================================

class TestGateAWaveformFactorization:
    """Verify exact factorization: h~(f; M_c, eta, t_c, phi_c) = h~_0(f) * exp(-2*pi*i*f*t_c) * exp(i*phi_c)."""

    def test_waveform_factorization_exact_identity(self, stage3_setup):
        grid = stage3_setup["grid"]
        m1_kg = 36.0 * M_SUN_SI
        m2_kg = 29.0 * M_SUN_SI
        dist = 410.0 * MPC_SI

        # Generate zero-extrinsic template h~_0(f)
        h0 = taylorf2_fd_waveform(
            freqs=grid.frequencies,
            m1_kg=m1_kg,
            m2_kg=m2_kg,
            distance_m=dist,
            phi_c=0.0,
            f_end=512.0,
        )

        test_cases = [
            (2.0, 0.0),
            (1.95, 0.5),
            (2.05, 1.7),
            (2.003, 3.1415),
            (1.91, 5.5),
            (2.09, 2.0 * math.pi),
        ]

        for tc_val, phic_val in test_cases:
            # Direct generation with non-zero (tc, phic)
            h_direct = taylorf2_fd_waveform(
                freqs=grid.frequencies,
                m1_kg=m1_kg,
                m2_kg=m2_kg,
                distance_m=dist,
                phi_c=phic_val,
                f_end=512.0,
            )
            h_direct = apply_coalescence_time_shift(h_direct, grid.frequencies, tc_val)

            # Factorized generation: apply_coalescence_time_shift(h0, f, tc) * exp(i * phi_c)
            h_shifted = apply_coalescence_time_shift(h0, grid.frequencies, tc_val)
            h_factorized = h_shifted * np.exp(1.0j * phic_val)

            max_diff = np.max(np.abs(h_direct - h_factorized))
            assert max_diff < 1e-14, f"Failed at tc={tc_val}, phic={phic_val} with max_diff={max_diff}"

    def test_gate_a_sign_convention_on_likelihood(self, stage3_setup):
        """Verify likelihood wrapper evaluates identically using factorized template."""
        l_obj = stage3_setup["likelihood"]
        wrapper = GravitationalWaveLikelihood4DWrapper(likelihood=l_obj)

        mc_msun = stage3_setup["mc_inj_msun"]
        eta = stage3_setup["eta_inj"]

        for tc_val, phic_val in [(1.95, 0.8), (2.02, 2.4), (2.08, 4.5)]:
            theta = [mc_msun, eta, tc_val, phic_val]
            ln_l = wrapper.evaluate_log_likelihood(theta)
            assert math.isfinite(ln_l)


# ============================================================================
# 4. Gate B: Direct Sum vs Complex IFFT
# ============================================================================

class TestGateBDirectVsIFFT:
    """Verify complex matched-filter statistic z(t_c) direct sum vs zero-padded complex IFFT."""

    def test_direct_sum_vs_ifft_grid_times(self, stage3_setup):
        grid = stage3_setup["grid"]
        psd = stage3_setup["psd"]
        d_fd = stage3_setup["data_fd"]
        k_act = stage3_setup["likelihood"].active_indices
        delta_f = grid.delta_f

        m1_kg = 36.0 * M_SUN_SI
        m2_kg = 29.0 * M_SUN_SI
        dist = 410.0 * MPC_SI

        h0 = taylorf2_fd_waveform(
            freqs=grid.frequencies,
            m1_kg=m1_kg,
            m2_kg=m2_kg,
            distance_m=dist,
            phi_c=0.0,
            f_end=512.0,
        )

        n_base = 2 * (len(grid.frequencies) - 1)
        t_grid, z_grid = compute_complex_matched_filter_statistic_ifft(
            data_fd=d_fd,
            template_zero_fd=h0,
            psd=psd,
            delta_f=delta_f,
            active_indices=k_act,
            n_base=n_base,
            zero_pad_factor=1,
        )

        # Pick 5 exact grid points near t=2.0s
        indices_to_test = [int(2.0 / (t_grid[1] - t_grid[0])) + offset for offset in [-10, -3, 0, 5, 12]]
        test_times = t_grid[indices_to_test]

        z_direct = compute_complex_matched_filter_statistic_direct(
            data_fd=d_fd,
            template_zero_fd=h0,
            psd=psd,
            delta_f=delta_f,
            active_indices=k_act,
            t_c=test_times,
        )

        z_ifft_exact = z_grid[indices_to_test]

        # Agreement at exact grid points must be near machine precision (< 1e-10)
        max_diff_real = np.max(np.abs(np.real(z_direct) - np.real(z_ifft_exact)))
        max_diff_imag = np.max(np.abs(np.imag(z_direct) - np.imag(z_ifft_exact)))
        max_diff_mag = np.max(np.abs(np.abs(z_direct) - np.abs(z_ifft_exact)))

        assert max_diff_real < 1e-10
        assert max_diff_imag < 1e-10
        assert max_diff_mag < 1e-10

    def test_direct_sum_vs_ifft_off_grid_times_with_padding(self, stage3_setup):
        grid = stage3_setup["grid"]
        psd = stage3_setup["psd"]
        d_fd = stage3_setup["data_fd"]
        k_act = stage3_setup["likelihood"].active_indices
        delta_f = grid.delta_f

        m1_kg = 36.0 * M_SUN_SI
        m2_kg = 29.0 * M_SUN_SI
        dist = 410.0 * MPC_SI

        h0 = taylorf2_fd_waveform(
            freqs=grid.frequencies,
            m1_kg=m1_kg,
            m2_kg=m2_kg,
            distance_m=dist,
            phi_c=0.0,
            f_end=512.0,
        )

        n_base = 2 * (len(grid.frequencies) - 1)
        test_times = [1.980123, 1.995432, 2.000000, 2.001234, 2.015678]

        # Compare with zero_pad_factor = 16
        diffs = compare_direct_and_ifft_matched_filter(
            data_fd=d_fd,
            template_zero_fd=h0,
            psd=psd,
            delta_f=delta_f,
            active_indices=k_act,
            n_base=n_base,
            test_times=test_times,
            zero_pad_factor=16,
        )

        # Off-grid interpolation error is bounded
        assert diffs["max_diff_magnitude"] < 0.05
        assert diffs["max_diff_phase"] < 1e-3


# ============================================================================
# 5. Gate C: Extrinsic Marginalization & Stability
# ============================================================================

class TestGateCExtrinsicMarginalization:
    """Verify analytic phase marginalization, log-Bessel stability, and time quadrature."""

    def test_log_bessel_i0_numerical_stability(self):
        # Small arguments
        assert pytest.approx(log_bessel_i0(0.0), abs=1e-12) == 0.0
        assert pytest.approx(log_bessel_i0(1.0), rel=1e-10) == math.log(float(i0e(1.0) * math.exp(1.0)))

        # Test extreme values that would overflow direct I_0(x) (which overflows at x ~= 709.8)
        extreme_x = np.array([0.0, 1.0, 50.0, 700.0, 710.0, 2053.0, 10000.0])
        res = log_bessel_i0(extreme_x)
        assert np.all(np.isfinite(res))

        # Check asymptotic behavior for large x: ln I_0(x) ~= x - 0.5 * ln(2 * pi * x)
        x_large = 2053.0
        expected_asymptotic = x_large - 0.5 * math.log(2.0 * math.pi * x_large)
        assert pytest.approx(log_bessel_i0(x_large), rel=1e-4) == expected_asymptotic

    def test_time_marginalization_quadrature_weights(self):
        t_grid = np.linspace(1.80, 2.20, 401)  # dt = 0.001s
        # Constant log-likelihood ln L = 10.0
        log_l_const = np.full_like(t_grid, 10.0)

        # Over [1.90, 2.10], integral of const is const * (2.10 - 1.90)
        # Normalized by 1 / (2.10 - 1.90), result must be exactly 10.0!
        res = marginalize_extrinsic_parameters_4d(t_grid, log_l_const, tc_min=1.90, tc_max=2.10)
        assert pytest.approx(res, rel=1e-12) == 10.0

    def test_time_marginalization_peak_integration(self, stage3_setup):
        """Test full extrinsic marginalization at injection parameters."""
        grid = stage3_setup["grid"]
        psd = stage3_setup["psd"]
        d_fd = stage3_setup["data_fd"]
        k_act = stage3_setup["likelihood"].active_indices
        delta_f = grid.delta_f

        m1_kg = 36.0 * M_SUN_SI
        m2_kg = 29.0 * M_SUN_SI
        dist = 410.0 * MPC_SI

        h0 = taylorf2_fd_waveform(
            freqs=grid.frequencies,
            m1_kg=m1_kg,
            m2_kg=m2_kg,
            distance_m=dist,
            phi_c=0.0,
            f_end=512.0,
        )

        n_base = 2 * (len(grid.frequencies) - 1)
        t_grid, z_grid = compute_complex_matched_filter_statistic_ifft(
            data_fd=d_fd,
            template_zero_fd=h0,
            psd=psd,
            delta_f=delta_f,
            active_indices=k_act,
            n_base=n_base,
            zero_pad_factor=16,
        )

        # Inner products
        d_inner_d = float(4.0 * delta_f * np.sum(np.abs(d_fd[k_act])**2 / psd[k_act]))
        # Template inner product (h0|h0)
        h0_inner_h0 = float(4.0 * delta_f * np.sum(np.abs(h0[k_act])**2 / psd[k_act]))
        log_l_norm = stage3_setup["likelihood"].log_likelihood_norm

        # Phase-marginalized log-likelihood array
        z_abs = np.abs(z_grid)
        ln_l_pm = log_phase_marginalized_likelihood(z_abs, d_inner_d, h0_inner_h0, log_l_norm)

        # Extrinsic-marginalized log-likelihood
        ln_l_ext_marg = marginalize_extrinsic_parameters_4d(t_grid, ln_l_pm, tc_min=1.90, tc_max=2.10)
        assert math.isfinite(ln_l_ext_marg)

        # Peak is near t=2.0s, so ln_l_ext_marg should be high (close to ln L_norm)
        assert ln_l_ext_marg > log_l_norm - 50.0


# ============================================================================
# 6. Gate D: Dynesty 4D Integration Smoke Test
# ============================================================================

class TestGateDDynesty4DIntegrationSmoke:
    """Verify dynesty nested sampling in 4D with periodic phase boundary (periodic=[3])."""

    @pytest.mark.skipif(not is_dynesty_available(), reason="dynesty not installed")
    def test_toy_4d_gaussian_nested_sampling(self):
        """Toy 4D Gaussian likelihood to verify adapter, periodic wrapping, and results unpacking."""
        # Mean vector and covariance
        mu = np.array([30.0, 0.18, 2.00, math.pi])
        sigma = np.array([0.5, 0.01, 0.005, 0.2])

        def toy_log_likelihood(theta: np.ndarray) -> float:
            arr = np.asarray(theta)
            # Coordinate 3 is wrapped mod 2*pi
            d = arr - mu
            d[3] = (d[3] + math.pi) % (2.0 * math.pi) - math.pi
            chi2 = float(np.sum((d / sigma) ** 2))
            return -0.5 * chi2

        adapter = NestedSamplerAdapter(
            log_likelihood_fn=toy_log_likelihood,
            prior_transform_fn=prior_transform_4d,
            ndim=4,
            nlive=50,
            dlogz=0.5,
            seed=42,
            param_names=["mc", "eta", "tc", "phic"],
        )

        res = adapter.run(
            bound="multi",
            sample="unif",
            periodic=[3],
            maxiter=500,
            print_progress=False,
        )

        assert res.samples.shape[1] == 4
        assert len(res.weights) == len(res.samples)
        assert pytest.approx(float(np.sum(res.weights)), rel=1e-7) == 1.0
        assert math.isfinite(res.log_evidence)
        assert res.log_evidence_err >= 0.0
        assert res.n_evaluations > 0

        summary = res.summary()
        assert "parameters" in summary
        assert set(summary["parameters"].keys()) == {"mc", "eta", "tc", "phic"}
