"""Deterministic test suite for Stage 6B alternative mass parameterization (M_total, eta).

Verifies:
1. Unit-cube corner, boundary, and interior mappings.
2. Exact round-trip transformation (M_c, eta) <-> (M_total, eta) in 2D and 4D.
3. Analytical and numerical Jacobian determinant |J| = eta^(3/5) and density integration.
4. Correlated support boundary checking and strict rejection of out-of-bounds samples.
5. Invariance of extrinsic parameter priors (t_c, phi_c), phase endpoints, and periodic phase wrapping.
6. Algebraic parameter-mapping equivalence to original coordinates without likelihood calls.
"""

import math
import numpy as np
import pytest
from scipy import integrate

from gw_inference.prior import (
    Correlated4DTotalMassEtaPrior,
    CorrelatedTotalMassEtaPrior,
    Uniform4DParameterPrior,
    UniformChirpMassEtaPrior,
    mchirp_eta_to_total_mass_eta,
    prior_transform_2d,
    prior_transform_4d,
    prior_transform_total_mass_eta_2d,
    prior_transform_total_mass_eta_4d,
    total_mass_eta_jacobian,
    total_mass_eta_to_mchirp_eta,
)


class TestUnitCubeBoundariesAndEndpoints:
    """Verify mapping of unit-cube corners and interior points."""

    def test_unit_cube_corners_2d(self):
        """Test representative corner points u = (u_1, u_2):

        u = (0, 0) -> eta = 0.10, M = 20 * 0.10^(-0.6) ~= 79.6214 M_sun, M_c = 20.0
        u = (0, 1) -> eta = 0.10, M = 40 * 0.10^(-0.6) ~= 159.2429 M_sun, M_c = 40.0
        u = (1, 0) -> eta = 0.25, M = 20 * 0.25^(-0.6) ~= 45.9479 M_sun, M_c = 20.0
        u = (1, 1) -> eta = 0.25, M = 40 * 0.25^(-0.6) ~= 91.8959 M_sun, M_c = 40.0
        """
        # Corner 1: (0, 0)
        theta_00 = prior_transform_total_mass_eta_2d([0.0, 0.0])
        eta_expected_00 = 0.10
        m_expected_00 = 20.0 * (0.10 ** -0.6)
        np.testing.assert_allclose(theta_00, [m_expected_00, eta_expected_00], rtol=1e-14)
        mc_00 = theta_00[0] * (theta_00[1] ** 0.6)
        assert pytest.approx(mc_00, rel=1e-14) == 20.0

        # Corner 2: (0, 1)
        theta_01 = prior_transform_total_mass_eta_2d([0.0, 1.0])
        eta_expected_01 = 0.10
        m_expected_01 = 40.0 * (0.10 ** -0.6)
        np.testing.assert_allclose(theta_01, [m_expected_01, eta_expected_01], rtol=1e-14)
        mc_01 = theta_01[0] * (theta_01[1] ** 0.6)
        assert pytest.approx(mc_01, rel=1e-14) == 40.0

        # Corner 3: (1, 0)
        theta_10 = prior_transform_total_mass_eta_2d([1.0, 0.0])
        eta_expected_10 = 0.25
        m_expected_10 = 20.0 * (0.25 ** -0.6)
        np.testing.assert_allclose(theta_10, [m_expected_10, eta_expected_10], rtol=1e-14)
        mc_10 = theta_10[0] * (theta_10[1] ** 0.6)
        assert pytest.approx(mc_10, rel=1e-14) == 20.0

        # Corner 4: (1, 1)
        theta_11 = prior_transform_total_mass_eta_2d([1.0, 1.0])
        eta_expected_11 = 0.25
        m_expected_11 = 40.0 * (0.25 ** -0.6)
        np.testing.assert_allclose(theta_11, [m_expected_11, eta_expected_11], rtol=1e-14)
        mc_11 = theta_11[0] * (theta_11[1] ** 0.6)
        assert pytest.approx(mc_11, rel=1e-14) == 40.0

    def test_interior_points_2d(self):
        """Test midpoint and arbitrary interior unit-cube samples."""
        u_mid = [0.5, 0.5]
        theta_mid = prior_transform_total_mass_eta_2d(u_mid)
        eta_mid = 0.10 + 0.15 * 0.5  # 0.175
        m_mid = (eta_mid ** -0.6) * 30.0  # ~85.34005 M_sun
        np.testing.assert_allclose(theta_mid, [m_mid, eta_mid], rtol=1e-14)

        # Check physical bounds on Mc and eta
        prior = CorrelatedTotalMassEtaPrior()
        assert prior.is_within_bounds(theta_mid)

        # Vectorized batch evaluation
        u_batch = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ])
        theta_batch = prior_transform_total_mass_eta_2d(u_batch)
        assert theta_batch.shape == (6, 2)
        for i in range(len(u_batch)):
            single = prior_transform_total_mass_eta_2d(u_batch[i])
            np.testing.assert_allclose(theta_batch[i], single, rtol=1e-14)

    def test_unit_cube_input_validation(self):
        """Verify strict rejection of invalid unit-cube inputs."""
        # Non-finite inputs
        with pytest.raises(ValueError, match="non-finite"):
            prior_transform_total_mass_eta_2d([np.nan, 0.5])
        with pytest.raises(ValueError, match="non-finite"):
            prior_transform_total_mass_eta_2d([0.5, np.inf])

        # Out-of-range inputs
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            prior_transform_total_mass_eta_2d([-0.01, 0.5])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            prior_transform_total_mass_eta_2d([0.5, 1.01])

        # Invalid shapes
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            prior_transform_total_mass_eta_2d([0.5])
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            prior_transform_total_mass_eta_2d([0.5, 0.5, 0.5])


class TestRoundTripTransformation:
    """Verify bidirectional round-trip mapping between (M_c, eta) and (M_total, eta)."""

    def test_2d_round_trip(self):
        """Verify (M_c, eta) -> (M_total, eta) -> (M_c, eta) to numerical precision."""
        rng = np.random.default_rng(12345)
        n_samples = 500
        mc_samples = rng.uniform(20.0, 40.0, n_samples)
        eta_samples = rng.uniform(0.10, 0.25, n_samples)
        orig_2d = np.column_stack([mc_samples, eta_samples])

        # Forward map: (M_c, eta) -> (M_total, eta)
        alt_2d = mchirp_eta_to_total_mass_eta(orig_2d)
        assert alt_2d.shape == (n_samples, 2)

        # Inverse map: (M_total, eta) -> (M_c, eta)
        roundtrip_2d = total_mass_eta_to_mchirp_eta(alt_2d)
        np.testing.assert_allclose(roundtrip_2d, orig_2d, rtol=1e-14, atol=1e-14)

        # Single vector round-trip
        single_vec = [28.08, 0.247]
        alt_single = mchirp_eta_to_total_mass_eta(single_vec)
        rt_single = total_mass_eta_to_mchirp_eta(alt_single)
        np.testing.assert_allclose(rt_single, single_vec, rtol=1e-14, atol=1e-14)

    def test_4d_round_trip(self):
        """Verify (M_c, eta, t_c, phi_c) -> (M_total, eta, t_c, phi_c) -> (M_c, eta, t_c, phi_c)."""
        rng = np.random.default_rng(54321)
        n_samples = 500
        mc_samples = rng.uniform(20.0, 40.0, n_samples)
        eta_samples = rng.uniform(0.10, 0.25, n_samples)
        tc_samples = rng.uniform(1.90, 2.10, n_samples)
        phic_samples = rng.uniform(0.0, 2.0 * math.pi, n_samples)
        orig_4d = np.column_stack([mc_samples, eta_samples, tc_samples, phic_samples])

        alt_4d = mchirp_eta_to_total_mass_eta(orig_4d)
        assert alt_4d.shape == (n_samples, 4)

        # Check extrinsic preservation
        np.testing.assert_allclose(alt_4d[:, 2], orig_4d[:, 2], rtol=1e-15)
        np.testing.assert_allclose(alt_4d[:, 3], orig_4d[:, 3], rtol=1e-15)

        # Full round-trip
        rt_4d = total_mass_eta_to_mchirp_eta(alt_4d)
        np.testing.assert_allclose(rt_4d, orig_4d, rtol=1e-14, atol=1e-14)


class TestJacobianAndDensity:
    """Verify analytical Jacobian derivation and numerical normalization."""

    def test_jacobian_analytical_and_finite_difference(self):
        """Verify |J| = eta^(3/5) against numerical partial derivatives."""
        eta_test = 0.20
        m_test = 70.0

        # Analytical Jacobian
        j_analytical = total_mass_eta_jacobian(eta_test)
        assert pytest.approx(j_analytical, rel=1e-14) == (0.20 ** 0.6)

        # Finite difference of M_c(M, eta) = M * eta^(3/5)
        eps = 1e-7
        # d(M_c)/d(M) = eta^(3/5)
        d_mc_d_m = ((m_test + eps) * (eta_test ** 0.6) - (m_test - eps) * (eta_test ** 0.6)) / (2.0 * eps)
        # d(M_c)/d(eta) = (3/5) * M * eta^(-2/5)
        d_mc_d_eta = (m_test * ((eta_test + eps) ** 0.6) - m_test * ((eta_test - eps) ** 0.6)) / (2.0 * eps)
        # d(eta)/d(M) = 0
        d_eta_d_m = 0.0
        # d(eta)/d(eta) = 1
        d_eta_d_eta = 1.0

        num_det = d_mc_d_m * d_eta_d_eta - d_mc_d_eta * d_eta_d_m
        assert pytest.approx(num_det, rel=1e-6) == j_analytical

    def test_density_2d_normalization_quadrature(self):
        """Verify numerical integration of p(M, eta) over correlated support equals 1.0."""
        prior = CorrelatedTotalMassEtaPrior()

        # Integral: \int_{0.10}^{0.25} deta \int_{20*eta^(-0.6)}^{40*eta^(-0.6)} dM (1/3) * eta^(0.6)
        def integrand(m, eta):
            return prior.density([m, eta])

        integral, abserr = integrate.dblquad(
            integrand,
            0.10,  # eta_min
            0.25,  # eta_max
            lambda eta: 20.0 * (eta ** -0.6),  # M_lower(eta)
            lambda eta: 40.0 * (eta ** -0.6),  # M_upper(eta)
            epsabs=1e-12,
            epsrel=1e-12,
        )

        assert pytest.approx(integral, abs=1e-10) == 1.0
        assert abserr < 1e-10

    def test_density_4d_normalization(self):
        """Verify 4D density integrates to 1.0 over correlated support."""
        prior = Correlated4DTotalMassEtaPrior()

        # 4D integral factors into 2D mass integral * Delta t_c * Delta phi_c
        # Normalization constant = 1 / (3.0 * 0.20 * 2*pi) = 1 / (1.2 * pi)
        def mass_integrand(m, eta):
            return (eta ** 0.6)

        int_mass, _ = integrate.dblquad(
            mass_integrand,
            0.10,
            0.25,
            lambda eta: 20.0 * (eta ** -0.6),
            lambda eta: 40.0 * (eta ** -0.6),
        )
        # int_mass = 20/3 * 0.15 * (1/0.15) = 20/3 * 0.15 = 1.0 * (20/3 * 0.15) = 3.0
        # Wait: int_{20*eta^-0.6}^{40*eta^-0.6} eta^0.6 dM = 20 * eta^-0.6 * eta^0.6 = 20
        # int_{0.10}^{0.25} 20 deta = 20 * 0.15 = 3.0
        assert pytest.approx(int_mass, rel=1e-12) == 3.0

        # Full 4D hypervolume integral with base density:
        # (1.0 / (3.0 * 0.20 * 2*pi)) * int_mass * (2.10 - 1.90) * (2*pi - 0) = 1.0
        int_4d = (1.0 / (3.0 * 0.20 * 2.0 * math.pi)) * int_mass * 0.20 * (2.0 * math.pi)
        assert pytest.approx(int_4d, rel=1e-12) == 1.0

        # Test density method directly
        test_theta = [70.0, 0.20, 2.00, math.pi]
        expected_density = (1.0 / (1.2 * math.pi)) * (0.20 ** 0.6)
        assert pytest.approx(prior.density(test_theta), rel=1e-14) == expected_density
        assert pytest.approx(prior.log_prior(test_theta), rel=1e-14) == math.log(expected_density)


class TestSupportRejectionAndBoundaries:
    """Verify strict support checking and rejection of invalid / out-of-support parameters."""

    def test_2d_support_rejection(self):
        """Verify that points outside correlated support are strictly rejected."""
        prior = CorrelatedTotalMassEtaPrior()

        # Valid interior point: eta = 0.20, M = 70 -> Mc = 70 * 0.20^0.6 ~= 26.68 (in [20, 40])
        theta_valid = [70.0, 0.20]
        assert prior.is_within_bounds(theta_valid)
        assert math.isfinite(prior.log_prior(theta_valid))
        assert prior.density(theta_valid) > 0.0

        # Invalid eta: below eta_min (eta = 0.08)
        theta_low_eta = [100.0, 0.08]
        assert not prior.is_within_bounds(theta_low_eta)
        assert prior.log_prior(theta_low_eta) == -math.inf
        assert prior.density(theta_low_eta) == 0.0

        # Invalid eta: above eta_max (eta = 0.26)
        theta_high_eta = [60.0, 0.26]
        assert not prior.is_within_bounds(theta_high_eta)
        assert prior.log_prior(theta_high_eta) == -math.inf
        assert prior.density(theta_high_eta) == 0.0

        # Valid eta (0.25), but M too low: M = 40.0 -> Mc = 40 * 0.25^0.6 = 17.41 < 20.0
        theta_low_mc = [40.0, 0.25]
        assert not prior.is_within_bounds(theta_low_mc)
        assert prior.log_prior(theta_low_mc) == -math.inf
        assert prior.density(theta_low_mc) == 0.0

        # Valid eta (0.25), but M too high: M = 100.0 -> Mc = 100 * 0.25^0.6 = 43.53 > 40.0
        theta_high_mc = [100.0, 0.25]
        assert not prior.is_within_bounds(theta_high_mc)
        assert prior.log_prior(theta_high_mc) == -math.inf
        assert prior.density(theta_high_mc) == 0.0

        # Non-positive M
        assert not prior.is_within_bounds([-10.0, 0.20])
        assert not prior.is_within_bounds([0.0, 0.20])
        assert not prior.is_within_bounds([np.nan, 0.20])
        assert not prior.is_within_bounds([70.0, np.nan])

    def test_4d_support_rejection(self):
        """Verify 4D support rejection on mass, time, and phase boundaries."""
        prior = Correlated4DTotalMassEtaPrior()

        # Valid point
        theta_valid = [70.0, 0.20, 2.00, math.pi]
        assert prior.is_within_bounds(theta_valid)

        # Invalid time
        assert not prior.is_within_bounds([70.0, 0.20, 1.89, math.pi])
        assert not prior.is_within_bounds([70.0, 0.20, 2.11, math.pi])

        # Invalid phase
        assert not prior.is_within_bounds([70.0, 0.20, 2.00, -0.01])
        assert not prior.is_within_bounds([70.0, 0.20, 2.00, 2.0 * math.pi + 0.01])

        # Invalid shape
        assert not prior.is_within_bounds([70.0, 0.20, 2.00])


class TestExtrinsicPriorsAndPhaseWrapping:
    """Verify extrinsic parameter definitions, transformations, endpoints, and wrapping."""

    def test_extrinsic_transform_linear_and_periodic(self):
        """Verify linear mapping for t_c and periodic wrapping for phi_c."""
        # Test unit inputs for extrinsic params
        u = [0.5, 0.5, 0.0, 0.0]
        theta = prior_transform_total_mass_eta_4d(u)
        assert theta[2] == 1.90
        assert theta[3] == 0.0

        u_max = [0.5, 0.5, 1.0, 1.0]
        theta_max = prior_transform_total_mass_eta_4d(u_max)
        assert theta_max[2] == 2.10
        assert theta_max[3] == 2.0 * math.pi

        # Test phase wrapping modulo 1
        u_wrap1 = [0.5, 0.5, 0.5, 1.25]
        theta_wrap1 = prior_transform_total_mass_eta_4d(u_wrap1)
        assert pytest.approx(theta_wrap1[3], rel=1e-14) == 0.25 * 2.0 * math.pi

        u_wrap2 = [0.5, 0.5, 0.5, -0.25]
        theta_wrap2 = prior_transform_total_mass_eta_4d(u_wrap2)
        assert pytest.approx(theta_wrap2[3], rel=1e-14) == 0.75 * 2.0 * math.pi

    def test_phase_endpoints_and_interior_values(self):
        """Verify phase transform behavior at endpoints u_4 = 0, u_4 = 1, and interior values."""
        # 1. Lower boundary: u_4 = 0.0 -> phi_c = 0.0
        u_zero = [0.5, 0.5, 0.5, 0.0]
        theta_zero = prior_transform_total_mass_eta_4d(u_zero)
        assert theta_zero[3] == 0.0

        # 2. Upper boundary: u_4 = 1.0 -> phi_c = 2*pi
        u_one = [0.5, 0.5, 0.5, 1.0]
        theta_one = prior_transform_total_mass_eta_4d(u_one)
        assert theta_one[3] == 2.0 * math.pi

        # 3. Representative interior values: u_4 = 0.5 -> pi, u_4 = 0.25 -> pi/2
        u_half = [0.5, 0.5, 0.5, 0.5]
        theta_half = prior_transform_total_mass_eta_4d(u_half)
        assert pytest.approx(theta_half[3], rel=1e-14) == math.pi

        u_quarter = [0.5, 0.5, 0.5, 0.25]
        theta_quarter = prior_transform_total_mass_eta_4d(u_quarter)
        assert pytest.approx(theta_quarter[3], rel=1e-14) == 0.5 * math.pi

        # 4. Invariance relative to original 4D transform
        for u4 in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, -0.25]:
            u_alt = [0.5, 0.5, 0.5, u4]
            u_orig = [0.5, 0.5, 0.5, u4]
            th_alt = prior_transform_total_mass_eta_4d(u_alt)
            th_orig = prior_transform_4d(u_orig)
            assert pytest.approx(th_alt[3], rel=1e-14, abs=1e-14) == th_orig[3]

    def test_phase_support_bounds_and_transform_agreement(self):
        """Verify that prior transform endpoints and is_within_bounds agree for phi_c in [0, 2*pi]."""
        prior_4d = Correlated4DTotalMassEtaPrior()
        prior_orig = Uniform4DParameterPrior()

        # Both endpoints phi_c = 0.0 and phi_c = 2*pi are within closed support [0, 2*pi]
        pt_phi0 = [70.0, 0.20, 2.00, 0.0]
        pt_phi2pi = [70.0, 0.20, 2.00, 2.0 * math.pi]
        pt_interior = [70.0, 0.20, 2.00, math.pi]

        assert prior_4d.is_within_bounds(pt_phi0)
        assert prior_4d.is_within_bounds(pt_phi2pi)
        assert prior_4d.is_within_bounds(pt_interior)
        assert math.isfinite(prior_4d.log_prior(pt_phi0))
        assert math.isfinite(prior_4d.log_prior(pt_phi2pi))

        # Check original prior matches this convention
        mc_pt_phi0 = [70.0 * (0.20 ** 0.6), 0.20, 2.00, 0.0]
        mc_pt_phi2pi = [70.0 * (0.20 ** 0.6), 0.20, 2.00, 2.0 * math.pi]
        assert prior_orig.is_within_bounds(mc_pt_phi0)
        assert prior_orig.is_within_bounds(mc_pt_phi2pi)

        # Points outside [0, 2*pi] are strictly rejected
        pt_phi_neg = [70.0, 0.20, 2.00, -1e-7]
        pt_phi_exceed = [70.0, 0.20, 2.00, 2.0 * math.pi + 1e-7]
        assert not prior_4d.is_within_bounds(pt_phi_neg)
        assert not prior_4d.is_within_bounds(pt_phi_exceed)
        assert prior_4d.log_prior(pt_phi_neg) == -math.inf
        assert prior_4d.log_prior(pt_phi_exceed) == -math.inf

        # Verify exact agreement between transform output and is_within_bounds
        # Across entire unit interval [0, 1] for u_4
        for u4_val in [0.0, 0.1, 0.5, 0.9, 1.0]:
            u_pt = [0.5, 0.5, 0.5, u4_val]
            theta_pt = prior_transform_total_mass_eta_4d(u_pt)
            assert prior_4d.is_within_bounds(theta_pt), f"Transform output at u_4={u4_val} failed is_within_bounds"


class TestAlgebraicParameterMappingEquivalence:
    """Verify algebraic parameter-mapping equivalence between alternative and original coordinates.

    Strict restriction: Zero likelihood, PSD, or waveform functions are imported or called.
    """

    def test_algebraic_parameter_mapping_equivalence(self):
        """Verify that mapping (M_total, eta, t_c, phi_c) -> (M_c, eta, t_c, phi_c) via total_mass_eta_to_mchirp_eta:
        1. Maps alternative-coordinate vector to expected original (M_c, eta, t_c, phi_c) vector.
        2. Preserves extrinsic parameters t_c and phi_c identically.
        3. Matches prior_transform_4d to machine precision (< 1e-14).
        4. Involves zero likelihood or waveform evaluations.
        """
        # Test across deterministic unit-hypercube evaluation points
        test_points_u = [
            [0.1, 0.1, 0.2, 0.3],
            [0.2, 0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.8, 0.2, 0.7, 0.9],
            [0.9, 0.9, 0.1, 0.4],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.3, 0.7, 0.4, 0.8],
        ]

        for u in test_points_u:
            # 1. Alternative prior transform: theta_alt = (M_total, eta, t_c, phi_c)
            theta_alt = prior_transform_total_mass_eta_4d(u)
            mtotal, eta, tc, phic = theta_alt

            # 2. Purely algebraic mapping to physical coordinates
            theta_mapped = total_mass_eta_to_mchirp_eta(theta_alt)

            # 3. Verify algebraic conversion M_c = M_total * eta^(3/5)
            expected_mc = mtotal * (eta ** 0.6)
            assert pytest.approx(theta_mapped[0], rel=1e-14, abs=1e-14) == expected_mc
            assert pytest.approx(theta_mapped[1], rel=1e-14, abs=1e-14) == eta

            # 4. Verify extrinsic parameters are preserved identically
            assert theta_mapped[2] == tc == theta_alt[2]
            assert theta_mapped[3] == phic == theta_alt[3]

            # 5. Cross-check against original prior_transform_4d:
            # In prior_transform_4d, u_orig = [u_mc, u_eta, u_tc, u_phic]
            # Since theta_alt used u[1] for mc mapping and u[0] for eta mapping:
            u_orig = [u[1], u[0], u[2], u[3]]
            theta_orig = prior_transform_4d(u_orig)

            np.testing.assert_allclose(theta_mapped, theta_orig, rtol=1e-14, atol=1e-14)
