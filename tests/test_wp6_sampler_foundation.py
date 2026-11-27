"""Focused unit tests for WP6 Stage 1: Nested-Sampling Foundation and Prior Transforms.

Tests cover:
1. Analytic unit-square prior transformation (corners, interior points, bounds, error rejection).
2. Prior normalization and absence of component-mass Jacobians.
3. NestedSamplerAdapter configuration validation and interface contracts.
4. Graceful handling of unavailable optional dependencies (dynesty).
5. Mocked nested-sampling execution and result extraction.
6. GravitationalWaveLikelihoodWrapper integration and normalization preservation.
7. Minimal integration smoke test (skipped when dynesty is not installed).
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gw_inference.constants import C_ISCO_SI, M_SUN_SI, MPC_SI
from gw_inference.grid import FrequencyGrid
from gw_inference.likelihood import (
    FrequencyDomainGaussianLikelihood,
    compute_log_likelihood_normalization,
)
from gw_inference.posterior import GravitationalWaveLikelihoodWrapper
from gw_inference.prior import (
    UniformChirpMassEtaPrior,
    prior_transform_2d,
)
from gw_inference.psd import analytical_detector_psd
from gw_inference.sampler import (
    NestedSamplerAdapter,
    NestedSamplingResult,
    extract_total_evaluations,
    is_dynesty_available,
)
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    component_masses_to_chirp_mass,
    taylorf2_fd_waveform,
)


# ==============================================================================
# 1. Prior Transformation Tests
# ==============================================================================

class TestPriorTransformation2D:
    """Tests for standalone analytic prior transform `prior_transform_2d`."""

    def test_output_dimension_and_parameter_ordering(self) -> None:
        """Verify output has shape (2,) and ordering [M_c, eta]."""
        u = np.array([0.5, 0.5])
        theta = prior_transform_2d(u)
        assert isinstance(theta, np.ndarray)
        assert theta.shape == (2,)
        assert theta.dtype == np.float64
        # M_c = 20 + 20 * 0.5 = 30.0, eta = 0.10 + 0.15 * 0.5 = 0.175
        assert math.isclose(theta[0], 30.0, rel_tol=1e-14)
        assert math.isclose(theta[1], 0.175, rel_tol=1e-14)

    def test_hypercube_corners(self) -> None:
        """Verify transformation at all 4 corners of the unit square [0, 1]^2."""
        # Lower-left corner: (0, 0) -> (20.0, 0.10)
        theta_00 = prior_transform_2d([0.0, 0.0])
        assert math.isclose(theta_00[0], 20.0, abs_tol=1e-14)
        assert math.isclose(theta_00[1], 0.10, abs_tol=1e-14)

        # Upper-right corner: (1, 1) -> (40.0, 0.25)
        theta_11 = prior_transform_2d([1.0, 1.0])
        assert math.isclose(theta_11[0], 40.0, abs_tol=1e-14)
        assert math.isclose(theta_11[1], 0.25, abs_tol=1e-14)

        # Lower-right corner: (1, 0) -> (40.0, 0.10)
        theta_10 = prior_transform_2d([1.0, 0.0])
        assert math.isclose(theta_10[0], 40.0, abs_tol=1e-14)
        assert math.isclose(theta_10[1], 0.10, abs_tol=1e-14)

        # Upper-left corner: (0, 1) -> (20.0, 0.25)
        theta_01 = prior_transform_2d([0.0, 1.0])
        assert math.isclose(theta_01[0], 20.0, abs_tol=1e-14)
        assert math.isclose(theta_01[1], 0.25, abs_tol=1e-14)

    def test_interior_points(self) -> None:
        """Verify transformation at multiple interior points."""
        test_cases = [
            ([0.25, 0.75], 25.0, 0.10 + 0.15 * 0.75),
            ([0.10, 0.20], 22.0, 0.10 + 0.15 * 0.20),
            ([0.80, 0.90], 36.0, 0.10 + 0.15 * 0.90),
        ]
        for u, expected_mc, expected_eta in test_cases:
            theta = prior_transform_2d(u)
            assert math.isclose(theta[0], expected_mc, rel_tol=1e-14)
            assert math.isclose(theta[1], expected_eta, rel_tol=1e-14)

    def test_batch_2d_input(self) -> None:
        """Verify vectorized 2D array input of shape (N, 2)."""
        u_batch = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])
        theta_batch = prior_transform_2d(u_batch)
        assert theta_batch.shape == (3, 2)
        assert math.isclose(theta_batch[0, 0], 20.0)
        assert math.isclose(theta_batch[0, 1], 0.10)
        assert math.isclose(theta_batch[1, 0], 40.0)
        assert math.isclose(theta_batch[1, 1], 0.25)
        assert math.isclose(theta_batch[2, 0], 30.0)
        assert math.isclose(theta_batch[2, 1], 0.175)

    def test_deterministic_output(self) -> None:
        """Verify that calling the transformation repeatedly yields bitwise identical results."""
        u = np.array([0.314159, 0.271828])
        res1 = prior_transform_2d(u)
        res2 = prior_transform_2d(u)
        assert np.array_equal(res1, res2)

    def test_rejection_of_invalid_dimensions(self) -> None:
        """Verify ValueError is raised for inputs that are not length 2 or shape (N, 2)."""
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            prior_transform_2d([0.5])
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            prior_transform_2d([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="shape \\(N, 2\\)"):
            prior_transform_2d(np.ones((4, 3)))
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            prior_transform_2d(np.ones((2, 2, 2)))

    def test_rejection_of_non_finite_inputs(self) -> None:
        """Verify ValueError is raised for NaN and infinite values."""
        with pytest.raises(ValueError, match="non-finite values"):
            prior_transform_2d([np.nan, 0.5])
        with pytest.raises(ValueError, match="non-finite values"):
            prior_transform_2d([0.5, np.nan])
        with pytest.raises(ValueError, match="non-finite values"):
            prior_transform_2d([np.inf, 0.5])
        with pytest.raises(ValueError, match="non-finite values"):
            prior_transform_2d([0.5, -np.inf])

    def test_rejection_of_out_of_range_values(self) -> None:
        """Verify ValueError is raised for inputs outside the unit interval [0, 1]."""
        with pytest.raises(ValueError, match="must lie in the unit interval"):
            prior_transform_2d([-0.001, 0.5])
        with pytest.raises(ValueError, match="must lie in the unit interval"):
            prior_transform_2d([1.0001, 0.5])
        with pytest.raises(ValueError, match="must lie in the unit interval"):
            prior_transform_2d([0.5, -0.01])
        with pytest.raises(ValueError, match="must lie in the unit interval"):
            prior_transform_2d([0.5, 1.001])


# ==============================================================================
# 2. Prior Class and Mathematical Measure Tests
# ==============================================================================

class TestUniformChirpMassEtaPrior:
    """Tests for `UniformChirpMassEtaPrior` class properties and normalization."""

    def test_default_properties_and_normalization(self) -> None:
        """Verify default bounds, coordinate volume, and normalized prior density."""
        prior = UniformChirpMassEtaPrior()
        assert prior.mc_min == 20.0
        assert prior.mc_max == 40.0
        assert prior.eta_min == 0.10
        assert prior.eta_max == 0.25

        # Volume: (40 - 20) * (0.25 - 0.10) = 20 * 0.15 = 3.0 M_sun
        expected_volume = 3.0
        assert math.isclose(prior.volume, expected_volume, rel_tol=1e-14)
        assert math.isclose(prior.normalized_density, 1.0 / 3.0, rel_tol=1e-14)
        assert math.isclose(prior.log_density, -math.log(3.0), rel_tol=1e-14)

    def test_log_prior_evaluation_and_support_boundaries(self) -> None:
        """Verify log_prior returns -ln(3) inside support and -inf outside."""
        prior = UniformChirpMassEtaPrior()
        expected_log_p = -math.log(3.0)

        # Inside points
        assert math.isclose(prior.log_prior([25.0, 0.20]), expected_log_p, rel_tol=1e-14)
        assert math.isclose(prior.log_prior([20.0, 0.10]), expected_log_p, rel_tol=1e-14)
        assert math.isclose(prior.log_prior([40.0, 0.25]), expected_log_p, rel_tol=1e-14)

        # Outside points
        assert prior.log_prior([19.99, 0.20]) == -math.inf
        assert prior.log_prior([40.01, 0.20]) == -math.inf
        assert prior.log_prior([30.0, 0.099]) == -math.inf
        assert prior.log_prior([30.0, 0.251]) == -math.inf
        assert prior.log_prior([np.nan, 0.20]) == -math.inf

    def test_numerical_normalization_quadrature(self) -> None:
        """Verify that the 2D Riemann integral of the prior density equals 1.0000."""
        prior = UniformChirpMassEtaPrior()
        mc_grid = np.linspace(prior.mc_min, prior.mc_max, 201)
        eta_grid = np.linspace(prior.eta_min, prior.eta_max, 151)
        d_mc = mc_grid[1] - mc_grid[0]
        d_eta = eta_grid[1] - eta_grid[0]

        # 2D midpoint integration
        mc_mid = 0.5 * (mc_grid[:-1] + mc_grid[1:])
        eta_mid = 0.5 * (eta_grid[:-1] + eta_grid[1:])
        MC, ETA = np.meshgrid(mc_mid, eta_mid)

        density_values = np.full(MC.shape, prior.normalized_density)
        integral = float(np.sum(density_values) * d_mc * d_eta)
        assert math.isclose(integral, 1.0, rel_tol=1e-10)

    def test_absence_of_component_mass_jacobian(self) -> None:
        """Verify that the prior density is strictly constant in (M_c, eta) space.

        The approved specification requires a uniform prior in (M_c, eta), NOT a
        prior uniform in (m_1, m_2). A uniform (m_1, m_2) prior would introduce a
        variable factor proportional to M_c / (eta^(6/5) * sqrt(1 - 4*eta)).
        """
        prior = UniformChirpMassEtaPrior()
        # Test multiple points with different symmetric mass ratios
        p1 = prior.normalized_density
        # Check that log_prior is identically constant across varying eta and M_c
        for mc in [21.0, 28.0, 35.0, 39.0]:
            for eta in [0.11, 0.18, 0.22, 0.249]:
                assert math.isclose(prior.log_prior([mc, eta]), -math.log(3.0), rel_tol=1e-14)

    def test_invalid_constructor_bounds(self) -> None:
        """Verify constructor validates physical bounds."""
        with pytest.raises(ValueError, match="mc_min must be positive"):
            UniformChirpMassEtaPrior(mc_min=0.0)
        with pytest.raises(ValueError, match="mc_max must be greater"):
            UniformChirpMassEtaPrior(mc_min=30.0, mc_max=20.0)
        with pytest.raises(ValueError, match="eta_min must be positive"):
            UniformChirpMassEtaPrior(eta_min=-0.05)
        with pytest.raises(ValueError, match="eta_max must satisfy"):
            UniformChirpMassEtaPrior(eta_max=0.30)  # Exceeds equal-mass boundary 0.25


# ==============================================================================
# 3. NestedSamplerAdapter Interface and Configuration Tests
# ==============================================================================

class TestNestedSamplerAdapter:
    """Tests for `NestedSamplerAdapter` input validation, wiring, and mocked execution."""

    def test_valid_initialization(self) -> None:
        """Verify proper attribute assignment on valid initialization."""
        dummy_logl = lambda theta: -0.5 * np.sum(theta ** 2)
        dummy_ptform = lambda u: u * 10.0

        adapter = NestedSamplerAdapter(
            log_likelihood_fn=dummy_logl,
            prior_transform_fn=dummy_ptform,
            ndim=2,
            nlive=250,
            dlogz=0.05,
            seed=42,
            param_names=["p1", "p2"],
        )
        assert adapter.ndim == 2
        assert adapter.nlive == 250
        assert adapter.dlogz == 0.05
        assert adapter.param_names == ("p1", "p2")

    def test_configuration_validation(self) -> None:
        """Verify constructor rejects non-callable and out-of-bound arguments."""
        dummy_fn = lambda x: x

        with pytest.raises(TypeError, match="log_likelihood_fn must be callable"):
            NestedSamplerAdapter(log_likelihood_fn="not_a_callable", prior_transform_fn=dummy_fn)  # type: ignore

        with pytest.raises(TypeError, match="prior_transform_fn must be callable"):
            NestedSamplerAdapter(log_likelihood_fn=dummy_fn, prior_transform_fn=123)  # type: ignore

        with pytest.raises(ValueError, match="ndim must be an integer >= 1"):
            NestedSamplerAdapter(log_likelihood_fn=dummy_fn, prior_transform_fn=dummy_fn, ndim=0)

        with pytest.raises(ValueError, match="nlive must be an integer >= 1"):
            NestedSamplerAdapter(log_likelihood_fn=dummy_fn, prior_transform_fn=dummy_fn, nlive=-10)

        with pytest.raises(ValueError, match="dlogz must be a positive finite float"):
            NestedSamplerAdapter(log_likelihood_fn=dummy_fn, prior_transform_fn=dummy_fn, dlogz=0.0)

        with pytest.raises(ValueError, match="param_names length"):
            NestedSamplerAdapter(
                log_likelihood_fn=dummy_fn,
                prior_transform_fn=dummy_fn,
                ndim=2,
                param_names=["only_one"],
            )

    def test_graceful_handling_when_dynesty_unavailable(self) -> None:
        """Verify informative ImportError is raised when dynesty is not available."""
        adapter = NestedSamplerAdapter(
            log_likelihood_fn=lambda theta: 0.0,
            prior_transform_fn=lambda u: u,
            ndim=2,
        )
        # Halt `import dynesty` deterministically via sys.modules regardless of module aliasing
        with patch.dict("sys.modules", {"dynesty": None}):
            with pytest.raises(ImportError, match="dynesty is required to run nested sampling"):
                adapter.run()

    def test_mocked_dynesty_execution_and_result_unpacking(self) -> None:
        """Verify dynesty.NestedSampler is called with correct parameters and results unpacked."""
        dummy_logl = MagicMock(return_value=-10.0)
        dummy_ptform = MagicMock(side_effect=lambda u: u * 2.0)

        adapter = NestedSamplerAdapter(
            log_likelihood_fn=dummy_logl,
            prior_transform_fn=dummy_ptform,
            ndim=2,
            nlive=100,
            dlogz=0.1,
            seed=12345,
            param_names=["x", "y"],
        )

        # Create mock dynesty module and result structure representing actual per-iteration ncall array
        mock_results = MagicMock()
        mock_results.samples = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
        mock_results.logwt = np.array([-5.0, -4.0, -3.0])
        mock_results.logz = np.array([-4.5, -3.5, -2.5])
        mock_results.logzerr = np.array([0.2, 0.15, 0.1])
        mock_results.ncall = np.array([120, 150, 150])  # Per-iteration calls summing to 420

        mock_sampler_instance = MagicMock()
        mock_sampler_instance.results = mock_results
        mock_sampler_cls = MagicMock(return_value=mock_sampler_instance)

        mock_dynesty = MagicMock()
        mock_dynesty.NestedSampler = mock_sampler_cls

        with patch.dict("sys.modules", {"dynesty": mock_dynesty}):
            with patch("gw_inference.sampler.is_dynesty_available", return_value=True):
                result = adapter.run()

                # Verify constructor was called with expected arguments
                mock_sampler_cls.assert_called_once()
                call_kwargs = mock_sampler_cls.call_args[1]
                assert call_kwargs["loglikelihood"] == dummy_logl
                assert call_kwargs["prior_transform"] == dummy_ptform
                assert call_kwargs["ndim"] == 2
                assert call_kwargs["nlive"] == 100

                # Verify run_nested was called with dlogz
                mock_sampler_instance.run_nested.assert_called_once_with(dlogz=0.1)

                # Verify unpacked result dataclass
                assert isinstance(result, NestedSamplingResult)
                assert np.array_equal(result.samples, mock_results.samples)
                assert math.isclose(result.log_evidence, -2.5)
                assert math.isclose(result.log_evidence_err, 0.1)
                assert result.n_evaluations == 420
                assert result.param_names == ("x", "y")
                assert math.isclose(float(np.sum(result.weights)), 1.0, rel_tol=1e-12)
                assert result.posterior_samples.shape[1] == 2

                # Verify summary dictionary
                summary = result.summary()
                assert summary["log_evidence"] == -2.5
                assert summary["n_evaluations"] == 420
                assert "x" in summary["parameters"]
                assert "y" in summary["parameters"]


# ==============================================================================
# 4. Sampler Result Extraction and Evaluation Count Tests
# ==============================================================================

class TestSamplerResultExtraction:
    """Tests for robust extraction of likelihood evaluation counts and posterior weights."""

    def test_ncall_array_sum(self) -> None:
        """Verify that an array-valued ncall produces the exact integer sum."""
        # 1D NumPy integer array
        res_np = MagicMock()
        res_np.ncall = np.array([100, 200, 120])
        total = extract_total_evaluations(res_np, n_samples=3)
        assert isinstance(total, int)
        assert total == 420

        # Python list of per-iteration call counts
        res_list = MagicMock()
        res_list.ncall = [50, 75, 125]
        total_list = extract_total_evaluations(res_list, n_samples=3)
        assert isinstance(total_list, int)
        assert total_list == 250

    def test_ncall_scalar_support(self) -> None:
        """Verify that scalar-valued ncall (int, float, 0D array) is supported."""
        # Integer scalar
        res_int = MagicMock()
        res_int.ncall = 500
        assert extract_total_evaluations(res_int, n_samples=10) == 500

        # Float scalar representing an integer count
        res_float = MagicMock()
        res_float.ncall = 500.0
        assert extract_total_evaluations(res_float, n_samples=10) == 500

        # 0D NumPy array scalar
        res_0d = MagicMock()
        res_0d.ncall = np.array(500)
        assert extract_total_evaluations(res_0d, n_samples=10) == 500

    def test_ncall_missing_or_none(self) -> None:
        """Verify ValueError is raised when ncall is missing or None."""
        # Missing attribute on mock
        res_missing = MagicMock(spec=[])  # No attributes
        with pytest.raises(ValueError, match="missing 'ncall'"):
            extract_total_evaluations(res_missing)

        # Explicitly None
        res_none = MagicMock()
        res_none.ncall = None
        with pytest.raises(ValueError, match="missing 'ncall'"):
            extract_total_evaluations(res_none)

        # Dictionary with missing key
        with pytest.raises(ValueError, match="missing 'ncall'"):
            extract_total_evaluations({})

    def test_ncall_empty(self) -> None:
        """Verify ValueError is raised when ncall is empty."""
        res_empty_arr = MagicMock()
        res_empty_arr.ncall = np.array([])
        with pytest.raises(ValueError, match="ncall array is empty"):
            extract_total_evaluations(res_empty_arr)

        res_empty_list = MagicMock()
        res_empty_list.ncall = []
        with pytest.raises(ValueError, match="ncall array is empty"):
            extract_total_evaluations(res_empty_list)

    def test_ncall_invalid_shapes(self) -> None:
        """Verify ValueError is raised for multi-dimensional ncall arrays."""
        res_2d = MagicMock()
        res_2d.ncall = np.array([[10, 20], [30, 40]])
        with pytest.raises(ValueError, match="ncall array must be 1D"):
            extract_total_evaluations(res_2d)

        res_3d = MagicMock()
        res_3d.ncall = np.ones((2, 2, 2))
        with pytest.raises(ValueError, match="ncall array must be 1D"):
            extract_total_evaluations(res_3d)

    def test_ncall_invalid_values(self) -> None:
        """Verify ValueError/TypeError is raised for invalid counts (negative, non-finite, etc.)."""
        # Negative in array
        res_neg_arr = MagicMock()
        res_neg_arr.ncall = np.array([10, -5, 20])
        with pytest.raises(ValueError, match="negative evaluation counts"):
            extract_total_evaluations(res_neg_arr)

        # Negative scalar
        res_neg_scalar = MagicMock()
        res_neg_scalar.ncall = -10
        with pytest.raises(ValueError, match="non-negative"):
            extract_total_evaluations(res_neg_scalar)

        # NaN in array
        res_nan = MagicMock()
        res_nan.ncall = np.array([10.0, np.nan, 20.0])
        with pytest.raises(ValueError, match="non-finite values"):
            extract_total_evaluations(res_nan)

        # Inf in array
        res_inf = MagicMock()
        res_inf.ncall = np.array([10.0, np.inf, 20.0])
        with pytest.raises(ValueError, match="non-finite values"):
            extract_total_evaluations(res_inf)

        # Non-integer float
        res_non_int = MagicMock()
        res_non_int.ncall = 42.5
        with pytest.raises(ValueError, match="integer count"):
            extract_total_evaluations(res_non_int)

        # Non-numeric type
        res_str = MagicMock()
        res_str.ncall = "invalid_string"
        with pytest.raises(TypeError):
            extract_total_evaluations(res_str)

        # Total call count less than number of dead points
        res_inconsistent = MagicMock()
        res_inconsistent.ncall = np.array([1, 1])  # Total = 2
        with pytest.raises(ValueError, match="cannot be less than number of dead points"):
            extract_total_evaluations(res_inconsistent, n_samples=10)

    def test_ncall_fractional_array_rejection(self) -> None:
        """Verify ValueError is raised when array entries contain fractional values."""
        res_frac1 = MagicMock()
        res_frac1.ncall = np.array([10.5, 20.0])
        with pytest.raises(ValueError, match="fractional"):
            extract_total_evaluations(res_frac1)

        res_frac2 = MagicMock()
        res_frac2.ncall = [10.0, 20.25]
        with pytest.raises(ValueError, match="fractional"):
            extract_total_evaluations(res_frac2)

    def test_ncall_complex_rejection(self) -> None:
        """Verify TypeError is raised when ncall is complex-valued."""
        res_cplx_arr = MagicMock()
        res_cplx_arr.ncall = np.array([10 + 0j, 20 + 0j])
        with pytest.raises(TypeError, match="complex-valued"):
            extract_total_evaluations(res_cplx_arr)

        res_cplx_scalar = MagicMock()
        res_cplx_scalar.ncall = 10 + 0j
        with pytest.raises(TypeError, match="complex-valued"):
            extract_total_evaluations(res_cplx_scalar)

    def test_ncall_accumulation_overflow_protection(self) -> None:
        """Verify safe accumulation avoiding fixed-width integer overflow."""
        # 32-bit signed int overflow: 3 * 2^30 = 3221225472 > 2^31 - 1
        arr_32 = np.array([2**30, 2**30, 2**30], dtype=np.int32)
        res_32 = MagicMock()
        res_32.ncall = arr_32
        total_32 = extract_total_evaluations(res_32, n_samples=3)
        assert isinstance(total_32, int)
        assert total_32 == 3 * (2**30)

        # 64-bit signed int overflow: 3 * 2^62 > 2^63 - 1
        arr_64 = np.array([2**62, 2**62, 2**62], dtype=np.int64)
        res_64 = MagicMock()
        res_64.ncall = arr_64
        total_64 = extract_total_evaluations(res_64, n_samples=3)
        assert isinstance(total_64, int)
        assert total_64 == 3 * (2**62)

    def test_posterior_weight_normalization_numerical_stability(self) -> None:
        """Verify that posterior weights sum to 1.0 without arbitrary clipping distortion."""
        # Simulated logwt and logz with wide dynamic range
        # Top log-weight is at -100.0, lowest at -600.0 and -1000.0
        logwt = np.array([-100.0, -101.0, -102.0, -600.0, -1000.0])
        logz = -99.5

        # Perform the stable log-domain normalization from sampler.py
        rel_log_weights = logwt - logz
        max_rel = float(np.max(rel_log_weights))
        shifted_weights = np.exp(rel_log_weights - max_rel)
        sum_w = float(np.sum(shifted_weights))
        weights = shifted_weights / sum_w

        # Check normalization
        assert math.isclose(float(np.sum(weights)), 1.0, rel_tol=1e-14)
        assert np.all(weights >= 0.0)

        # Check that -600.0 preserves its tiny true magnitude without artificial clipping to exp(-300)
        assert 0.0 < weights[3] < 1e-200

        # Check that -1000.0 underflows naturally to 0.0 without distorting the top points
        assert weights[4] == 0.0

        # Check that relative probabilities of top points match exact odds
        # p(theta_0) / p(theta_1) = exp(-100.0 - (-101.0)) = e^1
        assert math.isclose(weights[0] / weights[1], math.e, rel_tol=1e-12)
        # p(theta_0) / p(theta_2) = exp(-100.0 - (-102.0)) = e^2
        assert math.isclose(weights[0] / weights[2], math.e ** 2, rel_tol=1e-12)

        # Check deep-negative regime (where unshifted exp would underflow all to 0.0)
        deep_logwt = np.array([-1500.0, -1501.0])
        deep_logz = -1499.5
        deep_rel = deep_logwt - deep_logz
        deep_shifted = np.exp(deep_rel - float(np.max(deep_rel)))
        deep_weights = deep_shifted / float(np.sum(deep_shifted))

        assert math.isclose(float(np.sum(deep_weights)), 1.0, rel_tol=1e-14)
        assert math.isclose(deep_weights[0] / deep_weights[1], math.e, rel_tol=1e-12)


# ==============================================================================
# 5. Fail-Safe Result Validation Tests in NestedSamplerAdapter
# ==============================================================================

class TestNestedSamplerAdapterResultValidation:
    """Tests for fail-safe validation of samples, logwt, and evidence in NestedSamplerAdapter."""

    @pytest.fixture
    def dummy_adapter(self) -> NestedSamplerAdapter:
        return NestedSamplerAdapter(
            log_likelihood_fn=lambda theta: -0.5 * np.sum(theta ** 2),
            prior_transform_fn=lambda u: u * 10.0,
            ndim=2,
            nlive=50,
        )

    def _run_with_mock_results(self, adapter: NestedSamplerAdapter, mock_results: MagicMock) -> Any:
        mock_sampler = MagicMock()
        mock_sampler.results = mock_results
        mock_dynesty = MagicMock()
        mock_dynesty.NestedSampler = MagicMock(return_value=mock_sampler)

        with patch.dict("sys.modules", {"dynesty": mock_dynesty}):
            with patch("gw_inference.sampler.is_dynesty_available", return_value=True):
                return adapter.run()

    def test_run_rejects_nan_logwt(self, dummy_adapter: NestedSamplerAdapter) -> None:
        """Verify ValueError is raised if logwt contains NaN."""
        res = MagicMock()
        res.samples = np.array([[1.0, 2.0], [3.0, 4.0]])
        res.logwt = np.array([-5.0, np.nan])
        res.logz = np.array([-4.0])
        res.logzerr = np.array([0.1])
        res.ncall = np.array([10, 10])

        with pytest.raises(ValueError, match="contains NaN values"):
            self._run_with_mock_results(dummy_adapter, res)

    def test_run_rejects_positive_infinite_logwt(self, dummy_adapter: NestedSamplerAdapter) -> None:
        """Verify ValueError is raised if logwt contains +inf."""
        res = MagicMock()
        res.samples = np.array([[1.0, 2.0], [3.0, 4.0]])
        res.logwt = np.array([-5.0, np.inf])
        res.logz = np.array([-4.0])
        res.logzerr = np.array([0.1])
        res.ncall = np.array([10, 10])

        with pytest.raises(ValueError, match="positive-infinite"):
            self._run_with_mock_results(dummy_adapter, res)

    def test_run_rejects_all_negative_infinite_logwt(self, dummy_adapter: NestedSamplerAdapter) -> None:
        """Verify ValueError is raised if all logwt entries are -inf."""
        res = MagicMock()
        res.samples = np.array([[1.0, 2.0], [3.0, 4.0]])
        res.logwt = np.array([-np.inf, -np.inf])
        res.logz = np.array([-4.0])
        res.logzerr = np.array([0.1])
        res.ncall = np.array([10, 10])

        with pytest.raises(ValueError, match="All logwt entries are negative-infinite"):
            self._run_with_mock_results(dummy_adapter, res)

    def test_run_permits_individual_negative_infinite_logwt(self, dummy_adapter: NestedSamplerAdapter) -> None:
        """Verify individual -inf logwt entries are permitted and assigned 0.0 weight."""
        res = MagicMock()
        res.samples = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        res.logwt = np.array([-5.0, -np.inf, -3.0])
        res.logz = np.array([-2.5])
        res.logzerr = np.array([0.1])
        res.ncall = np.array([10, 15, 20])

        result = self._run_with_mock_results(dummy_adapter, res)
        assert isinstance(result, NestedSamplingResult)
        assert result.weights[1] == 0.0
        assert math.isclose(float(np.sum(result.weights)), 1.0, rel_tol=1e-12)
        assert result.weights[0] > 0.0
        assert result.weights[2] > 0.0

    def test_run_rejects_invalid_evidence(self, dummy_adapter: NestedSamplerAdapter) -> None:
        """Verify ValueError is raised for non-finite logz or logzerr."""
        for bad_z in [np.nan, np.inf, -np.inf]:
            res = MagicMock()
            res.samples = np.array([[1.0, 2.0], [3.0, 4.0]])
            res.logwt = np.array([-5.0, -4.0])
            res.logz = np.array([bad_z])
            res.logzerr = np.array([0.1])
            res.ncall = np.array([10, 10])

            with pytest.raises(ValueError, match="Evidence logz must be finite"):
                self._run_with_mock_results(dummy_adapter, res)

    def test_run_rejects_mismatched_and_empty_arrays(self, dummy_adapter: NestedSamplerAdapter) -> None:
        """Verify ValueError is raised for empty arrays or shape/count mismatches."""
        # Empty samples
        res_empty_s = MagicMock()
        res_empty_s.samples = np.empty((0, 2))
        res_empty_s.logwt = np.array([-5.0])
        res_empty_s.logz = np.array([-4.0])
        res_empty_s.logzerr = np.array([0.1])
        res_empty_s.ncall = np.array([10])
        with pytest.raises(ValueError, match="samples is empty"):
            self._run_with_mock_results(dummy_adapter, res_empty_s)

        # Empty logwt
        res_empty_w = MagicMock()
        res_empty_w.samples = np.array([[1.0, 2.0]])
        res_empty_w.logwt = np.array([])
        res_empty_w.logz = np.array([-4.0])
        res_empty_w.logzerr = np.array([0.1])
        res_empty_w.ncall = np.array([10])
        with pytest.raises(ValueError, match="logwt is empty"):
            self._run_with_mock_results(dummy_adapter, res_empty_w)

        # Count mismatch
        res_mismatch = MagicMock()
        res_mismatch.samples = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        res_mismatch.logwt = np.array([-5.0, -4.0])
        res_mismatch.logz = np.array([-4.0])
        res_mismatch.logzerr = np.array([0.1])
        res_mismatch.ncall = np.array([10, 10, 10])
        with pytest.raises(ValueError, match="Sample count mismatch"):
            self._run_with_mock_results(dummy_adapter, res_mismatch)

        # Wrong sample dimensions (1D samples)
        res_1d = MagicMock()
        res_1d.samples = np.array([1.0, 2.0])
        res_1d.logwt = np.array([-5.0, -4.0])
        res_1d.logz = np.array([-4.0])
        res_1d.logzerr = np.array([0.1])
        res_1d.ncall = np.array([10, 10])
        with pytest.raises(ValueError, match="samples must be a 2D array"):
            self._run_with_mock_results(dummy_adapter, res_1d)

        # Wrong parameter count (ndim=3 when adapter ndim=2)
        res_3d = MagicMock()
        res_3d.samples = np.array([[1.0, 2.0, 3.0]])
        res_3d.logwt = np.array([-5.0])
        res_3d.logz = np.array([-4.0])
        res_3d.logzerr = np.array([0.1])
        res_3d.ncall = np.array([10])
        with pytest.raises(ValueError, match="must match adapter ndim"):
            self._run_with_mock_results(dummy_adapter, res_3d)


# ==============================================================================
# 6. GravitationalWaveLikelihoodWrapper Integration Tests
# ==============================================================================

class TestGravitationalWaveLikelihoodWrapper:
    """Tests integrating LikelihoodWrapper with FrequencyDomainGaussianLikelihood."""

    @pytest.fixture
    def setup_likelihood_env(self) -> tuple[FrequencyDomainGaussianLikelihood, FrequencyGrid, np.ndarray]:
        """Construct synthetic likelihood environment with flat PSD and active band."""
        sampling_rate = 2048.0
        n_samples = 4096
        delta_t = 1.0 / sampling_rate
        grid = FrequencyGrid(n_samples=n_samples, delta_t=delta_t)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)

        # Flat PSD S_n = 1e-46
        psd = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)

        # Synthetic BBH injection template
        m1_inj = 30.0 * M_SUN_SI
        m2_inj = 20.0 * M_SUN_SI
        dist_inj = 410.0 * MPC_SI
        h_inj = taylorf2_fd_waveform(grid.frequencies, m1_inj, m2_inj, dist_inj, phi_c=0.0)

        # Zero-noise data: data_fd = h_inj
        likelihood = FrequencyDomainGaussianLikelihood(
            data_fd=h_inj,
            psd=psd,
            grid=grid,
            mask=mask,
        )
        return likelihood, grid, h_inj

    def test_likelihood_normalization_preservation(
        self,
        setup_likelihood_env: tuple[FrequencyDomainGaussianLikelihood, FrequencyGrid, np.ndarray],
    ) -> None:
        """Verify wrapper preserves the full likelihood normalization ln L_norm."""
        likelihood, grid, _ = setup_likelihood_env
        wrapper = GravitationalWaveLikelihoodWrapper(
            likelihood=likelihood,
            distance_m=410.0 * MPC_SI,
            phi_c=0.0,
            t_c=0.0,
            f_start=20.0,
        )

        # Injected parameters: M = 50 M_sun, q = 20/30 = 2/3
        # eta = (30 * 20) / (50^2) = 600 / 2500 = 0.24
        # M_c = 50 * (0.24)^0.6 approx 21.2868 M_sun
        m1_msun = 30.0
        m2_msun = 20.0
        M_total = m1_msun + m2_msun
        eta_inj = (m1_msun * m2_msun) / (M_total ** 2)
        mc_inj = M_total * (eta_inj ** (3.0 / 5.0))

        # At the exact injection parameters in zero-noise, residual r_k = 0
        # Therefore, ln L must equal ln L_norm exactly!
        ln_l = wrapper.evaluate_log_likelihood([mc_inj, eta_inj])
        expected_ln_norm = likelihood.log_likelihood_norm

        assert math.isclose(ln_l, expected_ln_norm, rel_tol=1e-12)

    def test_out_of_bounds_and_unphysical_parameter_rejection(
        self,
        setup_likelihood_env: tuple[FrequencyDomainGaussianLikelihood, FrequencyGrid, np.ndarray],
    ) -> None:
        """Verify non-physical parameters return -inf."""
        likelihood, _, _ = setup_likelihood_env
        wrapper = GravitationalWaveLikelihoodWrapper(likelihood=likelihood)

        # eta > 0.25 (unphysical mass ratio)
        assert wrapper.evaluate_log_likelihood([30.0, 0.26]) == -math.inf
        # eta <= 0
        assert wrapper.evaluate_log_likelihood([30.0, -0.05]) == -math.inf
        # M_c <= 0
        assert wrapper.evaluate_log_likelihood([-10.0, 0.20]) == -math.inf
        # Total mass so high that f_ISCO <= f_start (20 Hz)
        # f_ISCO = 4397.17 / M_total <= 20 => M_total >= 219.86 M_sun
        # For eta = 0.20, M_total = M_c * 0.20^(-0.6) = M_c * 2.6265
        # M_c = 100.0 => M_total approx 262.6 M_sun => f_ISCO approx 16.7 Hz <= 20 Hz
        assert wrapper.evaluate_log_likelihood([100.0, 0.20]) == -math.inf


# ==============================================================================
# 7. Minimal Integration Smoke Test (Optional Dependency Aware)
# ==============================================================================

class TestDynestyIntegrationSmoke:
    """Minimal integration test executing dynesty if installed in the environment."""

    @pytest.mark.skipif(
        not is_dynesty_available(),
        reason="dynesty is not installed in the environment (optional dependency)",
    )
    def test_minimal_toy_gaussian_nested_sampling(self) -> None:
        """Inexpensive smoke test on a 2D Gaussian likelihood using actual dynesty.

        Only executed if dynesty is installed.
        """
        # 2D independent Gaussian with mean at (0.5, 0.5) and sigma = 0.1
        target_mean = np.array([0.5, 0.5])
        inv_cov = np.diag([100.0, 100.0])  # sigma = 0.1

        def toy_logl(theta: np.ndarray) -> float:
            diff = theta - target_mean
            return float(-0.5 * diff.T @ inv_cov @ diff)

        def toy_ptform(u: np.ndarray) -> np.ndarray:
            return np.asarray(u, dtype=np.float64)  # Unit square prior [0, 1]^2

        adapter = NestedSamplerAdapter(
            log_likelihood_fn=toy_logl,
            prior_transform_fn=toy_ptform,
            ndim=2,
            nlive=50,  # Minimal live points for fast smoke test
            dlogz=0.5,
            seed=42,
            param_names=["x", "y"],
        )

        result = adapter.run(print_progress=False)

        assert isinstance(result, NestedSamplingResult)
        assert len(result.samples) > 50
        assert math.isfinite(result.log_evidence)
        # Check posterior median is near 0.5 within statistical tolerance
        median_x = float(np.median(result.posterior_samples[:, 0]))
        median_y = float(np.median(result.posterior_samples[:, 1]))
        assert 0.4 <= median_x <= 0.6
        assert 0.4 <= median_y <= 0.6


# ==============================================================================
# 8. WP6 Stage 2 Inference and Grid Validation Tests
# ==============================================================================

class TestWP6Stage2InferenceAndGridValidation:
    """Focused tests for WP6 Stage 2 inference, grid quadrature, and statistical comparison."""

    def test_real_dynesty_toy_integration(self) -> None:
        """Verify real dynesty nested sampling on a normalized 2D Gaussian with analytical evidence."""
        target_mean = np.array([0.5, 0.5])
        sigma = 0.1
        norm_const = -math.log(2.0 * math.pi * (sigma ** 2))

        def toy_logl(theta: np.ndarray) -> float:
            diff = theta - target_mean
            return float(norm_const - 0.5 * np.sum(diff ** 2) / (sigma ** 2))

        def toy_ptform(u: np.ndarray) -> np.ndarray:
            return np.asarray(u, dtype=np.float64)

        adapter = NestedSamplerAdapter(
            log_likelihood_fn=toy_logl,
            prior_transform_fn=toy_ptform,
            ndim=2,
            nlive=100,
            dlogz=0.1,
            seed=42,
            param_names=["x", "y"],
        )

        res = adapter.run(bootstrap=0, print_progress=False)
        assert isinstance(res, NestedSamplingResult)
        assert res.n_evaluations > 1000
        assert math.isclose(float(np.sum(res.weights)), 1.0, rel_tol=1e-12)
        assert np.all(res.weights >= 0.0)

        # Exact analytical evidence on [0, 1]^2: [erf(0.5 / (sqrt(2) * sigma))]^2
        # ln Z_analytical = 2 * ln(erf(5 / sqrt(2))) ≈ -1.146607e-6
        arg = 0.5 / (math.sqrt(2.0) * sigma)
        ln_z_analytical = 2.0 * math.log(math.erf(arg))
        assert abs(ln_z_analytical - 0.0) < 2e-6

        # Verify sampler evidence recovery within 3 sigma_lnZ
        assert abs(res.log_evidence - ln_z_analytical) < 3.0 * res.log_evidence_err

        # Posterior means within sampling tolerance
        mean_x = float(np.mean(res.posterior_samples[:, 0]))
        mean_y = float(np.mean(res.posterior_samples[:, 1]))
        assert abs(mean_x - 0.5) < 0.05
        assert abs(mean_y - 0.5) < 0.05

    def test_2d_gw_inference_execution_path(self) -> None:
        """Verify the complete 2D GW inference execution path producing valid NestedSamplingResult."""
        duration = 4.0
        sampling_rate = 2048.0
        grid = FrequencyGrid(n_samples=int(duration * sampling_rate), delta_t=1.0 / sampling_rate)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)
        psd = analytical_detector_psd(grid.frequencies, model="aligo_representative")

        m1_kg = 36.0 * M_SUN_SI
        m2_kg = 29.0 * M_SUN_SI
        dist_m = 410.0 * MPC_SI

        M_total, eta_inj, mc_inj_kg = component_masses_to_chirp_mass(m1_kg, m2_kg)
        mc_inj = float(mc_inj_kg / M_SUN_SI)

        h_raw = taylorf2_fd_waveform(grid.frequencies, m1_kg, m2_kg, dist_m, phi_c=0.0)
        h_inj = apply_coalescence_time_shift(h_raw, grid.frequencies, 2.0)

        likelihood = FrequencyDomainGaussianLikelihood(data_fd=h_inj, psd=psd, grid=grid, mask=mask)
        wrapper = GravitationalWaveLikelihoodWrapper(likelihood=likelihood, distance_m=dist_m, phi_c=0.0, t_c=2.0, f_start=20.0)

        # Fast execution with nlive=100 and dlogz=0.5
        adapter = NestedSamplerAdapter(
            log_likelihood_fn=wrapper.evaluate_log_likelihood,
            prior_transform_fn=prior_transform_2d,
            ndim=2,
            nlive=100,
            dlogz=0.5,
            seed=42,
            param_names=["Mc", "eta"],
        )

        res = adapter.run(bootstrap=0, print_progress=False)
        assert isinstance(res, NestedSamplingResult)
        assert res.param_names == ("Mc", "eta")
        assert res.n_evaluations > 2000
        assert math.isclose(float(np.sum(res.weights)), 1.0, rel_tol=1e-12)

        # Injected parameters recovered within statistical uncertainty
        mean_mc = float(np.sum(res.samples[:, 0] * res.weights))
        mean_eta = float(np.sum(res.samples[:, 1] * res.weights))
        assert abs(mean_mc - mc_inj) < 0.05
        assert abs(mean_eta - eta_inj) < 0.001

    def test_prior_support_and_parameter_ordering_stage2(self) -> None:
        """Verify prior boundaries and parameter ordering for Stage 2."""
        prior = UniformChirpMassEtaPrior()
        assert prior.mc_min == 20.0 and prior.mc_max == 40.0
        assert prior.eta_min == 0.10 and prior.eta_max == 0.25

        # Corners
        pt_00 = prior_transform_2d([0.0, 0.0])
        pt_11 = prior_transform_2d([1.0, 1.0])
        assert math.isclose(pt_00[0], 20.0) and math.isclose(pt_00[1], 0.10)
        assert math.isclose(pt_11[0], 40.0) and math.isclose(pt_11[1], 0.25)

        # Injected values strictly inside
        m1_kg = 36.0 * M_SUN_SI
        m2_kg = 29.0 * M_SUN_SI
        _, eta_inj, mc_inj_kg = component_masses_to_chirp_mass(m1_kg, m2_kg)
        mc_inj = float(mc_inj_kg / M_SUN_SI)
        assert 20.0 < mc_inj < 40.0
        assert 0.10 < eta_inj < 0.25

    def test_stable_grid_log_posterior_normalization(self) -> None:
        """Verify that grid log-posterior normalization avoids overflow on large log-likelihoods."""
        # Simulated log-likelihood with large baseline offset ~200,000
        offset = 203647.0
        x = np.linspace(-1.0, 1.0, 51)
        y = np.linspace(-1.0, 1.0, 51)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        log_l = offset - 0.5 * (X ** 2 + Y ** 2) / (0.2 ** 2)
        log_prior = -math.log(3.0)

        log_unnorm = log_l + log_prior
        max_log = float(np.max(log_unnorm))
        shifted = np.exp(log_unnorm - max_log)
        integral = float(np.sum(shifted) * dx * dy)
        log_z = max_log + math.log(integral)

        # Normalized density
        p_density = shifted / integral
        total_prob = float(np.sum(p_density) * dx * dy)
        assert math.isclose(total_prob, 1.0, rel_tol=1e-14)
        assert math.isfinite(log_z)

    def test_grid_evidence_quadrature_correctness(self) -> None:
        """Verify grid evidence Riemann sum matches analytical Gaussian integral."""
        # 2D Gaussian with sigma=0.2 over [-1, 1]^2
        sigma = 0.2
        x = np.linspace(-1.0, 1.0, 101)
        y = np.linspace(-1.0, 1.0, 101)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        # f(x, y) = exp(-0.5 * (x^2 + y^2) / sigma^2)
        # Analytical integral over infinite plane = 2 * pi * sigma^2 = 2 * pi * 0.04 ≈ 0.2513274
        f = np.exp(-0.5 * (X ** 2 + Y ** 2) / (sigma ** 2))
        riemann_integral = float(np.sum(f) * dx * dy)
        analytical_integral = 2.0 * math.pi * (sigma ** 2)

        # Truncation error at 5 sigma (1.0 / 0.2 = 5) is < 1e-6
        assert math.isclose(riemann_integral, analytical_integral, rel_tol=1e-4)

    def test_grid_resolution_convergence(self) -> None:
        """Verify that refining grid resolution from 51x51 to 101x101 converges."""
        # Coarse and fine grid over a synthetic Gaussian posterior
        def compute_grid_integral(n: int) -> tuple[float, float, float]:
            x = np.linspace(-0.5, 0.5, n)
            y = np.linspace(-0.5, 0.5, n)
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            X, Y = np.meshgrid(x, y, indexing="ij")
            log_p = -0.5 * (X ** 2 / 0.05**2 + Y ** 2 / 0.02**2)
            max_log = float(np.max(log_p))
            shifted = np.exp(log_p - max_log)
            integral = float(np.sum(shifted) * dx * dy)
            log_z = max_log + math.log(integral)
            p = shifted / integral
            mean_x = float(np.sum(X * p) * dx * dy)
            std_x = math.sqrt(float(np.sum((X - mean_x) ** 2 * p) * dx * dy))
            return log_z, mean_x, std_x

        logz_51, mean_51, std_51 = compute_grid_integral(51)
        logz_101, mean_101, std_101 = compute_grid_integral(101)

        # Relative change in log-evidence and std < 0.1%
        assert abs(logz_101 - logz_51) / abs(logz_101) < 1e-3
        assert abs(std_101 - std_51) / std_51 < 1e-3

    def test_posterior_comparison_calculations_jsd(self) -> None:
        """Verify properties of Jensen-Shannon divergence computation."""
        from scripts.run_wp6_stage2_experiment import compute_jensen_shannon_divergence

        # Identical distributions have JSD = 0
        p = np.array([0.1, 0.4, 0.5])
        assert math.isclose(compute_jensen_shannon_divergence(p, p), 0.0, abs_tol=1e-12)

        # Disjoint distributions have JSD = ln(2)
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])
        assert math.isclose(compute_jensen_shannon_divergence(p1, p2), math.log(2.0), rel_tol=1e-12)

        # Symmetry
        q = np.array([0.3, 0.3, 0.4])
        assert math.isclose(
            compute_jensen_shannon_divergence(p, q),
            compute_jensen_shannon_divergence(q, p),
            rel_tol=1e-12,
        )

    def test_reproducibility_under_fixed_seed(self) -> None:
        """Verify that dynesty sampling with identical seed produces bitwise identical results."""
        target_mean = np.array([0.5, 0.5])
        def toy_logl(theta: np.ndarray) -> float:
            return float(-0.5 * np.sum((theta - target_mean) ** 2) / 0.01)

        def toy_ptform(u: np.ndarray) -> np.ndarray:
            return np.asarray(u, dtype=np.float64)

        adapter1 = NestedSamplerAdapter(
            log_likelihood_fn=toy_logl,
            prior_transform_fn=toy_ptform,
            ndim=2,
            nlive=50,
            dlogz=0.5,
            seed=123,
            param_names=["x", "y"],
        )
        res1 = adapter1.run(bootstrap=0, print_progress=False)

        adapter2 = NestedSamplerAdapter(
            log_likelihood_fn=toy_logl,
            prior_transform_fn=toy_ptform,
            ndim=2,
            nlive=50,
            dlogz=0.5,
            seed=123,
            param_names=["x", "y"],
        )
        res2 = adapter2.run(bootstrap=0, print_progress=False)

        assert np.array_equal(res1.samples, res2.samples)
        assert np.array_equal(res1.log_weights, res2.log_weights)
        assert math.isclose(res1.log_evidence, res2.log_evidence, rel_tol=1e-14)
        assert res1.n_evaluations == res2.n_evaluations

    def test_joint_2d_jensen_shannon_divergence_properties(self) -> None:
        """Verify mathematical properties of joint 2D Jensen-Shannon divergence."""
        from scripts.run_wp6_stage2_experiment import compute_joint_2d_jensen_shannon_divergence

        # Synthetic 2D Gaussian density on grid
        x = np.linspace(-1.0, 1.0, 21)
        y = np.linspace(-1.0, 1.0, 21)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        p2d = np.exp(-0.5 * (X ** 2 + Y ** 2) / (0.3 ** 2))
        p2d /= np.sum(p2d * dx * dy)

        # Generate samples matching the 2D Gaussian
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.0, scale=0.3, size=(5000, 2))
        weights = np.ones(5000) / 5000.0

        jsd_2d = compute_joint_2d_jensen_shannon_divergence(
            samples=samples,
            weights=weights,
            mc_grid_pts=x,
            eta_grid_pts=y,
            p_density_2d=p2d,
            dmc=dx,
            deta=dy,
            nbins_x=10,
            nbins_y=10,
        )
        # Well sampled 2D distribution should have small JSD < 0.05
        assert 0.0 <= jsd_2d < 0.05

    def test_partitioned_quadrature_exterior_bound_math(self) -> None:
        """Verify log-evidence combination and analytical exterior bound formulas."""
        # Main mode lnZ1 = 100.0, secondary fringe lnZ2 = 80.0
        ln_z1 = 100.0
        ln_z2 = 80.0
        diff = ln_z2 - ln_z1  # -20.0
        ratio = math.exp(diff)
        combined_ln_z = ln_z1 + math.log1p(ratio)
        direct_ln_z = math.log(math.exp(ln_z1) + math.exp(ln_z2))
        assert math.isclose(combined_ln_z, direct_ln_z, rel_tol=1e-12)

        # Exterior bound in log-ratio form to prevent float overflow:
        # If L_rest <= exp(lnL_norm - 35) over volume V=3 with density pi=1/3,
        # then integral <= exp(lnL_norm - 35), and ratio to Z_main is exp((lnL_norm - 35) - lnZ_main)
        ln_l_norm = 203647.0
        ln_z_main = 203634.8
        delta_threshold = -35.0
        # Ratio is exp((203647 - 35) - 203634.8) = exp(-22.8)
        log_ratio = (ln_l_norm + delta_threshold) - ln_z_main
        ratio_rest = math.exp(log_ratio)
        delta_logz_bound = math.log1p(ratio_rest)
        assert 0.0 < ratio_rest < 1e-9
        assert 0.0 < delta_logz_bound < 1e-9
