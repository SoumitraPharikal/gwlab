"""Unit tests for Work Package 6 (WP6) Stage 4: Calibration Diagnostics.

Tests verify:
1. Sample and weight validation (NaN, Inf, negatives, sum != 1, empty, shape mismatch).
2. Effective sample size (ESS) computation (equal weights, delta function, Kish formula).
3. Weighted linear percentile rank extraction (monotonicity, boundaries, known ranks).
4. Weighted circular percentile rank extraction on S^1 (branch-cut immunity, anti-mode unrolling).
5. Linear moments and normalized residuals (mean, variance, bias, z-score).
6. Circular directional moments and residuals (C, S, R, bar_phi, circular variance, circular std, wrapped bias).
7. Weighted quantiles and equal-tailed linear credible intervals (containment, widths).
8. Shortest circular credible arcs (wraparound near 0 and 2pi, containment, minimal arc length).
9. Empirical coverage aggregation across runs (SE, expected misses, 2sigma normal-theory bounds).
10. Kolmogorov-Smirnov test and 4-test Bonferroni multiple testing adjustments.
11. SingleRunDiagnosticResult structure, computation, and JSON serialization.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pytest

from gw_inference.diagnostics import (
    R_MIN_CIRCULAR_THRESHOLD,
    SingleRunDiagnosticResult,
    compute_circular_moments_and_residuals,
    compute_coverage_summary,
    compute_effective_sample_size,
    compute_ks_test_uniform,
    compute_linear_credible_interval,
    compute_linear_moments_and_residuals,
    compute_shortest_circular_credible_arc,
    compute_single_run_diagnostics,
    compute_weighted_percentile_rank_circular,
    compute_weighted_percentile_rank_linear,
    compute_weighted_quantile_linear,
    validate_posterior_samples_and_weights,
)


class TestDiagnosticsValidation:
    """Tests for input validation in diagnostics."""

    def test_valid_1d_and_2d_inputs(self):
        """Verify clean 1D and 2D arrays pass validation."""
        s1d = np.array([1.0, 2.0, 3.0])
        w = np.array([0.2, 0.5, 0.3])
        s_val, w_val = validate_posterior_samples_and_weights(s1d, w, expected_ndim=1)
        assert s_val.shape == (3,)
        assert math.isclose(np.sum(w_val), 1.0)

        s2d = np.ones((10, 4))
        w10 = np.ones(10) / 10.0
        s_val2, w_val2 = validate_posterior_samples_and_weights(s2d, w10, expected_ndim=4)
        assert s_val2.shape == (10, 4)
        assert math.isclose(np.sum(w_val2), 1.0)

    def test_reject_empty_arrays(self):
        """Verify empty arrays are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_posterior_samples_and_weights(np.array([]), np.array([]))

    def test_reject_nan_and_inf(self):
        """Verify NaN and Inf are rejected in samples and weights."""
        s = np.array([1.0, np.nan, 3.0])
        w = np.array([0.33, 0.33, 0.34])
        with pytest.raises(ValueError, match="non-finite"):
            validate_posterior_samples_and_weights(s, w)

        s_clean = np.array([1.0, 2.0, 3.0])
        w_inf = np.array([0.5, np.inf, 0.5])
        with pytest.raises(ValueError, match="non-finite"):
            validate_posterior_samples_and_weights(s_clean, w_inf)

    def test_reject_negative_weights(self):
        """Verify negative weights are rejected."""
        s = np.array([1.0, 2.0, 3.0])
        w = np.array([0.6, -0.1, 0.5])
        with pytest.raises(ValueError, match="negative"):
            validate_posterior_samples_and_weights(s, w)

    def test_reject_unnormalized_weights(self):
        """Verify weights that severely do not sum to 1.0 are rejected."""
        s = np.array([1.0, 2.0, 3.0])
        w = np.array([0.5, 0.5, 0.5])  # sum = 1.5
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_posterior_samples_and_weights(s, w)

    def test_reject_dimension_and_length_mismatch(self):
        """Verify mismatched sample/weight counts or dimensions are rejected."""
        s = np.array([1.0, 2.0])
        w = np.array([0.3, 0.3, 0.4])
        with pytest.raises(ValueError, match="Sample count mismatch"):
            validate_posterior_samples_and_weights(s, w)

        s4d = np.ones((5, 3))
        w5 = np.ones(5) / 5.0
        with pytest.raises(ValueError, match="Sample dimension mismatch"):
            validate_posterior_samples_and_weights(s4d, w5, expected_ndim=4)


class TestEffectiveSampleSize:
    """Tests for Kish effective sample size."""

    def test_equal_weights(self):
        """For N equal weights 1/N, ESS must equal N exactly."""
        N = 500
        w = np.ones(N) / float(N)
        assert math.isclose(compute_effective_sample_size(w), float(N), rel_tol=1e-12)

    def test_delta_weight(self):
        """If one sample carries all weight, ESS must be 1.0."""
        w = np.zeros(100)
        w[42] = 1.0
        assert math.isclose(compute_effective_sample_size(w), 1.0, rel_tol=1e-12)


class TestLinearPercentileRanksAndQuantiles:
    """Tests for linear percentile ranks and quantiles."""

    def test_known_uniform_percentiles(self):
        """Test linear percentile ranks on a uniform grid."""
        s = np.linspace(0.0, 10.0, 1001)
        w = np.ones_like(s) / len(s)

        # Value at 5.0 should have rank ~ 0.50
        rank_mid = compute_weighted_percentile_rank_linear(s, w, 5.0)
        assert math.isclose(rank_mid, 0.50, abs_tol=0.01)

        # Value below min is 0.0
        assert compute_weighted_percentile_rank_linear(s, w, -1.0) == 0.0
        # Value above max is 1.0
        assert compute_weighted_percentile_rank_linear(s, w, 11.0) == 1.0

    def test_weighted_quantiles(self):
        """Verify weighted quantile inversion."""
        s = np.array([10.0, 20.0, 30.0, 40.0])
        w = np.array([0.25, 0.25, 0.25, 0.25])

        q50 = compute_weighted_quantile_linear(s, w, 0.50)
        assert 10.0 <= q50 <= 40.0

        # Boundary quantiles
        assert compute_weighted_quantile_linear(s, w, 0.0) == 10.0
        assert compute_weighted_quantile_linear(s, w, 1.0) == 40.0


class TestCircularPhaseDiagnostics:
    """Tests for directional statistics and circular percentile ranks on S^1."""

    def test_circular_moments_and_std_around_zero(self):
        """Test circular moments for angles symmetric around 0 (and 2pi)."""
        # Samples symmetric around 0: [-0.1, +0.1] rad
        s = np.array([0.1, 2.0 * math.pi - 0.1])
        w = np.array([0.5, 0.5])

        res = compute_circular_moments_and_residuals(s, w, true_value=0.0)
        # Circular mean should be 0.0 (or 2pi)
        assert math.isclose(res["circular_mean"], 0.0, abs_tol=1e-10) or math.isclose(
            res["circular_mean"], 2.0 * math.pi, abs_tol=1e-10
        )
        assert res["resultant_r"] > 0.99
        assert math.isclose(res["angular_bias"], 0.0, abs_tol=1e-10)
        assert math.isclose(res["circular_z_score"], 0.0, abs_tol=1e-10)

    def test_circular_percentile_rank_branch_cut_immunity(self):
        """Verify circular percentile rank handles cluster straddling 0/2pi with cut away from cluster."""
        # Concentrated around 0 rad: samples straddling 0 and 2pi
        rng = np.random.default_rng(12345)
        raw = rng.normal(loc=0.0, scale=0.1, size=2000)
        samples = raw % (2.0 * math.pi)
        weights = np.ones(len(samples)) / float(len(samples))

        # With reference cut U = pi, cluster at 0 is rotated to pi
        rank_zero = compute_weighted_percentile_rank_circular(
            samples, weights, 0.0, rotation=math.pi
        )
        assert math.isclose(rank_zero, 0.50, abs_tol=0.05)

        # Injections slightly to the left (-0.1) and right (+0.1)
        rank_left = compute_weighted_percentile_rank_circular(
            samples, weights, 2.0 * math.pi - 0.1, rotation=math.pi
        )
        rank_right = compute_weighted_percentile_rank_circular(
            samples, weights, 0.1, rotation=math.pi
        )
        assert rank_left < rank_zero < rank_right

    def test_circular_percentile_rank_rotation_invariance(self):
        """Verify exact invariance under common rotation of samples, injection, and cut."""
        rng = np.random.default_rng(42)
        raw_samples = rng.uniform(0.0, 2.0 * math.pi, size=100)
        weights = rng.exponential(scale=1.0, size=100)
        weights /= np.sum(weights)
        val = 1.85
        u_cut = 0.75
        two_pi = 2.0 * math.pi

        # Baseline rank
        rank_orig = compute_weighted_percentile_rank_circular(
            raw_samples, weights, val, rotation=u_cut
        )

        # Apply common arbitrary global rotation alpha
        alpha = 2.3456
        samples_rot = (raw_samples + alpha) % two_pi
        val_rot = (val + alpha) % two_pi
        cut_rot = (u_cut + alpha) % two_pi

        rank_rot = compute_weighted_percentile_rank_circular(
            samples_rot, weights, val_rot, rotation=cut_rot
        )
        assert math.isclose(rank_orig, rank_rot, abs_tol=1e-12)

    def test_circular_percentile_rank_seed_reproducibility(self):
        """Verify reproducibility when using explicit PRNG seed."""
        samples = np.array([0.5, 1.2, 2.8, 4.5, 5.9])
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        val = 2.0

        r1, u1, s1 = compute_weighted_percentile_rank_circular(
            samples, weights, val, seed=777, return_details=True
        )
        r2, u2, s2 = compute_weighted_percentile_rank_circular(
            samples, weights, val, seed=777, return_details=True
        )

        assert r1 == r2
        assert u1 == u2
        assert s1 == s2 == 777
        assert 0.0 <= u1 < 2.0 * math.pi

    def test_circular_percentile_rank_broad_and_multimodal_zero_resultant(self):
        """Verify rank computation on multimodal/uniform distributions with resultant R ~ 0."""
        # 1. Bimodal antipodal distribution: 50% at 0.0, 50% at pi.
        # Trigonometric moments: C = 0.5*cos(0) + 0.5*cos(pi) = 0, S = 0 -> R = 0.
        s_bimodal = np.array([0.0, math.pi])
        w_bimodal = np.array([0.5, 0.5])
        circ_mom = compute_circular_moments_and_residuals(s_bimodal, w_bimodal, 0.0)
        assert math.isclose(circ_mom["resultant_r"], 0.0, abs_tol=1e-12)
        assert circ_mom["circular_mean"] is None
        assert circ_mom["circular_std"] is None
        assert circ_mom["angular_bias"] is None
        assert circ_mom["circular_z_score"] is None
        assert circ_mom["status"] == "undefined_low_resultant"

        # With rotation cut U = pi/2:
        # phi=0 rotated to -pi/2 == 3pi/2; phi=pi rotated to pi/2
        # So rotated values are [3pi/2, pi/2] with weights [0.5, 0.5]
        # In rotated coordinates: sample pi is at pi/2; sample 0 is at 3pi/2
        # Target at pi: rotated to pi/2 -> rank should include sample at pi/2 -> 0.50
        rank_pi = compute_weighted_percentile_rank_circular(
            s_bimodal, w_bimodal, math.pi, rotation=math.pi / 2.0
        )
        assert math.isclose(rank_pi, 0.50, abs_tol=1e-12)

        # Target at 0: rotated to 3pi/2 -> rank includes both pi/2 and 3pi/2 -> 1.00
        rank_zero = compute_weighted_percentile_rank_circular(
            s_bimodal, w_bimodal, 0.0, rotation=math.pi / 2.0
        )
        assert math.isclose(rank_zero, 1.00, abs_tol=1e-12)

        # 2. 4-fold symmetric distribution: points at 0, pi/2, pi, 3pi/2 with equal weight
        s_4fold = np.array([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])
        w_4fold = np.ones(4) / 4.0
        mom_4fold = compute_circular_moments_and_residuals(s_4fold, w_4fold, 0.0)
        assert math.isclose(mom_4fold["resultant_r"], 0.0, abs_tol=1e-12)
        assert mom_4fold["circular_mean"] is None
        assert mom_4fold["circular_std"] is None
        assert mom_4fold["angular_bias"] is None
        assert mom_4fold["circular_z_score"] is None
        assert mom_4fold["status"] == "undefined_low_resultant"

        # Ranks must evaluate cleanly to valid floats in [0, 1] without NaN or exception
        for target in s_4fold:
            rank_val = compute_weighted_percentile_rank_circular(
                s_4fold, w_4fold, target, rotation=0.1
            )
            assert 0.0 <= rank_val <= 1.0
            assert math.isfinite(rank_val)

    def test_circular_moments_exact_antipodal_zero_resultant(self):
        """Verify directional statistics for exactly antipodal equal-weight samples (R = 0).

        Opposing angles cancel identically: C = 0, S = 0 -> R = 0.
        Verifies:
        - Resultant R is 0.0.
        - Circular mean is None (not a spurious atan2(0,0)=0.0).
        - First-trigonometric-moment dispersion is None (not inf).
        - Angular bias and circular z-score are None (not 0.0 from division by inf).
        - Status is 'undefined_low_resultant' with descriptive status message.
        """
        # Two antipodal points separated by pi
        samples = np.array([0.25 * math.pi, 1.25 * math.pi])
        weights = np.array([0.5, 0.5])
        res = compute_circular_moments_and_residuals(samples, weights, true_value=0.0)

        assert math.isclose(res["c_moment"], 0.0, abs_tol=1e-15)
        assert math.isclose(res["s_moment"], 0.0, abs_tol=1e-15)
        assert res["resultant_r"] < 1e-15
        assert math.isclose(res["circular_variance"], 1.0, abs_tol=1e-15)
        assert res["circular_mean"] is None
        assert res["circular_std"] is None
        assert res["angular_bias"] is None
        assert res["circular_z_score"] is None
        assert res["status"] == "undefined_low_resultant"
        assert "below numerical threshold" in res["status_message"]

    def test_circular_moments_near_zero_resultant_below_threshold(self):
        """Verify directional statistics when resultant length R is below threshold (0 < R < R_min).

        Verifies that slightly broken symmetry yielding tiny positive R < 1e-6
        correctly triggers the indeterminate guard rather than calculating unstable
        directional estimates. Also tests custom r_min_threshold and invalid thresholds.
        """
        # Perturb antipodal points by 1e-8 rad so R ~ 5e-9 < 1e-6
        samples = np.array([0.0, math.pi + 1e-8])
        weights = np.array([0.5, 0.5])
        res = compute_circular_moments_and_residuals(samples, weights, true_value=0.5)

        assert 0.0 < res["resultant_r"] < R_MIN_CIRCULAR_THRESHOLD
        assert res["circular_mean"] is None
        assert res["circular_std"] is None
        assert res["angular_bias"] is None
        assert res["circular_z_score"] is None
        assert res["status"] == "undefined_low_resultant"

        # With relaxed threshold r_min = 1e-9, the same data is well-defined
        res_relaxed = compute_circular_moments_and_residuals(
            samples, weights, true_value=0.5, r_min_threshold=1e-9
        )
        assert res_relaxed["status"] == "well_defined"
        assert res_relaxed["circular_mean"] is not None
        assert res_relaxed["circular_z_score"] is not None

        # Rejection of invalid thresholds
        with pytest.raises(ValueError, match="r_min_threshold must be non-negative and finite"):
            compute_circular_moments_and_residuals(samples, weights, 0.0, r_min_threshold=-1.0)
        with pytest.raises(ValueError, match="r_min_threshold must be non-negative and finite"):
            compute_circular_moments_and_residuals(samples, weights, 0.0, r_min_threshold=float("nan"))

    def test_circular_moments_concentrated_posterior(self):
        """Verify directional statistics for a concentrated unimodal posterior (R -> 1).

        Verifies:
        - Circular mean accurately recovers the center of mass.
        - First-trigonometric-moment dispersion sigma_circ matches sample standard deviation.
        - Angular bias and circular z-score are finite, accurate, and status is 'well_defined'.
        """
        rng = np.random.default_rng(2026)
        true_phi = 1.25  # rad
        sigma_true = 0.05  # rad
        N = 5000
        raw_samples = rng.normal(loc=true_phi, scale=sigma_true, size=N)
        samples = raw_samples % (2.0 * math.pi)
        weights = np.ones(N) / float(N)

        res = compute_circular_moments_and_residuals(samples, weights, true_value=true_phi)

        assert res["status"] == "well_defined"
        assert res["resultant_r"] > 0.99
        # Circular mean close to true_phi
        assert math.isclose(res["circular_mean"], true_phi, abs_tol=0.01)
        # For narrow distributions, sigma_circ = sqrt(-2 ln R) ~ sigma_sample ~ 0.05
        assert math.isclose(res["circular_std"], sigma_true, abs_tol=0.005)
        # Bias is small
        assert abs(res["angular_bias"]) < 0.01
        # z-score is finite and reasonable
        assert abs(res["circular_z_score"]) < 1.0

    def test_circular_moments_json_serialization_null(self):
        """Verify that undefined circular diagnostics serialize to standard JSON nulls.

        Verifies:
        - json.dumps produces valid JSON with "null" for undefined fields.
        - No non-standard tokens ('Infinity', '-Infinity', 'NaN') appear in JSON.
        - json.loads successfully round-trips all fields and restores None.
        """
        # Exactly antipodal distribution
        s_antipodal = np.array([0.0, math.pi])
        w = np.array([0.5, 0.5])
        res = compute_circular_moments_and_residuals(s_antipodal, w, true_value=0.0)

        # Serialize to standard JSON (allow_nan=False ensures strict JSON conformance)
        json_str = json.dumps(res, allow_nan=False, indent=2)

        # Assert JSON representation of undefined fields is null
        assert '"circular_mean": null' in json_str
        assert '"circular_std": null' in json_str
        assert '"angular_bias": null' in json_str
        assert '"circular_z_score": null' in json_str
        assert '"status": "undefined_low_resultant"' in json_str
        assert "Infinity" not in json_str
        assert "NaN" not in json_str

        # Round-trip deserialization
        loaded = json.loads(json_str)
        assert loaded["circular_mean"] is None
        assert loaded["circular_std"] is None
        assert loaded["angular_bias"] is None
        assert loaded["circular_z_score"] is None
        assert loaded["resultant_r"] < 1e-15
        assert loaded["status"] == "undefined_low_resultant"

    def test_circular_percentile_rank_tie_conventions(self):
        """Test inclusive, strict, and midpoint weighted tie conventions."""
        # Samples with exact ties at phi = 1.0
        samples = np.array([1.0, 1.0, 2.5, 4.0])
        weights = np.array([0.2, 0.3, 0.3, 0.2])
        # Use rotation cut U = 0.0 so coordinates are unchanged
        val = 1.0

        rank_inc = compute_weighted_percentile_rank_circular(
            samples, weights, val, rotation=0.0, tie_convention="inclusive"
        )
        rank_strict = compute_weighted_percentile_rank_circular(
            samples, weights, val, rotation=0.0, tie_convention="strict"
        )
        rank_mid = compute_weighted_percentile_rank_circular(
            samples, weights, val, rotation=0.0, tie_convention="midpoint"
        )

        # Inclusive includes tied samples (0.2 + 0.3 = 0.5)
        assert math.isclose(rank_inc, 0.50, abs_tol=1e-12)
        # Strict excludes tied samples (0.0)
        assert math.isclose(rank_strict, 0.00, abs_tol=1e-12)
        # Midpoint adds half the tied mass (0.0 + 0.5 * 0.5 = 0.25)
        assert math.isclose(rank_mid, 0.25, abs_tol=1e-12)

    def test_circular_percentile_rank_invalid_inputs(self):
        """Test rejection of non-finite angles, rotations, and invalid tie conventions."""
        s = np.array([0.1, 0.5])
        w = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="Reference value must be finite"):
            compute_weighted_percentile_rank_circular(s, w, float("nan"))

        with pytest.raises(ValueError, match="Reference value must be finite"):
            compute_weighted_percentile_rank_circular(s, w, float("inf"))

        with pytest.raises(ValueError, match="Rotation must be finite"):
            compute_weighted_percentile_rank_circular(s, w, 0.0, rotation=float("nan"))

        with pytest.raises(ValueError, match="Invalid tie_convention"):
            compute_weighted_percentile_rank_circular(s, w, 0.0, tie_convention="unsupported")

    def test_shortest_circular_credible_arc_wrapping(self):
        """Test shortest circular credible arc wrapping across the 0/2pi boundary."""
        # Angle cluster spanning [350 deg, 10 deg] == [6.108, 0.175] rad
        angles_deg = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
        s = np.deg2rad(angles_deg)
        w = np.ones(5) / 5.0

        # 80% credible interval should enclose 4 out of 5 points
        arc = compute_shortest_circular_credible_arc(s, w, cred_level=0.80, true_value=0.0)
        assert arc["is_contained"] == 1
        assert arc["arc_length"] <= np.deg2rad(25.0)
        assert arc["enclosed_weight"] >= 0.80 - 1e-12

        # An angle on the opposite side (180 deg) must NOT be contained
        arc_opp = compute_shortest_circular_credible_arc(
            s, w, cred_level=0.80, true_value=math.pi
        )
        assert arc_opp["is_contained"] == 0

    def test_shortest_circular_credible_arc_multimodal_distinction(self):
        """Demonstrate that shortest connected arc differs from disconnected HPD region on multimodal data."""
        # Bimodal distribution: 50% cluster near 0, 50% cluster near pi
        cluster_0 = np.array([0.01, 0.02, 0.03])
        cluster_pi = np.array([math.pi + 0.01, math.pi + 0.02, math.pi + 0.03])
        s = np.concatenate([cluster_0, cluster_pi])
        w = np.ones(6) / 6.0

        # Request 90% mass: a single connected arc must span across the gap between 0 and pi (length >= pi)
        arc = compute_shortest_circular_credible_arc(s, w, cred_level=0.90)
        assert arc["enclosed_weight"] >= 0.90 - 1e-12
        assert arc["arc_length"] >= math.pi - 0.1
        # In contrast, an arbitrary HPD region could consist of two disjoint small intervals of length ~0.02 each


class TestCredibleIntervalContainmentAndCoverage:
    """Tests for linear credible interval containment and multi-run coverage aggregation."""

    def test_linear_credible_interval_containment(self):
        """Test linear credible interval calculation and containment."""
        s = np.linspace(10.0, 20.0, 100)
        w = np.ones(100) / 100.0

        # 90% CI for uniform on [10, 20] is [10.5, 19.5]
        ci90 = compute_linear_credible_interval(s, w, cred_level=0.90, true_value=15.0)
        assert ci90["is_contained"] == 1
        assert math.isclose(ci90["lower"], 10.5, abs_tol=0.2)
        assert math.isclose(ci90["upper"], 19.5, abs_tol=0.2)

        # Value outside interval
        ci_out = compute_linear_credible_interval(s, w, cred_level=0.90, true_value=25.0)
        assert ci_out["is_contained"] == 0

    def test_coverage_summary_aggregation(self):
        """Test coverage aggregation math and 2sigma envelope."""
        # 35 runs with 32 hits (3 misses) at nominal 90%
        flags = [1] * 32 + [0] * 3
        summary = compute_coverage_summary(flags, nominal_level=0.90)

        assert summary["n_runs"] == 35
        assert summary["n_hits"] == 32
        assert summary["n_misses"] == 3
        assert math.isclose(summary["empirical_coverage"], 32.0 / 35.0, abs_tol=1e-12)
        assert math.isclose(summary["binomial_se"], math.sqrt(0.09 / 35.0), abs_tol=1e-12)
        assert summary["is_within_2sigma"] is True


class TestKolmogorovSmirnovAndBonferroni:
    """Tests for Kolmogorov-Smirnov test and Bonferroni multiple testing adjustments."""

    def test_ks_test_on_uniform_ranks(self):
        """Verify standard uniform ranks pass KS and Bonferroni tests."""
        # Deterministic perfectly uniform grid
        ranks = np.linspace(0.01, 0.99, 35)
        res = compute_ks_test_uniform(ranks, alpha=0.05, n_tests=4)

        assert res["n_samples"] == 35
        assert res["passes_unadjusted"] is True
        assert res["passes_bonferroni"] is True
        assert math.isclose(res["d_crit_5pct"], 1.358 / math.sqrt(35), rel_tol=1e-4)
        assert math.isclose(res["alpha_bonferroni"], 0.05 / 4, rel_tol=1e-12)

    def test_ks_test_on_heavily_biased_ranks(self):
        """Verify skewed ranks fail KS test."""
        # Heavily clumped near 0
        ranks = np.full(35, 0.02)
        res = compute_ks_test_uniform(ranks, alpha=0.05, n_tests=4)
        assert res["passes_unadjusted"] is False
        assert res["passes_bonferroni"] is False


class TestSingleRunDiagnosticsAndSerialization:
    """Tests for SingleRunDiagnosticResult structure and JSON export."""

    def test_compute_single_run_diagnostics_and_json(self):
        """Test complete 4D diagnostic extraction and JSON round-trip."""
        rng = np.random.default_rng(999)
        N = 1000

        # Generate toy 4D samples around canonical truth
        mc_true, eta_true, tc_true, phic_true = 28.0956, 0.2471, 2.000, 0.000
        mc_s = rng.normal(mc_true, 0.01, size=N)
        eta_s = rng.normal(eta_true, 0.001, size=N)
        tc_s = rng.normal(tc_true, 0.002, size=N)
        phic_s = (rng.normal(phic_true, 0.1, size=N)) % (2.0 * math.pi)

        samples = np.column_stack([mc_s, eta_s, tc_s, phic_s])
        weights = np.ones(N) / float(N)

        diag = compute_single_run_diagnostics(
            samples=samples,
            weights=weights,
            injected_params=[mc_true, eta_true, tc_true, phic_true],
            param_names=("chirp_mass", "symmetric_mass_ratio", "tc", "phic"),
            run_id="test_run_01",
            log_evidence=203628.126,
            log_evidence_err=0.152,
            n_evaluations=335000,
            runtime_seconds=12.5,
            timestamp="2026-11-17T00:00:00Z",
        )

        assert diag.run_id == "test_run_01"
        assert math.isclose(diag.n_eff, float(N), rel_tol=1e-10)
        assert "chirp_mass" in diag.recovered_parameters
        assert "phic" in diag.recovered_parameters
        assert diag.recovered_parameters["phic"]["type"] == "circular"
        assert diag.recovered_parameters["chirp_mass"]["type"] == "linear"

        # Verify JSON serializability
        d_dict = diag.to_dict()
        json_str = json.dumps(d_dict, indent=2)
        assert len(json_str) > 0
        loaded = json.loads(json_str)
        assert loaded["run_id"] == "test_run_01"
        assert loaded["log_evidence"] == 203628.126
