"""Statistical calibration and parameter recovery diagnostics for Bayesian inference.

This module implements the diagnostic foundation for WP6 Stage 4:
- Weighted posterior percentile ranks (simulation-based calibration) for linear and circular coordinates.
- Parameter estimation bias, variance, and normalized residuals (z-scores).
- Directional circular statistics on S^1 (first trigonometric moments, circular mean,
  resultant vector length, circular variance, circular standard deviation, wrapped bias).
- Equal-tailed linear credible intervals and shortest circular credible arcs.
- Empirical credible interval coverage aggregation with normal-theory uncertainty envelopes.
- Kolmogorov-Smirnov uniformity tests with Bonferroni multiple-comparison corrections.
- Effective sample size (ESS) and input validation for Bayesian posterior samples.

All calculations strictly enforce the mathematical conventions established in
Stage 0-3 and the Stage 4 Experimental Design Freeze.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats


def validate_posterior_samples_and_weights(
    samples: np.ndarray,
    weights: np.ndarray,
    expected_ndim: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate sample array and normalized weights.

    Parameters
    ----------
    samples : np.ndarray
        Array of parameter samples of shape (N,) or (N, D).
    weights : np.ndarray
        Array of statistical weights of shape (N,).
    expected_ndim : int, optional
        Expected dimensionality D for 2D sample array.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Validated float64 samples and strictly normalized float64 weights.

    Raises
    ------
    ValueError
        If inputs contain NaNs, Infs, mismatched shapes, non-positive weights,
        or if weights fail to sum to 1.0 within tolerance.
    """
    s = np.asarray(samples, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if s.size == 0 or w.size == 0:
        raise ValueError("Samples and weights arrays cannot be empty.")

    if not np.all(np.isfinite(s)):
        raise ValueError("Samples array contains non-finite values (NaN or Inf).")

    if not np.all(np.isfinite(w)):
        raise ValueError("Weights array contains non-finite values (NaN or Inf).")

    if w.ndim != 1:
        raise ValueError(f"Weights array must be 1D, got ndim={w.ndim}.")

    n_samples = len(w)
    if len(s) != n_samples:
        raise ValueError(
            f"Sample count mismatch: len(samples)={len(s)} != len(weights)={n_samples}."
        )

    if expected_ndim is not None:
        if s.ndim == 1 and expected_ndim != 1:
            raise ValueError(f"Expected {expected_ndim}D samples, got 1D array.")
        if s.ndim == 2 and s.shape[1] != expected_ndim:
            raise ValueError(
                f"Sample dimension mismatch: expected {expected_ndim}, got {s.shape[1]}."
            )

    if np.any(w < 0.0):
        raise ValueError("Weights array contains negative values.")

    sum_w = float(np.sum(w))
    if not (math.isfinite(sum_w) and sum_w > 0.0):
        raise ValueError(f"Weight sum is non-finite or non-positive ({sum_w}).")

    if not math.isclose(sum_w, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        # Renormalize if slightly drifted, or raise if severely unnormalized
        if abs(sum_w - 1.0) > 1e-3:
            raise ValueError(f"Weights do not sum to 1.0 (sum={sum_w}).")
        w = w / sum_w

    return s, w


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """Compute Kish effective sample size (ESS) from normalized weights.

    N_eff = 1.0 / sum(w_i^2)

    Parameters
    ----------
    weights : np.ndarray
        1D array of normalized posterior weights summing to 1.0.

    Returns
    -------
    float
        Effective sample size N_eff >= 1.0.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("Weights must be a non-empty 1D array.")
    sum_sq = float(np.sum(w**2))
    if sum_sq <= 0.0 or not math.isfinite(sum_sq):
        raise ValueError("Sum of squared weights must be positive and finite.")
    return float(1.0 / sum_sq)


R_MIN_CIRCULAR_THRESHOLD: float = 1e-6
"""Minimum mean resultant vector length threshold for directional circular statistics.

Below this threshold, the mean direction on S^1 is mathematically indeterminate,
and first-trigonometric-moment dispersion diagnostics diverge.
"""


def compute_weighted_percentile_rank_linear(
    samples: np.ndarray,
    weights: np.ndarray,
    value: float,
) -> float:
    """Compute empirical cumulative posterior percentile rank for a linear parameter.

    p(theta_inj) = sum_{i: theta_i <= theta_inj} w_i

    Parameters
    ----------
    samples : np.ndarray
        1D array of parameter samples.
    weights : np.ndarray
        1D array of normalized weights summing to 1.0.
    value : float
        Injected or reference parameter value.

    Returns
    -------
    float
        Percentile rank p in [0.0, 1.0].
    """
    s, w = validate_posterior_samples_and_weights(samples, weights, expected_ndim=1)
    if not math.isfinite(value):
        raise ValueError(f"Reference value must be finite, got {value}.")

    rank = float(np.sum(w[s <= float(value)]))
    return float(np.clip(rank, 0.0, 1.0))


def compute_weighted_percentile_rank_circular(
    samples: np.ndarray,
    weights: np.ndarray,
    value: float,
    rotation: Optional[float] = None,
    seed: Optional[int] = None,
    tie_convention: str = "inclusive",
    return_details: bool = False,
) -> Union[float, Tuple[float, float, Optional[int]]]:
    """Compute cumulative posterior percentile rank for a circular phase parameter on S^1.

    Implements a random-cut circular percentile rank by rotating all posterior
    samples and the reference/injected value by a reference cut U drawn uniformly
    from [0, 2pi) independently of the posterior and injected value:
        U ~ Uniform(0, 2pi)
        phi_i' = (phi_i - U) mod 2pi
        phi_inj' = (phi_inj - U) mod 2pi

    The weighted empirical CDF is then computed at phi_inj'. The cut U is strictly
    independent of the posterior mean, mode, or resultant vector, ensuring distributional
    invariance under SO(2) global rotations of the underlying circular probability distribution,
    branch-cut immunity, and robust behavior even for broad, multimodal, or nearly uniform
    circular posteriors where the resultant vector length R approaches 0.

    Parameters
    ----------
    samples : np.ndarray
        1D array of phase samples in radians.
    weights : np.ndarray
        1D array of normalized weights summing to 1.0.
    value : float
        Injected or reference phase angle in radians.
    rotation : float, optional
        Explicit rotation cut U in radians in [0, 2pi). If provided, `seed` is ignored.
        Must be finite. Wrapped to [0, 2pi).
    seed : int, optional
        PRNG seed used to draw U ~ Uniform(0, 2pi) via numpy.random.default_rng(seed).
        If both `rotation` and `seed` are None, defaults to seed=42 for deterministic
        reproducibility.
    tie_convention : {"inclusive", "strict", "midpoint"}, default="inclusive"
        Convention for evaluating samples tied with the reference value phi_inj':
        - "inclusive": sum_{i: phi_i' <= phi_inj'} w_i (standard right-continuous empirical CDF)
        - "strict":    sum_{i: phi_i' < phi_inj'} w_i (left-continuous empirical CDF)
        - "midpoint":  sum_{i: phi_i' < phi_inj'} w_i + 0.5 * sum_{i: phi_i' == phi_inj'} w_i
    return_details : bool, default=False
        If True, returns tuple (percentile_rank, rotation_u, recorded_seed).
        If False, returns percentile_rank as a float.

    Returns
    -------
    float or tuple of (float, float, int or None)
        If return_details is False:
            Circular percentile rank p in [0.0, 1.0].
        If return_details is True:
            Tuple of (percentile_rank, rotation_u, recorded_seed).

    Raises
    ------
    ValueError
        If `value` or `rotation` is non-finite, or if `tie_convention` is not one of
        {"inclusive", "strict", "midpoint"}.
    """
    s, w = validate_posterior_samples_and_weights(samples, weights, expected_ndim=1)
    if not math.isfinite(value):
        raise ValueError(f"Reference value must be finite, got {value}.")

    valid_tie_conventions = ("inclusive", "strict", "midpoint")
    if tie_convention not in valid_tie_conventions:
        raise ValueError(
            f"Invalid tie_convention: '{tie_convention}'. Must be one of {valid_tie_conventions}."
        )

    two_pi = 2.0 * math.pi

    # Resolve rotation cut U independently of posterior moments
    if rotation is not None:
        if not math.isfinite(rotation):
            raise ValueError(f"Rotation must be finite, got {rotation}.")
        u_cut = float(rotation) % two_pi
        recorded_seed: Optional[int] = None
    elif seed is not None:
        rng = np.random.default_rng(seed)
        u_cut = float(rng.uniform(0.0, two_pi))
        recorded_seed = int(seed)
    else:
        recorded_seed = 42
        rng = np.random.default_rng(recorded_seed)
        u_cut = float(rng.uniform(0.0, two_pi))

    # Wrap angles to [0, 2pi) and apply rotation cut U
    s_rot = (s - u_cut) % two_pi
    val_rot = (float(value) - u_cut) % two_pi

    # Evaluate weighted empirical CDF with explicit tie convention
    tol = 1e-12
    is_less = s_rot < val_rot - tol
    is_equal = np.abs(s_rot - val_rot) <= tol

    if tie_convention == "inclusive":
        rank = float(np.sum(w[is_less | is_equal]))
    elif tie_convention == "strict":
        rank = float(np.sum(w[is_less]))
    elif tie_convention == "midpoint":
        rank = float(np.sum(w[is_less]) + 0.5 * np.sum(w[is_equal]))

    rank_clipped = float(np.clip(rank, 0.0, 1.0))

    if return_details:
        return rank_clipped, u_cut, recorded_seed
    return rank_clipped


def compute_linear_moments_and_residuals(
    samples: np.ndarray,
    weights: np.ndarray,
    true_value: float,
) -> Dict[str, float]:
    """Compute weighted posterior mean, variance, std, bias, and z-score for a linear parameter.

    Parameters
    ----------
    samples : np.ndarray
        1D array of parameter samples.
    weights : np.ndarray
        1D array of normalized weights.
    true_value : float
        True injected parameter value.

    Returns
    -------
    dict
        Dictionary containing:
        - "mean": Weighted posterior mean.
        - "variance": Weighted posterior variance.
        - "std": Weighted posterior standard deviation.
        - "bias": Difference (mean - true_value).
        - "abs_error": Absolute difference |mean - true_value|.
        - "z_score": Normalized residual (mean - true_value) / std.
    """
    s, w = validate_posterior_samples_and_weights(samples, weights, expected_ndim=1)
    if not math.isfinite(true_value):
        raise ValueError(f"true_value must be finite, got {true_value}.")

    mean = float(np.sum(w * s))
    var = float(np.sum(w * (s - mean) ** 2))
    std = math.sqrt(max(0.0, var))

    bias = mean - float(true_value)
    abs_err = abs(bias)
    z_score = bias / std if std > 0.0 else 0.0

    return {
        "mean": mean,
        "variance": var,
        "std": std,
        "bias": bias,
        "abs_error": abs_err,
        "z_score": z_score,
    }


def compute_circular_moments_and_residuals(
    samples: np.ndarray,
    weights: np.ndarray,
    true_value: float,
    r_min_threshold: float = R_MIN_CIRCULAR_THRESHOLD,
) -> Dict[str, Any]:
    """Compute directional circular moments, angular bias, and circular z-score on S^1.

    Computes the sample first trigonometric moments:
        C = sum_i w_i cos(phi_i)
        S = sum_i w_i sin(phi_i)
        R = sqrt(C^2 + S^2) in [0, 1]
        V = 1 - R (circular variance)

    When the mean resultant vector length R >= r_min_threshold, the directional
    circular mean is well-defined:
        bar_phi = atan2(S, C) mod 2pi
        sigma_circ = sqrt(-2 ln R)
        angular_bias = atan2(sin(bar_phi - phi_true), cos(bar_phi - phi_true)) in [-pi, pi)
        z_circ = angular_bias / sigma_circ

    Statistical interpretation of dispersion diagnostic:
    sigma_circ = sqrt(-2 ln R) is a first-trigonometric-moment dispersion diagnostic
    derived for unimodal circular distributions (e.g., von Mises or wrapped normal).
    It is NOT a general uncertainty measure for multimodal circular posteriors.
    For multimodal or symmetric distributions (e.g. equal antipodal lobes at phi and
    phi + pi), opposing vector components cancel out, yielding R ~ 0 and causing
    sigma_circ -> inf, even if each individual mode is sharply concentrated.
    For multimodal posteriors, the shortest circular credible arc or disconnected HPD
    regions, along with the random-cut circular percentile rank, provide reliable
    coverage and calibration diagnostics.

    Low-resultant regime (R < r_min_threshold):
    When R is below the numerical threshold `r_min_threshold`, the directional mean
    is mathematically indeterminate. To prevent presenting an arbitrary angle from
    atan2(0, 0) as a well-defined estimate, and to prevent division by an infinite
    circular standard deviation from producing a misleading z_circ = 0, the following
    fields are explicitly returned as None (serializing to JSON `null`):
        - circular_mean: None
        - circular_std: None
        - angular_bias: None
        - circular_z_score: None
    with status="undefined_low_resultant" and a descriptive status message.

    Parameters
    ----------
    samples : np.ndarray
        1D array of phase samples in radians.
    weights : np.ndarray
        1D array of normalized weights.
    true_value : float
        True injected phase angle in radians.
    r_min_threshold : float, default=1e-6
        Numerical threshold below which the resultant vector length R is considered
        indeterminate. Must be non-negative and finite.

    Returns
    -------
    dict
        Dictionary containing:
        - "c_moment": First trigonometric cosine moment C.
        - "s_moment": First trigonometric sine moment S.
        - "resultant_r": Mean resultant vector length R in [0, 1].
        - "circular_mean": Circular mean angle in radians in [0, 2pi), or None if R < r_min.
        - "circular_variance": Circular variance V = 1 - R.
        - "circular_std": Circular dispersion diagnostic sqrt(-2 ln R), or None if R < r_min.
        - "angular_bias": Wrapped angular difference in [-pi, pi), or None if R < r_min.
        - "circular_z_score": Normalized circular residual, or None if R < r_min.
        - "status": "well_defined" or "undefined_low_resultant".
        - "status_message": Human-readable explanation of status.
    """
    s, w = validate_posterior_samples_and_weights(samples, weights, expected_ndim=1)
    if not math.isfinite(true_value):
        raise ValueError(f"true_value must be finite, got {true_value}.")
    if not math.isfinite(r_min_threshold) or r_min_threshold < 0.0:
        raise ValueError(f"r_min_threshold must be non-negative and finite, got {r_min_threshold}.")

    s_wrapped = s % (2.0 * math.pi)
    true_wrapped = float(true_value) % (2.0 * math.pi)

    c_mom = float(np.sum(w * np.cos(s_wrapped)))
    s_mom = float(np.sum(w * np.sin(s_wrapped)))
    r_len = float(math.sqrt(c_mom**2 + s_mom**2))
    r_len_bounded = float(min(1.0, max(0.0, r_len)))
    circ_var = float(max(0.0, 1.0 - r_len_bounded))

    if r_len_bounded < r_min_threshold:
        return {
            "c_moment": c_mom,
            "s_moment": s_mom,
            "resultant_r": r_len_bounded,
            "circular_mean": None,
            "circular_variance": circ_var,
            "circular_std": None,
            "angular_bias": None,
            "circular_z_score": None,
            "status": "undefined_low_resultant",
            "status_message": (
                f"Resultant vector length R={r_len_bounded:.6e} is below numerical threshold "
                f"{r_min_threshold:.1e}; directional circular mean, circular dispersion, "
                "angular bias, and normalized circular residual are undefined."
            ),
        }

    circ_mean = float(math.atan2(s_mom, c_mom) % (2.0 * math.pi))

    # First-trigonometric-moment dispersion diagnostic
    if r_len_bounded >= 1.0:
        circ_std = 0.0
    else:
        circ_std = float(math.sqrt(-2.0 * math.log(r_len_bounded)))

    # Wrapped angular difference in [-pi, pi)
    d_phi = float(
        math.atan2(math.sin(circ_mean - true_wrapped), math.cos(circ_mean - true_wrapped))
    )
    z_circ = float(d_phi / circ_std) if circ_std > 0.0 else 0.0

    return {
        "c_moment": c_mom,
        "s_moment": s_mom,
        "resultant_r": r_len_bounded,
        "circular_mean": circ_mean,
        "circular_variance": circ_var,
        "circular_std": circ_std,
        "angular_bias": d_phi,
        "circular_z_score": z_circ,
        "status": "well_defined",
        "status_message": (
            f"Resultant vector length R={r_len_bounded:.6e} exceeds threshold {r_min_threshold:.1e}; "
            "circular directional statistics are well-defined."
        ),
    }


def compute_weighted_quantile_linear(
    samples: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """Compute weighted quantile for a 1D linear distribution.

    Parameters
    ----------
    samples : np.ndarray
        1D array of parameter samples.
    weights : np.ndarray
        1D array of normalized weights.
    q : float
        Quantile level in [0.0, 1.0].

    Returns
    -------
    float
        Quantile value.
    """
    s, w = validate_posterior_samples_and_weights(samples, weights, expected_ndim=1)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile q must be in [0.0, 1.0], got {q}.")

    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]

    cum_w = np.cumsum(w_sorted)
    # Ensure cumulative weights strictly start and end correctly
    cum_w = cum_w / cum_w[-1]

    if q <= cum_w[0]:
        return float(s_sorted[0])
    if q >= cum_w[-1]:
        return float(s_sorted[-1])

    return float(np.interp(q, cum_w, s_sorted))


def compute_linear_credible_interval(
    samples: np.ndarray,
    weights: np.ndarray,
    cred_level: float,
    true_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute equal-tailed credible interval for a linear parameter.

    Parameters
    ----------
    samples : np.ndarray
        1D array of parameter samples.
    weights : np.ndarray
        1D array of normalized weights.
    cred_level : float
        Nominal credible level c in (0.0, 1.0), e.g. 0.50, 0.90, 0.95.
    true_value : float, optional
        If provided, evaluates containment indicator I_c = 1 if true_value in [low, high] else 0.

    Returns
    -------
    dict
        Dictionary containing:
        - "cred_level": Nominal level c.
        - "lower": Lower bound theta_low.
        - "upper": Upper bound theta_high.
        - "width": Interval width (upper - lower).
        - "is_contained": Containment indicator (int 0 or 1), or None if true_value is omitted.
    """
    if not (0.0 < cred_level < 1.0):
        raise ValueError(f"cred_level must be in (0.0, 1.0), got {cred_level}.")

    alpha = 1.0 - cred_level
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0

    lower = compute_weighted_quantile_linear(samples, weights, q_low)
    upper = compute_weighted_quantile_linear(samples, weights, q_high)

    is_contained = None
    if true_value is not None:
        if not math.isfinite(true_value):
            raise ValueError(f"true_value must be finite, got {true_value}.")
        is_contained = 1 if (lower <= float(true_value) <= upper) else 0

    return {
        "cred_level": float(cred_level),
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "is_contained": is_contained,
    }


def compute_shortest_circular_credible_arc(
    samples: np.ndarray,
    weights: np.ndarray,
    cred_level: float,
    true_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the shortest connected circular arc on S^1 enclosing probability mass >= cred_level.

    Finds the single connected interval [arc_start, arc_end] on S^1 of minimum angular
    length Delta phi = (arc_end - arc_start) mod 2pi such that the enclosed posterior
    weight satisfies sum_{phi_i on arc} w_i >= cred_level.

    Important distinction:
    This function computes the single shortest connected arc on the circle. It is NOT
    in general an arbitrary or disconnected highest-posterior-density (HPD) credible region.
    For multimodal posteriors (e.g., mixtures of antipodal clusters), a true HPD region
    consists of multiple disjoint intervals, whereas this function strictly enforces a single
    connected circular interval.

    Parameters
    ----------
    samples : np.ndarray
        1D array of phase samples in radians.
    weights : np.ndarray
        1D array of normalized weights.
    cred_level : float
        Nominal credible level c in (0.0, 1.0), e.g. 0.50, 0.90, 0.95.
    true_value : float, optional
        If provided, evaluates circular containment indicator I_c = 1 if true_value is on arc else 0.

    Returns
    -------
    dict
        Dictionary containing:
        - "cred_level": Nominal level c.
        - "arc_start": Starting angle in radians in [0, 2pi).
        - "arc_end": Ending angle in radians in [0, 2pi).
        - "arc_length": Arc length Delta phi in radians in [0, 2pi].
        - "enclosed_weight": Total posterior weight mass contained on the returned arc.
        - "is_contained": Containment indicator (int 0 or 1), or None if true_value is omitted.
    """
    s, w = validate_posterior_samples_and_weights(samples, weights, expected_ndim=1)
    if not (0.0 < cred_level < 1.0):
        raise ValueError(f"cred_level must be in (0.0, 1.0), got {cred_level}.")

    # Wrap angles to [0, 2pi)
    two_pi = 2.0 * math.pi
    s_wrapped = s % two_pi

    # Filter out exact zero weights to streamline window search
    nonzero_mask = w > 0.0
    if not np.any(nonzero_mask):
        raise ValueError("All sample weights are zero.")
    s_nz = s_wrapped[nonzero_mask]
    w_nz = w[nonzero_mask] / float(np.sum(w[nonzero_mask]))

    order = np.argsort(s_nz)
    s_sorted = s_nz[order]
    w_sorted = w_nz[order]
    n = len(s_sorted)

    # Extend array periodically over circle
    s_ext = np.concatenate([s_sorted, s_sorted + two_pi])
    w_ext = np.concatenate([w_sorted, w_sorted])
    cum_w_ext = np.cumsum(w_ext)

    min_arc_len = float("inf")
    best_i = 0
    best_j = 0

    j = 0
    for i in range(n):
        target_w = (cum_w_ext[i - 1] if i > 0 else 0.0) + float(cred_level)
        j = max(j, i)
        # Advance j until target weight is enclosed
        while j < 2 * n and cum_w_ext[j] < target_w - 1e-12:
            j += 1
        if j >= 2 * n:
            break
        arc_len = float(s_ext[j] - s_ext[i])
        if arc_len < min_arc_len:
            min_arc_len = arc_len
            best_i = i
            best_j = j

    arc_start = float(s_ext[best_i] % two_pi)
    arc_end = float(s_ext[best_j] % two_pi)
    arc_length = min_arc_len
    enclosed_w = float(cum_w_ext[best_j] - (cum_w_ext[best_i - 1] if best_i > 0 else 0.0))

    is_contained = None
    if true_value is not None:
        if not math.isfinite(true_value):
            raise ValueError(f"true_value must be finite, got {true_value}.")
        if arc_length >= two_pi - 1e-12:
            is_contained = 1
        else:
            val_wrapped = float(true_value) % two_pi
            dist = (val_wrapped - arc_start) % two_pi
            is_contained = 1 if dist <= arc_length + 1e-12 else 0

    return {
        "cred_level": float(cred_level),
        "arc_start": arc_start,
        "arc_end": arc_end,
        "arc_length": arc_length,
        "enclosed_weight": enclosed_w,
        "is_contained": is_contained,
    }


def compute_coverage_summary(
    containment_flags: Sequence[Union[int, bool]],
    nominal_level: float,
) -> Dict[str, Any]:
    """Compute empirical coverage fraction and normal-theory uncertainty envelopes across M runs.

    Parameters
    ----------
    containment_flags : sequence of int or bool
        Array-like sequence of binary containment indicators (1 if contained, 0 otherwise).
    nominal_level : float
        Nominal credible level c in (0.0, 1.0), e.g. 0.50, 0.90, 0.95.

    Returns
    -------
    dict
        Dictionary containing:
        - "n_runs": Number of runs M.
        - "nominal_level": Nominal level c.
        - "n_hits": Number of runs where truth was contained.
        - "n_misses": Number of runs where truth was outside interval.
        - "empirical_coverage": hat_c = n_hits / M.
        - "binomial_se": sigma_hat_c = sqrt(c * (1 - c) / M).
        - "expected_misses": M * (1 - c).
        - "normal_envelope_2sigma": Tuple of [c - 2*sigma, c + 2*sigma] clipped to [0, 1].
        - "is_within_2sigma": Boolean indicating if hat_c is within the 2sigma envelope.
    """
    flags = [int(bool(x)) for x in containment_flags]
    m = len(flags)
    if m == 0:
        raise ValueError("containment_flags sequence cannot be empty.")
    if not (0.0 < nominal_level < 1.0):
        raise ValueError(f"nominal_level must be in (0.0, 1.0), got {nominal_level}.")

    c = float(nominal_level)
    n_hits = sum(flags)
    n_misses = m - n_hits
    hat_c = n_hits / m

    se = math.sqrt((c * (1.0 - c)) / m)
    low_2s = max(0.0, c - 2.0 * se)
    high_2s = min(1.0, c + 2.0 * se)
    is_within = (low_2s <= hat_c <= high_2s)

    return {
        "n_runs": m,
        "nominal_level": c,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "empirical_coverage": float(hat_c),
        "binomial_se": float(se),
        "expected_misses": float(m * (1.0 - c)),
        "normal_envelope_2sigma": [float(low_2s), float(high_2s)],
        "is_within_2sigma": bool(is_within),
    }


def compute_ks_test_uniform(
    ranks: Sequence[float],
    alpha: float = 0.05,
    n_tests: int = 4,
) -> Dict[str, Any]:
    """Perform one-sample Kolmogorov-Smirnov test for standard uniformity with Bonferroni correction.

    Parameters
    ----------
    ranks : sequence of float
        Array-like sequence of percentile ranks in [0.0, 1.0] from M independent inferences.
    alpha : float, default=0.05
        Nominal unadjusted significance level (default 0.05).
    n_tests : int, default=4
        Number of simultaneous hypothesis tests for Bonferroni adjustment (e.g. 4 parameters).

    Returns
    -------
    dict
        Dictionary containing:
        - "n_samples": Number of inferences M.
        - "ks_statistic": KS distance D_KS.
        - "p_value": Unadjusted two-sided KS p-value.
        - "d_crit_5pct": Approximate 5% critical distance 1.358 / sqrt(M).
        - "d_crit_bonferroni": Approximate Bonferroni critical distance (at alpha / n_tests).
        - "p_value_bonferroni": Bonferroni-adjusted p-value min(1.0, p_value * n_tests).
        - "alpha_unadjusted": alpha.
        - "alpha_bonferroni": alpha / n_tests.
        - "passes_unadjusted": Boolean (p_value > alpha).
        - "passes_bonferroni": Boolean (p_value > alpha_bonferroni).
    """
    r = np.asarray(ranks, dtype=np.float64)
    if r.size == 0:
        raise ValueError("Ranks array cannot be empty.")
    if not np.all(np.isfinite(r)):
        raise ValueError("Ranks array contains non-finite values.")
    if np.any(r < 0.0) or np.any(r > 1.0):
        raise ValueError("Percentile ranks must strictly lie in [0.0, 1.0].")
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1, got {n_tests}.")

    m = len(r)
    ks_res = stats.kstest(r, "uniform")
    d_ks = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    # Asymptotic critical values
    d_crit_5 = float(1.358 / math.sqrt(m))
    # For alpha = 0.0125 (n_tests = 4), c(alpha) ~ 1.59
    # In general for alpha_adj = 0.05 / n_tests, c(alpha) ~ sqrt(-0.5 * ln(alpha_adj / 2))
    alpha_adj = float(alpha / n_tests)
    c_bonf = math.sqrt(-0.5 * math.log(alpha_adj / 2.0))
    d_crit_bonf = float(c_bonf / math.sqrt(m))

    p_bonf = float(min(1.0, p_val * n_tests))
    passes_unadj = bool(p_val > alpha)
    passes_bonf = bool(p_val > alpha_adj)

    return {
        "n_samples": m,
        "ks_statistic": d_ks,
        "p_value": p_val,
        "d_crit_5pct": d_crit_5,
        "d_crit_bonferroni": d_crit_bonf,
        "p_value_bonferroni": p_bonf,
        "alpha_unadjusted": float(alpha),
        "alpha_bonferroni": float(alpha_adj),
        "passes_unadjusted": passes_unadj,
        "passes_bonferroni": passes_bonf,
    }


@dataclass
class SingleRunDiagnosticResult:
    """Diagnostic summaries for a single injection-recovery inference."""

    run_id: str
    injected_params: Dict[str, float]
    recovered_parameters: Dict[str, Dict[str, Any]]
    log_evidence: float
    log_evidence_err: float
    n_evaluations: int
    n_samples: int
    n_eff: float
    runtime_seconds: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic result to a JSON-serializable dictionary."""
        return asdict(self)


def compute_single_run_diagnostics(
    samples: np.ndarray,
    weights: np.ndarray,
    injected_params: Sequence[float],
    param_names: Sequence[str] = ("chirp_mass", "symmetric_mass_ratio", "tc", "phic"),
    cred_levels: Sequence[float] = (0.50, 0.90, 0.95),
    run_id: str = "run_unknown",
    log_evidence: float = 0.0,
    log_evidence_err: float = 0.0,
    n_evaluations: int = 0,
    runtime_seconds: float = 0.0,
    timestamp: str = "",
    circular_rotation: Optional[float] = None,
    circular_rotation_seed: Optional[int] = None,
    circular_tie_convention: str = "inclusive",
    circular_r_min_threshold: float = R_MIN_CIRCULAR_THRESHOLD,
) -> SingleRunDiagnosticResult:
    """Compute comprehensive parameter-by-parameter diagnostics for a single inference run.

    Parameters
    ----------
    samples : np.ndarray
        Array of parameter samples of shape (N, D).
    weights : np.ndarray
        1D array of normalized weights of shape (N,).
    injected_params : sequence of float
        Sequence of true injected parameter values of length D.
    param_names : sequence of str
        Names of parameters of length D.
    cred_levels : sequence of float
        Credible levels to evaluate.
    run_id : str
        Identifier for this run.
    log_evidence : float
        Log-evidence ln(Z).
    log_evidence_err : float
        Statistical uncertainty Delta ln(Z).
    n_evaluations : int
        Total likelihood evaluations.
    runtime_seconds : float
        Elapsed execution time.
    timestamp : str
        ISO timestamp string.
    circular_rotation : float, optional
        Explicit rotation cut U in radians for circular parameters.
    circular_rotation_seed : int, optional
        PRNG seed for drawing rotation cut U for circular parameters.
    circular_tie_convention : {"inclusive", "strict", "midpoint"}, default="inclusive"
        Tie convention for circular percentile rank.
    circular_r_min_threshold : float, default=1e-6
        Numerical threshold below which resultant vector length R is considered
        indeterminate.

    Returns
    -------
    SingleRunDiagnosticResult
        Structured container with all diagnostics.
    """
    s, w = validate_posterior_samples_and_weights(
        samples, weights, expected_ndim=len(param_names)
    )
    if len(injected_params) != len(param_names):
        raise ValueError(
            f"injected_params length ({len(injected_params)}) must match param_names ({len(param_names)})."
        )

    inj_dict = {name: float(val) for name, val in zip(param_names, injected_params)}
    n_eff = compute_effective_sample_size(w)

    param_results = {}
    for i, name in enumerate(param_names):
        p_samples = s[:, i]
        inj_val = float(injected_params[i])

        if name in ("phic", "phi_c", "phase", "coalescence_phase", "coalescence_phase_rad"):
            # Directional circular statistics
            circ_mom = compute_circular_moments_and_residuals(
                p_samples, w, inj_val, r_min_threshold=circular_r_min_threshold
            )
            p_rank, u_cut, rot_seed = compute_weighted_percentile_rank_circular(
                p_samples,
                w,
                inj_val,
                rotation=circular_rotation,
                seed=circular_rotation_seed,
                tie_convention=circular_tie_convention,
                return_details=True,
            )

            intervals = {}
            for lvl in cred_levels:
                arc_dict = compute_shortest_circular_credible_arc(
                    p_samples, w, lvl, true_value=inj_val
                )
                intervals[f"ci_{int(round(lvl*100))}"] = arc_dict

            param_results[name] = {
                "type": "circular",
                "injected": inj_val,
                "percentile_rank": p_rank,
                "rotation_cut_rad": u_cut,
                "rotation_seed": rot_seed,
                "tie_convention": circular_tie_convention,
                "circular_moments": circ_mom,
                "credible_intervals": intervals,
            }
        else:
            # Linear statistics
            lin_mom = compute_linear_moments_and_residuals(p_samples, w, inj_val)
            p_rank = compute_weighted_percentile_rank_linear(p_samples, w, inj_val)

            intervals = {}
            for lvl in cred_levels:
                ci_dict = compute_linear_credible_interval(
                    p_samples, w, lvl, true_value=inj_val
                )
                intervals[f"ci_{int(round(lvl*100))}"] = ci_dict

            param_results[name] = {
                "type": "linear",
                "injected": inj_val,
                "percentile_rank": p_rank,
                "linear_moments": lin_mom,
                "credible_intervals": intervals,
            }

    return SingleRunDiagnosticResult(
        run_id=run_id,
        injected_params=inj_dict,
        recovered_parameters=param_results,
        log_evidence=float(log_evidence),
        log_evidence_err=float(log_evidence_err),
        n_evaluations=int(n_evaluations),
        n_samples=len(s),
        n_eff=n_eff,
        runtime_seconds=float(runtime_seconds),
        timestamp=timestamp,
    )
