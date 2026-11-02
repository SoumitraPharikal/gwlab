#!/usr/bin/env python3
"""Statistically calibrated numerical validation script for Welch PSD estimation.

Performs rigorous Monte Carlo and theoretical validation of:
1. One-sided Welch mean PSD estimation under discrete Gaussian white noise,
   using realization-level band statistics and standardized z-score deviations.
2. Statistically calibrated adjacent-segment correlation vs proper-complex prediction r_1^2
   using a paired bootstrap (B=1000) across independent Monte Carlo realizations (Correction 1).
   The bootstrap estimates sampling uncertainty of the correlation statistic across independent
   realizations; it does not establish that the theoretical correlation model is universally valid.
3. Direct numerical evaluation of exact discrete pseudo-covariance window sums P_11(k), P_12(k)
   and full Isserlis/Wick periodogram correlation across the analysis band (Correction 3).
4. Finite-sample bias-corrected Welch median PSD estimation:
   - Part 3A: Exact theoretical baseline experiment using rectangular window on independent
     Gaussian white-noise segments, where Fourier coefficients are strictly proper and periodograms
     follow exact independent exponential distributions (Correction 2A).
   - Part 3B: Numerical robustness diagnostic using Tukey window, quantifying the effect of
     small non-zero window pseudo-covariance on finite-sample median estimation (Correction 2B).
5. Windowed discrete Parseval energy conservation to machine precision.
6. Diagnostic broad spectral recovery of an AR(1) coloured noise process.

Usage:
------
python scripts/validate_welch_psd.py
python scripts/validate_welch_psd.py --n-realizations 300 --seed 42
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, Optional, Union

import numpy as np

# Ensure src is in sys.path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gw_inference.conventions import expected_white_noise_psd
from gw_inference.noise import generate_white_noise
from gw_inference.psd import (
    discrete_pseudo_covariance_ratios,
    extract_segments,
    median_bias_factor,
    single_segment_psd,
    tukey_window,
    welch_psd,
    welch_variance_inflation,
    window_coherent_gain,
    window_multi_lag_correlations,
    window_overlap_correlation,
    window_power,
)


def validate_white_noise_welch(
    f_s: float = 4096.0,
    total_duration: float = 8.0,
    segment_duration: float = 1.0,
    overlap_fraction: float = 0.5,
    sigma: float = 1.0,
    n_realizations: int = 300,
    seed: int = 42,
    alpha: float = 0.25,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """Run statistically calibrated Monte Carlo validation of Welch mean PSD on Gaussian white noise."""
    delta_t = 1.0 / f_s
    segment_length = int(round(segment_duration * f_s))
    overlap_samples = int(round(overlap_fraction * segment_length))
    hop_length = segment_length - overlap_samples

    w = tukey_window(segment_length, alpha=alpha)
    s2 = window_power(w)
    s1 = window_coherent_gain(w)
    r1 = window_overlap_correlation(w, overlap_samples=overlap_samples)
    n_samples = int(round(total_duration * f_s))
    n_segments = (n_samples - overlap_samples) // hop_length

    # All-lags variance inflation factor
    var_inflation = welch_variance_inflation(
        w, segment_length=segment_length,
        overlap_samples=overlap_samples, n_segments=n_segments
    )

    # Multi-lag correlations
    r_lags = window_multi_lag_correlations(w, hop_length=hop_length, max_lag=n_segments - 1)

    # Expected values
    expected_interior = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
    expected_dc = expected_white_noise_psd(sigma, delta_t, bin_type="dc")
    expected_nyquist = expected_white_noise_psd(sigma, delta_t, bin_type="nyquist")

    # Theoretical single-bin standard error
    se_single_bin = (expected_interior / math.sqrt(n_realizations * n_segments)) * math.sqrt(var_inflation)

    # Accumulate realization-level statistics
    rng = np.random.default_rng(seed)
    n_freqs = segment_length // 2 + 1
    interior_slice = slice(10, n_freqs - 10)

    q_realizations = np.empty(n_realizations, dtype=np.float64)
    dc_realizations = np.empty(n_realizations, dtype=np.float64)
    nyquist_realizations = np.empty(n_realizations, dtype=np.float64)

    # Adjacent segments for empirical correlation
    if n_segments >= 2:
        seg0_p = np.empty((n_realizations, n_freqs), dtype=np.float64)
        seg1_p = np.empty((n_realizations, n_freqs), dtype=np.float64)

    parseval_rel_diffs = np.empty(n_realizations, dtype=np.float64)

    for r in range(n_realizations):
        x = generate_white_noise(n_samples, sigma=sigma, generator=rng)

        # Parseval check on segment 0
        seg0 = x[:segment_length]
        p0 = single_segment_psd(seg0, delta_t, window=w)
        delta_f = 1.0 / (segment_length * delta_t)
        freq_energy = np.sum(p0) * delta_f
        time_energy = np.sum((seg0 * w) ** 2) / (segment_length * s2)
        parseval_rel_diffs[r] = abs(freq_energy - time_energy) / time_energy

        if n_segments >= 2:
            seg0_p[r] = p0
            seg1 = x[hop_length : hop_length + segment_length]
            seg1_p[r] = single_segment_psd(seg1, delta_t, window=w)

        _, psd = welch_psd(
            x, delta_t, segment_length=segment_length,
            overlap_samples=overlap_samples, window=w, method="mean"
        )

        q_realizations[r] = np.mean(psd[interior_slice])
        dc_realizations[r] = psd[0]
        nyquist_realizations[r] = psd[-1]

    # Realization-level band statistics across R realizations
    q_bar = float(np.mean(q_realizations))
    s_q = float(np.std(q_realizations, ddof=1))
    se_q_bar = float(s_q / math.sqrt(n_realizations))
    z_interior = float((q_bar - expected_interior) / se_q_bar) if se_q_bar > 0 else 0.0

    # DC statistics
    dc_bar = float(np.mean(dc_realizations))
    s_dc = float(np.std(dc_realizations, ddof=1))
    se_dc_bar = float(s_dc / math.sqrt(n_realizations))
    z_dc = float((dc_bar - expected_dc) / se_dc_bar) if se_dc_bar > 0 else 0.0

    # Nyquist statistics
    nyq_bar = float(np.mean(nyquist_realizations))
    s_nyq = float(np.std(nyquist_realizations, ddof=1))
    se_nyq_bar = float(s_nyq / math.sqrt(n_realizations))
    z_nyquist = float((nyq_bar - expected_nyquist) / se_nyq_bar) if se_nyq_bar > 0 else 0.0

    # Adjacent segment correlation and paired bootstrap standard error (Correction 1)
    measured_rho_p = 0.0
    se_rho_p = 0.0
    z_rho_p = 0.0
    r1_squared = r1 ** 2

    if n_segments >= 2:
        # Vectorized correlation across interior bins
        x_bins = seg0_p[:, interior_slice]
        y_bins = seg1_p[:, interior_slice]

        def _calc_mean_r(x_mat: np.ndarray, y_mat: np.ndarray) -> float:
            dx = x_mat - np.mean(x_mat, axis=0, keepdims=True)
            dy = y_mat - np.mean(y_mat, axis=0, keepdims=True)
            cov = np.sum(dx * dy, axis=0)
            std_prod = np.sqrt(np.sum(dx ** 2, axis=0) * np.sum(dy ** 2, axis=0))
            nonzero = std_prod > 0
            return float(np.mean(cov[nonzero] / std_prod[nonzero]))

        measured_rho_p = _calc_mean_r(x_bins, y_bins)

        # Paired bootstrap across realizations to estimate sampling uncertainty
        boot_rng = np.random.default_rng(seed + 999)
        boot_rhos = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = boot_rng.choice(n_realizations, size=n_realizations, replace=True)
            boot_rhos[b] = _calc_mean_r(x_bins[idx], y_bins[idx])

        se_rho_p = float(np.std(boot_rhos, ddof=1))
        z_rho_p = float((measured_rho_p - r1_squared) / se_rho_p) if se_rho_p > 0 else 0.0

    return {
        "f_s": f_s,
        "delta_t": delta_t,
        "total_duration": total_duration,
        "n_samples": n_samples,
        "segment_duration": segment_duration,
        "segment_length": segment_length,
        "overlap_fraction": overlap_fraction,
        "overlap_samples": overlap_samples,
        "hop_length": hop_length,
        "n_segments": n_segments,
        "alpha": alpha,
        "S2": s2,
        "S1": s1,
        "r1": r1,
        "r1_squared": r1_squared,
        "n_shared_lags": len(r_lags),
        "var_inflation": var_inflation,
        "n_realizations": n_realizations,
        "n_bootstrap": n_bootstrap,
        "expected_interior": expected_interior,
        "q_bar": q_bar,
        "s_q": s_q,
        "se_q_bar": se_q_bar,
        "se_single_bin": se_single_bin,
        "z_interior": z_interior,
        "expected_dc": expected_dc,
        "dc_bar": dc_bar,
        "se_dc_bar": se_dc_bar,
        "z_dc": z_dc,
        "expected_nyquist": expected_nyquist,
        "nyq_bar": nyq_bar,
        "se_nyq_bar": se_nyq_bar,
        "z_nyquist": z_nyquist,
        "measured_rho_p": measured_rho_p,
        "se_rho_p": se_rho_p,
        "z_rho_p": z_rho_p,
        "mean_parseval_rel_diff": float(np.mean(parseval_rel_diffs)),
    }


def validate_pseudo_covariance_sums(
    segment_length: int = 4096,
    alpha: float = 0.25,
    overlap_fraction: float = 0.5,
) -> Dict[str, float]:
    """Direct numerical evaluation of discrete window covariance and pseudo-covariance sums (Correction 3)."""
    overlap_samples = int(round(overlap_fraction * segment_length))
    w = tukey_window(segment_length, alpha=alpha)
    res = discrete_pseudo_covariance_ratios(w, overlap_samples=overlap_samples)
    n_freqs = segment_length // 2 + 1
    interior_slice = slice(5, n_freqs - 5)

    return {
        "C11": float(res["C11"]),
        "C12": float(res["C12"]),
        "rw_squared": float(res["rw_squared"]),
        "ratio_P11_dc": float(res["ratio_P11"][0]),
        "ratio_P12_dc": float(res["ratio_P12"][0]),
        "rho_full_dc": float(res["rho_full"][0]),
        "ratio_P11_k1": float(res["ratio_P11"][1]),
        "ratio_P12_k1": float(res["ratio_P12"][1]),
        "rho_full_k1": float(res["rho_full"][1]),
        "rel_diff_k1": float(res["rel_diff"][1]),
        "ratio_P11_k2": float(res["ratio_P11"][2]),
        "ratio_P12_k2": float(res["ratio_P12"][2]),
        "rho_full_k2": float(res["rho_full"][2]),
        "rel_diff_k2": float(res["rel_diff"][2]),
        "ratio_P11_k5": float(res["ratio_P11"][5]),
        "ratio_P12_k5": float(res["ratio_P12"][5]),
        "rho_full_k5": float(res["rho_full"][5]),
        "rel_diff_k5": float(res["rel_diff"][5]),
        "ratio_P11_nyq": float(res["ratio_P11"][-1]),
        "ratio_P12_nyq": float(res["ratio_P12"][-1]),
        "rho_full_nyq": float(res["rho_full"][-1]),
        "max_ratio_P11_interior": float(np.max(res["ratio_P11"][interior_slice])),
        "mean_ratio_P11_interior": float(np.mean(res["ratio_P11"][interior_slice])),
        "max_ratio_P12_interior": float(np.max(res["ratio_P12"][interior_slice])),
        "mean_ratio_P12_interior": float(np.mean(res["ratio_P12"][interior_slice])),
        "max_rel_diff_interior": float(np.max(np.abs(res["rel_diff"][interior_slice]))),
        "mean_rel_diff_interior": float(np.mean(np.abs(res["rel_diff"][interior_slice]))),
    }


def validate_median_white_noise_rectangular(
    f_s: float = 4096.0,
    segment_duration: float = 0.5,
    n_segments: int = 15,
    sigma: float = 1.0,
    n_realizations: int = 300,
    seed: int = 142,
) -> Dict[str, float]:
    """Exact theoretical baseline: validate median-Welch on rectangular-window independent white noise (Correction 2A).

    Under discrete Gaussian white noise and a rectangular window (w_j = 1), Fourier coefficients
    at interior frequencies are strictly proper complex Gaussian random variables (E[X_k^2] = 0).
    Therefore, the segment periodograms are identically, independently exponentially distributed,
    providing an exact test of the finite-sample median bias correction factor C(K).
    """
    delta_t = 1.0 / f_s
    segment_length = int(round(segment_duration * f_s))
    total_samples = n_segments * segment_length

    expected_interior = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
    ck = median_bias_factor(n_segments)

    rng = np.random.default_rng(seed)
    n_freqs = segment_length // 2 + 1
    interior_slice = slice(5, n_freqs - 5)

    raw_realizations = np.empty(n_realizations, dtype=np.float64)
    corr_realizations = np.empty(n_realizations, dtype=np.float64)

    for r in range(n_realizations):
        x = generate_white_noise(total_samples, sigma=sigma, generator=rng)

        # Raw median (uncorrected)
        _, psd_raw = welch_psd(
            x, delta_t, segment_length=segment_length,
            overlap_samples=0, window="rect", method="median",
            bias_correct=False,
        )
        raw_realizations[r] = np.mean(psd_raw[interior_slice])

        # Corrected median
        _, psd_corr = welch_psd(
            x, delta_t, segment_length=segment_length,
            overlap_samples=0, window="rect", method="median",
            bias_correct=True,
            assume_independent_exponential_periodograms=True,
        )
        corr_realizations[r] = np.mean(psd_corr[interior_slice])

    raw_bar = float(np.mean(raw_realizations))
    corr_bar = float(np.mean(corr_realizations))
    s_corr = float(np.std(corr_realizations, ddof=1))
    se_corr_bar = float(s_corr / math.sqrt(n_realizations))
    z_median = float((corr_bar - expected_interior) / se_corr_bar) if se_corr_bar > 0 else 0.0

    return {
        "n_segments": n_segments,
        "ck": ck,
        "expected_interior": expected_interior,
        "expected_raw": ck * expected_interior,
        "raw_bar": raw_bar,
        "corr_bar": corr_bar,
        "se_corr_bar": se_corr_bar,
        "z_median": z_median,
    }


def validate_median_white_noise_tukey(
    f_s: float = 4096.0,
    segment_duration: float = 0.5,
    n_segments: int = 15,
    sigma: float = 1.0,
    n_realizations: int = 300,
    seed: int = 142,
    alpha: float = 0.25,
) -> Dict[str, float]:
    """Numerical robustness diagnostic: evaluate median-Welch on Tukey-window independent white noise (Correction 2B).

    Tapered windows introduce slight non-zero pseudo-covariance P_11(k) != 0, so periodograms are
    only approximately exponential. This experiment evaluates the empirical accuracy of C(K)
    under this mild departure from the theoretical assumptions.
    """
    delta_t = 1.0 / f_s
    segment_length = int(round(segment_duration * f_s))
    total_samples = n_segments * segment_length

    expected_interior = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
    ck = median_bias_factor(n_segments)

    rng = np.random.default_rng(seed)
    n_freqs = segment_length // 2 + 1
    interior_slice = slice(5, n_freqs - 5)

    raw_realizations = np.empty(n_realizations, dtype=np.float64)
    corr_realizations = np.empty(n_realizations, dtype=np.float64)

    w = tukey_window(segment_length, alpha=alpha)

    for r in range(n_realizations):
        x = generate_white_noise(total_samples, sigma=sigma, generator=rng)

        # Raw median (uncorrected)
        _, psd_raw = welch_psd(
            x, delta_t, segment_length=segment_length,
            overlap_samples=0, window=w, method="median",
            bias_correct=False,
        )
        raw_realizations[r] = np.mean(psd_raw[interior_slice])

        # Corrected median
        _, psd_corr = welch_psd(
            x, delta_t, segment_length=segment_length,
            overlap_samples=0, window=w, method="median",
            bias_correct=True,
            assume_independent_exponential_periodograms=True,
        )
        corr_realizations[r] = np.mean(psd_corr[interior_slice])

    raw_bar = float(np.mean(raw_realizations))
    corr_bar = float(np.mean(corr_realizations))
    s_corr = float(np.std(corr_realizations, ddof=1))
    se_corr_bar = float(s_corr / math.sqrt(n_realizations))
    z_median = float((corr_bar - expected_interior) / se_corr_bar) if se_corr_bar > 0 else 0.0

    return {
        "n_segments": n_segments,
        "ck": ck,
        "expected_interior": expected_interior,
        "expected_raw": ck * expected_interior,
        "raw_bar": raw_bar,
        "corr_bar": corr_bar,
        "se_corr_bar": se_corr_bar,
        "z_median": z_median,
    }


def validate_median_white_noise(
    f_s: float = 4096.0,
    segment_duration: float = 0.5,
    n_segments: int = 15,
    sigma: float = 1.0,
    n_realizations: int = 300,
    seed: int = 142,
    window: str = "rect",
) -> Dict[str, float]:
    """Backward-compatible entry point for median-Welch validation."""
    if window == "rect":
        return validate_median_white_noise_rectangular(
            f_s=f_s, segment_duration=segment_duration, n_segments=n_segments,
            sigma=sigma, n_realizations=n_realizations, seed=seed,
        )
    return validate_median_white_noise_tukey(
        f_s=f_s, segment_duration=segment_duration, n_segments=n_segments,
        sigma=sigma, n_realizations=n_realizations, seed=seed,
    )


def validate_coloured_noise_ar1(
    phi: float = 0.65,
    sigma_eps: float = 1.0,
    f_s: float = 2048.0,
    total_duration: float = 8.0,
    segment_duration: float = 1.0,
    n_realizations: int = 250,
    seed: int = 242,
) -> Dict[str, float]:
    """Diagnostic broad spectral recovery of an AR(1) coloured noise process."""
    delta_t = 1.0 / f_s
    n_samples = int(round(total_duration * f_s))
    segment_length = int(round(segment_duration * f_s))
    overlap_samples = segment_length // 2

    rng = np.random.default_rng(seed)
    n_freqs = segment_length // 2 + 1
    psd_ensemble = np.zeros(n_freqs, dtype=np.float64)

    burn_in = 500
    for _ in range(n_realizations):
        eps = rng.normal(0.0, sigma_eps, size=n_samples + burn_in)
        x_full = np.empty_like(eps)
        x_full[0] = eps[0] / math.sqrt(1.0 - phi ** 2)
        for j in range(1, len(eps)):
            x_full[j] = phi * x_full[j - 1] + eps[j]
        x = x_full[burn_in:]

        freqs, psd = welch_psd(
            x, delta_t, segment_length=segment_length,
            overlap_samples=overlap_samples, window="tukey", method="mean"
        )
        psd_ensemble += psd

    psd_ensemble /= n_realizations

    # Theoretical one-sided AR(1) spectrum
    denom = 1.0 + phi ** 2 - 2.0 * phi * np.cos(2.0 * np.pi * freqs * delta_t)
    expected_psd = 2.0 * (sigma_eps ** 2) * delta_t / denom

    interior_slice = slice(5, n_freqs - 5)
    rel_diffs = np.abs(psd_ensemble[interior_slice] - expected_psd[interior_slice]) / expected_psd[interior_slice]
    mean_rel_err = float(np.mean(rel_diffs))
    max_rel_err = float(np.max(rel_diffs))

    return {
        "phi": phi,
        "sigma_eps": sigma_eps,
        "f_s": f_s,
        "n_realizations": n_realizations,
        "mean_rel_err": mean_rel_err,
        "max_rel_err": max_rel_err,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Statistically Calibrated Welch PSD Validation.")
    parser.add_argument("--fs", type=float, default=4096.0, help="Sampling frequency in Hz")
    parser.add_argument("--duration", type=float, default=8.0, help="Total duration in seconds")
    parser.add_argument("--seg-duration", type=float, default=1.0, help="Segment duration in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction")
    parser.add_argument("--sigma", type=float, default=1.0, help="White noise standard deviation")
    parser.add_argument("--n-realizations", type=int, default=300, help="Number of Monte Carlo realizations")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of paired bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.25, help="Tukey alpha parameter")

    args = parser.parse_args()

    # Input validation
    if args.fs <= 0 or not math.isfinite(args.fs):
        print(f"Error: fs must be positive and finite, got {args.fs}", file=sys.stderr)
        return 1
    if args.duration <= 0 or not math.isfinite(args.duration):
        print(f"Error: duration must be positive and finite, got {args.duration}", file=sys.stderr)
        return 1
    if args.seg_duration <= 0 or args.seg_duration > args.duration:
        print(f"Error: seg-duration must be in (0, duration], got {args.seg_duration}", file=sys.stderr)
        return 1
    if not (0.0 <= args.overlap < 1.0):
        print(f"Error: overlap must be in [0.0, 1.0), got {args.overlap}", file=sys.stderr)
        return 1
    if args.sigma <= 0 or not math.isfinite(args.sigma):
        print(f"Error: sigma must be positive and finite, got {args.sigma}", file=sys.stderr)
        return 1
    if args.n_realizations < 2:
        print(f"Error: n-realizations must be at least 2, got {args.n_realizations}", file=sys.stderr)
        return 1
    if args.n_bootstrap < 100:
        print(f"Error: n-bootstrap must be at least 100, got {args.n_bootstrap}", file=sys.stderr)
        return 1

    print("================================================================================")
    print("STAGE 1.2: STATISTICALLY CALIBRATED WELCH PSD VALIDATION")
    print("================================================================================")

    # 1. White Noise Welch Mean Validation
    print("\n--- Part 1: White Noise Welch Mean Estimation (Calibrated Statistics) ---")
    res_white = validate_white_noise_welch(
        f_s=args.fs,
        total_duration=args.duration,
        segment_duration=args.seg_duration,
        overlap_fraction=args.overlap,
        sigma=args.sigma,
        n_realizations=args.n_realizations,
        seed=args.seed,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
    )

    print(f"Sampling frequency (f_s):           {res_white['f_s']:.1f} Hz")
    print(f"Sampling interval (delta_t):         {res_white['delta_t']:.6e} s")
    print(f"Total duration:                      {res_white['total_duration']:.2f} s ({res_white['n_samples']} samples)")
    print(f"Segment duration:                    {res_white['segment_duration']:.2f} s ({res_white['segment_length']} samples)")
    print(f"Overlap fraction:                    {res_white['overlap_fraction'] * 100:.1f}% ({res_white['overlap_samples']} samples)")
    print(f"Hop length:                          {res_white['hop_length']} samples")
    print(f"Number of valid segments (K):        {res_white['n_segments']}")
    print(f"Number of shared lags (ell_max):     {res_white['n_shared_lags']}")
    print(f"Tukey window alpha:                  {res_white['alpha']}")
    print(f"Discrete window power (S_2):         {res_white['S2']:.6f}")
    print(f"Adjacent correlation (r_1):          {res_white['r1']:.6f}")
    print(f"All-lags variance inflation (kappa): {res_white['var_inflation']:.4f}")
    print(f"Number of Monte Carlo trials (R):    {res_white['n_realizations']}")
    print(f"Interior expected PSD:               {res_white['expected_interior']:.6e} strain^2/Hz")
    print(f"Interior ensemble band mean Q_bar:   {res_white['q_bar']:.6e} strain^2/Hz")
    print(f"Realization band std dev (s_Q):      {res_white['s_q']:.6e} strain^2/Hz")
    print(f"Ensemble band SE (SE(Q_bar)):        {res_white['se_q_bar']:.6e} strain^2/Hz")
    print(f"Single-bin theoretical SE:           {res_white['se_single_bin']:.6e} strain^2/Hz")
    print(f"Interior standardized deviation z_Q: {res_white['z_interior']:+.3f}")
    print(f"DC expected PSD:                     {res_white['expected_dc']:.6e} strain^2/Hz")
    print(f"DC ensemble mean:                    {res_white['dc_bar']:.6e} strain^2/Hz (SE: {res_white['se_dc_bar']:.6e}, z = {res_white['z_dc']:+.3f})")
    print(f"Nyquist expected PSD:                {res_white['expected_nyquist']:.6e} strain^2/Hz")
    print(f"Nyquist ensemble mean:               {res_white['nyq_bar']:.6e} strain^2/Hz (SE: {res_white['se_nyq_bar']:.6e}, z = {res_white['z_nyquist']:+.3f})")
    print(f"Empirical adj-seg corr (rho_P):      {res_white['measured_rho_p']:.4f} (pred: r_1^2 = {res_white['r1_squared']:.4f})")
    print(f"Paired bootstrap replicates (B):     {res_white['n_bootstrap']}")
    print(f"Paired bootstrap SE (SE(rho_P)):     {res_white['se_rho_p']:.6e}")
    print(f"Correlation standardized z-score:    {res_white['z_rho_p']:+.3f}")
    print(f"Mean Parseval relative error:        {res_white['mean_parseval_rel_diff']:.2e}")

    # 2. Exact Discrete Window Pseudo-Covariance Sums (Correction 3)
    print("\n--- Part 2: Exact Discrete Pseudo-Covariance Window Sums (Correction 3) ---")
    res_pseudo = validate_pseudo_covariance_sums(
        segment_length=res_white["segment_length"],
        alpha=args.alpha,
        overlap_fraction=args.overlap,
    )
    print(f"Discrete window total energy (C_11): {res_pseudo['C11']:.4f}")
    print(f"Discrete window overlap prod (C_12): {res_pseudo['C12']:.4f}")
    print(f"Proper-complex prediction (r_w^2):   {res_pseudo['rw_squared']:.6f}")
    print(f"DC (k=0):      |P_11|/C_11 = {res_pseudo['ratio_P11_dc']:.4e}, |P_12|/|C_12| = {res_pseudo['ratio_P12_dc']:.4e}, rho_full = {res_pseudo['rho_full_dc']:.6f}")
    print(f"Bin k=1:       |P_11|/C_11 = {res_pseudo['ratio_P11_k1']:.4e}, |P_12|/|C_12| = {res_pseudo['ratio_P12_k1']:.4e}, rho_full = {res_pseudo['rho_full_k1']:.6f} (rel diff: {res_pseudo['rel_diff_k1']:+.4e})")
    print(f"Bin k=2:       |P_11|/C_11 = {res_pseudo['ratio_P11_k2']:.4e}, |P_12|/|C_12| = {res_pseudo['ratio_P12_k2']:.4e}, rho_full = {res_pseudo['rho_full_k2']:.6f} (rel diff: {res_pseudo['rel_diff_k2']:+.4e})")
    print(f"Bin k=5:       |P_11|/C_11 = {res_pseudo['ratio_P11_k5']:.4e}, |P_12|/|C_12| = {res_pseudo['ratio_P12_k5']:.4e}, rho_full = {res_pseudo['rho_full_k5']:.6f} (rel diff: {res_pseudo['rel_diff_k5']:+.4e})")
    print(f"Nyquist (k=M/2): |P_11|/C_11 = {res_pseudo['ratio_P11_nyq']:.4e}, |P_12|/|C_12| = {res_pseudo['ratio_P12_nyq']:.4e}, rho_full = {res_pseudo['rho_full_nyq']:.6f}")
    print(f"Analysis band [5, M/2-5] Max |P_11|/C_11:   {res_pseudo['max_ratio_P11_interior']:.4e} (Mean: {res_pseudo['mean_ratio_P11_interior']:.4e})")
    print(f"Analysis band [5, M/2-5] Max |P_12|/|C_12|: {res_pseudo['max_ratio_P12_interior']:.4e} (Mean: {res_pseudo['mean_ratio_P12_interior']:.4e})")
    print(f"Analysis band Max |rho_full - r_w^2|/r_w^2:  {res_pseudo['max_rel_diff_interior']:.4e} (Mean: {res_pseudo['mean_rel_diff_interior']:.4e})")

    # 3A. Median-Welch Exact Rectangular Baseline (Correction 2A)
    print("\n--- Part 3A: Median-Welch Exact Theoretical Baseline (Rectangular Window) ---")
    res_med_rect = validate_median_white_noise_rectangular(
        f_s=args.fs,
        segment_duration=0.5,
        n_segments=15,
        sigma=args.sigma,
        n_realizations=args.n_realizations,
        seed=args.seed + 100,
    )
    print(f"Number of independent segments (K):  {res_med_rect['n_segments']}")
    print(f"Stage 0.6 odd bias factor C(15):     {res_med_rect['ck']:.6f}")
    print(f"Expected uncorrected raw median:     {res_med_rect['expected_raw']:.6e} strain^2/Hz")
    print(f"Measured uncorrected raw median:     {res_med_rect['raw_bar']:.6e} strain^2/Hz")
    print(f"Corrected median PSD ensemble mean:  {res_med_rect['corr_bar']:.6e} strain^2/Hz")
    print(f"Corrected median ensemble SE:        {res_med_rect['se_corr_bar']:.6e} strain^2/Hz")
    print(f"Exact baseline median z-score:       {res_med_rect['z_median']:+.3f}")

    # 3B. Median-Welch Tukey Window Diagnostic (Correction 2B)
    print("\n--- Part 3B: Median-Welch Robustness Diagnostic (Tukey Window, alpha=0.25) ---")
    res_med_tukey = validate_median_white_noise_tukey(
        f_s=args.fs,
        segment_duration=0.5,
        n_segments=15,
        sigma=args.sigma,
        n_realizations=args.n_realizations,
        seed=args.seed + 100,
        alpha=args.alpha,
    )
    print(f"Number of independent segments (K):  {res_med_tukey['n_segments']}")
    print(f"Expected uncorrected raw median:     {res_med_tukey['expected_raw']:.6e} strain^2/Hz")
    print(f"Measured uncorrected raw median:     {res_med_tukey['raw_bar']:.6e} strain^2/Hz")
    print(f"Corrected median PSD ensemble mean:  {res_med_tukey['corr_bar']:.6e} strain^2/Hz")
    print(f"Corrected median ensemble SE:        {res_med_tukey['se_corr_bar']:.6e} strain^2/Hz")
    print(f"Tukey diagnostic median z-score:     {res_med_tukey['z_median']:+.3f}")

    # 4. Coloured Noise AR(1) Diagnostic Spectral Recovery
    print("\n--- Part 4: Coloured Noise AR(1) Broad Spectral Shape (Diagnostic) ---")
    res_ar1 = validate_coloured_noise_ar1(
        phi=0.65,
        sigma_eps=args.sigma,
        f_s=2048.0,
        total_duration=8.0,
        segment_duration=1.0,
        n_realizations=250,
        seed=args.seed + 200,
    )
    print(f"AR(1) parameter phi:                 {res_ar1['phi']}")
    print(f"Mean relative error across band:     {res_ar1['mean_rel_err']*100:.3f}% (diagnostic spectral agreement)")
    print(f"Max relative error across band:      {res_ar1['max_rel_err']*100:.3f}% (reflects window spectral smoothing)")

    print("\n================================================================================")
    print("STATISTICALLY CALIBRATED ACCEPTANCE CRITERIA CHECK")
    print("================================================================================")

    # Acceptance criteria:
    pass_interior = abs(res_white["z_interior"]) < 5.0
    pass_dc = abs(res_white["z_dc"]) < 5.0
    pass_nyq = abs(res_white["z_nyquist"]) < 5.0
    pass_parseval = res_white["mean_parseval_rel_diff"] < 1e-10
    pass_corr = abs(res_white["z_rho_p"]) < 5.0
    pass_med_rect = abs(res_med_rect["z_median"]) < 5.0
    pass_pseudo = res_pseudo["max_rel_diff_interior"] < 1e-3

    print(f"Criteria 1: Interior band mean z-score |z_Q| < 5.0:            {'PASS' if pass_interior else 'FAIL'} (z = {res_white['z_interior']:+.3f})")
    print(f"Criteria 2: DC bin z-score |z_DC| < 5.0:                       {'PASS' if pass_dc else 'FAIL'} (z = {res_white['z_dc']:+.3f})")
    print(f"Criteria 3: Nyquist bin z-score |z_Nyq| < 5.0:                 {'PASS' if pass_nyq else 'FAIL'} (z = {res_white['z_nyquist']:+.3f})")
    print(f"Criteria 4: Parseval energy conservation error < 1e-10:        {'PASS' if pass_parseval else 'FAIL'} ({res_white['mean_parseval_rel_diff']:.2e})")
    print(f"Criteria 5: Paired bootstrap correlation |z_rho| < 5.0:        {'PASS' if pass_corr else 'FAIL'} (z = {res_white['z_rho_p']:+.3f}, B = {res_white['n_bootstrap']})")
    print(f"Criteria 6: Rectangular median exact baseline |z_M| < 5.0:     {'PASS' if pass_med_rect else 'FAIL'} (z = {res_med_rect['z_median']:+.3f})")
    print(f"Criteria 7: Analysis band pseudo-covariance rel diff < 1e-3:   {'PASS' if pass_pseudo else 'FAIL'} ({res_pseudo['max_rel_diff_interior']:.2e})")
    print(f"Diagnostic: Tukey median robustness diagnostic z-score:        {res_med_tukey['z_median']:+.3f}")
    print(f"Diagnostic: AR(1) coloured-noise band mean error:             {res_ar1['mean_rel_err']*100:.3f}% (max: {res_ar1['max_rel_err']*100:.3f}%)")

    all_passed = pass_interior and pass_dc and pass_nyq and pass_parseval and pass_corr and pass_med_rect and pass_pseudo
    print(f"\nOverall Validation Status: {'ALL CRITERIA PASSED' if all_passed else 'VALIDATION FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
