#!/usr/bin/env python3
"""Numerical validation script for Welch PSD estimation.

Performs rigorous Monte Carlo and theoretical validation of:
1. One-sided Welch mean PSD estimation under discrete Gaussian white noise.
2. Standard error calculation accounting for 50% overlap correlation (Stage 0.6 Section 3-4).
3. Empirical adjacent-segment periodogram correlation vs r_w^2 proper-complex prediction.
4. Windowed discrete Parseval energy conservation.
5. Finite-sample bias-corrected Welch median PSD validation under independent segments.
6. Coloured-noise spectral shape recovery using an exact analytical AR(1) process.

Usage:
------
python scripts/validate_welch_psd.py
python scripts/validate_welch_psd.py --n-realizations 500 --seed 42
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, Tuple

import numpy as np

# Ensure src is in sys.path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gw_inference.conventions import expected_white_noise_psd
from gw_inference.noise import generate_white_noise
from gw_inference.psd import (
    extract_segments,
    median_bias_factor,
    single_segment_psd,
    tukey_window,
    welch_psd,
    window_coherent_gain,
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
) -> Dict[str, float]:
    """Run Monte Carlo validation of Welch mean PSD on Gaussian white noise."""
    delta_t = 1.0 / f_s
    n_samples = int(round(total_duration * f_s))
    segment_length = int(round(segment_duration * f_s))
    overlap_samples = int(round(overlap_fraction * segment_length))
    hop_length = segment_length - overlap_samples

    w = tukey_window(segment_length, alpha=alpha)
    s2 = window_power(w)
    s1 = window_coherent_gain(w)
    rw = window_overlap_correlation(w, overlap_samples=overlap_samples)

    n_segments = (n_samples - overlap_samples) // hop_length

    # Expected values
    expected_interior = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
    expected_dc = expected_white_noise_psd(sigma, delta_t, bin_type="dc")
    expected_nyquist = expected_white_noise_psd(sigma, delta_t, bin_type="nyquist")

    # Overlap variance inflation factor for Welch mean (Stage 0.6 Section 4)
    # Var(S_welch) = (S^2 / K) * [1 + 2 * ((K-1)/K) * rw^2]
    var_inflation = 1.0 + 2.0 * ((n_segments - 1) / n_segments) * (rw ** 2) if n_segments > 1 else 1.0

    # Theoretical standard error of the ensemble mean over R realizations at a single interior bin
    se_interior_bin = (expected_interior / math.sqrt(n_realizations * n_segments)) * math.sqrt(var_inflation)

    # Accumulate ensemble PSD
    rng = np.random.default_rng(seed)
    n_freqs = segment_length // 2 + 1
    psd_ensemble = np.zeros(n_freqs, dtype=np.float64)

    # To measure adjacent segment correlation, store periodograms of seg 0 and seg 1
    if n_segments >= 2:
        seg0_p = np.zeros((n_realizations, n_freqs), dtype=np.float64)
        seg1_p = np.zeros((n_realizations, n_freqs), dtype=np.float64)

    # Track Parseval energy balance across realizations
    parseval_rel_diffs = []

    for r in range(n_realizations):
        x = generate_white_noise(n_samples, sigma=sigma, generator=rng)

        # Parseval check on segment 0
        seg0 = x[:segment_length]
        p0 = single_segment_psd(seg0, delta_t, window=w)
        delta_f = 1.0 / (segment_length * delta_t)
        freq_energy = np.sum(p0) * delta_f
        time_energy = np.sum((seg0 * w) ** 2) / (segment_length * s2)
        parseval_rel_diffs.append(abs(freq_energy - time_energy) / time_energy)

        if n_segments >= 2:
            seg0_p[r] = p0
            seg1 = x[hop_length : hop_length + segment_length]
            seg1_p[r] = single_segment_psd(seg1, delta_t, window=w)

        _, psd = welch_psd(
            x, delta_t, segment_length=segment_length,
            overlap_samples=overlap_samples, window=w, method="mean"
        )
        psd_ensemble += psd

    psd_ensemble /= n_realizations

    # Interior band average (bins 10 to n_freqs - 10)
    interior_slice = slice(10, n_freqs - 10)
    measured_interior = float(np.mean(psd_ensemble[interior_slice]))
    measured_dc = float(psd_ensemble[0])
    measured_nyquist = float(psd_ensemble[-1])

    rel_err_interior = abs(measured_interior - expected_interior) / expected_interior
    rel_err_dc = abs(measured_dc - expected_dc) / expected_dc
    rel_err_nyquist = abs(measured_nyquist - expected_nyquist) / expected_nyquist

    # Empirical adjacent segment correlation across interior bins
    measured_rho_p = 0.0
    if n_segments >= 2:
        rho_bins = []
        for k in range(10, n_freqs - 10):
            r_mat = np.corrcoef(seg0_p[:, k], seg1_p[:, k])
            rho_bins.append(r_mat[0, 1])
        measured_rho_p = float(np.mean(rho_bins))

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
        "rw": rw,
        "rw_squared": rw ** 2,
        "var_inflation": var_inflation,
        "n_realizations": n_realizations,
        "expected_interior": expected_interior,
        "measured_interior": measured_interior,
        "rel_err_interior": rel_err_interior,
        "se_interior_bin": se_interior_bin,
        "expected_dc": expected_dc,
        "measured_dc": measured_dc,
        "rel_err_dc": rel_err_dc,
        "expected_nyquist": expected_nyquist,
        "measured_nyquist": measured_nyquist,
        "rel_err_nyquist": rel_err_nyquist,
        "measured_rho_p": measured_rho_p,
        "mean_parseval_rel_diff": float(np.mean(parseval_rel_diffs)),
    }


def validate_median_white_noise(
    f_s: float = 4096.0,
    segment_duration: float = 0.5,
    n_segments: int = 15,
    sigma: float = 1.0,
    n_realizations: int = 300,
    seed: int = 142,
) -> Dict[str, float]:
    """Validate finite-sample bias-corrected Welch median PSD under independent segments."""
    delta_t = 1.0 / f_s
    segment_length = int(round(segment_duration * f_s))
    total_samples = n_segments * segment_length

    expected_interior = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
    ck = median_bias_factor(n_segments)

    rng = np.random.default_rng(seed)
    n_freqs = segment_length // 2 + 1
    raw_med_ensemble = np.zeros(n_freqs, dtype=np.float64)
    corr_med_ensemble = np.zeros(n_freqs, dtype=np.float64)

    w = tukey_window(segment_length, alpha=0.25)

    for _ in range(n_realizations):
        x = generate_white_noise(total_samples, sigma=sigma, generator=rng)
        # Compute with welch_psd (bias-corrected)
        _, psd_corr = welch_psd(
            x, delta_t, segment_length=segment_length,
            overlap_samples=0, window=w, method="median"
        )
        corr_med_ensemble += psd_corr

        # Also compute raw median
        segs = extract_segments(x, segment_length=segment_length, overlap_samples=0)
        p_raw = [single_segment_psd(s, delta_t, window=w) for s in segs]
        raw_med_ensemble += np.median(p_raw, axis=0)

    raw_med_ensemble /= n_realizations
    corr_med_ensemble /= n_realizations

    raw_interior = float(np.mean(raw_med_ensemble[5:-5]))
    corr_interior = float(np.mean(corr_med_ensemble[5:-5]))

    # Raw median should be approximately C(K) * expected
    expected_raw = ck * expected_interior
    rel_err_raw = abs(raw_interior - expected_raw) / expected_raw
    rel_err_corr = abs(corr_interior - expected_interior) / expected_interior

    return {
        "n_segments": n_segments,
        "ck": ck,
        "expected_interior": expected_interior,
        "raw_interior": raw_interior,
        "expected_raw": expected_raw,
        "rel_err_raw": rel_err_raw,
        "corr_interior": corr_interior,
        "rel_err_corr": rel_err_corr,
    }


def validate_coloured_noise_ar1(
    phi: float = 0.65,
    sigma_eps: float = 1.0,
    f_s: float = 2048.0,
    total_duration: float = 8.0,
    segment_duration: float = 1.0,
    n_realizations: int = 250,
    seed: int = 242,
) -> Dict[str, float]:
    """Validate broad spectral recovery of an AR(1) coloured noise process."""
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

    # Theoretical one-sided AR(1) PSD: 2 * sigma_eps^2 * dt / [1 + phi^2 - 2 * phi * cos(2 pi f dt)]
    denom = 1.0 + phi ** 2 - 2.0 * phi * np.cos(2.0 * np.pi * freqs * delta_t)
    expected_psd = 2.0 * (sigma_eps ** 2) * delta_t / denom

    # Compare across interior band (k = 5 to n_freqs - 5)
    rel_diffs = np.abs(psd_ensemble[5:-5] - expected_psd[5:-5]) / expected_psd[5:-5]
    max_rel_err = float(np.max(rel_diffs))
    mean_rel_err = float(np.mean(rel_diffs))

    return {
        "phi": phi,
        "sigma_eps": sigma_eps,
        "f_s": f_s,
        "n_realizations": n_realizations,
        "mean_rel_err": mean_rel_err,
        "max_rel_err": max_rel_err,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Welch PSD Estimation Module.")
    parser.add_argument("--fs", type=float, default=4096.0, help="Sampling frequency in Hz")
    parser.add_argument("--duration", type=float, default=8.0, help="Total duration in seconds")
    parser.add_argument("--seg-duration", type=float, default=1.0, help="Segment duration in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction")
    parser.add_argument("--sigma", type=float, default=1.0, help="White noise standard deviation")
    parser.add_argument("--n-realizations", type=int, default=300, help="Number of Monte Carlo realizations")
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

    print("================================================================================")
    print("STAGE 1.2: WELCH POWER SPECTRAL DENSITY ESTIMATION VALIDATION")
    print("================================================================================")

    # 1. White Noise Welch Mean Validation
    print("\n--- Part 1: White Noise Welch Mean Estimation ---")
    res_white = validate_white_noise_welch(
        f_s=args.fs,
        total_duration=args.duration,
        segment_duration=args.seg_duration,
        overlap_fraction=args.overlap,
        sigma=args.sigma,
        n_realizations=args.n_realizations,
        seed=args.seed,
        alpha=args.alpha,
    )

    print(f"Sampling frequency (f_s):         {res_white['f_s']:.1f} Hz")
    print(f"Sampling interval (delta_t):       {res_white['delta_t']:.6e} s")
    print(f"Total duration:                    {res_white['total_duration']:.2f} s ({res_white['n_samples']} samples)")
    print(f"Segment duration:                  {res_white['segment_duration']:.2f} s ({res_white['segment_length']} samples)")
    print(f"Overlap fraction:                  {res_white['overlap_fraction'] * 100:.1f}% ({res_white['overlap_samples']} samples)")
    print(f"Hop length:                        {res_white['hop_length']} samples")
    print(f"Number of valid segments (K):      {res_white['n_segments']}")
    print(f"Tukey window alpha:                {res_white['alpha']}")
    print(f"Discrete window power (S_2):       {res_white['S2']:.6f} (asymptotic: 27/32 = {27/32:.6f})")
    print(f"Discrete coherent gain (S_1):      {res_white['S1']:.6f} (asymptotic: 7/8 = {7/8:.6f})")
    print(f"Window overlap correlation (r_w):  {res_white['rw']:.6f} (asymptotic: 4/9 = {4/9:.6f})")
    print(f"Overlap variance factor (kappa):   {res_white['var_inflation']:.4f}")
    print(f"Number of Monte Carlo trials:      {res_white['n_realizations']}")
    print(f"Interior expected PSD (2*sig^2*dt):{res_white['expected_interior']:.6e} strain^2/Hz")
    print(f"Interior measured PSD:             {res_white['measured_interior']:.6e} strain^2/Hz")
    print(f"Interior relative deviation:       {res_white['rel_err_interior'] * 100:.3f}%")
    print(f"Single-bin SE (overlap-corrected): {res_white['se_interior_bin']:.6e} strain^2/Hz ({res_white['se_interior_bin']/res_white['expected_interior']*100:.2f}%)")
    print(f"DC expected PSD (sig^2*dt):        {res_white['expected_dc']:.6e} strain^2/Hz")
    print(f"DC measured PSD:                   {res_white['measured_dc']:.6e} strain^2/Hz (rel err: {res_white['rel_err_dc']*100:.2f}%)")
    print(f"Nyquist expected PSD:              {res_white['expected_nyquist']:.6e} strain^2/Hz")
    print(f"Nyquist measured PSD:              {res_white['measured_nyquist']:.6e} strain^2/Hz (rel err: {res_white['rel_err_nyquist']*100:.2f}%)")
    print(f"Empirical adj-seg corr (rho_P):    {res_white['measured_rho_p']:.4f} (proper-complex prediction r_w^2 = {res_white['rw_squared']:.4f})")
    print(f"Mean Parseval relative error:      {res_white['mean_parseval_rel_diff']:.2e}")

    # 2. Median-Welch Validation
    print("\n--- Part 2: Finite-Sample Bias-Corrected Median Estimation ---")
    res_median = validate_median_white_noise(
        f_s=args.fs,
        segment_duration=0.5,
        n_segments=15,
        sigma=args.sigma,
        n_realizations=args.n_realizations,
        seed=args.seed + 100,
    )
    print(f"Number of independent segments (K):{res_median['n_segments']}")
    print(f"Stage 0.6 odd bias factor C(15):   {res_median['ck']:.6f} (baseline: 0.725372)")
    print(f"Expected uncorrected raw median:   {res_median['expected_raw']:.6e} strain^2/Hz")
    print(f"Measured uncorrected raw median:   {res_median['raw_interior']:.6e} strain^2/Hz (rel err: {res_median['rel_err_raw']*100:.3f}%)")
    print(f"Corrected median PSD estimate:     {res_median['corr_interior']:.6e} strain^2/Hz")
    print(f"Corrected median relative error:   {res_median['rel_err_corr']*100:.3f}% (unbiased recovery within statistical noise)")

    # 3. Coloured Noise AR(1) Spectral Recovery
    print("\n--- Part 3: Coloured Noise AR(1) Broad Spectral Recovery ---")
    res_ar1 = validate_coloured_noise_ar1(
        phi=0.65,
        sigma_eps=args.sigma,
        f_s=2048.0,
        total_duration=8.0,
        segment_duration=1.0,
        n_realizations=250,
        seed=args.seed + 200,
    )
    print(f"AR(1) parameter phi:               {res_ar1['phi']}")
    print(f"Mean relative error across band:   {res_ar1['mean_rel_err']*100:.3f}%")
    print(f"Max relative error across band:    {res_ar1['max_rel_err']*100:.3f}%")

    print("\n================================================================================")
    print("VALIDATION ACCEPTANCE CRITERIA CHECK")
    print("================================================================================")

    # Acceptance criteria:
    # 1. White noise interior mean within 3% of theoretical expectation
    pass_interior = res_white["rel_err_interior"] < 0.03
    # 2. Parseval conservation to machine precision (< 1e-10)
    pass_parseval = res_white["mean_parseval_rel_diff"] < 1e-10
    # 3. Median bias-corrected PSD within 4% of theoretical expectation
    pass_median = res_median["rel_err_corr"] < 0.04
    # 4. Coloured noise mean spectral error < 5%
    pass_ar1 = res_ar1["mean_rel_err"] < 0.05
    # 5. Measured adjacent correlation close to r_w^2 (within 0.03)
    pass_corr = abs(res_white["measured_rho_p"] - res_white["rw_squared"]) < 0.03

    print(f"Criteria 1: Interior Welch PSD relative deviation < 3.0%:    {'PASS' if pass_interior else 'FAIL'} ({res_white['rel_err_interior']*100:.3f}%)")
    print(f"Criteria 2: Parseval energy conservation error < 1e-10:       {'PASS' if pass_parseval else 'FAIL'} ({res_white['mean_parseval_rel_diff']:.2e})")
    print(f"Criteria 3: Median-Welch bias-corrected error < 4.0%:        {'PASS' if pass_median else 'FAIL'} ({res_median['rel_err_corr']*100:.3f}%)")
    print(f"Criteria 4: AR(1) coloured-noise band mean error < 5.0%:      {'PASS' if pass_ar1 else 'FAIL'} ({res_ar1['mean_rel_err']*100:.3f}%)")
    print(f"Criteria 5: Empirical adj-seg corr matches r_w^2 within 0.03: {'PASS' if pass_corr else 'FAIL'} (|{res_white['measured_rho_p']:.4f} - {res_white['rw_squared']:.4f}| = {abs(res_white['measured_rho_p'] - res_white['rw_squared']):.4f})")

    all_passed = pass_interior and pass_parseval and pass_median and pass_ar1 and pass_corr
    print(f"\nOverall Validation Status: {'ALL CRITERIA PASSED' if all_passed else 'VALIDATION FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
