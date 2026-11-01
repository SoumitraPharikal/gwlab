"""Validation script for one-sided white-noise periodogram normalization.

Stage 1.1: Discrete Fourier Conventions and Noise PSD Normalization.

This script executes a controlled numerical experiment generating an ensemble of
independent white Gaussian noise realizations and validating that:
1. The ensemble-averaged periodogram at interior bins matches S_0 = 2 * sigma^2 / f_s.
2. The DC bin (k = 0) matches S_0 / 2 = sigma^2 / f_s.
3. The Nyquist bin (k = N/2) matches S_0 / 2 = sigma^2 / f_s.
4. Total integrated periodogram matches time-domain signal power (Parseval consistency).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

# Ensure src is on sys.path if running as standalone script
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from gw_inference.conventions import (
    build_frequency_grid,
    expected_white_noise_psd,
    one_sided_periodogram,
)
from gw_inference.noise import generate_white_noise


def run_ensemble_validation(
    n_samples: int = 32768,
    f_s: float = 4096.0,
    sigma: float = 1.0,
    n_realizations: int = 1000,
    seed: int = 20261102,
) -> dict:
    """Run the white-noise periodogram ensemble validation.

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples per realization (default: 32768).
    f_s : float
        Sampling frequency in Hz (default: 4096.0).
    sigma : float
        Noise standard deviation in signal units (default: 1.0).
    n_realizations : int
        Number of independent ensemble realizations (default: 1000).
    seed : int
        Random seed for numpy.random.default_rng (default: 20261102).

    Returns
    -------
    dict
        Structured results containing numerical measurements and comparison metrics.
    """
    delta_t = 1.0 / f_s
    delta_f = 1.0 / (n_samples * delta_t)
    n_freqs = n_samples // 2 + 1
    freqs = build_frequency_grid(n_samples, delta_t)

    # Theoretical expected PSD values
    s_interior_expected = expected_white_noise_psd(sigma, delta_t, bin_type="interior")
    s_dc_expected = expected_white_noise_psd(sigma, delta_t, bin_type="dc")
    s_nyq_expected = expected_white_noise_psd(sigma, delta_t, bin_type="nyquist")

    rng = np.random.default_rng(seed)

    print("=" * 78)
    print("STAGE 1.1: WHITE-NOISE PERIODOGRAM ENSEMBLE VALIDATION")
    print("=" * 78)
    print(f"Sampling frequency (f_s) : {f_s:.1f} Hz (Delta t = {delta_t:.8e} s)")
    print(f"Sample count (N)        : {n_samples} (T = {n_samples * delta_t:.2f} s)")
    print(f"Frequency resolution    : {delta_f:.4f} Hz (Nyquist = {freqs[-1]:.1f} Hz)")
    print(f"Total frequency bins    : {n_freqs} (Interior bins: {n_freqs - 2})")
    print(f"Noise standard dev      : {sigma:.4f} (Variance = {sigma**2:.4f})")
    print(f"Ensemble realizations   : {n_realizations}")
    print(f"Random seed             : {seed}")
    print(f"Numerical precision     : IEEE 754 float64 (double precision)")
    print("-" * 78)

    t0 = time.perf_counter()

    # Accumulators for ensemble statistics
    psd_sum = np.zeros(n_freqs, dtype=np.float64)
    psd_sq_sum = np.zeros(n_freqs, dtype=np.float64)
    time_energy_sum = 0.0

    for i in range(n_realizations):
        noise = generate_white_noise(n_samples, sigma, generator=rng)
        psd = one_sided_periodogram(noise, delta_t)
        psd_sum += psd
        psd_sq_sum += psd ** 2
        time_energy_sum += np.sum(noise ** 2) * delta_t

        if (i + 1) % 200 == 0 or (i + 1) == n_realizations:
            elapsed = time.perf_counter() - t0
            print(f"  Processed {i + 1:4d} / {n_realizations} realizations ({elapsed:.2f} s)...")

    t_total = time.perf_counter() - t0

    # Ensemble statistics across realizations
    psd_mean = psd_sum / n_realizations
    # Sample variance across realizations for each frequency bin
    psd_var = (psd_sq_sum - n_realizations * (psd_mean ** 2)) / (n_realizations - 1)
    psd_sem = np.sqrt(psd_var / n_realizations)

    # Interior bins (1 <= k < N/2)
    interior_slice = slice(1, -1)
    interior_freqs = freqs[interior_slice]
    interior_means = psd_mean[interior_slice]
    interior_grand_mean = float(np.mean(interior_means))
    interior_grand_sem = float(np.sqrt(np.mean(psd_var[interior_slice]) / (n_realizations * (n_freqs - 2))))

    rel_diff_interior = (interior_grand_mean - s_interior_expected) / s_interior_expected

    # DC bin (k = 0)
    dc_mean = float(psd_mean[0])
    dc_sem = float(psd_sem[0])
    rel_diff_dc = (dc_mean - s_dc_expected) / s_dc_expected

    # Nyquist bin (k = N/2)
    nyq_mean = float(psd_mean[-1])
    nyq_sem = float(psd_sem[-1])
    rel_diff_nyq = (nyq_mean - s_nyq_expected) / s_nyq_expected

    # Parseval energy balance across ensemble
    mean_integrated_psd = float(np.sum(psd_mean) * delta_f)
    mean_time_power = float(time_energy_sum / (n_realizations * n_samples * delta_t))
    parseval_rel_diff = abs(mean_integrated_psd - mean_time_power) / mean_time_power

    print("-" * 78)
    print("RESULTS SUMMARY:")
    print("-" * 78)
    print(f"1. Interior Frequency Bins ({interior_freqs[0]:.3f} Hz to {interior_freqs[-1]:.3f} Hz, {len(interior_freqs)} bins):")
    print(f"   - Theoretical expectation S_0 = 2 sigma^2 / f_s : {s_interior_expected:.8e} [signal]^2/Hz")
    print(f"   - Measured ensemble grand mean                  : {interior_grand_mean:.8e} [signal]^2/Hz")
    print(f"   - Standard error of grand mean                  : {interior_grand_sem:.8e} [signal]^2/Hz")
    print(f"   - Relative deviation                            : {rel_diff_interior:+.4%}")
    print(f"   - Standardized residual (z-score)               : {rel_diff_interior * s_interior_expected / interior_grand_sem:+.2f} sigma")

    print(f"\n2. DC Frequency Bin (f = 0.0 Hz, k = 0):")
    print(f"   - Theoretical expectation S_DC = sigma^2 / f_s  : {s_dc_expected:.8e} [signal]^2/Hz")
    print(f"   - Measured ensemble mean                        : {dc_mean:.8e} [signal]^2/Hz")
    print(f"   - Standard error of mean                        : {dc_sem:.8e} [signal]^2/Hz")
    print(f"   - Relative deviation                            : {rel_diff_dc:+.4%}")
    print(f"   - Standardized residual (z-score)               : {rel_diff_dc * s_dc_expected / dc_sem:+.2f} sigma")

    print(f"\n3. Nyquist Frequency Bin (f = {freqs[-1]:.1f} Hz, k = {n_freqs - 1}):")
    print(f"   - Theoretical expectation S_Nyq = sigma^2 / f_s : {s_nyq_expected:.8e} [signal]^2/Hz")
    print(f"   - Measured ensemble mean                        : {nyq_mean:.8e} [signal]^2/Hz")
    print(f"   - Standard error of mean                        : {nyq_sem:.8e} [signal]^2/Hz")
    print(f"   - Relative deviation                            : {rel_diff_nyq:+.4%}")
    print(f"   - Standardized residual (z-score)               : {rel_diff_nyq * s_nyq_expected / nyq_sem:+.2f} sigma")

    print(f"\n4. Parseval Power Conservation:")
    print(f"   - Mean frequency-domain integrated PSD          : {mean_integrated_psd:.8e}")
    print(f"   - Mean time-domain mean-square power            : {mean_time_power:.8e}")
    print(f"   - Discrepancy                                   : {parseval_rel_diff:.2e}")
    print(f"\nTotal execution time: {t_total:.2f} seconds.")
    print("=" * 78)

    # Acceptance criteria
    # Interior mean relative error within 3 sigma of standard error (~ 0.08% for 1000 trials & 16383 bins)
    interior_pass = abs(rel_diff_interior) < 0.005  # < 0.5%
    # DC and Nyquist: each has standard error 1/sqrt(1000) ~ 3.16%; 3 sigma ~ 10%
    dc_pass = abs(rel_diff_dc) < 0.10
    nyq_pass = abs(rel_diff_nyq) < 0.10
    parseval_pass = parseval_rel_diff < 1e-12

    all_pass = interior_pass and dc_pass and nyq_pass and parseval_pass
    status_str = "PASSED" if all_pass else "FAILED"
    print(f"Validation status: {status_str}")
    print("=" * 78)

    return {
        "n_samples": n_samples,
        "f_s": f_s,
        "delta_t": delta_t,
        "delta_f": delta_f,
        "sigma": sigma,
        "n_realizations": n_realizations,
        "seed": seed,
        "s_interior_expected": s_interior_expected,
        "s_dc_expected": s_dc_expected,
        "s_nyq_expected": s_nyq_expected,
        "interior_grand_mean": interior_grand_mean,
        "interior_grand_sem": interior_grand_sem,
        "rel_diff_interior": rel_diff_interior,
        "dc_mean": dc_mean,
        "dc_sem": dc_sem,
        "rel_diff_dc": rel_diff_dc,
        "nyq_mean": nyq_mean,
        "nyq_sem": nyq_sem,
        "rel_diff_nyq": rel_diff_nyq,
        "parseval_rel_diff": parseval_rel_diff,
        "all_pass": all_pass,
        "execution_time_sec": t_total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1.1 White Noise PSD Ensemble Validation")
    parser.add_argument("--realizations", type=int, default=1000, help="Number of ensemble realizations")
    parser.add_argument("--seed", type=int, default=20261102, help="Random seed")
    args = parser.parse_args()

    results = run_ensemble_validation(n_realizations=args.realizations, seed=args.seed)
    if not results["all_pass"]:
        sys.exit(1)
