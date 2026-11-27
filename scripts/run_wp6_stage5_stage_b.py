#!/usr/bin/env python3
"""WP6 Stage 5 — Stage B: White-Noise Welch PSD Validation Campaign Runner.

Executes the pre-specified 50-realization Gaussian white-noise campaign to validate
the statistical behavior of the Welch PSD estimator against analytical predictions.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gw_inference.grid import FrequencyGrid
from gw_inference.noise import generate_white_noise
from gw_inference.psd import (
    discrete_pseudo_covariance_ratios,
    extract_segments,
    single_segment_psd,
    tukey_window,
    welch_psd,
    welch_variance_inflation,
    window_coherent_gain,
    window_multi_lag_correlations,
    window_overlap_correlation,
    window_power,
)


def run_stage_b_experiment() -> Dict[str, object]:
    """Execute Stage B experiment and compute comprehensive statistics."""
    print("======================================================================")
    print("WP6 Stage 5 — Stage B: White-Noise Welch PSD Validation Campaign")
    print("======================================================================")

    # 1. Configuration
    n_realizations = 50
    seeds = list(range(6101, 6151))
    f_s = 2048.0
    delta_t = 1.0 / f_s
    t_obs = 64.0
    n_total = int(t_obs * f_s)  # 131072
    m_seg = 8192
    t_seg = m_seg * delta_t  # 4.0 s
    overlap = 4096  # 50%
    hop = m_seg - overlap  # 4096
    alpha = 0.25
    sigma = 1.0
    s_0_true = 2.0 * (sigma ** 2) * delta_t  # 9.765625e-4

    f_start = 20.0
    f_end = 512.0
    delta_f = 1.0 / t_seg  # 0.25 Hz

    grid = FrequencyGrid.from_duration_and_sampling_rate(duration=t_seg, sampling_rate=f_s)
    mask = grid.create_active_mask(f_start=f_start, f_end=f_end)
    active_indices = grid.get_active_indices(mask)
    n_freq = grid.n_frequencies  # 4097
    n_act = len(active_indices)  # 1969
    active_freqs = grid.frequencies[active_indices]

    n_segments_expected = (n_total - overlap) // hop  # 31

    print(f"Sampling frequency: {f_s} Hz (delta_t = {delta_t:.6e} s)")
    print(f"Total duration: {t_obs} s ({n_total} samples)")
    print(f"Segment duration: {t_seg} s ({m_seg} samples), overlap = {overlap} samples (50%)")
    print(f"Number of segments per realization: {n_segments_expected}")
    print(f"Active band: [{f_start}, {f_end}] Hz (delta_f = {delta_f} Hz)")
    print(f"Full grid bins: {n_freq}, Active bins: {n_act}")
    print(f"Theoretical one-sided white noise PSD S_0 = {s_0_true:.10e} strain^2/Hz")
    print(f"Number of independent realizations: {n_realizations} (seeds {seeds[0]}..{seeds[-1]})")
    print("----------------------------------------------------------------------")

    # 2. Window and Overlap Math
    w = tukey_window(m_seg, alpha=alpha)
    s_2 = window_power(w)  # mean(w^2) = 27/32 = 0.84375
    s_1 = window_coherent_gain(w)  # mean(w) = 7/8 = 0.875
    ms2 = np.sum(w ** 2)  # 6912.0

    r1_exact = window_overlap_correlation(w, overlap)  # 4/9
    r_lags = window_multi_lag_correlations(w, hop_length=hop, max_lag=n_segments_expected - 1)
    kappa_exact = welch_variance_inflation(w, m_seg, overlap, n_segments_expected)  # 1157/837
    nu_eff_exact = 2.0 * n_segments_expected / kappa_exact  # 44.8522
    sigma_rel_single_theoretical = math.sqrt(2.0 / nu_eff_exact)  # sqrt(kappa/31) ~ 0.2111656
    sigma_rel_ens50_theoretical = sigma_rel_single_theoretical / math.sqrt(n_realizations)  # ~ 0.029863

    print(f"Window: discrete Tukey (alpha = {alpha}, length = {m_seg})")
    print(f"Window power S_2: {s_2:.10f} (exact 27/32 = {27/32:.10f})")
    print(f"Window coherent gain S_1: {s_1:.10f} (exact 7/8 = {7/8:.10f})")
    print(f"Adjacent-segment overlap correlation r_1: {r1_exact:.10f} (exact 4/9 = {4/9:.10f})")
    print(f"Multi-lag correlations: len = {len(r_lags)}, values = {r_lags}")
    print(f"Welch variance inflation factor kappa_overlap: {kappa_exact:.10f} (exact 1157/837 = {1157/837:.10f})")
    print(f"Effective degrees of freedom nu_eff: {nu_eff_exact:.10f}")
    print(f"Theoretical per-bin relative std (single realization): {sigma_rel_single_theoretical * 100:.4f}%")
    print(f"Theoretical per-bin relative std error (50-ensemble mean): {sigma_rel_ens50_theoretical * 100:.4f}%")
    print("----------------------------------------------------------------------")

    # 3. Frequency-Bin Covariance Derivation
    w2 = w ** 2
    w2_fft = np.fft.fft(w2)
    c0_lags = np.abs(w2_fft) ** 2 / (ms2 ** 2)

    # Cross-segment overlap product w_n * w_{n-H}
    prod_w = np.zeros(m_seg, dtype=np.float64)
    prod_w[overlap:] = w[overlap:] * w[: m_seg - overlap]
    w2_1_fft = np.fft.fft(prod_w)
    c1_lags = np.abs(w2_1_fft) ** 2 / (ms2 ** 2)

    # Full cross-frequency normalized correlation rho(d)
    k_float = float(n_segments_expected)
    cov_structure = (c0_lags + 2.0 * (1.0 - 1.0 / k_float) * c1_lags) / kappa_exact
    rho_theoretical = cov_structure[:100]  # first 100 lags

    # Pseudo-covariance inspection
    pseudo_dict = discrete_pseudo_covariance_ratios(w, overlap_samples=overlap)
    max_p11_ratio = float(np.max(pseudo_dict["ratio_P11"][active_indices]))
    max_p12_ratio = float(np.max(pseudo_dict["ratio_P12"][active_indices]))
    max_rel_diff_pseudo = float(np.max(np.abs(pseudo_dict["rel_diff"][active_indices])))

    print(f"Max active pseudo-covariance |P11|/C11: {max_p11_ratio:.6e}")
    print(f"Max active cross pseudo-covariance |P12|/|C12|: {max_p12_ratio:.6e}")
    print(f"Max relative difference (rho_full - rw^2)/rw^2: {max_rel_diff_pseudo:.6e}")
    print(f"Theoretical frequency correlation rho(0) = {rho_theoretical[0]:.6f}")
    print(f"Theoretical frequency correlation rho(1) = {rho_theoretical[1]:.6f} ({rho_theoretical[1]*100:.2f}%)")
    print(f"Theoretical frequency correlation rho(2) = {rho_theoretical[2]:.6f} ({rho_theoretical[2]*100:.2f}%)")
    print(f"Theoretical frequency correlation rho(3) = {rho_theoretical[3]:.6f} ({rho_theoretical[3]*100:.2f}%)")

    # Gamma factor for band average
    gamma_band_exact = float(
        rho_theoretical[0]
        + 2.0 * np.sum([(1.0 - d / n_act) * rho_theoretical[d] for d in range(1, len(rho_theoretical))])
    )
    n_eff_freq_exact = n_act / gamma_band_exact
    sigma_band_single_theoretical = math.sqrt(
        (kappa_exact / k_float) * (gamma_band_exact / n_act)
    )  # rel std of band mean
    sigma_band_ens50_theoretical = sigma_band_single_theoretical / math.sqrt(n_realizations)

    print(f"Band correlation sum Gamma: {gamma_band_exact:.6f}")
    print(f"Frequency-bin effective degrees of freedom N_eff,freq: {n_eff_freq_exact:.2f}")
    print(f"Theoretical band-average rel std (single run): {sigma_band_single_theoretical * 100:.4f}%")
    print(f"Theoretical band-average rel std (50-ensemble mean): {sigma_band_ens50_theoretical * 100:.4f}%")
    print("----------------------------------------------------------------------")

    # 4. Predeclared Frequency Blocks (8 blocks)
    n_blocks = 8
    block_size_base = n_act // n_blocks  # 246
    remainder = n_act % n_blocks  # 1

    block_defs: List[Dict[str, object]] = []
    idx_cursor = 0
    for b in range(n_blocks):
        b_len = block_size_base + (1 if b == n_blocks - 1 else 0)
        b_indices = active_indices[idx_cursor : idx_cursor + b_len]
        b_freqs = grid.frequencies[b_indices]
        gamma_b = float(
            rho_theoretical[0]
            + 2.0 * np.sum([(1.0 - d / b_len) * rho_theoretical[d] for d in range(1, min(len(rho_theoretical), b_len))])
        )
        sigma_b_single = math.sqrt((kappa_exact / k_float) * (gamma_b / b_len))
        sigma_b_ens50 = sigma_b_single / math.sqrt(n_realizations)

        block_defs.append({
            "block_idx": b + 1,
            "f_min": float(b_freqs[0]),
            "f_max": float(b_freqs[-1]),
            "n_bins": int(b_len),
            "indices": b_indices,
            "gamma": gamma_b,
            "sigma_single_rel": sigma_b_single,
            "sigma_ens50_rel": sigma_b_ens50,
        })
        idx_cursor += b_len

    print("Predeclared Frequency Blocks:")
    for bd in block_defs:
        print(f"  Block {bd['block_idx']}: [{bd['f_min']:.2f}, {bd['f_max']:.2f}] Hz, "
              f"N_bins = {bd['n_bins']}, sigma_single = {bd['sigma_single_rel']*100:.3f}%, "
              f"sigma_ens50 = {bd['sigma_ens50_rel']*100:.3f}%")
    print("----------------------------------------------------------------------")

    # 5. Execute 50 Realizations
    print(f"Generating and estimating Welch PSD across {n_realizations} realizations...")
    psd_realizations = np.empty((n_realizations, n_freq), dtype=np.float64)
    seed_manifest: List[Dict[str, object]] = []

    for r_idx, seed in enumerate(seeds):
        # Generate white noise
        x = generate_white_noise(n_samples=n_total, sigma=sigma, seed=seed)
        # Verify finite and variance
        var_x = float(np.var(x))

        # Estimate Welch PSD
        freqs_est, psd_est = welch_psd(
            x,
            delta_t=delta_t,
            segment_length=m_seg,
            overlap_samples=overlap,
            window="tukey",
            method="mean",
        )

        # Basic validations
        assert np.array_equal(freqs_est, grid.frequencies), "Frequency grid mismatch"
        assert np.all(np.isfinite(psd_est[active_indices])), f"Non-finite PSD in seed {seed}"
        assert np.all(psd_est[active_indices] > 0.0), f"Non-positive PSD in seed {seed}"

        psd_realizations[r_idx] = psd_est

        band_mean = float(np.mean(psd_est[active_indices]))
        rel_err = (band_mean - s_0_true) / s_0_true

        seed_manifest.append({
            "run_idx": r_idx + 1,
            "seed": seed,
            "sample_var": var_x,
            "band_mean_psd": band_mean,
            "band_mean_rel_err": rel_err,
        })

        if (r_idx + 1) % 10 == 0 or r_idx == 0:
            print(f"  Completed realization {r_idx + 1:2d}/50 (seed {seed}): "
                  f"band mean = {band_mean:.6e}, rel err = {rel_err * 100:+.3f}%")

    print("----------------------------------------------------------------------")
    print("Computing ensemble statistics across 50 realizations...")

    # 6. Ensemble Analysis
    ens_mean_psd = np.mean(psd_realizations, axis=0)  # (4097,)
    ens_std_psd = np.std(psd_realizations, axis=0, ddof=1)  # (4097,)
    ens_rel_std_psd = ens_std_psd / ens_mean_psd  # (4097,)

    act_mean_psd = ens_mean_psd[active_indices]
    act_std_psd = ens_std_psd[active_indices]
    act_rel_std = ens_rel_std_psd[active_indices]

    # Global active band metrics
    mean_active_psd = float(np.mean(act_mean_psd))
    global_bias_rel = (mean_active_psd - s_0_true) / s_0_true
    mean_empirical_rel_std = float(np.mean(act_rel_std))
    std_empirical_rel_std = float(np.std(act_rel_std, ddof=1))

    # Realization band means
    realization_band_means = np.mean(psd_realizations[:, active_indices], axis=1)  # (50,)
    emp_band_mean = float(np.mean(realization_band_means))
    emp_band_std = float(np.std(realization_band_means, ddof=1))
    emp_band_rel_std = emp_band_std / s_0_true
    emp_band_ens50_rel_std = emp_band_rel_std / math.sqrt(n_realizations)

    print(f"Theoretical S_0: {s_0_true:.10e}")
    print(f"Ensemble active-band mean PSD: {mean_active_psd:.10e} (bias = {global_bias_rel * 100:+.4f}%)")
    print(f"Theoretical single-run relative std: {sigma_rel_single_theoretical * 100:.4f}%")
    print(f"Empirical per-bin relative std (mean +/- std across 1969 bins): "
          f"{mean_empirical_rel_std * 100:.4f}% +/- {std_empirical_rel_std * 100:.4f}%")
    print(f"Theoretical band-average rel std: {sigma_band_single_theoretical * 100:.4f}%")
    print(f"Empirical band-average rel std across 50 runs: {emp_band_rel_std * 100:.4f}%")
    print(f"Ensemble mean band-average relative error: {((emp_band_mean - s_0_true) / s_0_true) * 100:+.4f}% "
          f"(uncertainty +/- {emp_band_ens50_rel_std * 100:.4f}%)")
    print("----------------------------------------------------------------------")

    # 7. Empirical Frequency Covariance
    # Centered active realizations: shape (50, 1969)
    act_reals = psd_realizations[:, active_indices]  # (50, 1969)
    act_devs = act_reals - np.mean(act_reals, axis=0, keepdims=True)  # (50, 1969)

    # Autocorrelation over lags d = 0..20
    emp_rho = np.empty(21, dtype=np.float64)
    cov_diag = np.mean(act_devs ** 2, axis=0)  # (1969,) variance per bin

    for d in range(21):
        if d == 0:
            emp_rho[d] = 1.0
        else:
            # Pair bins (k, k+d)
            v1 = act_devs[:, :-d]  # (50, 1969 - d)
            v2 = act_devs[:, d:]   # (50, 1969 - d)
            s1 = np.sqrt(cov_diag[:-d])
            s2 = np.sqrt(cov_diag[d:])
            cross_cov = np.sum(v1 * v2, axis=0) / 49.0  # (1969 - d,)
            corr_pairs = cross_cov / (s1 * s2)
            emp_rho[d] = float(np.mean(corr_pairs))

    print("Frequency-Bin Correlation (Empirical vs Theoretical):")
    print("  Lag d | Empirical rho(d) | Theoretical rho(d) | Discrepancy")
    print("  ------+------------------+--------------------+------------")
    for d in range(11):
        print(f"   {d:2d}   |     {emp_rho[d]*100:6.3f}%      |      {rho_theoretical[d]*100:6.3f}%      | { (emp_rho[d] - rho_theoretical[d])*100:+6.3f}%")
    print("----------------------------------------------------------------------")

    # 8. Block Summary
    block_results: List[Dict[str, object]] = []
    for bd in block_defs:
        b_idx = bd["block_idx"]
        b_indices = bd["indices"]
        reals_b = np.mean(psd_realizations[:, b_indices], axis=1)  # (50,)
        mean_b = float(np.mean(reals_b))
        std_b = float(np.std(reals_b, ddof=1))
        rel_std_b = std_b / s_0_true
        bias_b = (mean_b - s_0_true) / s_0_true
        z_b = (mean_b - s_0_true) / (bd["sigma_ens50_rel"] * s_0_true)

        block_results.append({
            "block_idx": b_idx,
            "f_min": bd["f_min"],
            "f_max": bd["f_max"],
            "n_bins": bd["n_bins"],
            "mean_psd": mean_b,
            "rel_bias_pct": bias_b * 100.0,
            "theo_single_rel_pct": bd["sigma_single_rel"] * 100.0,
            "emp_single_rel_pct": rel_std_b * 100.0,
            "theo_ens50_rel_pct": bd["sigma_ens50_rel"] * 100.0,
            "z_score_ens50": z_b,
        })

    print("Block Statistics (50 Realizations):")
    print("  Blk | Band [Hz]     | N_bins | Emp Mean PSD | Bias [%] | Theo Std [%] | Emp Std [%] | Z-score")
    print("  ----+---------------+--------+--------------+----------+--------------+-------------+--------")
    for br in block_results:
        print(f"   {br['block_idx']}  | {br['f_min']:5.1f}–{br['f_max']:5.1f} |  {br['n_bins']:4d}  | {br['mean_psd']:.6e} |  {br['rel_bias_pct']:+6.3f}% |    {br['theo_single_rel_pct']:6.3f}%   |   {br['emp_single_rel_pct']:6.3f}%   | {br['z_score_ens50']:+5.2f}")
    print("======================================================================")

    # 9. Output Directory and Artifacts
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "reports", "demonstrations", "wp6_stage5", "stage_b")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save NPZ
    npz_path = os.path.join(output_dir, "welch_stage_b_results.npz")
    np.savez_compressed(
        npz_path,
        frequencies=grid.frequencies,
        active_indices=active_indices,
        active_frequencies=active_freqs,
        psd_realizations=psd_realizations,
        psd_ensemble_mean=ens_mean_psd,
        psd_ensemble_std=ens_std_psd,
        psd_relative_std=ens_rel_std_psd,
        theoretical_psd=s_0_true,
        theoretical_relative_std=sigma_rel_single_theoretical,
        theoretical_ens50_relative_std=sigma_rel_ens50_theoretical,
        r_lags=r_lags,
        kappa_overlap=kappa_exact,
        nu_eff=nu_eff_exact,
        empirical_autocorr_lags=emp_rho,
        theoretical_autocorr_lags=rho_theoretical[:21],
        realization_band_means=realization_band_means,
        seeds=np.array(seeds, dtype=np.int64),
    )
    print(f"Saved NPZ results to: {npz_path}")

    # Save manifest JSON
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_data = {
        "campaign": "WP6 Stage 5 — Stage B White-Noise Welch Validation",
        "configuration": {
            "n_realizations": n_realizations,
            "seed_range": [seeds[0], seeds[-1]],
            "sampling_frequency_hz": f_s,
            "delta_t_s": delta_t,
            "segment_length_samples": m_seg,
            "segment_duration_s": t_seg,
            "overlap_samples": overlap,
            "overlap_fraction": 0.5,
            "n_segments_per_realization": n_segments_expected,
            "total_samples": n_total,
            "total_duration_s": t_obs,
            "window": "tukey",
            "alpha": alpha,
            "delta_f_hz": delta_f,
            "active_band_hz": [f_start, f_end],
            "n_frequencies_full": n_freq,
            "n_frequencies_active": n_act,
            "noise_sigma": sigma,
            "theoretical_s_0": s_0_true,
        },
        "analytical_validation": {
            "window_power_S2": s_2,
            "window_coherent_gain_S1": s_1,
            "r_1_exact": r1_exact,
            "r_lags": list(r_lags),
            "kappa_overlap": kappa_exact,
            "nu_eff": nu_eff_exact,
            "sigma_rel_single_theoretical": sigma_rel_single_theoretical,
            "sigma_rel_ens50_theoretical": sigma_rel_ens50_theoretical,
            "max_active_pseudo_covariance_P11_ratio": max_p11_ratio,
            "max_active_cross_pseudo_covariance_P12_ratio": max_p12_ratio,
            "max_rel_diff_pseudo": max_rel_diff_pseudo,
            "theoretical_correlation_lags_0_to_5": list(rho_theoretical[:6]),
            "gamma_band_exact": gamma_band_exact,
            "n_eff_freq_exact": n_eff_freq_exact,
            "sigma_band_single_theoretical": sigma_band_single_theoretical,
            "sigma_band_ens50_theoretical": sigma_band_ens50_theoretical,
        },
        "ensemble_results": {
            "global_active_mean_psd": mean_active_psd,
            "global_active_bias_pct": global_bias_rel * 100.0,
            "mean_empirical_rel_std_pct": mean_empirical_rel_std * 100.0,
            "std_empirical_rel_std_pct": std_empirical_rel_std * 100.0,
            "empirical_band_mean_psd": emp_band_mean,
            "empirical_band_mean_bias_pct": ((emp_band_mean - s_0_true) / s_0_true) * 100.0,
            "empirical_band_single_rel_std_pct": emp_band_rel_std * 100.0,
            "empirical_band_ens50_rel_std_pct": emp_band_ens50_rel_std * 100.0,
            "empirical_correlation_lags_0_to_5": list(emp_rho[:6]),
        },
        "blocks": block_results,
        "runs": seed_manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved manifest to: {manifest_path}")

    # Save summary CSV
    summary_csv_path = os.path.join(output_dir, "welch_stage_b_summary.csv")
    rep_indices = [
        active_indices[0],                     # 20.0 Hz
        active_indices[n_act // 8],            # ~81.5 Hz
        active_indices[n_act // 4],            # ~143.0 Hz
        active_indices[n_act // 2],            # ~266.0 Hz
        active_indices[3 * n_act // 4],        # ~389.0 Hz
        active_indices[-1],                    # 512.0 Hz
    ]
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        f.write("frequency_hz,bin_index,theoretical_psd,ens_mean_psd,rel_bias_pct,emp_rel_std_pct,theo_rel_std_pct,z_score_ens50\n")
        for idx in rep_indices:
            freq = grid.frequencies[idx]
            mean_v = ens_mean_psd[idx]
            bias_pct = ((mean_v - s_0_true) / s_0_true) * 100.0
            emp_rel = ens_rel_std_psd[idx] * 100.0
            theo_rel = sigma_rel_single_theoretical * 100.0
            z_score = (mean_v - s_0_true) / (sigma_rel_ens50_theoretical * s_0_true)
            f.write(f"{freq:.2f},{idx},{s_0_true:.10e},{mean_v:.10e},{bias_pct:+.4f},{emp_rel:.4f},{theo_rel:.4f},{z_score:+.3f}\n")
    print(f"Saved summary CSV to: {summary_csv_path}")

    # Save blocks CSV
    blocks_csv_path = os.path.join(output_dir, "welch_stage_b_blocks.csv")
    with open(blocks_csv_path, "w", encoding="utf-8") as f:
        f.write("block_idx,f_min_hz,f_max_hz,n_bins,ens_mean_psd,rel_bias_pct,theo_single_rel_std_pct,emp_single_rel_std_pct,theo_ens50_rel_std_pct,z_score_ens50\n")
        for br in block_results:
            f.write(f"{br['block_idx']},{br['f_min']:.2f},{br['f_max']:.2f},{br['n_bins']},{br['mean_psd']:.10e},"
                    f"{br['rel_bias_pct']:+.4f},{br['theo_single_rel_pct']:.4f},{br['emp_single_rel_pct']:.4f},"
                    f"{br['theo_ens50_rel_pct']:.4f},{br['z_score_ens50']:+.3f}\n")
    print(f"Saved blocks CSV to: {blocks_csv_path}")

    return manifest_data


if __name__ == "__main__":
    run_stage_b_experiment()
