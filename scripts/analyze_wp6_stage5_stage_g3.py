#!/usr/bin/env python3
"""
Comprehensive offline analysis script for WP6 Stage 5G.3.
Analyzes posterior geometry, chirp coordinate xi, likelihood profiles,
weighted correlations, and density valleys across all Stage 5G.2 runs.
"""

import math
import json
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde

OUT_DIR = Path("reports/demonstrations/wp6_stage5/stage_g2")

RUN_IDS = [
    "base_01_plug_inst_s1",
    "base_01_plug_inst_s2",
    "base_01_stud_inst_s1",
    "base_01_stud_inst_s2",
    "diag_01_plug_nl1000_s1",
    "diag_01_plug_nl1000_s2",
    "diag_01_plug_rslice_s1",
    "diag_01_plug_rslice_s2",
    "diag_01_stud_nl1000_s1",
    "diag_01_stud_nl1000_s2",
    "diag_01_stud_rslice_s1",
    "diag_01_stud_rslice_s2",
]

def weighted_quantile(values, quantiles, sample_weight=None):
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def weighted_stats(x, w):
    norm_w = w / np.sum(w)
    mean = np.sum(norm_w * x)
    var = np.sum(norm_w * (x - mean) ** 2)
    std = np.sqrt(var)
    q025, q50, q975 = weighted_quantile(x, [0.025, 0.50, 0.975], sample_weight=w)
    return mean, std, q025, q50, q975

def weighted_corr(x, y, w):
    norm_w = w / np.sum(w)
    mx = np.sum(norm_w * x)
    my = np.sum(norm_w * y)
    cov = np.sum(norm_w * (x - mx) * (y - my))
    sx = np.sqrt(np.sum(norm_w * (x - mx) ** 2))
    sy = np.sqrt(np.sum(norm_w * (y - my) ** 2))
    return cov / (sx * sy) if (sx > 0 and sy > 0) else 0.0

def main():
    print("=" * 100)
    print("WP6 STAGE 5G.3 — POSTERIOR GEOMETRY & CHIRP COORDINATE OFFLINE ANALYSIS")
    print("=" * 100)

    results = {}

    for run_id in RUN_IDS:
        spath = OUT_DIR / f"{run_id}_samples.npz"
        jpath = OUT_DIR / f"{run_id}.json"
        
        sdata = np.load(spath)
        jdata = json.load(open(jpath))
        
        samples = sdata["samples"]  # (N, 4): Mc, eta, tc, phic
        weights = sdata["weights"]  # (N,)
        logl = sdata["logl"]        # (N,)
        d_vals = sdata["d_vals"]    # (N,)
        shells = sdata["shells"]    # (N,)
        
        mc = samples[:, 0]
        eta = samples[:, 1]
        tc = samples[:, 2]
        phic = samples[:, 3]
        
        # Chirp coordinate xi = Mc * eta^(-3/5) = M_total
        xi = mc * (eta ** (-0.6))
        
        # Weighted stats
        mc_mean, mc_std, mc_025, mc_med, mc_975 = weighted_stats(mc, weights)
        eta_mean, eta_std, eta_025, eta_med, eta_975 = weighted_stats(eta, weights)
        tc_mean, tc_std, tc_025, tc_med, tc_975 = weighted_stats(tc, weights)
        xi_mean, xi_std, xi_025, xi_med, xi_975 = weighted_stats(xi, weights)
        
        # Correlations
        corr_mc_eta = weighted_corr(mc, eta, weights)
        corr_xi_eta = weighted_corr(xi, eta, weights)
        corr_mc_tc = weighted_corr(mc, tc, weights)
        corr_eta_tc = weighted_corr(eta, tc, weights)
        
        # Regional mass partition at eta = 0.244
        high_mask = eta > 0.244
        low_mask = eta <= 0.244
        p_high = np.sum(weights[high_mask]) / np.sum(weights)
        p_low = np.sum(weights[low_mask]) / np.sum(weights)
        
        # Log-likelihood overlap
        logl_high_min = np.min(logl[high_mask]) if np.any(high_mask) else np.nan
        logl_high_max = np.max(logl[high_mask]) if np.any(high_mask) else np.nan
        logl_high_med = weighted_quantile(logl[high_mask], [0.50], sample_weight=weights[high_mask])[0] if np.any(high_mask) else np.nan
        
        logl_low_min = np.min(logl[low_mask]) if np.any(low_mask) else np.nan
        logl_low_max = np.max(logl[low_mask]) if np.any(low_mask) else np.nan
        logl_low_med = weighted_quantile(logl[low_mask], [0.50], sample_weight=weights[low_mask])[0] if np.any(low_mask) else np.nan
        
        # Weighted circular statistics for phic
        phi_circ = np.mod(phic, 2.0 * math.pi)
        C_bar = np.sum(weights * np.cos(phi_circ)) / np.sum(weights)
        S_bar = np.sum(weights * np.sin(phi_circ)) / np.sum(weights)
        R_bar = np.sqrt(C_bar ** 2 + S_bar ** 2)
        phi_mean_circ = math.atan2(S_bar, C_bar) % (2.0 * math.pi)
        
        # Weighted KDE along eta to check for density valley
        # Evaluate at grid across [0.230, 0.250]
        grid_eta = np.linspace(0.232, 0.250, 100)
        # Use resampled points according to weights for standard scipy KDE (10,000 resamples)
        rng = np.random.default_rng(42)
        norm_w = weights / np.sum(weights)
        idx_resamp = rng.choice(len(eta), size=20000, p=norm_w)
        eta_resamp = eta[idx_resamp]
        
        kde_scott = gaussian_kde(eta_resamp, bw_method='scott')
        kde_silverman = gaussian_kde(eta_resamp, bw_method='silverman')
        kde_half = gaussian_kde(eta_resamp, bw_method=kde_scott.factor * 0.5)
        
        dens_scott = kde_scott(grid_eta)
        dens_half = kde_half(grid_eta)
        
        # Check if local minimum exists in density between 0.240 and 0.246
        mask_inter = (grid_eta >= 0.240) & (grid_eta <= 0.246)
        min_dens_inter = np.min(dens_half[mask_inter])
        max_dens_high = np.max(dens_half[grid_eta > 0.246])
        max_dens_low = np.max(dens_half[grid_eta < 0.240])
        valley_ratio = min_dens_inter / max(max_dens_high, max_dens_low)
        
        results[run_id] = {
            "mc_med": mc_med, "mc_std": mc_std, "mc_ci95": [mc_025, mc_975],
            "eta_med": eta_med, "eta_std": eta_std, "eta_ci95": [eta_025, eta_975],
            "tc_med": tc_med, "tc_std": tc_std,
            "xi_med": xi_med, "xi_std": xi_std, "xi_ci95": [xi_025, xi_975],
            "corr_mc_eta": corr_mc_eta, "corr_xi_eta": corr_xi_eta,
            "corr_mc_tc": corr_mc_tc, "corr_eta_tc": corr_eta_tc,
            "p_high": p_high, "p_low": p_low,
            "logl_high_max": logl_high_max, "logl_low_max": logl_low_max,
            "logl_high_med": logl_high_med, "logl_low_med": logl_low_med,
            "R_bar": R_bar, "phi_mean_circ": phi_mean_circ,
            "valley_ratio": valley_ratio,
        }
        
        print(f"[{run_id}]")
        print(f"  Mc: {mc_med:.5f} +/- {mc_std:.5f} | 95% CI: [{mc_025:.5f}, {mc_975:.5f}]")
        print(f"  eta: {eta_med:.5f} +/- {eta_std:.5f} | 95% CI: [{eta_025:.5f}, {eta_975:.5f}]")
        print(f"  xi (M_tot): {xi_med:.5f} +/- {xi_std:.5f} | 95% CI: [{xi_025:.5f}, {xi_975:.5f}]")
        print(f"  Corr(Mc, eta) = {corr_mc_eta:+.4f}  vs  Corr(xi, eta) = {corr_xi_eta:+.4f}")
        print(f"  P(eta > 0.244) = {p_high*100:5.1f}% | P(eta <= 0.244) = {p_low*100:5.1f}%")
        print(f"  LogL: High max = {logl_high_max:.2f} (med {logl_high_med:.2f}) | Low max = {logl_low_max:.2f} (med {logl_low_med:.2f})")
        print(f"  Phase: R_bar = {R_bar:.4f}, mean_circ = {phi_mean_circ:.4f} rad")
        print(f"  Density Valley Ratio (min_inter / max_peak) = {valley_ratio:.4f}")
        print("-" * 80)

    # Save summary dictionary
    out_json = OUT_DIR / "stage_g3_offline_analysis.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis results saved to: {out_json}")

if __name__ == "__main__":
    main()
