"""WP6 Stage 5 Stage B - Covariance and Pseudo-Covariance Mathematical Reconciliation.

This script executes the exact calculations requested by the supervisor:
1. Finite-window sums from tukey_window(8192, alpha=0.25)
2. Evaluation of all 4 covariance/pseudo-covariance terms across active band k, l in [80, 2048]
3. 2D finite-band variance double sum vs 1D lag-only stationary approximation
4. Block-level theoretical standard deviations with and without pseudo-terms
5. Comparison with empirical results from welch_stage_b_results.npz:
   - Empirical lag correlation mean and standard error across 50 runs
   - Omnibus chi-squared test across the 8 frequency blocks
"""

import json
from pathlib import Path
import numpy as np
import scipy.stats as stats

from gw_inference.psd import (
    tukey_window,
    window_power,
    window_coherent_gain,
    window_overlap_correlation,
    window_multi_lag_correlations,
    welch_variance_inflation,
)

def run_reconciliation():
    M = 8192
    alpha = 0.25
    H = 4096
    K = 31
    fs = 2048.0
    delta_t = 1.0 / fs
    delta_f = fs / M  # 0.25 Hz
    sigma = 1.0
    S0 = 2.0 * (sigma ** 2) * delta_t  # 9.765625e-4

    print("=" * 80)
    print("1. FINITE-WINDOW SUMS & OVERLAP CORRELATIONS")
    print("=" * 80)
    w = tukey_window(M, alpha=alpha)
    S1_actual = float(np.sum(w))
    S2_actual = float(np.sum(w ** 2))
    S1_mean = float(np.mean(w))
    S2_mean = float(np.mean(w ** 2))
    
    # Overlap sum
    overlap_sum_actual = float(np.sum(w[H:] * w[:-H]))
    r1_actual = overlap_sum_actual / S2_actual
    
    # Higher order lags
    r_lags_actual = window_multi_lag_correlations(w, hop_length=H, max_lag=5)
    
    # Idealized values
    S1_ideal = 7168.0
    S2_ideal = 6912.0
    S1_mean_ideal = 7.0 / 8.0
    S2_mean_ideal = 27.0 / 32.0
    overlap_sum_ideal = 3072.0
    r1_ideal = 4.0 / 9.0
    
    print(f"S1 sum:              {S1_actual:.16f} | Ideal: {S1_ideal:.16f} | Diff: {S1_actual - S1_ideal:+.2e}")
    print(f"S1 mean (S1/M):      {S1_mean:.16f} | Ideal 7/8: {S1_mean_ideal:.16f} | Diff: {S1_mean - S1_mean_ideal:+.2e}")
    print(f"S2 sum:              {S2_actual:.16f} | Ideal: {S2_ideal:.16f} | Diff: {S2_actual - S2_ideal:+.2e}")
    print(f"S2 mean (S2/M):      {S2_mean:.16f} | Ideal 27/32: {S2_mean_ideal:.16f} | Diff: {S2_mean - S2_mean_ideal:+.2e}")
    print(f"Overlap sum:         {overlap_sum_actual:.16f} | Ideal: {overlap_sum_ideal:.16f} | Diff: {overlap_sum_actual - overlap_sum_ideal:+.2e}")
    print(f"r1:                  {r1_actual:.16f} | Ideal 4/9: {r1_ideal:.16f} | Diff: {r1_actual - r1_ideal:+.2e}")
    for ell, r_val in enumerate(r_lags_actual, start=1):
        print(f"r_{ell}:                 {r_val:.16e}")
        
    kappa_actual = welch_variance_inflation(w, segment_length=M, overlap_samples=H, n_segments=K)
    kappa_ideal = 1.0 + 2.0 * (1.0 - 1.0 / K) * (r1_ideal ** 2)
    nu_eff_actual = 2.0 * K / kappa_actual
    nu_eff_ideal = 2.0 * K / kappa_ideal
    
    print(f"kappa_overlap:       {kappa_actual:.16f} | Ideal 1157/837: {kappa_ideal:.16f} | Diff: {kappa_actual - kappa_ideal:+.2e}")
    print(f"nu_eff:              {nu_eff_actual:.16f} | Ideal 51894/1157: {nu_eff_ideal:.16f} | Diff: {nu_eff_actual - nu_eff_ideal:+.2e}")
    
    print("\n" + "=" * 80)
    print("2. 2D COVARIANCE & PSEUDO-COVARIANCE EVALUATION ACROSS ACTIVE BAND")
    print("=" * 80)
    k_min = 80
    k_max = 2048
    active_bins = np.arange(k_min, k_max + 1)
    N_act = len(active_bins)
    print(f"Active band: k in [{k_min}, {k_max}], N_act = {N_act}")

    # Window FTs
    denom = S2_actual  # M * S2_mean = sum(w^2) = 6912.0
    w2 = w ** 2
    W2_fft = np.fft.fft(w2)
    
    w21_full = np.zeros(M, dtype=np.float64)
    w21_full[H:] = w[H:] * w[:-H]
    W21_fft = np.fft.fft(w21_full)

    # Precompute 1D normalized squared magnitude tables
    # diff index: d = (k - l) % M
    # sum index:  s = (k + l) % M
    W2_sq_norm = (np.abs(W2_fft) ** 2) / (denom ** 2)
    W21_sq_norm = (np.abs(W21_fft) ** 2) / (denom ** 2)

    # Build 2D matrices over active band
    # k_grid, l_grid of shape (N_act, N_act)
    K_grid, L_grid = np.meshgrid(active_bins, active_bins, indexing="ij")
    diff_idx = (K_grid - L_grid) % M
    sum_idx = (K_grid + L_grid) % M

    C0_ord = W2_sq_norm[diff_idx]
    C0_pseudo = W2_sq_norm[sum_idx]
    C1_ord = W21_sq_norm[diff_idx]
    C1_pseudo = W21_sq_norm[sum_idx]

    print("\n--- MATRIX NORMS ACROSS ACTIVE BAND (1969 x 1969) ---")
    print(f"C0_ord:        max = {np.max(C0_ord):.10e}, min = {np.min(C0_ord):.10e}, RMS = {np.sqrt(np.mean(C0_ord**2)):.10e}")
    print(f"C0_pseudo:     max = {np.max(C0_pseudo):.10e}, min = {np.min(C0_pseudo):.10e}, RMS = {np.sqrt(np.mean(C0_pseudo**2)):.10e}")
    print(f"C1_ord:        max = {np.max(C1_ord):.10e}, min = {np.min(C1_ord):.10e}, RMS = {np.sqrt(np.mean(C1_ord**2)):.10e}")
    print(f"C1_pseudo:     max = {np.max(C1_pseudo):.10e}, min = {np.min(C1_pseudo):.10e}, RMS = {np.sqrt(np.mean(C1_pseudo**2)):.10e}")

    # Check diagonal and near-diagonal
    diag_indices = np.arange(N_act)
    print("\n--- DIAGONAL VALUES (k == l) ---")
    print(f"C0_ord diag:           max = {np.max(np.diag(C0_ord)):.10f}, min = {np.min(np.diag(C0_ord)):.10f}")
    print(f"C1_ord diag (r1^2):    max = {np.max(np.diag(C1_ord)):.10f}, min = {np.min(np.diag(C1_ord)):.10f} (ideal 16/81 = {16/81:.10f})")
    print(f"C0_pseudo diag (2k):   max = {np.max(np.diag(C0_pseudo)):.10e}, min = {np.min(np.diag(C0_pseudo)):.10e}, mean = {np.mean(np.diag(C0_pseudo)):.10e}")
    print(f"C1_pseudo diag (2k):   max = {np.max(np.diag(C1_pseudo)):.10e}, min = {np.min(np.diag(C1_pseudo)):.10e}, mean = {np.mean(np.diag(C1_pseudo)):.10e}")

    print("\n--- NEAR-DIAGONAL VALUES (k - l = 1) ---")
    subdiag_C0_ord = np.diag(C0_ord, k=1)
    subdiag_C1_ord = np.diag(C1_ord, k=1)
    subdiag_C0_pseudo = np.diag(C0_pseudo, k=1)
    subdiag_C1_pseudo = np.diag(C1_pseudo, k=1)
    print(f"C0_ord lag-1:          max = {np.max(subdiag_C0_ord):.10e}, min = {np.min(subdiag_C0_ord):.10e}")
    print(f"C1_ord lag-1:          max = {np.max(subdiag_C1_ord):.10e}, min = {np.min(subdiag_C1_ord):.10e}")
    print(f"C0_pseudo lag-1:       max = {np.max(subdiag_C0_pseudo):.10e}, min = {np.min(subdiag_C0_pseudo):.10e}")
    print(f"C1_pseudo lag-1:       max = {np.max(subdiag_C1_pseudo):.10e}, min = {np.min(subdiag_C1_pseudo):.10e}")

    # Diagonal and near-diagonal ratios
    ratio_C0_diag = float(np.max(np.diag(C0_pseudo) / np.diag(C0_ord)))
    ratio_C1_diag = float(np.max(np.diag(C1_pseudo) / np.diag(C1_ord)))
    ratio_C0_lag1 = float(np.max(subdiag_C0_pseudo / subdiag_C0_ord))
    ratio_C1_lag1 = float(np.max(subdiag_C1_pseudo / subdiag_C1_ord))
    print(f"\nDiagonal ratio C0_pseudo / C0_ord: {ratio_C0_diag:.10e}")
    print(f"Diagonal ratio C1_pseudo / C1_ord: {ratio_C1_diag:.10e}")
    print(f"Lag-1 ratio C0_pseudo / C0_ord:    {ratio_C0_lag1:.10e}")
    print(f"Lag-1 ratio C1_pseudo / C1_ord:    {ratio_C1_lag1:.10e}")

    # Total covariance matrix
    # Cov(S_k, S_l) = (S0^2 / K) * [ C0_ord + C0_pseudo + 2(1 - 1/K)(C1_ord + C1_pseudo) ]
    # Note: K=31, weight_C0 = 1/K, weight_C1 = 2*(K-1)/K^2
    w_C0 = 1.0 / K
    w_C1 = 2.0 * (K - 1.0) / (K ** 2)

    Cov_full = (S0 ** 2) * (w_C0 * (C0_ord + C0_pseudo) + w_C1 * (C1_ord + C1_pseudo))
    Cov_ord_only = (S0 ** 2) * (w_C0 * C0_ord + w_C1 * C1_ord)
    Cov_pseudo_only = (S0 ** 2) * (w_C0 * C0_pseudo + w_C1 * C1_pseudo)

    print("\n" + "=" * 80)
    print("3. EXACT 2D FINITE-BAND VARIANCE VS 1D STATIONARY APPROXIMATION")
    print("=" * 80)
    var_band_2d_full = float(np.sum(Cov_full) / (N_act ** 2))
    var_band_2d_ord = float(np.sum(Cov_ord_only) / (N_act ** 2))
    var_band_2d_pseudo = float(np.sum(Cov_pseudo_only) / (N_act ** 2))

    std_band_2d_full = np.sqrt(var_band_2d_full)
    std_band_2d_ord = np.sqrt(var_band_2d_ord)
    rel_std_2d_full = std_band_2d_full / S0
    rel_std_2d_ord = std_band_2d_ord / S0

    print(f"Var(S_bar) 2D full:       {var_band_2d_full:.16e}")
    print(f"Var(S_bar) 2D ord-only:   {var_band_2d_ord:.16e}")
    print(f"Var(S_bar) 2D pseudo:     {var_band_2d_pseudo:.16e}")
    print(f"Rel Std 2D full:          {rel_std_2d_full * 100:.8f}%")
    print(f"Rel Std 2D ord-only:      {rel_std_2d_ord * 100:.8f}%")
    print(f"Difference (full - ord):  {(rel_std_2d_full - rel_std_2d_ord) * 100:.16e}%")

    # 1D lag-only stationary approximation
    # rho(d) = [C0_ord(d) + 2(1-1/K)C1_ord(d)] / kappa_overlap
    # Gamma = sum_{d = -(N_act-1)}^{N_act-1} (1 - |d|/N_act) rho(d)
    lags = np.arange(N_act)
    C0_lag = W2_sq_norm[lags]
    C1_lag = W21_sq_norm[lags]
    rho_lag = (C0_lag + 2.0 * (1.0 - 1.0 / K) * C1_lag) / kappa_actual

    # 1D double sum
    # Gamma = rho(0) + 2 * sum_{d=1}^{N_act-1} (1 - d/N_act) * rho(d)
    weights_1d = 1.0 - lags[1:] / float(N_act)
    Gamma_1d = float(rho_lag[0] + 2.0 * np.sum(weights_1d * rho_lag[1:]))
    var_single_bin = (S0 ** 2) * kappa_actual / K
    var_band_1d = (var_single_bin / N_act) * Gamma_1d
    rel_std_1d = np.sqrt(var_band_1d) / S0

    print(f"\n1D Gamma factor:          {Gamma_1d:.10f}")
    print(f"1D N_eff,freq:            {N_act / Gamma_1d:.6f}")
    print(f"Var(S_bar) 1D lag-only:   {var_band_1d:.16e}")
    print(f"Rel Std 1D lag-only:      {rel_std_1d * 100:.8f}%")
    print(f"2D full vs 1D lag diff:   {(rel_std_2d_full - rel_std_1d) * 100:+.16e}%")

    print("\n" + "=" * 80)
    print("4. BLOCK-LEVEL THEORETICAL STANDARD DEVIATIONS")
    print("=" * 80)
    # Define 8 blocks
    block_edges = [80, 326, 572, 818, 1064, 1310, 1556, 1802, 2049]
    block_results = []
    for b in range(8):
        start_idx = block_edges[b] - k_min
        end_idx = block_edges[b+1] - k_min
        n_b = end_idx - start_idx
        f_start = block_edges[b] * delta_f
        f_end = (block_edges[b+1] - 1) * delta_f
        
        # Submatrices
        cov_sub_full = Cov_full[start_idx:end_idx, start_idx:end_idx]
        cov_sub_ord = Cov_ord_only[start_idx:end_idx, start_idx:end_idx]
        
        var_b_full = float(np.sum(cov_sub_full) / (n_b ** 2))
        var_b_ord = float(np.sum(cov_sub_ord) / (n_b ** 2))
        
        rel_std_b_full = np.sqrt(var_b_full) / S0
        rel_std_b_ord = np.sqrt(var_b_ord) / S0
        
        # 1D stationary approximation for block
        lags_b = np.arange(n_b)
        weights_b = 1.0 - lags_b[1:] / float(n_b)
        Gamma_b = float(rho_lag[0] + 2.0 * np.sum(weights_b * rho_lag[1:n_b]))
        var_b_1d = (var_single_bin / n_b) * Gamma_b
        rel_std_b_1d = np.sqrt(var_b_1d) / S0
        
        block_results.append({
            "block": b + 1,
            "f_start": f_start,
            "f_end": f_end,
            "n_bins": n_b,
            "rel_std_2d_full": rel_std_b_full,
            "rel_std_2d_ord": rel_std_b_ord,
            "rel_std_1d": rel_std_b_1d,
            "diff_full_ord": rel_std_b_full - rel_std_b_ord,
            "diff_2d_1d": rel_std_b_full - rel_std_b_1d,
        })
        print(f"Block {b+1} ({f_start:.2f}-{f_end:.2f} Hz, {n_b} bins): 2D full = {rel_std_b_full*100:.6f}%, 2D ord = {rel_std_b_ord*100:.6f}%, 1D = {rel_std_b_1d*100:.6f}% (diff 2D-1D: {(rel_std_b_full-rel_std_b_1d)*100:+.2e}%)")

    print("\n" + "=" * 80)
    print("5. RECONCILIATION WITH SAVED EMPIRICAL RESULTS (welch_stage_b_results.npz)")
    print("=" * 80)
    results_path = Path("reports/demonstrations/wp6_stage5/stage_b/welch_stage_b_results.npz")
    npz_data = np.load(results_path)
    psd_realizations = npz_data["psd_realizations"]  # shape (50, 4097)
    active_psd = psd_realizations[:, active_bins]    # shape (50, 1969)
    n_runs = active_psd.shape[0]

    # Empirical lag correlations per realization
    # Compute autocorrelation across bins for each run
    max_corr_lag = 10
    autocorr_runs = np.zeros((n_runs, max_corr_lag + 1), dtype=np.float64)
    for r in range(n_runs):
        psd_r = active_psd[r]
        psd_mean = np.mean(psd_r)
        psd_dev = psd_r - psd_mean
        denom_r = np.sum(psd_dev ** 2)
        autocorr_runs[r, 0] = 1.0
        for d in range(1, max_corr_lag + 1):
            autocorr_runs[r, d] = np.sum(psd_dev[:-d] * psd_dev[d:]) / denom_r

    emp_mean_autocorr = np.mean(autocorr_runs, axis=0)
    emp_std_autocorr = np.std(autocorr_runs, axis=0, ddof=1)
    emp_se_autocorr = emp_std_autocorr / np.sqrt(n_runs)

    print("\n--- EMPIRICAL LAG CORRELATION WITH 50-RUN STANDARD ERROR ---")
    print("Lag d | Theory rho(d) | Emp Mean rho_hat(d) | Emp Std (run-to-run) | 50-Run SE | Residual | Residual / SE")
    for d in range(max_corr_lag + 1):
        th = rho_lag[d]
        em = emp_mean_autocorr[d]
        sd = emp_std_autocorr[d]
        se = emp_se_autocorr[d]
        res = em - th
        z_res = res / se if se > 0 else 0.0
        print(f"{d:5d} | {th*100:13.5f}% | {em*100:19.5f}% | {sd*100:20.5f}% | {se*100:9.5f}% | {res*100:+8.4f}% | {z_res:+6.2f} sigma")

    # Empirical block statistics and omnibus test
    print("\n--- EMPIRICAL BLOCK STATISTICS & STATISTICAL TESTING ---")
    block_emp_means = []
    block_z_scores = []
    for b in range(8):
        start_idx = block_edges[b] - k_min
        end_idx = block_edges[b+1] - k_min
        block_psd = active_psd[:, start_idx:end_idx]  # shape (50, n_b)
        block_band_means = np.mean(block_psd, axis=1) # shape (50,)
        
        emp_mean_b = float(np.mean(block_band_means))
        emp_std_b = float(np.std(block_band_means, ddof=1))
        
        th_std_b = block_results[b]["rel_std_2d_full"] * S0
        th_se_b = th_std_b / np.sqrt(n_runs)
        
        bias_b = emp_mean_b - S0
        rel_bias_b = bias_b / S0
        z_b = bias_b / th_se_b
        
        block_emp_means.append(emp_mean_b)
        block_z_scores.append(z_b)
        
        print(f"Block {b+1}: Emp Mean = {emp_mean_b:.8e} (bias {rel_bias_b*100:+.3f}%), Emp Std = {emp_std_b/S0*100:.3f}% (Th {th_std_b/S0*100:.3f}%), Z = {z_b:+.2f}")

    # Omnibus chi-square test across 8 blocks
    # Note: between blocks, are they independent?
    # Between block b and block b+1, only adjacent edge bins share correlation.
    # Total cross-block covariance:
    cross_cov_blocks = np.zeros((8, 8), dtype=np.float64)
    for b1 in range(8):
        s1 = block_edges[b1] - k_min
        e1 = block_edges[b1+1] - k_min
        n1 = e1 - s1
        for b2 in range(8):
            s2 = block_edges[b2] - k_min
            e2 = block_edges[b2+1] - k_min
            n2 = e2 - s2
            sub_cov = Cov_full[s1:e1, s2:e2]
            cross_cov_blocks[b1, b2] = np.sum(sub_cov) / (n1 * n2)

    # Convert to correlation matrix between block averages
    block_stds = np.sqrt(np.diag(cross_cov_blocks))
    block_corr = cross_cov_blocks / np.outer(block_stds, block_stds)
    print("\nBlock cross-correlation matrix (adjacent blocks):")
    for b in range(7):
        print(f"Corr(Block {b+1}, Block {b+2}) = {block_corr[b, b+1]*100:.4f}%")

    # Omnibus Mahalanobis / Chi-squared statistic
    # Under H0, Delta_b = mean_b - S0 ~ N(0, cross_cov_blocks / n_runs)
    Sigma_block_mean = cross_cov_blocks / float(n_runs)
    delta_vec = np.array(block_emp_means) - S0
    chi2_val = float(delta_vec @ np.linalg.inv(Sigma_block_mean) @ delta_vec)
    p_val_chi2 = float(1.0 - stats.chi2.cdf(chi2_val, df=8))
    
    # Also naive sum of Z^2
    sum_z2 = float(np.sum(np.array(block_z_scores) ** 2))
    p_val_naive = float(1.0 - stats.chi2.cdf(sum_z2, df=8))

    print(f"\nOmnibus Chi-Squared Test across all 8 blocks:")
    print(f"Chi2 statistic (full 8x8 covariance): {chi2_val:.4f} (df = 8)")
    print(f"p-value (full covariance):           {p_val_chi2:.4f}")
    print(f"Sum of Z_b^2 (diagonal assumption):   {sum_z2:.4f} (df = 8)")
    print(f"p-value (diagonal assumption):        {p_val_naive:.4f}")

    # Probability of max |Z| >= 2.79 under 8 independent tests
    max_z = np.max(np.abs(block_z_scores))
    p_single_tail = 2.0 * (1.0 - stats.norm.cdf(max_z))
    p_family_fwer = 1.0 - (1.0 - p_single_tail) ** 8
    print(f"Max |Z| across 8 blocks:             {max_z:.2f} (Block 8)")
    print(f"Single-test two-sided p-value:       {p_single_tail:.4f}")
    print(f"Family-wise error rate (FWER, k=8):  {p_family_fwer:.4f} ({p_family_fwer*100:.2f}%)")

    # Save reconciliation summary to JSON
    summary_out = {
        "fourier_normalization_convention": {
            "fft_type": "raw_unscaled_rfft",
            "fft_definition": "F_{m,k} = sum_{n=0}^{M-1} w_n x_{mH+n} e^{-2pi i kn/M}",
            "single_segment_psd_formula": "S_k = (2 * delta_t / (M * S_2)) * |F_{m,k}|^2",
            "theoretical_white_noise_psd_S0": "2 * sigma^2 * delta_t",
            "discrete_energy_denom": "M * S_2 = sum(w^2)"
        },
        "finite_window": {
            "S1_actual": S1_actual,
            "S1_ideal": S1_ideal,
            "S1_diff": S1_actual - S1_ideal,
            "S2_actual": S2_actual,
            "S2_ideal": S2_ideal,
            "S2_diff": S2_actual - S2_ideal,
            "overlap_sum_actual": overlap_sum_actual,
            "overlap_sum_ideal": overlap_sum_ideal,
            "overlap_sum_diff": overlap_sum_actual - overlap_sum_ideal,
            "r1_actual": r1_actual,
            "r1_ideal": r1_ideal,
            "r1_diff": r1_actual - r1_ideal,
            "kappa_actual": kappa_actual,
            "kappa_ideal": kappa_ideal,
            "nu_eff_actual": nu_eff_actual,
            "nu_eff_ideal": nu_eff_ideal,
        },
        "pseudo_covariance_bounds": {
            "C0_ord_max": float(np.max(C0_ord)),
            "C0_pseudo_max": float(np.max(C0_pseudo)),
            "C0_pseudo_rms": float(np.sqrt(np.mean(C0_pseudo**2))),
            "C1_ord_max": float(np.max(C1_ord)),
            "C1_pseudo_max": float(np.max(C1_pseudo)),
            "C1_pseudo_rms": float(np.sqrt(np.mean(C1_pseudo**2))),
            "ratio_C0_pseudo_ord_diag": ratio_C0_diag,
            "ratio_C1_pseudo_ord_diag": ratio_C1_diag,
            "ratio_C0_pseudo_ord_lag1": ratio_C0_lag1,
            "ratio_C1_pseudo_ord_lag1": ratio_C1_lag1,
        },
        "band_variance": {
            "var_band_2d_full": var_band_2d_full,
            "var_band_2d_ord": var_band_2d_ord,
            "var_band_2d_pseudo": var_band_2d_pseudo,
            "rel_std_2d_full": rel_std_2d_full,
            "rel_std_2d_ord": rel_std_2d_ord,
            "rel_std_1d": rel_std_1d,
            "Gamma_1d": Gamma_1d,
        },
        "block_results": block_results,
        "empirical_autocorr": {
            "lags": list(range(max_corr_lag + 1)),
            "theory": [float(rho_lag[d]) for d in range(max_corr_lag + 1)],
            "emp_mean": [float(emp_mean_autocorr[d]) for d in range(max_corr_lag + 1)],
            "emp_std": [float(emp_std_autocorr[d]) for d in range(max_corr_lag + 1)],
            "emp_se": [float(emp_se_autocorr[d]) for d in range(max_corr_lag + 1)],
        },
        "omnibus_test": {
            "chi2": chi2_val,
            "p_val_chi2": p_val_chi2,
            "sum_z2": sum_z2,
            "p_val_naive": p_val_naive,
            "max_z": float(max_z),
            "p_family_fwer": float(p_family_fwer),
        }
    }
    
    out_json_path = Path("reports/demonstrations/wp6_stage5/stage_b/welch_stage_b_reconciliation.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved reconciliation summary JSON to {out_json_path}")

if __name__ == "__main__":
    run_reconciliation()
