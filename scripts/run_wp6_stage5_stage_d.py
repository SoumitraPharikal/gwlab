#!/usr/bin/env python3
"""WP6 Stage 5D — Authorized Execution: Multi-Point Likelihood-Geometry Screening.

This script executes the authorized Stage D likelihood diagnostic:
1. Mandatory preflight checks (Git branch, HEAD, status, SHA-256 hashes, library versions).
2. Deterministic regeneration and hard-stop validation of canonical on-source data (run_2B_01).
3. Deterministic extraction and validation of the 25 stratified posterior points and 3 anchors.
4. Precomputation of frequency-domain waveforms and evaluation of analytical reference likelihood.
5. 50-realization off-source colored-noise Welch PSD estimation (seeds 6201–6250).
6. 1,400 empirical-PSD likelihood evaluations with strict numerical validation:
   - |D_j(theta_inj)| <= 1e-12
   - Normalization cancellation identity error <= 1e-10
   - Finiteness of all log-likelihoods and intermediate terms
   - Strict positivity and finiteness of PSDs on active band
7. Statistical analysis:
   - Per-point ensemble mean, SEM, and 5th/25th/50th/75th/95th percentiles
   - Per-realization spatial mean and spatial spread (sample std dev across 25 points)
   - Across-realization spatial mean and spatial spread means and SEMs
   - Per-realization and ensemble OLS regression against selected-sample reference likelihood depth
   - Anchor perturbations (posterior mean, maximum weight) and global maximum |D_bar|
8. Generation of machine-readable manifest.json, results.json, and summary.json.

Execution boundary: Stage D ONLY.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure src/ is on sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import scipy

# Internal library imports
from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.conventions import continuous_rfft
from gw_inference.grid import FrequencyGrid
from gw_inference.likelihood import FrequencyDomainGaussianLikelihood
from gw_inference.noise import generate_colored_noise
from gw_inference.psd import analytical_detector_psd, welch_psd
from gw_inference.waveform import taylorf2_fd_waveform, apply_coalescence_time_shift


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hexadecimal digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_git_command(args: List[str]) -> str:
    """Execute a git command and return stripped stdout."""
    res = subprocess.run(["git"] + args, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def run_preflight() -> Dict[str, Any]:
    """Execute mandatory preflight environment and code-provenance checks."""
    print("=" * 70)
    print("SECTION 1: MANDATORY PREFLIGHT & PROVENANCE CHECKS")
    print("=" * 70)

    branch = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    commit_head = run_git_command(["rev-parse", "HEAD"])
    git_status = run_git_command(["status", "--short"])

    print(f"Git Branch:      {branch}")
    print(f"Git HEAD Commit: {commit_head}")
    print(f"Python Version:  {sys.version.split()[0]}")
    print(f"NumPy Version:   {np.__version__}")
    print(f"SciPy Version:   {scipy.__version__}")

    # Check required input files
    input_files = [
        "src/gw_inference/waveform.py",
        "src/gw_inference/noise.py",
        "src/gw_inference/psd.py",
        "src/gw_inference/likelihood.py",
        "reports/demonstrations/wp6_stage4/tier2/run_2B_01_results.npz",
        "reports/demonstrations/wp6_stage4/tier2/run_2B_01_diagnostics.json",
        "reports/demonstrations/wp6_stage5/stage_c/manifest.json",
        "reports/demonstrations/wp6_stage5/stage_c/results.json",
    ]

    hashes = {}
    print("\nFile SHA-256 Provenance Hashes:")
    for path in input_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input file missing: {path}")
        h = compute_file_sha256(path)
        hashes[path] = h
        print(f"  {h}  {path}")

    preflight_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "git_head": commit_head,
        "git_status_short": git_status,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "file_sha256": hashes,
    }
    return preflight_data


def generate_canonical_waveform(
    grid: FrequencyGrid,
    mc_msun: float,
    eta: float,
    tc: float,
    phic: float,
    dist_mpc: float = 410.0,
    f_end: float = 512.0,
) -> np.ndarray:
    """Generate TaylorF2 frequency-domain waveform shifted to coalescence time."""
    delta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    m1_msun = 0.5 * mc_msun * (eta ** (-0.6)) * (1.0 + delta)
    m2_msun = 0.5 * mc_msun * (eta ** (-0.6)) * (1.0 - delta)
    m1_kg = m1_msun * M_SUN_SI
    m2_kg = m2_msun * M_SUN_SI
    dist_m = dist_mpc * MPC_SI

    h_fd = taylorf2_fd_waveform(
        grid.frequencies,
        m1_kg,
        m2_kg,
        dist_m,
        phi_c=phic,
        f_end=f_end,
    )
    return apply_coalescence_time_shift(h_fd, grid.frequencies, tc)


def regenerate_and_validate_on_source(
    grid: FrequencyGrid,
    psd_ref: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Regenerate canonical on-source data and enforce hard-stop validation gates."""
    print("\n" + "=" * 70)
    print("SECTION 2: REGENERATE & VALIDATE CANONICAL ON-SOURCE DATA (run_2B_01)")
    print("=" * 70)

    # Injected parameters (Anchor: Injection)
    mc_inj = 28.095555795460438
    eta_inj = 0.24710059171597634
    tc_inj = 2.0
    phic_inj = 0.0
    dist_mpc = 410.0
    f_end = 512.0
    noise_seed = 4201

    k_act = np.where(mask)[0]
    delta_f = grid.delta_f

    # 1. Clean waveform at Injection
    h_inj = generate_canonical_waveform(
        grid, mc_inj, eta_inj, tc_inj, phic_inj, dist_mpc, f_end
    )

    # Optimal SNR
    snr_sq = float(4.0 * delta_f * np.sum(np.abs(h_inj[k_act]) ** 2 / psd_ref[k_act]))
    optimal_snr = math.sqrt(max(0.0, snr_sq))

    # 2. Colored noise
    rng = np.random.default_rng(noise_seed)
    noise_td = generate_colored_noise(
        n_samples=grid.n_samples, delta_t=grid.delta_t, psd=psd_ref, generator=rng
    )
    noise_fd = continuous_rfft(noise_td, delta_t=grid.delta_t)
    d_fd = h_inj + noise_fd

    # Matched-filter SNR
    dh = float(4.0 * delta_f * np.sum(np.real(d_fd[k_act] * np.conj(h_inj[k_act])) / psd_ref[k_act]))
    mf_snr = dh / optimal_snr

    # Baseline likelihood and quadratic form at Injection
    lik_ref = FrequencyDomainGaussianLikelihood(
        data_fd=d_fd,
        psd=psd_ref,
        grid=grid,
        mask=mask,
    )
    res_inj = lik_ref.evaluate_detailed(h_inj)
    q_inj = res_inj.quadratic_form
    lnL_inj = res_inj.log_likelihood
    lnL_norm_ref = res_inj.log_likelihood_norm

    # Verification Table
    checks = [
        ("Optimal SNR", optimal_snr, 45.31529969318099, 1e-12),
        ("Matched-filter SNR", mf_snr, 44.134845519112, 1e-10),
        ("Injection quadratic form Q", q_inj, 3866.7101210798, 1e-8),
        ("Injection log likelihood", lnL_inj, 201714.31255790, 1e-8),
    ]

    all_passed = True
    check_records = []
    for name, obs, exp, tol in checks:
        diff = abs(obs - exp)
        passed = diff <= tol
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {name:28s}: obs={obs:.14f}, exp={exp:.14f}, diff={diff:.2e}, tol={tol:.2e} -> {status}")
        check_records.append({
            "diagnostic": name,
            "observed": obs,
            "expected": exp,
            "difference": diff,
            "tolerance": tol,
            "passed": passed,
        })

    if not all_passed:
        raise RuntimeError("CRITICAL: Canonical on-source data validation FAILED. Aborting.")

    val_summary = {
        "status": "ALL_PASSED",
        "checks": check_records,
        "anchor_inj": {
            "mc": mc_inj,
            "eta": eta_inj,
            "tc": tc_inj,
            "phic": phic_inj,
            "optimal_snr": optimal_snr,
            "matched_filter_snr": mf_snr,
            "quadratic_form": q_inj,
            "log_likelihood_norm": lnL_norm_ref,
            "log_likelihood": lnL_inj,
        },
    }
    return d_fd, h_inj, val_summary


def extract_and_validate_parameter_points(
    posterior_npz_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract and validate the 25 stratified points and 3 anchors."""
    print("\n" + "=" * 70)
    print("SECTION 3: EXTRACT & VALIDATE 28 PARAMETER POINTS")
    print("=" * 70)

    s4_npz = np.load(posterior_npz_path)
    samples = s4_npz["samples"]
    weights = s4_npz["weights"]

    pos_mask = weights > 0
    cand_indices = np.where(pos_mask)[0]
    cand_weights = weights[cand_indices]
    n_positive = len(cand_indices)
    print(f"Total dead points: {len(weights)}, Positive-weight candidate pool: {n_positive}")
    if n_positive != 8671:
        raise ValueError(f"Expected 8671 positive-weight candidates, found {n_positive}")

    # Seed 7001 randomized weighted-stratum procedure
    rng = np.random.default_rng(7001)
    perm = rng.permutation(n_positive)
    perm_weights = cand_weights[perm]
    cum_weights = np.cumsum(perm_weights) / np.sum(perm_weights)

    targets = np.linspace(0.5 / 25.0, 1.0 - 0.5 / 25.0, 25)
    perm_sel = np.searchsorted(cum_weights, targets)
    sel_cand_indices = perm[perm_sel]
    sel_dead_indices = cand_indices[sel_cand_indices]

    expected_indices = [
        8532, 9867, 9686, 9268, 10727, 9203, 9338, 9583, 8481, 9158,
        8450, 10798, 8449, 8285, 8141, 8029, 9127, 8636, 8545, 10402,
        8770, 9056, 8899, 10882, 8914
    ]

    if list(sel_dead_indices) != expected_indices:
        raise ValueError(f"Selected dead indices mismatch!\nObserved: {list(sel_dead_indices)}\nExpected: {expected_indices}")
    if len(sel_dead_indices) != len(set(sel_dead_indices)):
        raise ValueError("Selected indices contain duplicates!")
    print(f"25 stratified posterior points reproduced exactly. Unique: {len(set(sel_dead_indices))}/25")

    # Max-importance-weight point
    max_w_cand_idx = int(np.argmax(cand_weights))
    max_w_dead_idx = int(cand_indices[max_w_cand_idx])
    max_w_val = float(cand_weights[max_w_cand_idx])
    if max_w_dead_idx != 8919:
        raise ValueError(f"Expected max-weight index 8919, observed {max_w_dead_idx}")
    print(f"Maximum importance weight point verified: index {max_w_dead_idx}, weight {max_w_val:.8e}")

    # Build the 28 parameter points list
    # Point 0: Anchor 0 (Injection)
    points: List[Dict[str, Any]] = []

    p_inj = {
        "point_id": 0,
        "point_name": "anchor_injection",
        "point_type": "anchor",
        "candidate_dead_index": None,
        "importance_weight": None,
        "mc": 28.095555795460438,
        "eta": 0.24710059171597634,
        "tc": 2.0,
        "phic": 0.0,
    }
    points.append(p_inj)

    # Points 1..25: Stratified Posterior Points
    for m, (d_idx, c_idx) in enumerate(zip(sel_dead_indices, sel_cand_indices), start=1):
        coord = samples[d_idx]
        w_val = float(weights[d_idx])
        points.append({
            "point_id": m,
            "point_name": f"stratified_point_{m:02d}",
            "point_type": "stratified_posterior",
            "candidate_dead_index": int(d_idx),
            "importance_weight": w_val,
            "mc": float(coord[0]),
            "eta": float(coord[1]),
            "tc": float(coord[2]),
            "phic": float(coord[3]),
        })

    # Point 26: Anchor 1 (Posterior Mean)
    points.append({
        "point_id": 26,
        "point_name": "anchor_posterior_mean",
        "point_type": "anchor",
        "candidate_dead_index": None,
        "importance_weight": None,
        "mc": 28.04322948443798,
        "eta": 0.2447929986739996,
        "tc": 2.000405919519245,
        "phic": 5.388460406503973,
    })

    # Point 27: Anchor 2 (Maximum Importance Weight)
    coord_mw = samples[max_w_dead_idx]
    points.append({
        "point_id": 27,
        "point_name": "anchor_max_weight",
        "point_type": "anchor",
        "candidate_dead_index": max_w_dead_idx,
        "importance_weight": max_w_val,
        "mc": float(coord_mw[0]),
        "eta": float(coord_mw[1]),
        "tc": float(coord_mw[2]),
        "phic": float(coord_mw[3]),
    })

    pts_summary = {
        "n_total_points": len(points),
        "n_stratified_points": 25,
        "n_anchors": 3,
        "selected_dead_indices": [int(x) for x in sel_dead_indices],
        "max_weight_dead_index": max_w_dead_idx,
        "max_weight_value": max_w_val,
    }
    return points, pts_summary


def precompute_reference_evaluations(
    grid: FrequencyGrid,
    psd_ref: np.ndarray,
    mask: np.ndarray,
    d_fd: np.ndarray,
    points: List[Dict[str, Any]],
) -> Tuple[List[np.ndarray], FrequencyDomainGaussianLikelihood, float]:
    """Precompute waveforms and reference likelihoods for all 28 points."""
    print("\n" + "=" * 70)
    print("SECTION 4: PRECOMPUTE TEMPLATES & REFERENCE LIKELIHOODS")
    print("=" * 70)

    lik_ref = FrequencyDomainGaussianLikelihood(
        data_fd=d_fd,
        psd=psd_ref,
        grid=grid,
        mask=mask,
    )

    templates = []
    t0 = time.time()
    for p in points:
        h = generate_canonical_waveform(
            grid=grid,
            mc_msun=p["mc"],
            eta=p["eta"],
            tc=p["tc"],
            phic=p["phic"],
            dist_mpc=410.0,
            f_end=512.0,
        )
        templates.append(h)

        res = lik_ref.evaluate_detailed(h)
        p["lnL_ref"] = float(res.log_likelihood)
        p["Q_ref"] = float(res.quadratic_form)
        p["lnL_norm_ref"] = float(res.log_likelihood_norm)

    # Reference likelihood depth for the 25 stratified points (m=1..25)
    strat_lnL = [points[m]["lnL_ref"] for m in range(1, 26)]
    max_strat_lnL = max(strat_lnL)

    for m in range(1, 26):
        points[m]["delta_lnL_depth"] = float(max_strat_lnL - points[m]["lnL_ref"])
        if points[m]["delta_lnL_depth"] < -1e-12:
            raise ValueError(f"Negative likelihood depth at point {m}: {points[m]['delta_lnL_depth']}")

    # For anchors, also compute depth relative to max_strat_lnL for reference
    for p in points:
        if "delta_lnL_depth" not in p:
            p["delta_lnL_depth"] = float(max_strat_lnL - p["lnL_ref"])

    elapsed = time.time() - t0
    print(f"Precomputed 28 templates and reference likelihoods in {elapsed:.3f} s.")
    print(f"Stratified points max lnL: {max_strat_lnL:.4f}")
    depths = [points[m]["delta_lnL_depth"] for m in range(1, 26)]
    print(f"Stratified reference likelihood depth range: [{min(depths):.4f}, {max(depths):.4f}] nats")

    return templates, lik_ref, max_strat_lnL


def execute_stage_d_campaign(
    grid_on: FrequencyGrid,
    psd_ref: np.ndarray,
    mask_on: np.ndarray,
    d_fd: np.ndarray,
    templates: List[np.ndarray],
    points: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the 50-realization, 1400-evaluation Stage D campaign with hard stops."""
    print("\n" + "=" * 70)
    print("SECTION 5: EXECUTE 50-REALIZATION STAGE D CAMPAIGN (1,400 EVALUATIONS)")
    print("=" * 70)

    n_samples_off = 131072
    fs = 2048.0
    delta_t = 1.0 / fs
    seg_len = 8192
    hop = 4096
    window_name = "tukey"
    alpha = 0.25
    seeds = list(range(6201, 6251))  # 6201..6250 inclusive

    grid_off = FrequencyGrid(n_samples=n_samples_off, delta_t=delta_t)
    psd_off_ref = analytical_detector_psd(grid_off.frequencies, model="aligo_representative")
    k_act = np.where(mask_on)[0]

    # Data structures for results
    # We will record per-realization data:
    # realization_records[j] contains:
    # - seed
    # - lnL_norm_emp
    # - delta_lnL_norm
    # - points: list of 28 dicts with {point_id, lnL_emp, Q_emp, delta_lnL, delta_Q, D}
    # - spatial_mean_D_strat (over 25 stratified points)
    # - spatial_std_D_strat (over 25 stratified points, ddof=1)
    # - ols_slope_beta (against delta_lnL_depth over 25 stratified points)
    # - ols_intercept_alpha
    # - ols_r
    # - ols_r2

    realization_records = []
    n_points = len(points)

    # 2D arrays for easy statistical aggregation across realizations:
    # shape: (n_realizations=50, n_points=28)
    D_matrix = np.empty((len(seeds), n_points), dtype=np.float64)
    delta_matrix = np.empty((len(seeds), n_points), dtype=np.float64)
    delta_Q_matrix = np.empty((len(seeds), n_points), dtype=np.float64)

    spatial_means = np.empty(len(seeds), dtype=np.float64)
    spatial_stds = np.empty(len(seeds), dtype=np.float64)
    ols_slopes = np.empty(len(seeds), dtype=np.float64)
    ols_intercepts = np.empty(len(seeds), dtype=np.float64)
    ols_corrs = np.empty(len(seeds), dtype=np.float64)

    depth_25 = np.array([points[m]["delta_lnL_depth"] for m in range(1, 26)], dtype=np.float64)
    x_mean = np.mean(depth_25)
    x_var = np.var(depth_25, ddof=1)
    x_std = math.sqrt(x_var)

    t_start = time.time()

    for idx, seed in enumerate(seeds):
        # 1. Generate off-source colored noise
        rng = np.random.default_rng(seed)
        noise_off_td = generate_colored_noise(
            n_samples=n_samples_off, delta_t=delta_t, psd=psd_off_ref, generator=rng
        )

        # 2. Estimate empirical Welch PSD
        freqs_welch, psd_est = welch_psd(
            noise_off_td,
            delta_t=delta_t,
            segment_length=seg_len,
            overlap_samples=hop,
            window=window_name,
            method="mean",
        )

        # Numerical health check on empirical PSD
        if not np.all(np.isfinite(psd_est)):
            raise ValueError(f"[Seed {seed}] Empirical PSD contains non-finite values.")
        if np.any(psd_est[k_act] <= 0.0):
            raise ValueError(f"[Seed {seed}] Empirical PSD contains non-positive values on active band.")

        # 3. Ingest empirical PSD
        lik_emp = FrequencyDomainGaussianLikelihood.from_empirical_psd(
            data_fd=d_fd,
            psd=psd_est,
            grid=grid_on,
            mask=mask_on,
            psd_frequencies=freqs_welch,
        )

        # 4. Evaluate on all 28 points
        rec_points = []
        delta_inj = None
        delta_Q_inj = None

        for p_idx, p in enumerate(points):
            h = templates[p_idx]
            res_emp = lik_emp.evaluate_detailed(h)

            lnL_emp = float(res_emp.log_likelihood)
            Q_emp = float(res_emp.quadratic_form)
            lnL_norm_emp = float(res_emp.log_likelihood_norm)

            if not (math.isfinite(lnL_emp) and math.isfinite(Q_emp) and math.isfinite(lnL_norm_emp)):
                raise ValueError(f"[Seed {seed}, Point {p['point_name']}] Non-finite likelihood output.")

            delta_lnL = lnL_emp - p["lnL_ref"]
            delta_Q = Q_emp - p["Q_ref"]

            if p_idx == 0:  # Injection anchor
                delta_inj = delta_lnL
                delta_Q_inj = delta_Q

            rec_points.append({
                "point_id": p["point_id"],
                "point_name": p["point_name"],
                "lnL_emp": lnL_emp,
                "Q_emp": Q_emp,
                "delta_lnL": delta_lnL,
                "delta_Q": delta_Q,
            })

            delta_matrix[idx, p_idx] = delta_lnL
            delta_Q_matrix[idx, p_idx] = delta_Q

        # 5. Compute D_j(theta_m) and enforce numerical health gates
        delta_lnL_norm = rec_points[0]["lnL_emp"] - points[0]["lnL_ref"]  # normalization term difference

        for p_idx, pt_rec in enumerate(rec_points):
            D_val = pt_rec["delta_lnL"] - delta_inj
            D_quad = -0.5 * (pt_rec["delta_Q"] - delta_Q_inj)

            # Gate 1: Injection anchor D_j(theta_inj) == 0 within 1e-12
            if p_idx == 0:
                if abs(D_val) > 1e-12:
                    raise RuntimeError(f"[Seed {seed}] Injection anchor gate violated: |D| = {abs(D_val):.2e} > 1e-12")

            # Gate 2: Normalization cancellation identity within 1e-10
            diff_quad = abs(D_val - D_quad)
            if diff_quad > 1e-10:
                raise RuntimeError(f"[Seed {seed}, Point {pt_rec['point_name']}] Normalization cancellation gate violated: diff = {diff_quad:.2e} > 1e-10")

            pt_rec["D"] = float(D_val)
            pt_rec["D_quad"] = float(D_quad)
            pt_rec["cancellation_discrepancy"] = float(diff_quad)
            D_matrix[idx, p_idx] = D_val

        # 6. Spatial metrics over the 25 stratified points (m=1..25)
        strat_D = D_matrix[idx, 1:26]
        mu_D_j = float(np.mean(strat_D))
        s_D_j = float(np.std(strat_D, ddof=1))
        spatial_means[idx] = mu_D_j
        spatial_stds[idx] = s_D_j

        # 7. OLS regression of D_j against delta_lnL_depth over 25 stratified points
        cov_xD = float(np.cov(depth_25, strat_D, ddof=1)[0, 1])
        beta_j = cov_xD / x_var
        alpha_j = mu_D_j - beta_j * x_mean
        r_j = cov_xD / (x_std * s_D_j) if s_D_j > 1e-15 else 0.0
        r2_j = r_j ** 2

        ols_slopes[idx] = beta_j
        ols_intercepts[idx] = alpha_j
        ols_corrs[idx] = r_j

        realization_records.append({
            "seed": seed,
            "delta_lnL_norm": float(rec_points[0]["delta_lnL"] + 0.5 * rec_points[0]["delta_Q"]),
            "points": rec_points,
            "spatial_metrics_25": {
                "spatial_mean_D": mu_D_j,
                "spatial_spread_sD": s_D_j,
            },
            "ols_regression_25": {
                "slope_beta": float(beta_j),
                "intercept_alpha": float(alpha_j),
                "correlation_r": float(r_j),
                "r_squared": float(r2_j),
            },
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Completed realization {idx+1:2d}/50 (Seed {seed}) in {time.time()-t_start:.2f} s | mu_D={mu_D_j:+.4f}, s_D={s_D_j:.4f}, beta={beta_j:+.4f}")

    total_time = time.time() - t_start
    print(f"\nAll 50 realizations (1,400 likelihood evaluations) completed successfully in {total_time:.2f} s.")

    # Aggregate Statistics
    # Per-point statistics
    per_point_summary = []
    for p_idx, p in enumerate(points):
        d_vals = D_matrix[:, p_idx]
        delta_vals = delta_matrix[:, p_idx]
        delta_Q_vals = delta_Q_matrix[:, p_idx]

        mean_D = float(np.mean(d_vals))
        std_D = float(np.std(d_vals, ddof=1))
        sem_D = std_D / math.sqrt(len(seeds))
        p5, p25, p50, p75, p95 = [float(x) for x in np.percentile(d_vals, [5, 25, 50, 75, 95])]

        per_point_summary.append({
            "point_id": p["point_id"],
            "point_name": p["point_name"],
            "point_type": p["point_type"],
            "candidate_dead_index": p["candidate_dead_index"],
            "importance_weight": p["importance_weight"],
            "coordinates": {
                "mc": p["mc"],
                "eta": p["eta"],
                "tc": p["tc"],
                "phic": p["phic"],
            },
            "lnL_ref": p["lnL_ref"],
            "delta_lnL_depth": p["delta_lnL_depth"],
            "D_stats": {
                "mean": mean_D,
                "std": std_D,
                "sem": sem_D,
                "p05": p5,
                "p25": p25,
                "median_p50": p50,
                "p75": p75,
                "p95": p95,
            },
            "delta_lnL_stats": {
                "mean": float(np.mean(delta_vals)),
                "std": float(np.std(delta_vals, ddof=1)),
                "sem": float(np.std(delta_vals, ddof=1) / math.sqrt(len(seeds))),
            },
            "delta_Q_stats": {
                "mean": float(np.mean(delta_Q_vals)),
                "std": float(np.std(delta_Q_vals, ddof=1)),
                "sem": float(np.std(delta_Q_vals, ddof=1) / math.sqrt(len(seeds))),
            },
        })

    # Spatial spread and spatial mean summary across realizations
    mean_spatial_mean = float(np.mean(spatial_means))
    sem_spatial_mean = float(np.std(spatial_means, ddof=1) / math.sqrt(len(seeds)))

    mean_spatial_std = float(np.mean(spatial_stds))
    sem_spatial_std = float(np.std(spatial_stds, ddof=1) / math.sqrt(len(seeds)))
    min_spatial_std = float(np.min(spatial_stds))
    max_spatial_std = float(np.max(spatial_stds))

    # OLS regression summary across realizations
    mean_beta = float(np.mean(ols_slopes))
    sem_beta = float(np.std(ols_slopes, ddof=1) / math.sqrt(len(seeds)))

    mean_alpha = float(np.mean(ols_intercepts))
    sem_alpha = float(np.std(ols_intercepts, ddof=1) / math.sqrt(len(seeds)))

    mean_corr = float(np.mean(ols_corrs))
    sem_corr = float(np.std(ols_corrs, ddof=1) / math.sqrt(len(seeds)))

    mean_r2 = float(np.mean(ols_corrs ** 2))

    # Anchors
    anchor_inj = per_point_summary[0]
    anchor_mean = per_point_summary[26]
    anchor_max_w = per_point_summary[27]

    # Global max absolute mean differential perturbation across all points
    all_mean_D = [abs(pt["D_stats"]["mean"]) for pt in per_point_summary]
    max_abs_mean_D = float(max(all_mean_D))
    max_abs_mean_D_point = per_point_summary[int(np.argmax(all_mean_D))]["point_name"]

    summary_data = {
        "metadata": {
            "experiment": "WP6 Stage 5D — Multi-Point Likelihood-Geometry Screening",
            "n_realizations": len(seeds),
            "n_points_total": n_points,
            "n_stratified_points": 25,
            "n_anchors": 3,
            "runtime_seconds": total_time,
        },
        "numerical_checks": {
            "injection_anchor_identity_max_abs": float(np.max(np.abs(D_matrix[:, 0]))),
            "injection_anchor_gate_passed": bool(np.max(np.abs(D_matrix[:, 0])) <= 1e-12),
            "cancellation_identity_max_discrepancy": float(np.max([
                r["points"][p_idx]["cancellation_discrepancy"]
                for r in realization_records
                for p_idx in range(n_points)
            ])),
            "cancellation_identity_gate_passed": bool(np.max([
                r["points"][p_idx]["cancellation_discrepancy"]
                for r in realization_records
                for p_idx in range(n_points)
            ]) <= 1e-10),
            "all_evaluated_quantities_finite": True,
            "psd_active_band_strictly_positive": True,
        },
        "spatial_metrics_25": {
            "mean_spatial_mean_D": mean_spatial_mean,
            "sem_spatial_mean_D": sem_spatial_mean,
            "mean_spatial_spread_sD": mean_spatial_std,
            "sem_spatial_spread_sD": sem_spatial_std,
            "min_spatial_spread_sD": min_spatial_std,
            "max_spatial_spread_sD": max_spatial_std,
        },
        "ols_depth_regression_25": {
            "reference_likelihood_depth_range": [float(np.min(depth_25)), float(np.max(depth_25))],
            "mean_slope_beta": mean_beta,
            "sem_slope_beta": sem_beta,
            "mean_intercept_alpha": mean_alpha,
            "sem_intercept_alpha": sem_alpha,
            "mean_correlation_r": mean_corr,
            "sem_correlation_r": sem_corr,
            "mean_r_squared": mean_r2,
        },
        "anchors": {
            "anchor_injection": {
                "point_name": anchor_inj["point_name"],
                "mean_D": anchor_inj["D_stats"]["mean"],
                "sem_D": anchor_inj["D_stats"]["sem"],
            },
            "anchor_posterior_mean": {
                "point_name": anchor_mean["point_name"],
                "mean_D": anchor_mean["D_stats"]["mean"],
                "sem_D": anchor_mean["D_stats"]["sem"],
                "median_D": anchor_mean["D_stats"]["median_p50"],
                "p05": anchor_mean["D_stats"]["p05"],
                "p95": anchor_mean["D_stats"]["p95"],
            },
            "anchor_max_weight": {
                "point_name": anchor_max_w["point_name"],
                "candidate_dead_index": anchor_max_w["candidate_dead_index"],
                "importance_weight": anchor_max_w["importance_weight"],
                "mean_D": anchor_max_w["D_stats"]["mean"],
                "sem_D": anchor_max_w["D_stats"]["sem"],
                "median_D": anchor_max_w["D_stats"]["median_p50"],
                "p05": anchor_max_w["D_stats"]["p05"],
                "p95": anchor_max_w["D_stats"]["p95"],
            },
        },
        "global_extremum": {
            "max_abs_mean_D": max_abs_mean_D,
            "max_abs_mean_D_point": max_abs_mean_D_point,
        },
        "per_point_summary": per_point_summary,
    }

    results_data = {
        "metadata": {
            "experiment": "WP6 Stage 5D — Multi-Point Likelihood-Geometry Screening",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_realizations": len(seeds),
            "seeds": seeds,
            "n_points": n_points,
        },
        "realizations": realization_records,
    }

    return results_data, summary_data


def main() -> None:
    """Orchestrate Stage D preflight, validation, execution, and artifact saving."""
    output_dir = "reports/demonstrations/wp6_stage5/stage_d"

    # Strict Scope Check: Do not overwrite if directory or files exist
    if os.path.exists(output_dir):
        files_inside = os.listdir(output_dir)
        if len(files_inside) > 0:
            raise RuntimeError(f"ABORT: Output directory {output_dir} already exists and is non-empty ({files_inside}). Overwriting is prohibited.")
    os.makedirs(output_dir, exist_ok=False)

    # 1. Preflight
    preflight = run_preflight()

    # 2. Setup on-source grid & PSD
    n_samples = 8192
    fs = 2048.0
    delta_t = 1.0 / fs
    grid_on = FrequencyGrid(n_samples=n_samples, delta_t=delta_t)
    psd_ref = analytical_detector_psd(grid_on.frequencies, model="aligo_representative")
    mask_on = (grid_on.frequencies >= 20.0) & (grid_on.frequencies <= 512.0)

    # Regenerate & validate canonical on-source data
    d_fd, h_inj, val_summary = regenerate_and_validate_on_source(grid_on, psd_ref, mask_on)

    # 3. Extract & validate 28 parameter points
    s4_npz_path = "reports/demonstrations/wp6_stage4/tier2/run_2B_01_results.npz"
    points, pts_summary = extract_and_validate_parameter_points(s4_npz_path)

    # 4. Precompute templates & reference evaluations
    templates, lik_ref, max_strat_lnL = precompute_reference_evaluations(
        grid=grid_on,
        psd_ref=psd_ref,
        mask=mask_on,
        d_fd=d_fd,
        points=points,
    )

    # 5. Execute Stage D Campaign
    results_data, summary_data = execute_stage_d_campaign(
        grid_on=grid_on,
        psd_ref=psd_ref,
        mask_on=mask_on,
        d_fd=d_fd,
        templates=templates,
        points=points,
    )

    # Add runner SHA-256 to preflight
    runner_path = "scripts/run_wp6_stage5_stage_d.py"
    if os.path.exists(runner_path):
        preflight["file_sha256"][runner_path] = compute_file_sha256(runner_path)

    # 6. Save manifest.json
    manifest_data = {
        "experiment": "WP6 Stage 5D — Multi-Point Likelihood-Geometry Screening",
        "authorization": "Stage D ONLY",
        "preflight": preflight,
        "on_source_validation": val_summary,
        "points_summary": pts_summary,
        "parameter_points": [
            {
                "point_id": p["point_id"],
                "point_name": p["point_name"],
                "point_type": p["point_type"],
                "candidate_dead_index": p["candidate_dead_index"],
                "importance_weight": p["importance_weight"],
                "mc": p["mc"],
                "eta": p["eta"],
                "tc": p["tc"],
                "phic": p["phic"],
                "lnL_ref": p["lnL_ref"],
                "Q_ref": p["Q_ref"],
                "delta_lnL_depth": p["delta_lnL_depth"],
            }
            for p in points
        ],
        "configuration": {
            "on_source_duration_s": 4.0,
            "on_source_n_samples": n_samples,
            "off_source_duration_s": 64.0,
            "off_source_n_samples": 131072,
            "sampling_rate_hz": fs,
            "welch_segment_length": 8192,
            "welch_overlap_samples": 4096,
            "welch_window": "tukey",
            "welch_tukey_alpha": 0.25,
            "welch_average_method": "mean",
            "f_low_hz": 20.0,
            "f_high_hz": 512.0,
            "n_active_bins": int(np.sum(mask_on)),
            "on_source_noise_seed": 4201,
            "off_source_seeds": list(range(6201, 6251)),
            "point_selection_seed": 7001,
            "n_realizations": 50,
            "n_points_total": len(points),
            "n_stratified_points": 25,
            "n_anchors": 3,
            "total_empirical_evaluations": len(points) * 50,
        },
    }

    manifest_file = os.path.join(output_dir, "manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\nSaved manifest to {manifest_file}")

    # Save results.json
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved results to {results_file}")

    # Save summary.json
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary to {summary_file}")

    print("\n" + "=" * 70)
    print("STAGE D EXECUTION COMPLETE — ALL ARTIFACTS WRITTEN SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
