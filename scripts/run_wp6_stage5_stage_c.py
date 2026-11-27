#!/usr/bin/env python3
"""WP6 Stage 5C — Authorized Bounded Empirical Welch PSD Experiment Runner.

This script executes the authorized Stage C characterization experiment:
1. Mandatory preflight checks (Git HEAD, branch, status, SHA-256 hashes, library versions).
2. Deterministic regeneration and hard-stop validation of canonical on-source data (run_2B_01).
3. 50-realization off-source colored-noise Welch PSD estimation (seeds 6201–6250).
4. Fixed-parameter likelihood evaluations at Point A (injection) and Point B (posterior mean).
5. Generation of machine-readable manifest, per-realization results, and aggregate summary.

Execution boundary: Stage C ONLY. Samplers, evidence calculation, and Stage D are strictly excluded.
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
    """Execute Section 1 mandatory preflight environment and code-provenance checks."""
    print("=" * 70)
    print("SECTION 1: MANDATORY PREFLIGHT CHECKS")
    print("=" * 70)

    branch = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    commit_head = run_git_command(["rev-parse", "HEAD"])
    git_status = run_git_command(["status", "--short"])

    print(f"Git Branch:      {branch}")
    print(f"Git HEAD Commit: {commit_head}")
    print(f"Python Version:  {sys.version.split()[0]}")
    print(f"NumPy Version:   {np.__version__}")
    print(f"SciPy Version:   {scipy.__version__}")

    source_files = [
        "src/gw_inference/waveform.py",
        "src/gw_inference/noise.py",
        "src/gw_inference/psd.py",
        "src/gw_inference/likelihood.py",
        "src/gw_inference/grid.py",
        "src/gw_inference/conventions.py",
        "src/gw_inference/constants.py",
    ]

    hashes = {}
    print("\nSource File SHA-256 Hashes:")
    for path in source_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required source file not found: {path}")
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
        "source_sha256": hashes,
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Execute Section 2 hard-stop validation on regenerated canonical on-source data."""
    print("\n" + "=" * 70)
    print("SECTION 2: REGENERATE & VALIDATE CANONICAL ON-SOURCE DATA (run_2B_01)")
    print("=" * 70)

    # Injected parameters (Point A)
    mc_inj = 28.095555795460438
    eta_inj = 0.24710059171597634
    tc_inj = 2.0
    phic_inj = 0.0
    dist_mpc = 410.0
    f_end = 512.0
    noise_seed = 4201

    k_act = np.where(mask)[0]
    delta_f = grid.delta_f

    # 1. Clean waveform at Point A
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

    # Baseline likelihood and quadratic form at Point A
    lik_ref = FrequencyDomainGaussianLikelihood(
        data_fd=d_fd,
        psd=psd_ref,
        grid=grid,
        mask=mask,
    )
    res_inj = lik_ref.evaluate_detailed(h_inj)
    q_inj = res_inj.quadratic_form
    lnL_inj = res_inj.log_likelihood

    # Point B: Stored posterior mean waveform
    mc_mean = 28.04322948443798
    eta_mean = 0.2447929986739996
    tc_mean = 2.000405919519245
    phic_mean = 5.388460406503973
    h_mean = generate_canonical_waveform(
        grid, mc_mean, eta_mean, tc_mean, phic_mean, dist_mpc, f_end
    )
    res_mean = lik_ref.evaluate_detailed(h_mean)
    q_mean = res_mean.quadratic_form
    lnL_mean = res_mean.log_likelihood

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
            "diff": diff,
            "tolerance": tol,
            "status": status,
        })

    if not all_passed:
        print("\nHARD STOP TRIGGERED: One or more preflight diagnostic checks failed.")
        sys.exit(1)

    print("\nAll Section 2 preflight diagnostic checks PASSED.")

    validation_summary = {
        "status": "ALL_PASSED",
        "checks": check_records,
        "point_a_inj": {
            "mc_msun": mc_inj,
            "eta": eta_inj,
            "tc_s": tc_inj,
            "phic_rad": phic_inj,
            "optimal_snr": optimal_snr,
            "matched_filter_snr": mf_snr,
            "quadratic_form": q_inj,
            "log_likelihood_norm": res_inj.log_likelihood_norm,
            "log_likelihood": lnL_inj,
        },
        "point_b_mean": {
            "mc_msun": mc_mean,
            "eta": eta_mean,
            "tc_s": tc_mean,
            "phic_rad": phic_mean,
            "quadratic_form": q_mean,
            "log_likelihood_norm": res_mean.log_likelihood_norm,
            "log_likelihood": lnL_mean,
            "delta_lnL_vs_inj": lnL_mean - lnL_inj,
        },
    }

    return d_fd, h_inj, h_mean, validation_summary


def execute_stage_c_campaign(
    grid_on: FrequencyGrid,
    psd_ref: np.ndarray,
    mask_on: np.ndarray,
    d_fd: np.ndarray,
    h_inj: np.ndarray,
    h_mean: np.ndarray,
    ref_vals: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute Section 3 & 4 50-realization campaign and evaluate diagnostics."""
    print("\n" + "=" * 70)
    print("SECTION 3: EXECUTING 50-REALIZATION OFF-SOURCE WELCH PSD CAMPAIGN")
    print("=" * 70)

    n_samples_off = 131072
    fs = 2048.0
    delta_t = 1.0 / fs
    duration_s = 64.0
    seg_len = 8192
    hop = 4096
    window_name = "tukey"
    alpha = 0.25
    seeds = list(range(6201, 6251))  # 6201 through 6250 inclusive

    grid_off = FrequencyGrid(n_samples=n_samples_off, delta_t=delta_t)
    psd_off_ref = analytical_detector_psd(grid_off.frequencies, model="aligo_representative")

    k_act = np.where(mask_on)[0]
    n_act = len(k_act)
    delta_f = grid_on.delta_f

    # Divide active band (20 - 512 Hz) into 8 frequency blocks
    n_blocks = 8
    block_size = n_act // n_blocks
    blocks = []
    for b in range(n_blocks):
        start_idx = b * block_size
        end_idx = (b + 1) * block_size if b < n_blocks - 1 else n_act
        b_k = k_act[start_idx:end_idx]
        blocks.append({
            "block_id": b + 1,
            "f_min": float(grid_on.frequencies[b_k[0]]),
            "f_max": float(grid_on.frequencies[b_k[-1]]),
            "k_indices": b_k,
            "n_bins": len(b_k),
        })

    q_inj_ref = ref_vals["point_a_inj"]["quadratic_form"]
    lnL_norm_ref = ref_vals["point_a_inj"]["log_likelihood_norm"]
    lnL_inj_ref = ref_vals["point_a_inj"]["log_likelihood"]

    q_mean_ref = ref_vals["point_b_mean"]["quadratic_form"]
    lnL_mean_ref = ref_vals["point_b_mean"]["log_likelihood"]

    realization_results = []
    psd_ratios_all = np.empty((len(seeds), n_act), dtype=np.float64)

    t_start = time.time()
    for idx, seed in enumerate(seeds):
        # 1. Generate independent off-source colored noise
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

        # Sanity checks on empirical PSD
        if np.any(np.isnan(psd_est)) or np.any(np.isinf(psd_est)):
            raise ValueError(f"[Seed {seed}] Empirical PSD contains NaN or Inf.")
        if np.any(psd_est[k_act] <= 0.0):
            raise ValueError(f"[Seed {seed}] Empirical PSD contains non-positive values on active bins.")

        psd_ratio_active = psd_est[k_act] / psd_ref[k_act]
        psd_ratios_all[idx] = psd_ratio_active

        # 3. Ingest empirical PSD into Whittle likelihood with full frequency validation
        lik_emp = FrequencyDomainGaussianLikelihood.from_empirical_psd(
            data_fd=d_fd,
            psd=psd_est,
            grid=grid_on,
            mask=mask_on,
            psd_frequencies=freqs_welch,
        )

        # 4. Evaluate at Point A (Injection)
        res_inj_emp = lik_emp.evaluate_detailed(h_inj)
        delta_lnL_norm = res_inj_emp.log_likelihood_norm - lnL_norm_ref
        delta_Q_inj = res_inj_emp.quadratic_form - q_inj_ref
        delta_lnL_inj = res_inj_emp.log_likelihood - lnL_inj_ref

        # Direct equation check: delta_lnL = delta_lnL_norm - 0.5 * delta_Q
        decomp_diff_inj = abs(delta_lnL_inj - (delta_lnL_norm - 0.5 * delta_Q_inj))
        if decomp_diff_inj > 1e-10:
            raise ValueError(f"[Seed {seed}] Decomposition discrepancy at Point A: {decomp_diff_inj}")

        # 5. Evaluate at Point B (Posterior Mean)
        res_mean_emp = lik_emp.evaluate_detailed(h_mean)
        delta_Q_mean = res_mean_emp.quadratic_form - q_mean_ref
        delta_lnL_mean = res_mean_emp.log_likelihood - lnL_mean_ref

        decomp_diff_mean = abs(delta_lnL_mean - (delta_lnL_norm - 0.5 * delta_Q_mean))
        if decomp_diff_mean > 1e-10:
            raise ValueError(f"[Seed {seed}] Decomposition discrepancy at Point B: {decomp_diff_mean}")

        diff_mean_inj = delta_lnL_mean - delta_lnL_inj

        # 6. Band-resolved breakdown across 8 frequency blocks
        band_breakdown = []
        for b_info in blocks:
            b_k = b_info["k_indices"]
            b_norm = -float(np.sum(np.log(psd_est[b_k] / psd_ref[b_k])))
            b_q_inj = float(4.0 * delta_f * np.sum(
                np.abs(d_fd[b_k] - h_inj[b_k]) ** 2 * (1.0 / psd_est[b_k] - 1.0 / psd_ref[b_k])
            ))
            b_q_mean = float(4.0 * delta_f * np.sum(
                np.abs(d_fd[b_k] - h_mean[b_k]) ** 2 * (1.0 / psd_est[b_k] - 1.0 / psd_ref[b_k])
            ))
            band_breakdown.append({
                "block_id": b_info["block_id"],
                "f_min": b_info["f_min"],
                "f_max": b_info["f_max"],
                "delta_lnL_norm_b": b_norm,
                "delta_Q_inj_b": b_q_inj,
                "delta_lnL_inj_b": b_norm - 0.5 * b_q_inj,
                "delta_Q_mean_b": b_q_mean,
                "delta_lnL_mean_b": b_norm - 0.5 * b_q_mean,
                "mean_psd_ratio_b": float(np.mean(psd_est[b_k] / psd_ref[b_k])),
            })

        rec = {
            "realization_index": idx,
            "seed": seed,
            "delta_lnL_norm": float(delta_lnL_norm),
            "point_a_inj": {
                "quadratic_form": float(res_inj_emp.quadratic_form),
                "delta_Q": float(delta_Q_inj),
                "log_likelihood": float(res_inj_emp.log_likelihood),
                "delta_lnL": float(delta_lnL_inj),
            },
            "point_b_mean": {
                "quadratic_form": float(res_mean_emp.quadratic_form),
                "delta_Q": float(delta_Q_mean),
                "log_likelihood": float(res_mean_emp.log_likelihood),
                "delta_lnL": float(delta_lnL_mean),
            },
            "delta_lnL_mean_minus_inj": float(diff_mean_inj),
            "band_breakdown": band_breakdown,
        }
        realization_results.append(rec)

        if (idx + 1) % 10 == 0 or idx == len(seeds) - 1:
            print(f"  Completed realization {idx+1:2d}/50 (seed {seed}): "
                  f"delta_lnL_norm={delta_lnL_norm:+7.2f}, "
                  f"delta_Q_inj={delta_Q_inj:+7.2f}, "
                  f"delta_lnL_inj={delta_lnL_inj:+7.2f}, "
                  f"diff_mean_inj={diff_mean_inj:+6.3f}")

    elapsed = time.time() - t_start
    print(f"\n50 realizations completed in {elapsed:.2f} seconds ({elapsed/50:.4f} s/realization).")

    # Aggregate Statistics
    def compute_stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "sem": float(np.std(arr, ddof=1) / math.sqrt(len(arr))),
            "median": float(np.median(arr)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "p05": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    delta_norm_arr = np.array([r["delta_lnL_norm"] for r in realization_results])
    delta_Q_inj_arr = np.array([r["point_a_inj"]["delta_Q"] for r in realization_results])
    delta_lnL_inj_arr = np.array([r["point_a_inj"]["delta_lnL"] for r in realization_results])

    delta_Q_mean_arr = np.array([r["point_b_mean"]["delta_Q"] for r in realization_results])
    delta_lnL_mean_arr = np.array([r["point_b_mean"]["delta_lnL"] for r in realization_results])
    diff_arr = np.array([r["delta_lnL_mean_minus_inj"] for r in realization_results])

    # Per-band aggregate stats
    band_summary = []
    for b_idx in range(n_blocks):
        b_norm = np.array([r["band_breakdown"][b_idx]["delta_lnL_norm_b"] for r in realization_results])
        b_q = np.array([r["band_breakdown"][b_idx]["delta_Q_inj_b"] for r in realization_results])
        b_lnL = np.array([r["band_breakdown"][b_idx]["delta_lnL_inj_b"] for r in realization_results])
        b_ratio = np.array([r["band_breakdown"][b_idx]["mean_psd_ratio_b"] for r in realization_results])
        band_summary.append({
            "block_id": blocks[b_idx]["block_id"],
            "f_min": blocks[b_idx]["f_min"],
            "f_max": blocks[b_idx]["f_max"],
            "n_bins": blocks[b_idx]["n_bins"],
            "delta_lnL_norm": compute_stats(b_norm),
            "delta_Q_inj": compute_stats(b_q),
            "delta_lnL_inj": compute_stats(b_lnL),
            "mean_psd_ratio": compute_stats(b_ratio),
        })

    # Global empirical PSD ratio across active band
    mean_ratio_per_bin = np.mean(psd_ratios_all, axis=0)
    std_ratio_per_bin = np.std(psd_ratios_all, axis=0, ddof=1)

    summary_data = {
        "n_realizations": len(seeds),
        "seeds": seeds,
        "runtime_seconds": elapsed,
        "delta_lnL_norm": compute_stats(delta_norm_arr),
        "point_a_inj": {
            "delta_Q": compute_stats(delta_Q_inj_arr),
            "delta_lnL": compute_stats(delta_lnL_inj_arr),
        },
        "point_b_mean": {
            "delta_Q": compute_stats(delta_Q_mean_arr),
            "delta_lnL": compute_stats(delta_lnL_mean_arr),
        },
        "delta_lnL_mean_minus_inj": compute_stats(diff_arr),
        "band_breakdown_summary": band_summary,
        "spectral_ratio_summary": {
            "global_active_mean_ratio": float(np.mean(mean_ratio_per_bin)),
            "global_active_std_ratio": float(np.mean(std_ratio_per_bin)),
            "block_1_mean_ratio": float(np.mean(mean_ratio_per_bin[:blocks[0]["n_bins"]])),
            "block_8_mean_ratio": float(np.mean(mean_ratio_per_bin[-blocks[-1]["n_bins"]:])),
        },
    }

    return realization_results, summary_data


def main() -> None:
    """Orchestrate Section 1 preflight, Section 2 validation, and Section 3/4 execution."""
    output_dir = "reports/demonstrations/wp6_stage5/stage_c"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Preflight
    preflight = run_preflight()

    # 2. Setup on-source grid & PSD
    n_samples = 8192
    fs = 2048.0
    delta_t = 1.0 / fs
    grid_on = FrequencyGrid(n_samples=n_samples, delta_t=delta_t)
    psd_ref = analytical_detector_psd(grid_on.frequencies, model="aligo_representative")
    mask_on = (grid_on.frequencies >= 20.0) & (grid_on.frequencies <= 512.0)

    # Regenerate & validate
    d_fd, h_inj, h_mean, val_summary = regenerate_and_validate_on_source(
        grid_on, psd_ref, mask_on
    )

    # 3 & 4. Execute 50-realization campaign
    results, summary = execute_stage_c_campaign(
        grid_on, psd_ref, mask_on, d_fd, h_inj, h_mean, val_summary
    )

    # Save manifest.json
    manifest_data = {
        "experiment": "WP6 Stage 5C — Empirical Welch PSD Characterization",
        "authorization": "Stage C ONLY",
        "preflight": preflight,
        "on_source_validation": val_summary,
        "configuration": {
            "off_source_duration_s": 64.0,
            "off_source_n_samples": 131072,
            "sampling_rate_hz": fs,
            "welch_segment_length": 8192,
            "welch_overlap_samples": 4096,
            "welch_window": "tukey",
            "welch_alpha": 0.25,
            "welch_aggregation": "mean",
            "seeds": list(range(6201, 6251)),
            "active_band_hz": [20.0, 512.0],
            "n_active_bins": int(np.sum(mask_on)),
        },
        "script_sha256": compute_file_sha256(__file__),
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    # Save results.json
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Per-realization results saved: {results_path}")

    # Save summary.json
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregate summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("STAGE 5C EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Delta lnL norm (mean +/- sem):  {summary['delta_lnL_norm']['mean']:.4f} +/- {summary['delta_lnL_norm']['sem']:.4f}")
    print(f"  Delta Q inj    (mean +/- sem):  {summary['point_a_inj']['delta_Q']['mean']:.4f} +/- {summary['point_a_inj']['delta_Q']['sem']:.4f}")
    print(f"  Delta lnL inj  (mean +/- sem):  {summary['point_a_inj']['delta_lnL']['mean']:.4f} +/- {summary['point_a_inj']['delta_lnL']['sem']:.4f}")
    print(f"  Delta Q mean   (mean +/- sem):  {summary['point_b_mean']['delta_Q']['mean']:.4f} +/- {summary['point_b_mean']['delta_Q']['sem']:.4f}")
    print(f"  Delta lnL mean (mean +/- sem):  {summary['point_b_mean']['delta_lnL']['mean']:.4f} +/- {summary['point_b_mean']['delta_lnL']['sem']:.4f}")
    print(f"  Diff (mean - inj) [mean +/- sem]: {summary['delta_lnL_mean_minus_inj']['mean']:.4f} +/- {summary['delta_lnL_mean_minus_inj']['sem']:.4f}")


if __name__ == "__main__":
    main()
