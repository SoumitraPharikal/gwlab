"""WP6 Stage 5E.1: White-Noise Predictive Calibration Prototype Runner.

This script executes the authorized WP6 Stage 5E.1 predictive calibration prototype:
1. Oracle Gaussian Whittle model (known true PSD S_0).
2. Plug-in Gaussian Whittle model (Welch estimated PSD S_hat).
3. Candidate Student-t predictive model (Welch estimated PSD S_hat, nu_eff = 51894/1157).
4. Nominal 90% inverse-gamma PSD credible interval coverage evaluation.

Conventions & Boundaries:
- Realization replication unit = 100 independent pairs (seeds 6301-6400 off, 6401-6500 on).
- Unit-variance Gaussian white noise, fs = 2048 Hz, S_0 = 2 / fs = 9.765625e-4.
- Active band 20 - 512 Hz (1969 bins, delta_f = 0.25 Hz).
- Zero production code modification, strictly reproducible standalone execution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy
from scipy import stats

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gw_inference.grid import FrequencyGrid
from gw_inference.noise import generate_white_noise
from gw_inference.psd import welch_psd


def get_git_info() -> Dict[str, str]:
    """Retrieve Git branch and HEAD commit."""
    try:
        branch = (
            subprocess.check_output(
                ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--short"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return {"branch": branch, "head": head, "status_summary": status}
    except Exception as exc:
        return {"branch": "unknown", "head": "unknown", "error": str(exc)}


def run_stage_e1(output_dir: Path) -> Dict[str, Any]:
    """Execute the 100-realization calibration prototype."""
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Physical and Signal Processing Parameters
    fs = 2048.0
    dt = 1.0 / fs
    duration_off = 64.0
    duration_on = 4.0
    n_off = int(round(duration_off * fs))  # 131072
    n_on = int(round(duration_on * fs))    # 8192
    m_seg = 8192
    h_overlap = 4096                       # 50% overlap
    tukey_alpha = 0.25
    s0_true = 2.0 / fs                     # 9.765625e-4
    df = 1.0 / (n_on * dt)                 # 0.25 Hz
    f_low = 20.0
    f_high = 512.0

    # Effective degrees of freedom under Stage 5B white-noise overlap treatment
    nu_eff = 51894.0 / 1157.0              # 44.8522039758
    c_chi2_05 = float(stats.chi2.ppf(0.05, df=nu_eff))
    c_chi2_95 = float(stats.chi2.ppf(0.95, df=nu_eff))

    # Construct and validate frequency grid & active mask
    grid = FrequencyGrid(n_on, dt)
    mask = grid.create_active_mask(f_low, f_high)
    act_idx = grid.get_active_indices(mask)
    act_freqs = grid.get_active_frequencies(mask)
    n_act = len(act_idx)

    assert n_act == 1969, f"Expected 1969 active bins, got {n_act}"
    assert act_freqs[0] == 20.0, f"Expected first frequency 20.0 Hz, got {act_freqs[0]}"
    assert act_freqs[-1] == 512.0, f"Expected last frequency 512.0 Hz, got {act_freqs[-1]}"
    assert 0 not in act_idx, "DC must be excluded"
    assert (n_on // 2) not in act_idx, "Nyquist must be excluded"

    # Preflight seeds
    off_seeds = list(range(6301, 6401))
    on_seeds = list(range(6401, 6500 + 1))
    n_rep = len(off_seeds)
    assert n_rep == 100, f"Expected 100 experiments, got {n_rep}"
    assert len(on_seeds) == 100, f"Expected 100 on-source seeds, got {len(on_seeds)}"
    assert len(set(off_seeds).intersection(set(on_seeds))) == 0, "Seeds must be disjoint"

    print("================================================================================")
    print("WP6 Stage 5E.1: White-Noise Predictive Calibration Prototype Execution")
    print("================================================================================")
    print(f"Sampling frequency: {fs} Hz (dt = {dt:.6e} s)")
    print(f"Off-source duration: {duration_off} s ({n_off} samples, 31 segments, 50% overlap)")
    print(f"On-source duration:  {duration_on} s ({n_on} samples, df = {df} Hz)")
    print(f"Active band:         [{f_low}, {f_high}] Hz ({n_act} bins)")
    print(f"True PSD S0:         {s0_true:.6e} strain^2/Hz")
    print(f"Effective DOF:       {nu_eff:.10f} (51894/1157)")
    print(f"Nominal 90% chi2:    [{c_chi2_05:.6f}, {c_chi2_95:.6f}]")
    print(f"Replications:        {n_rep} independent pairs (off: 6301-6400, on: 6401-6500)")
    print("--------------------------------------------------------------------------------")

    percentiles_target = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    # Storage arrays
    results_per_realization: List[Dict[str, Any]] = []
    
    # Coverage tracking
    coverage_fractions = np.zeros(n_rep, dtype=np.float64)
    per_bin_covered = np.zeros((n_rep, n_act), dtype=bool)

    # Realization-level PIT summaries
    mean_pit_oracle = np.zeros(n_rep, dtype=np.float64)
    var_pit_oracle = np.zeros(n_rep, dtype=np.float64)
    ks_pit_oracle = np.zeros(n_rep, dtype=np.float64)
    quantiles_oracle = np.zeros((n_rep, len(percentiles_target)), dtype=np.float64)

    mean_pit_plug = np.zeros(n_rep, dtype=np.float64)
    var_pit_plug = np.zeros(n_rep, dtype=np.float64)
    ks_pit_plug = np.zeros(n_rep, dtype=np.float64)
    quantiles_plug = np.zeros((n_rep, len(percentiles_target)), dtype=np.float64)

    mean_pit_t = np.zeros(n_rep, dtype=np.float64)
    var_pit_t = np.zeros(n_rep, dtype=np.float64)
    ks_pit_t = np.zeros(n_rep, dtype=np.float64)
    quantiles_t = np.zeros((n_rep, len(percentiles_target)), dtype=np.float64)

    # Pooled PIT arrays
    all_pit_oracle = np.zeros((n_rep, n_act), dtype=np.float64)
    all_pit_plug = np.zeros((n_rep, n_act), dtype=np.float64)
    all_pit_t = np.zeros((n_rep, n_act), dtype=np.float64)

    # Realization loop
    for r_idx in range(n_rep):
        s_off = off_seeds[r_idx]
        s_on = on_seeds[r_idx]

        # 1. Off-source noise generation & PSD estimation
        x_off = generate_white_noise(n_off, sigma=1.0, seed=s_off)
        freqs_est, psd_full = welch_psd(
            x_off,
            delta_t=dt,
            segment_length=m_seg,
            overlap_samples=h_overlap,
            window="tukey",
            method="mean",
        )
        assert np.array_equal(freqs_est, grid.frequencies), "Frequency grid mismatch"
        psd_act = psd_full[act_idx]
        assert np.all(np.isfinite(psd_act)), f"Non-finite PSD in seed {s_off}"
        assert np.all(psd_act > 0.0), f"Non-positive PSD in seed {s_off}"

        # 2. On-source noise generation & continuous-equivalent Fourier residual
        x_on = generate_white_noise(n_on, sigma=1.0, seed=s_on)
        r_tilde_full = dt * np.fft.rfft(x_on)
        r_tilde_act = r_tilde_full[act_idx]
        r_power_act = np.abs(r_tilde_act) ** 2

        # 3. Model A: Oracle Gaussian Control
        q_oracle = (2.0 * df * r_power_act) / s0_true
        pit_oracle = 1.0 - np.exp(-q_oracle)

        # 4. Model B: Plug-in Gaussian Model
        q_plug = (2.0 * df * r_power_act) / psd_act
        pit_plug = 1.0 - np.exp(-q_plug)

        # 5. Model C: Candidate Student-t Predictive Model
        u_cand = (4.0 * df * r_power_act) / (nu_eff * psd_act)
        pit_t = 1.0 - (1.0 + u_cand) ** (-nu_eff / 2.0)

        # Basic range assertions
        assert np.all(pit_oracle >= 0.0) and np.all(pit_oracle <= 1.0), "Oracle PIT out of [0, 1]"
        assert np.all(pit_plug >= 0.0) and np.all(pit_plug <= 1.0), "Plug-in PIT out of [0, 1]"
        assert np.all(pit_t >= 0.0) and np.all(pit_t <= 1.0), "Student-t PIT out of [0, 1]"

        # 6. Nominal 90% InvGamma Credible Interval Coverage for S0
        # S_k | S_hat_k ~ InvGamma(nu/2, nu*S_hat_k/2)
        # S_0 is in CI iff c_chi2_05 <= nu*S_hat_k/S0 <= c_chi2_95
        s_ci_low = (nu_eff * psd_act) / c_chi2_95
        s_ci_high = (nu_eff * psd_act) / c_chi2_05
        covered_k = (s0_true >= s_ci_low) & (s0_true <= s_ci_high)
        cov_frac = float(np.mean(covered_k))

        coverage_fractions[r_idx] = cov_frac
        per_bin_covered[r_idx, :] = covered_k

        # Store pooled PITs
        all_pit_oracle[r_idx, :] = pit_oracle
        all_pit_plug[r_idx, :] = pit_plug
        all_pit_t[r_idx, :] = pit_t

        # Realization-level statistics
        m_ora = float(np.mean(pit_oracle))
        v_ora = float(np.var(pit_oracle))
        ks_ora = float(stats.kstest(pit_oracle, "uniform").statistic)
        q_ora = np.quantile(pit_oracle, percentiles_target)

        m_plg = float(np.mean(pit_plug))
        v_plg = float(np.var(pit_plug))
        ks_plg = float(stats.kstest(pit_plug, "uniform").statistic)
        q_plg = np.quantile(pit_plug, percentiles_target)

        m_t = float(np.mean(pit_t))
        v_t = float(np.var(pit_t))
        ks_t = float(stats.kstest(pit_t, "uniform").statistic)
        q_t = np.quantile(pit_t, percentiles_target)

        mean_pit_oracle[r_idx] = m_ora
        var_pit_oracle[r_idx] = v_ora
        ks_pit_oracle[r_idx] = ks_ora
        quantiles_oracle[r_idx, :] = q_ora

        mean_pit_plug[r_idx] = m_plg
        var_pit_plug[r_idx] = v_plg
        ks_pit_plug[r_idx] = ks_plg
        quantiles_plug[r_idx, :] = q_plg

        mean_pit_t[r_idx] = m_t
        var_pit_t[r_idx] = v_t
        ks_pit_t[r_idx] = ks_t
        quantiles_t[r_idx, :] = q_t

        results_per_realization.append(
            {
                "realization_index": r_idx + 1,
                "off_source_seed": s_off,
                "on_source_seed": s_on,
                "psd_mean_active": float(np.mean(psd_act)),
                "psd_min_active": float(np.min(psd_act)),
                "psd_max_active": float(np.max(psd_act)),
                "psd_rel_error_to_s0": float((np.mean(psd_act) - s0_true) / s0_true),
                "psd_90ci_coverage": cov_frac,
                "oracle": {
                    "mean_pit": m_ora,
                    "var_pit": v_ora,
                    "ks_statistic": ks_ora,
                    "quantiles": {
                        f"p_{int(round(p*100)):02d}": float(val)
                        for p, val in zip(percentiles_target, q_ora)
                    },
                },
                "plug_in": {
                    "mean_pit": m_plg,
                    "var_pit": v_plg,
                    "ks_statistic": ks_plg,
                    "quantiles": {
                        f"p_{int(round(p*100)):02d}": float(val)
                        for p, val in zip(percentiles_target, q_plg)
                    },
                },
                "student_t": {
                    "mean_pit": m_t,
                    "var_pit": v_t,
                    "ks_statistic": ks_t,
                    "quantiles": {
                        f"p_{int(round(p*100)):02d}": float(val)
                        for p, val in zip(percentiles_target, q_t)
                    },
                },
            }
        )

        if (r_idx + 1) % 20 == 0 or (r_idx + 1) == n_rep:
            print(
                f"Completed {r_idx+1:3d}/{n_rep:3d} pairs | "
                f"Coverage: {cov_frac:.4f} | "
                f"PIT Mean [Ora: {m_ora:.4f}, Plg: {m_plg:.4f}, Stu-t: {m_t:.4f}]"
            )

    t_elapsed = time.time() - t_start
    print(f"Simulation completed in {t_elapsed:.2f} s.")

    # Flatten pooled arrays
    flat_pit_oracle = all_pit_oracle.ravel()
    flat_pit_plug = all_pit_plug.ravel()
    flat_pit_t = all_pit_t.ravel()

    # 7. Aggregate Statistical Evaluation
    # A. PSD Coverage Summary (Statistical Unit = Realization)
    mean_cov = float(np.mean(coverage_fractions))
    std_cov = float(np.std(coverage_fractions, ddof=1))
    se_cov = std_cov / math.sqrt(n_rep)
    t_crit = float(stats.t.ppf(0.975, df=n_rep - 1))
    ci95_cov = [mean_cov - t_crit * se_cov, mean_cov + t_crit * se_cov]

    # Frequency-dependent coverage across the 100 realizations
    bin_cov_fractions = np.mean(per_bin_covered, axis=0)  # shape (n_act,)
    mean_bin_cov = float(np.mean(bin_cov_fractions))
    min_bin_cov = float(np.min(bin_cov_fractions))
    max_bin_cov = float(np.max(bin_cov_fractions))
    std_bin_cov = float(np.std(bin_cov_fractions, ddof=1))

    # B. Model Comparison Across Realizations (Mean ± SE)
    def calc_mean_se(arr: np.ndarray) -> Dict[str, float]:
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1))
        se = s / math.sqrt(len(arr))
        return {"mean": m, "std": s, "se": se}

    summary_oracle = {
        "mean_pit": calc_mean_se(mean_pit_oracle),
        "var_pit": calc_mean_se(var_pit_oracle),
        "ks_stat": calc_mean_se(ks_pit_oracle),
        "quantiles": {
            f"p_{int(round(p*100)):02d}": {
                "nominal": float(p),
                "mean": float(np.mean(quantiles_oracle[:, idx])),
                "std": float(np.std(quantiles_oracle[:, idx], ddof=1)),
                "se": float(np.std(quantiles_oracle[:, idx], ddof=1) / math.sqrt(n_rep)),
            }
            for idx, p in enumerate(percentiles_target)
        },
    }

    summary_plug = {
        "mean_pit": calc_mean_se(mean_pit_plug),
        "var_pit": calc_mean_se(var_pit_plug),
        "ks_stat": calc_mean_se(ks_pit_plug),
        "quantiles": {
            f"p_{int(round(p*100)):02d}": {
                "nominal": float(p),
                "mean": float(np.mean(quantiles_plug[:, idx])),
                "std": float(np.std(quantiles_plug[:, idx], ddof=1)),
                "se": float(np.std(quantiles_plug[:, idx], ddof=1) / math.sqrt(n_rep)),
            }
            for idx, p in enumerate(percentiles_target)
        },
    }

    summary_t = {
        "mean_pit": calc_mean_se(mean_pit_t),
        "var_pit": calc_mean_se(var_pit_t),
        "ks_stat": calc_mean_se(ks_pit_t),
        "quantiles": {
            f"p_{int(round(p*100)):02d}": {
                "nominal": float(p),
                "mean": float(np.mean(quantiles_t[:, idx])),
                "std": float(np.std(quantiles_t[:, idx], ddof=1)),
                "se": float(np.std(quantiles_t[:, idx], ddof=1) / math.sqrt(n_rep)),
            }
            for idx, p in enumerate(percentiles_target)
        },
    }

    # C. Pooled Statistics (with explicit dependence qualification)
    pooled_stats = {
        "sample_size": int(len(flat_pit_oracle)),
        "oracle": {
            "mean": float(np.mean(flat_pit_oracle)),
            "var": float(np.var(flat_pit_oracle)),
            "ks_stat": float(stats.kstest(flat_pit_oracle, "uniform").statistic),
            "tail_p01": float(np.mean(flat_pit_oracle <= 0.01)),
            "tail_p05": float(np.mean(flat_pit_oracle <= 0.05)),
            "tail_p95": float(np.mean(flat_pit_oracle >= 0.95)),
            "tail_p99": float(np.mean(flat_pit_oracle >= 0.99)),
        },
        "plug_in": {
            "mean": float(np.mean(flat_pit_plug)),
            "var": float(np.var(flat_pit_plug)),
            "ks_stat": float(stats.kstest(flat_pit_plug, "uniform").statistic),
            "tail_p01": float(np.mean(flat_pit_plug <= 0.01)),
            "tail_p05": float(np.mean(flat_pit_plug <= 0.05)),
            "tail_p95": float(np.mean(flat_pit_plug >= 0.95)),
            "tail_p99": float(np.mean(flat_pit_plug >= 0.99)),
        },
        "student_t": {
            "mean": float(np.mean(flat_pit_t)),
            "var": float(np.var(flat_pit_t)),
            "ks_stat": float(stats.kstest(flat_pit_t, "uniform").statistic),
            "tail_p01": float(np.mean(flat_pit_t <= 0.01)),
            "tail_p05": float(np.mean(flat_pit_t <= 0.05)),
            "tail_p95": float(np.mean(flat_pit_t >= 0.95)),
            "tail_p99": float(np.mean(flat_pit_t >= 0.99)),
        },
    }

    summary_data = {
        "metadata": {
            "title": "WP6 Stage 5E.1 White-Noise Predictive Calibration Prototype",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": t_elapsed,
            "git": get_git_info(),
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "parameters": {
                "sampling_frequency": fs,
                "delta_t": dt,
                "delta_f": df,
                "n_on_samples": n_on,
                "n_off_samples": n_off,
                "duration_on": duration_on,
                "duration_off": duration_off,
                "active_frequency_band": [f_low, f_high],
                "active_bin_count": n_act,
                "true_psd_s0": s0_true,
                "nu_eff": nu_eff,
                "n_replications": n_rep,
                "off_source_seed_range": [off_seeds[0], off_seeds[-1]],
                "on_source_seed_range": [on_seeds[0], on_seeds[-1]],
            },
        },
        "psd_coverage_summary": {
            "nominal_coverage": 0.90,
            "realization_level": {
                "mean_coverage": mean_cov,
                "std_coverage": std_cov,
                "se_coverage": se_cov,
                "ci95_coverage": ci95_cov,
                "min_coverage": float(np.min(coverage_fractions)),
                "max_coverage": float(np.max(coverage_fractions)),
            },
            "frequency_dependent": {
                "mean_per_bin_coverage": mean_bin_cov,
                "min_per_bin_coverage": min_bin_cov,
                "max_per_bin_coverage": max_bin_cov,
                "std_per_bin_coverage": std_bin_cov,
            },
        },
        "model_comparison_realization_level": {
            "oracle": summary_oracle,
            "plug_in": summary_plug,
            "student_t": summary_t,
        },
        "pooled_statistics_qualified": pooled_stats,
    }

    # Save JSON files
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary_data,
                "realizations": results_per_realization,
                "bin_coverage_frequencies": act_freqs.tolist(),
                "bin_coverage_fractions": bin_cov_fractions.tolist(),
            },
            f,
            indent=2,
        )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    manifest_path = output_dir / "manifest.json"
    manifest_data = {
        "stage": "WP6_Stage_5E.1",
        "description": "White-noise predictive likelihood calibration prototype artifacts",
        "files": {
            "summary.json": "Aggregated calibration statistics and model comparisons",
            "results.json": "Detailed per-realization data across 100 paired experiments",
            "manifest.json": "Provenance and artifact catalog",
            "pit_histograms.png": "Three-panel PIT distribution comparison against uniform reference",
            "pp_plots.png": "Probability-Probability empirical CDF comparison",
            "coverage_diagnostics.png": "PSD 90% credible interval coverage across realizations and frequencies",
            "quantile_deviations.png": "Quantile deviation plot (q_p - p) across percentiles",
            "WP6_Stage5E1_White_Noise_Calibration_Report.md": "Comprehensive scientific calibration report",
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 8. Generate Diagnostic Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: PIT Histograms
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        bins = np.linspace(0.0, 1.0, 21)

        axes[0].hist(flat_pit_oracle, bins=bins, density=True, color="#2b5c8f", alpha=0.8, edgecolor="black")
        axes[0].axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Uniform Density")
        axes[0].set_title(f"Oracle Gaussian (Known $S_0$)\nMean: {pooled_stats['oracle']['mean']:.4f}, Var: {pooled_stats['oracle']['var']:.4f}")
        axes[0].set_xlabel("PIT Value $U$")
        axes[0].set_ylabel("Probability Density")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(flat_pit_plug, bins=bins, density=True, color="#d95f02", alpha=0.8, edgecolor="black")
        axes[1].axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Uniform Density")
        axes[1].set_title(f"Plug-in Gaussian (Welch $\\widehat{{S}}_k$)\nMean: {pooled_stats['plug_in']['mean']:.4f}, Var: {pooled_stats['plug_in']['var']:.4f}")
        axes[1].set_xlabel("PIT Value $U$")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(flat_pit_t, bins=bins, density=True, color="#2ca02c", alpha=0.8, edgecolor="black")
        axes[2].axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Uniform Density")
        axes[2].set_title(f"Candidate Student-$t$ ($\\nu_{{\\rm eff}}={nu_eff:.2f}$)\nMean: {pooled_stats['student_t']['mean']:.4f}, Var: {pooled_stats['student_t']['var']:.4f}")
        axes[2].set_xlabel("PIT Value $U$")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "pit_histograms.png", dpi=300)
        plt.close()

        # Plot 2: P-P Plots
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        u_eval = np.linspace(0.0, 1.0, 1001)

        # Pooled empirical CDFs
        sorted_ora = np.sort(flat_pit_oracle)
        sorted_plg = np.sort(flat_pit_plug)
        sorted_t = np.sort(flat_pit_t)
        n_pts = len(sorted_ora)
        cdf_y = np.arange(1, n_pts + 1) / n_pts

        # Downsample for smooth plotting
        step = max(1, n_pts // 2000)
        axes[0].plot(sorted_ora[::step], cdf_y[::step], label="Oracle Gaussian", color="#2b5c8f", linewidth=1.8)
        axes[0].plot(sorted_plg[::step], cdf_y[::step], label="Plug-in Gaussian", color="#d95f02", linewidth=1.8)
        axes[0].plot(sorted_t[::step], cdf_y[::step], label=f"Student-$t$ ($\\nu={nu_eff:.1f}$)", color="#2ca02c", linewidth=1.8)
        axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", label="Ideal Uniform (Identity)")
        axes[0].set_title("Probability-Probability (P-P) Plot")
        axes[0].set_xlabel("Empirical PIT Quantile")
        axes[0].set_ylabel("Theoretical Uniform CDF")
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # Deviation plot: F_emp(u) - u
        f_emp_ora = np.interp(u_eval, sorted_ora, cdf_y)
        f_emp_plg = np.interp(u_eval, sorted_plg, cdf_y)
        f_emp_t = np.interp(u_eval, sorted_t, cdf_y)

        axes[1].plot(u_eval, f_emp_ora - u_eval, label="Oracle Gaussian", color="#2b5c8f", linewidth=1.8)
        axes[1].plot(u_eval, f_emp_plg - u_eval, label="Plug-in Gaussian", color="#d95f02", linewidth=1.8)
        axes[1].plot(u_eval, f_emp_t - u_eval, label=f"Student-$t$ ($\\nu={nu_eff:.1f}$)", color="#2ca02c", linewidth=1.8)
        axes[1].axhline(0.0, color="black", linestyle="--")
        axes[1].set_title("Calibration Deviation: $\\widehat{F}(u) - u$")
        axes[1].set_xlabel("Nominal Quantile $u$")
        axes[1].set_ylabel("Excess Cumulative Probability")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "pp_plots.png", dpi=300)
        plt.close()

        # Plot 3: Coverage Diagnostics
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(coverage_fractions, bins=15, color="#7570b3", edgecolor="black", alpha=0.8)
        axes[0].axvline(0.90, color="crimson", linestyle="--", linewidth=2.0, label="Nominal 90% Coverage")
        axes[0].axvline(mean_cov, color="navy", linestyle="-", linewidth=2.0, label=f"Mean: {mean_cov:.4f} $\\pm$ {se_cov:.4f}")
        axes[0].set_title("Realization-Level 90% InvGamma CI Coverage")
        axes[0].set_xlabel("Empirical Coverage Fraction Across Active Bins")
        axes[0].set_ylabel("Number of Realizations")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(act_freqs, bin_cov_fractions, color="#1b9e77", linewidth=1.2)
        axes[1].axhline(0.90, color="crimson", linestyle="--", linewidth=2.0, label="Nominal 90% Coverage")
        axes[1].axhline(mean_bin_cov, color="navy", linestyle="-", linewidth=1.5, label=f"Frequency Average: {mean_bin_cov:.4f}")
        axes[1].set_title("Frequency-Resolved 90% CI Coverage Across 100 Realizations")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Empirical Coverage Fraction")
        axes[1].set_ylim(0.80, 1.0)
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "coverage_diagnostics.png", dpi=300)
        plt.close()

        # Plot 4: Quantile Deviations
        fig, ax = plt.subplots(figsize=(8, 5))
        p_vals = percentiles_target
        dev_ora = [summary_oracle["quantiles"][f"p_{int(round(p*100)):02d}"]["mean"] - p for p in p_vals]
        se_ora = [summary_oracle["quantiles"][f"p_{int(round(p*100)):02d}"]["se"] for p in p_vals]

        dev_plg = [summary_plug["quantiles"][f"p_{int(round(p*100)):02d}"]["mean"] - p for p in p_vals]
        se_plg = [summary_plug["quantiles"][f"p_{int(round(p*100)):02d}"]["se"] for p in p_vals]

        dev_t = [summary_t["quantiles"][f"p_{int(round(p*100)):02d}"]["mean"] - p for p in p_vals]
        se_t = [summary_t["quantiles"][f"p_{int(round(p*100)):02d}"]["se"] for p in p_vals]

        ax.errorbar(p_vals, dev_ora, yerr=se_ora, fmt="o-", color="#2b5c8f", label="Oracle Gaussian", capsize=4)
        ax.errorbar(p_vals, dev_plg, yerr=se_plg, fmt="s-", color="#d95f02", label="Plug-in Gaussian", capsize=4)
        ax.errorbar(p_vals, dev_t, yerr=se_t, fmt="^-", color="#2ca02c", label=f"Student-$t$ ($\\nu={nu_eff:.1f}$)", capsize=4)
        ax.axhline(0.0, color="black", linestyle="--")
        ax.set_title("Quantile Deviations Across Percentiles: $\\hat{q}_p - p$")
        ax.set_xlabel("Nominal Percentile $p$")
        ax.set_ylabel("Quantile Deviation $\\hat{q}_p - p$")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "quantile_deviations.png", dpi=300)
        plt.close()

        print(f"Diagnostic plots successfully saved to {output_dir}")
    except Exception as exc:
        print(f"Warning: Failed to generate plots: {exc}")

    print("Stage 5E.1 Prototype Execution Complete.")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WP6 Stage 5E.1 Predictive Calibration Prototype")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/demonstrations/wp6_stage5/stage_e1",
        help="Path to output directory",
    )
    args = parser.parse_args()
    run_stage_e1(Path(args.output_dir))
