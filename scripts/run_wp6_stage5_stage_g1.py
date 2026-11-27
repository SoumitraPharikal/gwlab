#!/usr/bin/env python3
"""WP6 Stage 5G.1: Bounded Baseline Sampler Pilot Campaign Runner.

This script executes the authorized 18-run baseline sampler pilot:
- 3 paired realizations (Realizations 01, 02, 03)
- 3 likelihood models:
    * Model 1: Oracle Gaussian Likelihood (Controlled Reference, true S_n(f))
    * Model 2: Plug-in Gaussian Whittle Likelihood (Production Baseline, Welch S_hat(f))
    * Model 3: Candidate Independent-Bin Student-t Product Likelihood (nu_eff = 51894/1157)
- 2 independent sampler repeat seeds per model per realization
- Standard nested sampling via dynesty (wrapped by NestedSamplerAdapter):
    * nlive = 500
    * dlogz = 0.10
    * bound = 'multi'
    * sample = 'rwalk'
    * walks = 50
    * periodic = [3]  (coalescence phase phi_c in [0, 2*pi))

Strict Authorization Boundary:
- Exactly 18 baseline runs authorized.
- High-resolution runs (nlive=1000), Model 4 (Leakage Diagnostic), and Expansion Tier
  (Realizations 04-10) are strictly NOT authorized and prohibited.
- Production likelihood implementations in src/ are untouched.
- Full lossless artifact serialization with safe restart behavior.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.conventions import continuous_rfft
from gw_inference.grid import FrequencyGrid
from gw_inference.noise import generate_colored_noise
from gw_inference.psd import analytical_detector_psd, welch_psd
from gw_inference.sampler import NestedSamplerAdapter, NestedSamplingResult, is_dynesty_available
from gw_inference.waveform import apply_coalescence_time_shift, taylorf2_fd_waveform


# =====================================================================
# CONSTANTS & FROZEN PARAMETERS (WP6 Stage 5G Revision 1.3)
# =====================================================================

# Canonical TaylorF2 injection parameters
CANONICAL_MC = 28.095555795460438
CANONICAL_ETA = 0.24710059171597634
CANONICAL_TC = 2.0
CANONICAL_PHIC = 0.0
CANONICAL_DIST_MPC = 410.0
CANONICAL_DIST_M = CANONICAL_DIST_MPC * MPC_SI
CANONICAL_F_END = 512.0

# Analysis band and grid
FS = 2048.0
DT = 1.0 / FS
T_ON = 4.0
N_ON = int(FS * T_ON)  # 8192
DF = 1.0 / T_ON       # 0.25 Hz
F_START = 20.0
F_END = 512.0
K_START = int(round(F_START / DF))  # 80
K_END = int(round(F_END / DF))      # 2048
N_ACT = K_END - K_START + 1         # 1969

# Preserved 64 s off-source Welch PSD configuration
T_OFF = 64.0
N_OFF = int(FS * T_OFF)  # 131072
M_SEG = 8192
H_HOP = 4096
TUKEY_ALPHA = 0.25
K_SEG = 31
NU_EFF = 51894.0 / 1157.0  # approx 44.8522039758

# 4D parameter boundaries (uniform prior)
MC_MIN, MC_MAX = 20.0, 40.0
ETA_MIN, ETA_MAX = 0.10, 0.25
TC_MIN, TC_MAX = 1.90, 2.10
PHIC_MIN, PHIC_MAX = 0.0, 2.0 * math.pi
PRIOR_VOLUME = (MC_MAX - MC_MIN) * (ETA_MAX - ETA_MIN) * (TC_MAX - TC_MIN) * (PHIC_MAX - PHIC_MIN)
LN_PRIOR_DENSITY = -math.log(PRIOR_VOLUME)

# Point F diagnostic anchor coordinates
POINT_F_MC = 28.1500
POINT_F_ETA = 0.2400
POINT_F_TC = 2.0020
POINT_F_PHIC = 0.5 * math.pi

# Baseline Nested Sampler Settings
BASELINE_NLIVE = 500
BASELINE_DLOGZ = 0.10
BASELINE_BOUND = "multi"
BASELINE_SAMPLE = "rwalk"
BASELINE_WALKS = 50
BASELINE_PERIODIC = [3]

# Pilot Tier Seed Schedule (Table 4.1 in Stage 5G Design)
PILOT_SCHEDULE = [
    {
        "realization": 1,
        "seed_off": 7101,
        "seed_on": 7151,
        "sampler_seeds": [8101, 8201],
    },
    {
        "realization": 2,
        "seed_off": 7102,
        "seed_on": 7152,
        "sampler_seeds": [8102, 8202],
    },
    {
        "realization": 3,
        "seed_off": 7103,
        "seed_on": 7153,
        "sampler_seeds": [8103, 8203],
    },
]

MODELS = ["oracle", "plug", "student_t"]


# =====================================================================
# HELPER FUNCTIONS & PROVENANCE
# =====================================================================


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy scalars and arrays."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(path)
    if not p.is_file():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


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
    except Exception as e:
        return {"branch": "UNKNOWN", "head": "UNKNOWN", "error": str(e)}


def prior_transform_4d(u: np.ndarray) -> np.ndarray:
    """Map unit hypercube u in [0, 1]^4 to physical parameter vector theta."""
    theta = np.empty_like(u)
    theta[0] = MC_MIN + (MC_MAX - MC_MIN) * u[0]
    theta[1] = ETA_MIN + (ETA_MAX - ETA_MIN) * u[1]
    theta[2] = TC_MIN + (TC_MAX - TC_MIN) * u[2]
    theta[3] = PHIC_MIN + (PHIC_MAX - PHIC_MIN) * u[3]
    return theta


def generate_waveform_active(
    grid_on: FrequencyGrid,
    mc: float,
    eta: float,
    tc: float,
    phic: float,
    k_act: np.ndarray,
    dist_m: float = CANONICAL_DIST_M,
    f_end: float = CANONICAL_F_END,
) -> np.ndarray:
    """Generate TaylorF2 waveform evaluated on active frequency bins."""
    delta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    m1 = 0.5 * mc * (eta ** (-0.6)) * (1.0 + delta) * M_SUN_SI
    m2 = 0.5 * mc * (eta ** (-0.6)) * (1.0 - delta) * M_SUN_SI

    h_fd = taylorf2_fd_waveform(
        grid_on.frequencies,
        m1,
        m2,
        dist_m,
        phi_c=phic,
        f_end=f_end,
    )
    h_shifted = apply_coalescence_time_shift(h_fd, grid_on.frequencies, tc)
    return h_shifted[k_act]


def compute_waveform_distance_from_act(
    h_theta_act: np.ndarray,
    h_inj_act: np.ndarray,
    psd_true_act: np.ndarray,
    df: float = DF,
) -> float:
    """Compute Oracle waveform distance D(theta)."""
    dh = h_theta_act - h_inj_act
    return float(4.0 * df * np.sum((np.abs(dh) ** 2) / psd_true_act))


def assign_mismatch_shell(d_val: float) -> int:
    """Assign waveform distance to mutually exclusive shells S0, S1, S2, S3.

    S0: 0 <= D <= 9
    S1: 9 < D <= 25
    S2: 25 < D <= 100
    S3: D > 100
    """
    if d_val <= 9.0:
        return 0
    elif d_val <= 25.0:
        return 1
    elif d_val <= 100.0:
        return 2
    else:
        return 3


def in_point_f_ball(theta: np.ndarray) -> bool:
    """Check whether theta falls into the continuous coordinate ball B_eps(theta_F)."""
    mc, eta, tc, phic = theta
    return bool(
        abs(mc - POINT_F_MC) <= 0.01
        and abs(eta - POINT_F_ETA) <= 0.005
        and abs(tc - POINT_F_TC) <= 0.001
        and abs(phic - POINT_F_PHIC) <= 0.1
    )


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """Compute weighted quantiles of an array."""
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    cumsum = np.cumsum(w_sorted)
    cdf = (cumsum - 0.5 * w_sorted) / cumsum[-1]
    return np.interp(quantiles, cdf, v_sorted)


# =====================================================================
# DATA GENERATION & LIKELIHOOD FACTORY
# =====================================================================


def prepare_realization_data(sched: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize off-source Welch PSD and on-source paired data for a realization."""
    r_id = sched["realization"]
    seed_off = sched["seed_off"]
    seed_on = sched["seed_on"]

    grid_on = FrequencyGrid(n_samples=N_ON, delta_t=DT)
    k_act = np.arange(K_START, K_END + 1)
    psd_true_full = analytical_detector_psd(grid_on.frequencies, model="aligo_representative")
    psd_true_act = psd_true_full[k_act]

    # Off-source Welch PSD (64 s, 31 segments, Tukey 0.25)
    grid_off = FrequencyGrid(n_samples=N_OFF, delta_t=DT)
    psd_true_off = analytical_detector_psd(grid_off.frequencies, model="aligo_representative")
    rng_off = np.random.default_rng(seed_off)
    noise_off = generate_colored_noise(
        n_samples=grid_off.n_samples,
        delta_t=DT,
        psd=psd_true_off,
        generator=rng_off,
    )
    _, s_welch_full = welch_psd(
        noise_off,
        delta_t=DT,
        segment_length=M_SEG,
        overlap_samples=M_SEG - H_HOP,
        window="tukey",
    )
    s_welch_act = s_welch_full[k_act]

    # Precomputed Likelihood Normalization Constants
    ln_norm_oracle = -float(np.sum(np.log((math.pi * psd_true_act) / (2.0 * DF))))
    ln_norm_plug = -float(np.sum(np.log((math.pi * s_welch_act) / (2.0 * DF))))

    # On-source data synthesis: canonical injection + noise
    rng_on = np.random.default_rng(seed_on)
    noise_on = generate_colored_noise(
        n_samples=grid_on.n_samples,
        delta_t=DT,
        psd=psd_true_full,
        generator=rng_on,
    )
    noise_fd_act = continuous_rfft(noise_on, delta_t=DT)[k_act]
    h_inj_act = generate_waveform_active(
        grid_on, CANONICAL_MC, CANONICAL_ETA, CANONICAL_TC, CANONICAL_PHIC, k_act
    )
    d_act = h_inj_act + noise_fd_act

    # Injection SNR
    opt_snr = math.sqrt(float(4.0 * DF * np.sum((np.abs(h_inj_act) ** 2) / psd_true_act)))

    return {
        "realization": r_id,
        "seed_off": seed_off,
        "seed_on": seed_on,
        "grid_on": grid_on,
        "k_act": k_act,
        "psd_true_act": psd_true_act,
        "s_welch_act": s_welch_act,
        "d_act": d_act,
        "h_inj_act": h_inj_act,
        "ln_norm_oracle": ln_norm_oracle,
        "ln_norm_plug": ln_norm_plug,
        "optimal_snr": opt_snr,
    }


def make_likelihood_fn(
    model: str,
    real_data: Dict[str, Any],
) -> Callable[[np.ndarray], float]:
    """Construct standalone fast log-likelihood callable for nested sampling."""
    grid_on = real_data["grid_on"]
    k_act = real_data["k_act"]
    d_act = real_data["d_act"]
    psd_true_act = real_data["psd_true_act"]
    s_welch_act = real_data["s_welch_act"]
    ln_norm_oracle = real_data["ln_norm_oracle"]
    ln_norm_plug = real_data["ln_norm_plug"]
    df = DF
    nu_eff = NU_EFF

    if model == "oracle":
        def logl_oracle(theta: np.ndarray) -> float:
            mc, eta, tc, phic = theta
            if not (MC_MIN <= mc <= MC_MAX and ETA_MIN <= eta <= ETA_MAX and TC_MIN <= tc <= TC_MAX and PHIC_MIN <= phic <= PHIC_MAX):
                return -np.inf
            h_th = generate_waveform_active(grid_on, mc, eta, tc, phic, k_act)
            res_sq = np.abs(d_act - h_th) ** 2
            q = 4.0 * df * np.sum(res_sq / psd_true_act)
            return float(ln_norm_oracle - 0.5 * q)
        return logl_oracle

    elif model == "plug":
        def logl_plug(theta: np.ndarray) -> float:
            mc, eta, tc, phic = theta
            if not (MC_MIN <= mc <= MC_MAX and ETA_MIN <= eta <= ETA_MAX and TC_MIN <= tc <= TC_MAX and PHIC_MIN <= phic <= PHIC_MAX):
                return -np.inf
            h_th = generate_waveform_active(grid_on, mc, eta, tc, phic, k_act)
            res_sq = np.abs(d_act - h_th) ** 2
            q = 4.0 * df * np.sum(res_sq / s_welch_act)
            return float(ln_norm_plug - 0.5 * q)
        return logl_plug

    elif model == "student_t":
        def logl_student_t(theta: np.ndarray) -> float:
            mc, eta, tc, phic = theta
            if not (MC_MIN <= mc <= MC_MAX and ETA_MIN <= eta <= ETA_MAX and TC_MIN <= tc <= TC_MAX and PHIC_MIN <= phic <= PHIC_MAX):
                return -np.inf
            h_th = generate_waveform_active(grid_on, mc, eta, tc, phic, k_act)
            res_sq = np.abs(d_act - h_th) ** 2
            u_bins = (4.0 * df * res_sq) / (nu_eff * s_welch_act)
            ln_fit = -(0.5 * nu_eff + 1.0) * np.sum(np.log(1.0 + u_bins))
            return float(ln_norm_plug + ln_fit)
        return logl_student_t

    else:
        raise ValueError(f"Unknown model: {model}")


# =====================================================================
# SINGLE RUN WORKER WITH SAFE RESTART
# =====================================================================


def run_single_job(
    job_spec: Dict[str, Any],
    real_data: Dict[str, Any],
    output_dir: Path,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute a single nested sampling run with safe restart and lossless artifact serialization."""
    r_id = job_spec["realization"]
    model = job_spec["model"]
    seed = job_spec["sampler_seed"]
    run_id = f"run_{r_id:02d}_{model}_{seed}"

    json_path = output_dir / f"{run_id}.json"
    npz_path = output_dir / f"{run_id}_samples.npz"

    # Safe restart check: do not overwrite completed run artifacts
    if not force and json_path.is_file() and npz_path.is_file():
        try:
            with open(json_path, "r") as f:
                saved_res = json.load(f)
            # Verify file integrity
            if saved_res.get("status") == "SUCCESS" and "log_evidence" in saved_res:
                print(f"[{run_id}] Found existing valid artifact on disk. Skipping execution (Safe Restart).")
                return saved_res
        except Exception as e:
            print(f"[{run_id}] Existing artifact corrupted ({e}); re-running.")

    print(f"[{run_id}] Launching baseline nested sampling (nlive={BASELINE_NLIVE}, dlogz={BASELINE_DLOGZ}, seed={seed})...")
    start_time = time.perf_counter()

    log_l_fn = make_likelihood_fn(model, real_data)
    adapter = NestedSamplerAdapter(
        log_likelihood_fn=log_l_fn,
        prior_transform_fn=prior_transform_4d,
        ndim=4,
        nlive=BASELINE_NLIVE,
        dlogz=BASELINE_DLOGZ,
        seed=seed,
        param_names=["mc", "eta", "tc", "phic"],
    )

    try:
        res = adapter.run(
            bound=BASELINE_BOUND,
            sample=BASELINE_SAMPLE,
            walks=BASELINE_WALKS,
            periodic=BASELINE_PERIODIC,
            print_progress=False,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        err_res = {
            "run_id": run_id,
            "realization": r_id,
            "model": model,
            "sampler_seed": seed,
            "status": "FAILED",
            "error": str(e),
            "wall_time_s": elapsed,
        }
        with open(json_path, "w") as f:
            json.dump(err_res, f, indent=2, cls=NumpyEncoder)
        print(f"[{run_id}] FAILED after {elapsed:.2f} s: {e}")
        return err_res

    elapsed = time.perf_counter() - start_time

    # Extract results and diagnostics
    samples = np.asarray(res.samples)  # (N_samples, 4)
    weights = np.asarray(res.weights)  # (N_samples,)
    logwt = np.asarray(res.log_weights)
    logl = np.asarray(res.raw_results.logl)
    logvol = np.asarray(res.raw_results.logvol)
    ncall = int(res.n_evaluations)
    logz = float(res.log_evidence)
    logzerr = float(res.log_evidence_err)

    # Effective sample size N_eff = 1 / sum(w_s^2)
    n_eff = float(1.0 / np.sum(weights ** 2))

    # Profile integral ln(I_m) = ln(Z_m) - ln(L_norm^(m))
    ln_norm = real_data["ln_norm_oracle"] if model == "oracle" else real_data["ln_norm_plug"]
    log_integral = float(logz - ln_norm)

    # Compute marginal statistics for all 4 parameters
    param_names = ["mc", "eta", "tc", "phic"]
    inj_vals = [CANONICAL_MC, CANONICAL_ETA, CANONICAL_TC, CANONICAL_PHIC]
    param_stats = {}
    quantiles_to_eval = [0.01, 0.025, 0.05, 0.50, 0.95, 0.975, 0.99]
    q_keys = ["q_0010", "q_0025", "q_0050", "q_0500", "q_0950", "q_0975", "q_0990"]

    for p_idx, p_name in enumerate(param_names):
        vals = samples[:, p_idx]
        mean_val = float(np.sum(weights * vals))
        std_val = float(math.sqrt(np.sum(weights * ((vals - mean_val) ** 2))))
        q_vals = weighted_quantile(vals, weights, quantiles_to_eval)
        q_dict = {k: float(v) for k, v in zip(q_keys, q_vals)}
        q_dict["q_0.01"] = float(q_vals[0])
        q_dict["q_0.025"] = float(q_vals[1])
        q_dict["q_0.05"] = float(q_vals[2])
        q_dict["q_0.50"] = float(q_vals[3])
        q_dict["q_0.95"] = float(q_vals[4])
        q_dict["q_0.975"] = float(q_vals[5])
        q_dict["q_0.99"] = float(q_vals[6])

        inj_val = inj_vals[p_idx]
        z_score = float((mean_val - inj_val) / std_val) if std_val > 0 else 0.0

        param_stats[p_name] = {
            "mean": mean_val,
            "std": std_val,
            "injection": inj_val,
            "z_score": z_score,
            "quantiles": q_dict,
            "ci90_width": float(q_dict["q_0.95"] - q_dict["q_0.05"]),
            "ci95_width": float(q_dict["q_0.975"] - q_dict["q_0.025"]),
        }

    # Compute waveform distance D(theta_s) for each sample and shell masses
    grid_on = real_data["grid_on"]
    k_act = real_data["k_act"]
    h_inj_act = real_data["h_inj_act"]
    psd_true_act = real_data["psd_true_act"]

    n_samples = len(samples)
    d_vals = np.empty(n_samples, dtype=np.float64)
    point_f_hits = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        th_s = samples[i]
        h_s = generate_waveform_active(grid_on, th_s[0], th_s[1], th_s[2], th_s[3], k_act)
        d_vals[i] = compute_waveform_distance_from_act(h_s, h_inj_act, psd_true_act)
        point_f_hits[i] = in_point_f_ball(th_s)

    # Shell masses S0, S1, S2, S3
    mask_s0 = d_vals <= 9.0
    mask_s1 = (d_vals > 9.0) & (d_vals <= 25.0)
    mask_s2 = (d_vals > 25.0) & (d_vals <= 100.0)
    mask_s3 = d_vals > 100.0

    p_s0 = float(np.sum(weights[mask_s0]))
    p_s1 = float(np.sum(weights[mask_s1]))
    p_s2 = float(np.sum(weights[mask_s2]))
    p_s3 = float(np.sum(weights[mask_s3]))
    p_point_f = float(np.sum(weights[point_f_hits]))

    # Lossless NPZ artifact serialization
    np.savez_compressed(
        npz_path,
        samples=samples,
        weights=weights,
        logwt=logwt,
        logl=logl,
        logvol=logvol,
        d_vals=d_vals,
    )

    result_record = {
        "run_id": run_id,
        "realization": r_id,
        "model": model,
        "sampler_seed": seed,
        "status": "SUCCESS",
        "log_evidence": logz,
        "log_evidence_err": logzerr,
        "log_norm": ln_norm,
        "log_integral": log_integral,
        "n_evaluations": ncall,
        "wall_time_s": elapsed,
        "evals_per_sec": float(ncall / elapsed) if elapsed > 0 else 0.0,
        "n_dead_points": n_samples,
        "n_eff": n_eff,
        "parameters": param_stats,
        "shell_masses": {
            "S0": p_s0,
            "S1": p_s1,
            "S2": p_s2,
            "S3": p_s3,
            "point_f_neighborhood": p_point_f,
        },
        "waveform_distance": {
            "min": float(np.min(d_vals)),
            "max": float(np.max(d_vals)),
            "mean": float(np.sum(weights * d_vals)),
            "median": float(weighted_quantile(d_vals, weights, [0.5])[0]),
        },
        "artifacts": {
            "json": str(json_path.name),
            "npz": str(npz_path.name),
        },
    }

    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2, cls=NumpyEncoder)

    print(
        f"[{run_id}] Finished in {elapsed:.2f} s | "
        f"ln(Z) = {logz:.4f} +/- {logzerr:.4f} | "
        f"ncall = {ncall} ({ncall/elapsed:.1f} evals/s) | "
        f"N_eff = {n_eff:.1f} | "
        f"S0={p_s0:.4f}, S1={p_s1:.4f}, S2={p_s2:.4f}, S3={p_s3:.6f}"
    )

    return result_record


# =====================================================================
# AGGREGATION & SCIENTIFIC SUMMARY
# =====================================================================


def summarize_campaign(
    all_runs: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Aggregate per-run results into realization-level means, differences, and statistical tests."""
    # Organize runs by realization, model, and seed
    runs_by_key: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    for r in all_runs:
        runs_by_key[(r["realization"], r["model"], r["sampler_seed"])] = r

    realizations_summary = []
    delta_logz_32_list = []
    delta_logz_31_list = []

    for r_idx in [1, 2, 3]:
        r_sched = PILOT_SCHEDULE[r_idx - 1]
        s1, s2 = r_sched["sampler_seeds"]

        model_summaries = {}
        for m in MODELS:
            run_1 = runs_by_key[(r_idx, m, s1)]
            run_2 = runs_by_key[(r_idx, m, s2)]

            # Check repeat stability
            d_repeat_logz = run_1["log_evidence"] - run_2["log_evidence"]
            sigma_pooled = math.sqrt(run_1["log_evidence_err"] ** 2 + run_2["log_evidence_err"] ** 2)
            z_repeat = abs(d_repeat_logz) / sigma_pooled if sigma_pooled > 0 else 0.0

            mean_logz = 0.5 * (run_1["log_evidence"] + run_2["log_evidence"])
            mean_log_integral = 0.5 * (run_1["log_integral"] + run_2["log_integral"])

            # Shell masses mean
            mean_s0 = 0.5 * (run_1["shell_masses"]["S0"] + run_2["shell_masses"]["S0"])
            mean_s1 = 0.5 * (run_1["shell_masses"]["S1"] + run_2["shell_masses"]["S1"])
            mean_s2 = 0.5 * (run_1["shell_masses"]["S2"] + run_2["shell_masses"]["S2"])
            mean_s3 = 0.5 * (run_1["shell_masses"]["S3"] + run_2["shell_masses"]["S3"])
            mean_pf = 0.5 * (run_1["shell_masses"]["point_f_neighborhood"] + run_2["shell_masses"]["point_f_neighborhood"])

            # Parameter std dev means
            std_mc = 0.5 * (run_1["parameters"]["mc"]["std"] + run_2["parameters"]["mc"]["std"])
            std_eta = 0.5 * (run_1["parameters"]["eta"]["std"] + run_2["parameters"]["eta"]["std"])
            std_tc = 0.5 * (run_1["parameters"]["tc"]["std"] + run_2["parameters"]["tc"]["std"])
            std_phic = 0.5 * (run_1["parameters"]["phic"]["std"] + run_2["parameters"]["phic"]["std"])

            model_summaries[m] = {
                "mean_logz": mean_logz,
                "mean_log_integral": mean_log_integral,
                "repeat_d_logz": d_repeat_logz,
                "repeat_sigma_pooled": sigma_pooled,
                "repeat_z_score": z_repeat,
                "repeat_stable_flag": bool(z_repeat <= 3.0),
                "shell_masses": {
                    "S0": mean_s0,
                    "S1": mean_s1,
                    "S2": mean_s2,
                    "S3": mean_s3,
                    "point_f_neighborhood": mean_pf,
                },
                "param_std": {
                    "mc": std_mc,
                    "eta": std_eta,
                    "tc": std_tc,
                    "phic": std_phic,
                },
            }

        # Paired differences for realization r
        delta_logz_32 = model_summaries["student_t"]["mean_logz"] - model_summaries["plug"]["mean_logz"]
        delta_logz_31 = model_summaries["student_t"]["mean_logz"] - model_summaries["oracle"]["mean_logz"]
        delta_logz_32_list.append(delta_logz_32)
        delta_logz_31_list.append(delta_logz_31)

        delta_log_integral_32 = model_summaries["student_t"]["mean_log_integral"] - model_summaries["plug"]["mean_log_integral"]
        delta_log_integral_31 = model_summaries["student_t"]["mean_log_integral"] - model_summaries["oracle"]["mean_log_integral"]

        # Paired shell mass differences
        delta_p_31 = {
            s_key: model_summaries["student_t"]["shell_masses"][s_key] - model_summaries["oracle"]["shell_masses"][s_key]
            for s_key in ["S0", "S1", "S2", "S3"]
        }
        delta_p_32 = {
            s_key: model_summaries["student_t"]["shell_masses"][s_key] - model_summaries["plug"]["shell_masses"][s_key]
            for s_key in ["S0", "S1", "S2", "S3"]
        }

        # Credible interval width ratios: sigma_3 / sigma_1 and sigma_3 / sigma_2
        width_ratios_31 = {
            p: model_summaries["student_t"]["param_std"][p] / model_summaries["oracle"]["param_std"][p]
            for p in ["mc", "eta", "tc", "phic"]
        }
        width_ratios_32 = {
            p: model_summaries["student_t"]["param_std"][p] / model_summaries["plug"]["param_std"][p]
            for p in ["mc", "eta", "tc", "phic"]
        }

        realizations_summary.append({
            "realization": r_idx,
            "models": model_summaries,
            "paired_evidence_differences": {
                "delta_logz_3_minus_2": delta_logz_32,
                "delta_logz_3_minus_1": delta_logz_31,
                "delta_log_integral_3_minus_2": delta_log_integral_32,
                "delta_log_integral_3_minus_1": delta_log_integral_31,
            },
            "paired_shell_mass_differences": {
                "delta_p_3_minus_1": delta_p_31,
                "delta_p_3_minus_2": delta_p_32,
            },
            "credible_interval_width_ratios": {
                "sigma_3_over_sigma_1": width_ratios_31,
                "sigma_3_over_sigma_2": width_ratios_32,
            },
        })

    # Sample statistics across the 3 pilot realizations
    mean_delta_32 = float(np.mean(delta_logz_32_list))
    std_delta_32 = float(np.std(delta_logz_32_list, ddof=1)) if len(delta_logz_32_list) > 1 else 0.0
    se_delta_32 = float(std_delta_32 / math.sqrt(len(delta_logz_32_list)))
    se_10_projected = float(std_delta_32 / math.sqrt(10.0))

    mean_delta_31 = float(np.mean(delta_logz_31_list))
    std_delta_31 = float(np.std(delta_logz_31_list, ddof=1)) if len(delta_logz_31_list) > 1 else 0.0
    se_delta_31 = float(std_delta_31 / math.sqrt(len(delta_logz_31_list)))

    # Totals across all runs
    total_calls = sum(r["n_evaluations"] for r in all_runs)
    total_wall_s = sum(r["wall_time_s"] for r in all_runs)
    min_n_eff = min(r["n_eff"] for r in all_runs)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_realizations": 3,
        "n_models": 3,
        "n_seeds_per_model": 2,
        "total_runs": len(all_runs),
        "total_likelihood_evaluations": total_calls,
        "total_wall_time_s": total_wall_s,
        "total_core_hours": total_wall_s / 3600.0,
        "min_effective_sample_size": min_n_eff,
        "overall_status": "SUCCESS" if all(r["status"] == "SUCCESS" for r in all_runs) else "FAILED",
        "paired_evidence_statistics": {
            "delta_logz_32": {
                "mean": mean_delta_32,
                "std": std_delta_32,
                "se_pilot_n3": se_delta_32,
                "projected_se_n10": se_10_projected,
                "expansion_precision_pass": bool(se_10_projected <= 0.25),
            },
            "delta_logz_31": {
                "mean": mean_delta_31,
                "std": std_delta_31,
                "se_pilot_n3": se_delta_31,
            },
        },
        "realizations": realizations_summary,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    return summary


# =====================================================================
# FIGURE GENERATION
# =====================================================================


def generate_all_figures(
    all_runs: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_dir: Path,
) -> List[str]:
    """Generate high-quality diagnostic figures for the Stage 5G.1 baseline pilot."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated_figures = []

    # Map runs by (realization, model, seed)
    runs_by_key = {(r["realization"], r["model"], r["sampler_seed"]): r for r in all_runs}

    # 1. Posterior Contours Comparison (Realization 01, Seed 1)
    # Compare 1D and 2D marginal distributions for (mc, eta) and (tc, phic)
    fig_path_1 = fig_dir / "posterior_contours_comparison.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=200)

    # Colors for models
    colors = {"oracle": "#1f77b4", "plug": "#ff7f0e", "student_t": "#2ca02c"}
    labels = {
        "oracle": "Model 1: Oracle Gaussian",
        "plug": "Model 2: Plug-in Gaussian",
        "student_t": "Model 3: Student-t (nu=44.85)",
    }

    # Load samples from Realization 01 Seed 8101
    for m in MODELS:
        npz_p = output_dir / f"run_01_{m}_8101_samples.npz"
        if npz_p.is_file():
            data = np.load(npz_p)
            s_arr = data["samples"]
            w_arr = data["weights"]

            # mc 1D (ax 0,0)
            axes[0, 0].hist(
                s_arr[:, 0],
                bins=40,
                weights=w_arr,
                density=True,
                histtype="step",
                color=colors[m],
                label=labels[m],
                linewidth=1.8,
            )

            # eta 1D (ax 1,1)
            axes[1, 1].hist(
                s_arr[:, 1],
                bins=40,
                weights=w_arr,
                density=True,
                histtype="step",
                color=colors[m],
                linewidth=1.8,
            )

            # mc vs eta 2D scatter (ax 1,0)
            sub_idx = np.random.choice(len(s_arr), size=min(800, len(s_arr)), p=w_arr)
            axes[1, 0].scatter(
                s_arr[sub_idx, 0],
                s_arr[sub_idx, 1],
                s=6,
                color=colors[m],
                alpha=0.4,
                label=labels[m],
            )

            # tc vs phic 2D scatter (ax 0,1)
            axes[0, 1].scatter(
                s_arr[sub_idx, 2],
                s_arr[sub_idx, 3],
                s=6,
                color=colors[m],
                alpha=0.4,
            )

    # Injections
    axes[0, 0].axvline(CANONICAL_MC, color="black", linestyle="--", linewidth=1.2, label="Injection")
    axes[1, 1].axvline(CANONICAL_ETA, color="black", linestyle="--", linewidth=1.2)
    axes[1, 0].scatter(CANONICAL_MC, CANONICAL_ETA, color="red", marker="+", s=100, linewidth=2, zorder=5)
    axes[0, 1].scatter(CANONICAL_TC, CANONICAL_PHIC, color="red", marker="+", s=100, linewidth=2, zorder=5)

    axes[0, 0].set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$")
    axes[0, 0].set_ylabel("Posterior Density")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[1, 0].set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$")
    axes[1, 0].set_ylabel(r"$\eta$")

    axes[0, 1].set_xlabel(r"$t_c\ [\mathrm{s}]$")
    axes[0, 1].set_ylabel(r"$\phi_c\ [\mathrm{rad}]$")

    axes[1, 1].set_xlabel(r"$\eta$")
    axes[1, 1].set_ylabel("Posterior Density")

    fig.suptitle("WP6 Stage 5G.1: Posterior Marginal Distributions (Realization 01)", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(fig_path_1)
    plt.close(fig)
    generated_figures.append(str(fig_path_1.name))

    # 2. Evidence Accumulation Profile p(ln X) vs ln X
    fig_path_2 = fig_dir / "evidence_accumulation_profile.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    for m in MODELS:
        npz_p = output_dir / f"run_01_{m}_8101_samples.npz"
        if npz_p.is_file():
            data = np.load(npz_p)
            logvol = data["logvol"]
            logl = data["logl"]
            weights = data["weights"]

            # Sort by decreasing prior volume (increasing -logvol)
            sorter = np.argsort(logvol)
            lv = logvol[sorter]
            wt = weights[sorter]

            # Approximate p(ln X) = dZ / d(ln X)
            dlv = np.gradient(lv)
            dlv[dlv == 0] = 1e-10
            p_lnx = wt / np.abs(dlv)

            ax.plot(lv, p_lnx, color=colors[m], label=labels[m], linewidth=1.8)

    ax.set_xlabel(r"$\ln X$ (Prior Volume Compression)")
    ax.set_ylabel(r"Evidence Density $p(\ln X) \propto \mathcal{L}(X) X$")
    ax.set_title("WP6 Stage 5G.1: Evidence Accumulation Profiles (Realization 01)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path_2)
    plt.close(fig)
    generated_figures.append(str(fig_path_2.name))

    # 3. Waveform-Distance Shell Probability Distribution
    fig_path_3 = fig_dir / "shell_probability_distribution.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=200, sharey=True)
    shells = ["S0", "S1", "S2", "S3"]
    x = np.arange(len(shells))
    bar_width = 0.25

    for r_idx in [1, 2, 3]:
        ax = axes[r_idx - 1]
        r_sum = summary["realizations"][r_idx - 1]

        p_oracle = [r_sum["models"]["oracle"]["shell_masses"][s] for s in shells]
        p_plug = [r_sum["models"]["plug"]["shell_masses"][s] for s in shells]
        p_student = [r_sum["models"]["student_t"]["shell_masses"][s] for s in shells]

        ax.bar(x - bar_width, p_oracle, width=bar_width, color=colors["oracle"], label="Model 1 (Oracle)")
        ax.bar(x, p_plug, width=bar_width, color=colors["plug"], label="Model 2 (Plug-in)")
        ax.bar(x + bar_width, p_student, width=bar_width, color=colors["student_t"], label="Model 3 (Student-t)")

        ax.set_xticks(x)
        ax.set_xticklabels([r"$\mathcal{S}_0$" + "\n[0,9]", r"$\mathcal{S}_1$" + "\n(9,25]", r"$\mathcal{S}_2$" + "\n(25,100]", r"$\mathcal{S}_3$" + "\n>100"])
        ax.set_title(f"Realization {r_idx:02d}")
        ax.set_xlabel("Mismatch Shell")
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        if r_idx == 1:
            ax.set_ylabel("Posterior Probability Mass")
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("WP6 Stage 5G.1: Waveform-Distance Shell Mass Distribution Across Realizations", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path_3, bbox_inches="tight")
    plt.close(fig)
    generated_figures.append(str(fig_path_3.name))

    # 4. Repeat Seed Consistency
    fig_path_4 = fig_dir / "repeat_seed_consistency.png"
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

    # Plot ln(Z) with error bars for each model across realizations
    indices = []
    labels_tick = []
    idx = 0
    for r_idx in [1, 2, 3]:
        r_sched = PILOT_SCHEDULE[r_idx - 1]
        s1, s2 = r_sched["sampler_seeds"]
        for m in MODELS:
            r1 = runs_by_key[(r_idx, m, s1)]
            r2 = runs_by_key[(r_idx, m, s2)]

            ax.errorbar(idx - 0.15, r1["log_evidence"], yerr=r1["log_evidence_err"], fmt="o", color=colors[m], capsize=3)
            ax.errorbar(idx + 0.15, r2["log_evidence"], yerr=r2["log_evidence_err"], fmt="s", color=colors[m], capsize=3)

            indices.append(idx)
            m_short = "Orcl" if m == "oracle" else ("Plug" if m == "plug" else "Stud")
            labels_tick.append(f"R{r_idx}_{m_short}")
            idx += 1
        idx += 0.5  # gap between realizations

    ax.set_xticks(indices)
    ax.set_xticklabels(labels_tick, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"Estimated Log-Evidence $\ln\mathcal{Z}$")
    ax.set_title("WP6 Stage 5G.1: Sampler Repeat-Seed Consistency (Seed 1 = Circle, Seed 2 = Square)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_path_4)
    plt.close(fig)
    generated_figures.append(str(fig_path_4.name))

    return generated_figures


# =====================================================================
# MAIN RUNNER ENTRY POINT
# =====================================================================


def main() -> int:
    """Execute WP6 Stage 5G.1 campaign."""
    parser = argparse.ArgumentParser(description="WP6 Stage 5G.1 Baseline Sampler Pilot Runner")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Synthesize data, check schedule, and exit without running samplers")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing run artifacts")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/demonstrations/wp6_stage5/stage_g1",
        help="Output directory for artifacts and manifest",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WP6 STAGE 5G.1: BOUNDED BASELINE SAMPLER PILOT CAMPAIGN")
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Preflight Environment & Provenance
    git_info = get_git_info()
    print(f"Git Branch: {git_info['branch']}")
    print(f"Git Commit HEAD: {git_info['head']}")
    print(f"Working Tree Status: {'DIRTY (preserved uncommitted tree)' if git_info.get('status_summary') else 'CLEAN'}")
    print(f"Platform: {platform.platform()}")
    print(f"Python Executable: {sys.executable} (version {platform.python_version()})")
    print(f"Dynesty Available: {is_dynesty_available()}")

    if not is_dynesty_available():
        print("ERROR: dynesty is not available in the current environment.")
        return 1

    # 2. Build Schedule and Precompute Realization Data
    print("\n--- Precomputing Realization Data ---")
    realizations_data: Dict[int, Dict[str, Any]] = {}
    for sched in PILOT_SCHEDULE:
        r_id = sched["realization"]
        print(f"Realization {r_id:02d}: off-seed={sched['seed_off']}, on-seed={sched['seed_on']}")
        r_data = prepare_realization_data(sched)
        realizations_data[r_id] = r_data
        print(f"  Canonical Optimal SNR: {r_data['optimal_snr']:.6f}")
        print(f"  ln L_norm Oracle: {r_data['ln_norm_oracle']:.4f}")
        print(f"  ln L_norm Plug-in: {r_data['ln_norm_plug']:.4f}")

    # 3. Build List of 18 Authorized Jobs
    jobs = []
    for sched in PILOT_SCHEDULE:
        r_id = sched["realization"]
        for m in MODELS:
            for s in sched["sampler_seeds"]:
                jobs.append({
                    "realization": r_id,
                    "model": m,
                    "sampler_seed": s,
                })

    print(f"\nTotal authorized jobs: {len(jobs)} runs (3 realizations x 3 models x 2 sampler seeds)")

    if args.dry_run:
        print("\n[DRY RUN] Verification successful. Exiting without sampling.")
        return 0

    # 4. Save Manifest
    manifest = {
        "campaign": "WP6_Stage_5G1_Baseline_Sampler_Pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "configuration": {
            "canonical_injection": {
                "mc_msun": CANONICAL_MC,
                "eta": CANONICAL_ETA,
                "tc_s": CANONICAL_TC,
                "phic_rad": CANONICAL_PHIC,
                "dist_mpc": CANONICAL_DIST_MPC,
                "f_end_hz": CANONICAL_F_END,
            },
            "grid": {
                "fs_hz": FS,
                "t_on_s": T_ON,
                "n_on": N_ON,
                "df_hz": DF,
                "f_start_hz": F_START,
                "f_end_hz": F_END,
                "n_act": N_ACT,
            },
            "welch_psd": {
                "t_off_s": T_OFF,
                "n_off": N_OFF,
                "segment_length": M_SEG,
                "hop_length": H_HOP,
                "tukey_alpha": TUKEY_ALPHA,
                "k_segments": K_SEG,
                "nu_eff": NU_EFF,
            },
            "prior": {
                "mc_range": [MC_MIN, MC_MAX],
                "eta_range": [ETA_MIN, ETA_MAX],
                "tc_range": [TC_MIN, TC_MAX],
                "phic_range": [PHIC_MIN, PHIC_MAX],
                "prior_volume": PRIOR_VOLUME,
                "ln_prior_density": LN_PRIOR_DENSITY,
            },
            "sampler": {
                "engine": "dynesty",
                "nlive": BASELINE_NLIVE,
                "dlogz": BASELINE_DLOGZ,
                "bound": BASELINE_BOUND,
                "sample": BASELINE_SAMPLE,
                "walks": BASELINE_WALKS,
                "periodic": BASELINE_PERIODIC,
            },
            "mismatch_shells": {
                "S0": "[0.0, 9.0]",
                "S1": "(9.0, 25.0]",
                "S2": "(25.0, 100.0]",
                "S3": "> 100.0",
                "point_f": {
                    "mc": POINT_F_MC,
                    "eta": POINT_F_ETA,
                    "tc": POINT_F_TC,
                    "phic": POINT_F_PHIC,
                },
            },
            "schedule": PILOT_SCHEDULE,
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)

    # 5. Execute Jobs (Parallel or Sequential)
    print(f"\n--- Executing 18 Baseline Runs (Workers: {args.workers}) ---")
    start_campaign = time.perf_counter()
    completed_runs: List[Dict[str, Any]] = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(
                    run_single_job,
                    job,
                    realizations_data[job["realization"]],
                    out_dir,
                    args.force,
                ): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                res = future.result()
                completed_runs.append(res)
    else:
        for job in jobs:
            res = run_single_job(
                job,
                realizations_data[job["realization"]],
                out_dir,
                args.force,
            )
            completed_runs.append(res)

    total_elapsed = time.perf_counter() - start_campaign
    print(f"\nAll 18 jobs finished in {total_elapsed:.2f} s ({total_elapsed / 60.0:.2f} min)")

    # 6. Save results.json
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"runs": completed_runs}, f, indent=2, cls=NumpyEncoder)
    print(f"Saved complete run results to {results_path}")

    # 7. Summarize Campaign & Check Stop Conditions
    print("\n--- Generating Statistical Summary ---")
    summary = summarize_campaign(completed_runs, out_dir)

    # 8. Generate Figures
    if not args.skip_plots:
        print("\n--- Generating Diagnostic Figures ---")
        figs = generate_all_figures(completed_runs, summary, out_dir)
        for fig in figs:
            print(f"  Generated: {fig}")

    # 9. Print Campaign Summary Table
    print("\n" + "=" * 80)
    print("WP6 STAGE 5G.1 BASELINE PILOT: CAMPAIGN SUMMARY")
    print("=" * 80)
    print(f"Total Completed Runs: {len(completed_runs)} / 18")
    print(f"Total Likelihood Evaluations: {summary['total_likelihood_evaluations']}")
    print(f"Total CPU Core-Hours: {summary['total_core_hours']:.3f} core-hrs")
    print(f"Minimum Posterior ESS: {summary['min_effective_sample_size']:.1f}")

    print("\n--- Realization-Level Evidence Differences ---")
    for r_sum in summary["realizations"]:
        r_id = r_sum["realization"]
        d32 = r_sum["paired_evidence_differences"]["delta_logz_3_minus_2"]
        d31 = r_sum["paired_evidence_differences"]["delta_logz_3_minus_1"]
        print(f"Realization {r_id:02d}: Delta ln(Z) [3 - 2] = {d32:+.4f} nats | Delta ln(Z) [3 - 1] = {d31:+.4f} nats")
        for m in MODELS:
            m_s = r_sum["models"][m]
            print(f"  {m:10s}: ln(Z) = {m_s['mean_logz']:+.4f} | Repeat diff = {m_s['repeat_d_logz']:+.4f} (z = {m_s['repeat_z_score']:.2f}, stable={m_s['repeat_stable_flag']})")

    p_stats = summary["paired_evidence_statistics"]["delta_logz_32"]
    print("\n--- Paired Student-t vs. Plug-in Evidence Statistics Across 3 Pilot Realizations ---")
    print(f"Mean Delta ln(Z) [3 - 2]: {p_stats['mean']:+.4f} nats")
    print(f"Sample Std Dev: {p_stats['std']:.4f} nats")
    print(f"Pilot Standard Error (N=3): {p_stats['se_pilot_n3']:.4f} nats")
    print(f"Projected Standard Error at N=10: {p_stats['projected_se_n10']:.4f} nats (Target: <= 0.25 nats)")
    print(f"Expansion Precision Pass: {p_stats['expansion_precision_pass']}")

    print("\n--- Mismatch Shell Masses Across Realizations (Model 3 Student-t) ---")
    for r_sum in summary["realizations"]:
        r_id = r_sum["realization"]
        sm = r_sum["models"]["student_t"]["shell_masses"]
        print(f"Realization {r_id:02d}: S0={sm['S0']:.4f}, S1={sm['S1']:.4f}, S2={sm['S2']:.4f}, S3={sm['S3']:.6f}, PtF={sm['point_f_neighborhood']:.2e}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
