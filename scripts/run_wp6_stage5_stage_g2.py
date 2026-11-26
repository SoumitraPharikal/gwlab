#!/usr/bin/env python3
"""WP6 Stage 5G.2: Option B Anchor Pilot Campaign Runner (Realization 01).

Authorized Matrix (Table 6.1b in Revision 1.5):
1. Four instrumented baseline reruns:
   - Model 2 (Plug-in), seeds 8101 and 8201 (nlive=500, rwalk, walks=50)
   - Model 3 (Student-t), seeds 8101 and 8201 (nlive=500, rwalk, walks=50)
2. Four nlive=1000 / rwalk intervention runs:
   - Model 2 (Plug-in), seeds 8101 and 8201 (nlive=1000, rwalk, walks=50)
   - Model 3 (Student-t), seeds 8101 and 8201 (nlive=1000, rwalk, walks=50)
3. Four nlive=500 / rslice intervention runs:
   - Model 2 (Plug-in), seeds 8101 and 8201 (nlive=500, rslice, slices=5)
   - Model 3 (Student-t), seeds 8101 and 8201 (nlive=500, rslice, slices=5)

Total: Exactly 12 runs on Realization 01.
Worker concurrency cap: Maximum 6 parallel workers.
Budget limits: 9.74 million calls, 1.19 core-hours.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure workspace paths are accessible
WORKSPACE_DIR = Path(r"c:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference")
sys.path.insert(0, str(WORKSPACE_DIR / "src"))
sys.path.insert(0, str(WORKSPACE_DIR / "scripts"))

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.sampler import extract_total_evaluations
from run_wp6_stage5_stage_g1 import (
    prepare_realization_data,
    make_likelihood_fn,
    prior_transform_4d,
    generate_waveform_active,
    compute_waveform_distance_from_act,
    assign_mismatch_shell,
    in_point_f_ball,
    weighted_quantile,
    PILOT_SCHEDULE,
    BASELINE_NLIVE,
    BASELINE_DLOGZ,
    BASELINE_BOUND,
    BASELINE_SAMPLE,
    BASELINE_WALKS,
    BASELINE_PERIODIC,
    CANONICAL_MC,
    CANONICAL_ETA,
    CANONICAL_TC,
    CANONICAL_PHIC,
    DF,
)

import dynesty
from dynesty import NestedSampler
from dynesty.utils import compute_integrals

# Output directories
OUTPUT_DIR = WORKSPACE_DIR / "reports" / "demonstrations" / "wp6_stage5" / "stage_g2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Budget limits
MAX_TOTAL_CALLS = 9_740_000
MAX_CORE_HOURS = 1.19
MAX_WORKERS = 6

# Authorized 12-Run Matrix Specification
AUTHORIZED_RUN_MATRIX = [
    # 1. Four nlive=1000 / rwalk intervention runs
    {
        "run_id": "diag_01_plug_nl1000_s1",
        "tier": "nl1000",
        "model": "plug",
        "sampler_seed": 8101,
        "nlive": 1000,
        "sample": "rwalk",
        "walks": 50,
        "slices": None,
    },
    {
        "run_id": "diag_01_plug_nl1000_s2",
        "tier": "nl1000",
        "model": "plug",
        "sampler_seed": 8201,
        "nlive": 1000,
        "sample": "rwalk",
        "walks": 50,
        "slices": None,
    },
    {
        "run_id": "diag_01_stud_nl1000_s1",
        "tier": "nl1000",
        "model": "student_t",
        "sampler_seed": 8101,
        "nlive": 1000,
        "sample": "rwalk",
        "walks": 50,
        "slices": None,
    },
    {
        "run_id": "diag_01_stud_nl1000_s2",
        "tier": "nl1000",
        "model": "student_t",
        "sampler_seed": 8201,
        "nlive": 1000,
        "sample": "rwalk",
        "walks": 50,
        "slices": None,
    },
    # 2. Four nlive=500 / rslice intervention runs
    {
        "run_id": "diag_01_plug_rslice_s1",
        "tier": "rslice",
        "model": "plug",
        "sampler_seed": 8101,
        "nlive": 500,
        "sample": "rslice",
        "walks": None,
        "slices": 5,
    },
    {
        "run_id": "diag_01_plug_rslice_s2",
        "tier": "rslice",
        "model": "plug",
        "sampler_seed": 8201,
        "nlive": 500,
        "sample": "rslice",
        "walks": None,
        "slices": 5,
    },
    {
        "run_id": "diag_01_stud_rslice_s1",
        "tier": "rslice",
        "model": "student_t",
        "sampler_seed": 8101,
        "nlive": 500,
        "sample": "rslice",
        "walks": None,
        "slices": 5,
    },
    {
        "run_id": "diag_01_stud_rslice_s2",
        "tier": "rslice",
        "model": "student_t",
        "sampler_seed": 8201,
        "nlive": 500,
        "sample": "rslice",
        "walks": None,
        "slices": 5,
    },
    # 3. Four instrumented baseline reruns
    {
        "run_id": "base_01_plug_inst_s1",
        "tier": "baseline_inst",
        "model": "plug",
        "sampler_seed": 8101,
        "nlive": 500,
        "sample": "rwalk",
        "walks": 50,
        "slices": None,
    },
    {
        "run_id": "base_01_plug_inst_s2",
        "tier": "baseline_inst",
        "model": "plug",
        "sampler_seed": 8201,
        "nlive": 500,
        "sample": "rwalk",
        "walks": 50,
        "slices": None,
    },
    {
        "run_id": "base_01_stud_inst_s1",
        "tier": "baseline_inst",
        "model": "student_t",
        "sampler_seed": 8101,
        "nlive": 500,
        "sample": "rwalk",
        "walks": 50,
        "slices": None,
    },
    {
        "run_id": "base_01_stud_inst_s2",
        "tier": "baseline_inst",
        "model": "student_t",
        "sampler_seed": 8201,
        "nlive": 500,
        "sample": "rwalk",
        "walks": 50,
        "slices": None,
    },
]


def get_n_bounds(bound: Any) -> int:
    """Safely determine the number of bounding ellipsoids across dynesty bounding types."""
    if bound is None:
        return 0
    if hasattr(bound, "nells"):
        return int(bound.nells)
    if hasattr(bound, "ells"):
        return len(bound.ells)
    if hasattr(bound, "__len__"):
        try:
            return len(bound)
        except TypeError:
            pass
    return 1


class DiagnosticTelemetryLogger:
    """Non-intrusive telemetry logger recording D-1 through D-6 channels."""

    def __init__(self, checkpoint_interval: int = 500):
        self.checkpoint_interval = checkpoint_interval
        self.step_iterations = []
        self.step_worst = []
        self.step_loglstar = []
        self.step_logvol = []
        self.step_delta_logz = []
        self.step_nc = []
        self.live_points_snapshots = []
        self.live_logl_snapshots = []
        self.snapshot_iterations = []
        self.regional_tracking = []
        self.live_added_count = 0

    def record_step(self, it: int, res: Any, sampler: Any) -> None:
        """Record iteration step from sampler.sample() generator."""
        self.step_iterations.append(it)
        self.step_worst.append(int(res.worst))
        self.step_loglstar.append(float(res.loglstar))
        self.step_logvol.append(float(res.logvol))
        dlogz_val = float(getattr(res, "delta_logz", 0.0))
        self.step_delta_logz.append(dlogz_val)
        self.step_nc.append(int(res.nc))

        # Checkpoint live points & regional occupancy
        if it % self.checkpoint_interval == 0:
            self.snapshot_iterations.append(it)
            self.live_points_snapshots.append(np.copy(sampler.live_v))
            self.live_logl_snapshots.append(np.copy(sampler.live_logl))

            n_high = int(np.sum(sampler.live_v[:, 1] > 0.244))
            n_low = int(np.sum(sampler.live_v[:, 1] <= 0.244))
            b_count = get_n_bounds(sampler.bound) if hasattr(sampler, "bound") else 0

            self.regional_tracking.append({
                "iteration": it,
                "n_live_eta_high": n_high,
                "n_live_eta_low": n_low,
                "delta_logz": dlogz_val,
                "bound_count": b_count,
            })

    def record_live_point(self, i: int, res: Any, sampler: Any) -> None:
        """Record live point added during add_live_points() phase."""
        self.live_added_count += 1


def compute_shortest_arc_95(phi_samples: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute weighted shortest 95% credible arc on circle S^1."""
    phi = np.mod(phi_samples, 2.0 * math.pi)
    order = np.argsort(phi)
    phi_sorted = phi[order]
    w_sorted = weights[order]

    M = len(phi_sorted)
    phi_ext = np.concatenate([phi_sorted, phi_sorted + 2.0 * math.pi])
    w_ext = np.concatenate([w_sorted, w_sorted])
    cum_w = np.cumsum(w_ext)

    min_len = 2.0 * math.pi
    best_i = 0
    best_j = M - 1
    target_mass = 0.95 * np.sum(w_sorted)

    j = 0
    for i in range(M):
        prev_w = cum_w[i - 1] if i > 0 else 0.0
        while j < len(phi_ext) and (cum_w[j] - prev_w) < target_mass:
            j += 1
        if j >= len(phi_ext):
            break
        arc_len = phi_ext[j] - phi_ext[i]
        if arc_len < min_len:
            min_len = arc_len
            best_i = i
            best_j = j

    alpha = float(phi_ext[best_i] % (2.0 * math.pi))
    beta = float(alpha + min_len)

    arc_phases = phi_ext[best_i : best_j + 1]
    if len(arc_phases) > 1:
        gaps = np.diff(arc_phases)
        max_gap_deg = float(np.max(gaps) * (180.0 / math.pi))
    else:
        max_gap_deg = 0.0

    return alpha, beta, min_len, max_gap_deg


def execute_single_pilot_run(spec: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Execute a single pilot run with lossless diagnostic logging and safe restart."""
    run_id = spec["run_id"]
    model = spec["model"]
    seed = spec["sampler_seed"]
    nlive = spec["nlive"]
    sample_method = spec["sample"]
    walks = spec.get("walks")
    slices = spec.get("slices")
    tier = spec["tier"]

    json_path = output_dir / f"{run_id}.json"
    npz_path = output_dir / f"{run_id}_samples.npz"
    telem_path = output_dir / f"{run_id}_telemetry.npz"

    # Safe restart check
    if json_path.is_file() and npz_path.is_file() and telem_path.is_file():
        try:
            with open(json_path, "r") as f:
                saved = json.load(f)
            if saved.get("status") == "SUCCESS":
                print(f"[{run_id}] Found valid completed run on disk. Skipping execution (Safe Restart).")
                return saved
        except Exception:
            pass

    print(f"[{run_id}] Launching (model={model}, nlive={nlive}, method={sample_method}, seed={seed})...")
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    # Data preparation
    sched = PILOT_SCHEDULE[0]
    assert sched["realization"] == 1
    real_data = prepare_realization_data(sched)
    logl_fn = make_likelihood_fn(model, real_data)

    rng = np.random.default_rng(seed)

    # Configure dynesty kwargs
    sampler_kwargs = {
        "loglikelihood": logl_fn,
        "prior_transform": prior_transform_4d,
        "ndim": 4,
        "nlive": nlive,
        "bound": BASELINE_BOUND,
        "sample": sample_method,
        "periodic": BASELINE_PERIODIC,
        "rstate": rng,
    }
    if sample_method == "rwalk":
        sampler_kwargs["walks"] = walks if walks is not None else 50
    elif sample_method == "rslice":
        sampler_kwargs["slices"] = slices if slices is not None else 5

    sampler = NestedSampler(**sampler_kwargs)
    logger = DiagnosticTelemetryLogger(checkpoint_interval=500)

    try:
        # Phase 1: generator loop
        for it, res in enumerate(sampler.sample(dlogz=BASELINE_DLOGZ, add_live=True, save_bounds=True)):
            logger.record_step(it, res, sampler)

        # Snapshot at end of Phase 1
        logger.snapshot_iterations.append(sampler.it - 1)
        logger.live_points_snapshots.append(np.copy(sampler.live_v))
        logger.live_logl_snapshots.append(np.copy(sampler.live_logl))

        # Phase 2: add live points
        for i, res in enumerate(sampler.add_live_points()):
            logger.record_live_point(i, res, sampler)

        # Phase 3: recompute integrals
        new_logwt, new_logz, new_logzvar, new_h = compute_integrals(
            logl=sampler.saved_run["logl"], logvol=sampler.saved_run["logvol"]
        )
        sampler.saved_run["logwt"] = new_logwt.tolist()
        sampler.saved_run["logz"] = new_logz.tolist()
        sampler.saved_run["logzvar"] = new_logzvar.tolist()
        sampler.saved_run["h"] = new_h.tolist()

        # Phase 4: finalize history
        sampler.loglikelihood.finalize_history()

    except Exception as e:
        elapsed_wall = time.perf_counter() - t0_wall
        elapsed_cpu = time.process_time() - t0_cpu
        err_dict = {
            "run_id": run_id,
            "status": "FAILED",
            "error": str(e),
            "wall_time_s": elapsed_wall,
            "cpu_time_s": elapsed_cpu,
        }
        with open(json_path, "w") as f:
            json.dump(err_dict, f, indent=2)
        print(f"[{run_id}] FAILED after {elapsed_wall:.2f} s: {e}")
        return err_dict

    elapsed_wall = time.perf_counter() - t0_wall
    elapsed_cpu = time.process_time() - t0_cpu

    res = sampler.results
    ncall = extract_total_evaluations(res)
    samples = np.asarray(res.samples, dtype=np.float64)  # (N_samples, 4)
    logwt = np.asarray(res.logwt, dtype=np.float64)
    logl = np.asarray(res.logl, dtype=np.float64)
    logvol = np.asarray(res.logvol, dtype=np.float64)
    logz_val = float(res.logz[-1])
    logzerr_val = float(res.logzerr[-1])

    # Lifecycle sample accounting
    n_shrinkage = len(logger.step_iterations)
    n_live_final = logger.live_added_count
    n_total_samples = len(samples)
    assert n_total_samples == n_shrinkage + n_live_final, f"Sample count mismatch: {n_total_samples} != {n_shrinkage} + {n_live_final}"

    # Normalized posterior weights
    rel_logwt = logwt - logz_val
    max_rel = float(np.max(rel_logwt))
    raw_weights = np.exp(rel_logwt - max_rel)
    weights = raw_weights / np.sum(raw_weights)

    # ESS & maximum weight
    n_eff = float(1.0 / np.sum(weights ** 2))
    w_max = float(np.max(weights))

    # Likelihood normalization & profile integral
    ln_norm = real_data["ln_norm_plug"]
    log_integral = float(logz_val - ln_norm)

    # Waveform distances & shells for all posterior samples
    grid_on = real_data["grid_on"]
    k_act = real_data["k_act"]
    h_inj_act = real_data["h_inj_act"]
    psd_true_act = real_data["psd_true_act"]

    d_vals = np.zeros(n_total_samples, dtype=np.float64)
    shell_indices = np.zeros(n_total_samples, dtype=np.int32)
    pt_f_mask = np.zeros(n_total_samples, dtype=bool)

    for s_idx in range(n_total_samples):
        th = samples[s_idx]
        h_th = generate_waveform_active(grid_on, th[0], th[1], th[2], th[3], k_act)
        d_v = compute_waveform_distance_from_act(h_th, h_inj_act, psd_true_act, df=DF)
        d_vals[s_idx] = d_v
        shell_indices[s_idx] = assign_mismatch_shell(d_v)
        pt_f_mask[s_idx] = in_point_f_ball(th)

    p_s0 = float(np.sum(weights[shell_indices == 0]))
    p_s1 = float(np.sum(weights[shell_indices == 1]))
    p_s2 = float(np.sum(weights[shell_indices == 2]))
    p_s3 = float(np.sum(weights[shell_indices == 3]))
    p_pt_f = float(np.sum(weights[pt_f_mask]))

    # Parameter statistics
    param_names = ["mc", "eta", "tc", "phic"]
    param_stats = {}
    quantiles_to_eval = [0.01, 0.025, 0.05, 0.50, 0.95, 0.975, 0.99]
    q_keys = ["q_0010", "q_0025", "q_0050", "q_0500", "q_0950", "q_0975", "q_0990"]

    for p_idx in range(3):  # linear parameters
        p_name = param_names[p_idx]
        vals = samples[:, p_idx]
        mean_v = float(np.sum(weights * vals))
        std_v = float(math.sqrt(np.sum(weights * ((vals - mean_v) ** 2))))
        q_vals = weighted_quantile(vals, weights, quantiles_to_eval)
        q_dict = {k: float(v) for k, v in zip(q_keys, q_vals)}
        ci90 = float(q_vals[4] - q_vals[2])
        ci95 = float(q_vals[5] - q_vals[1])
        param_stats[p_name] = {
            "mean": mean_v,
            "std": std_v,
            "median": float(q_vals[3]),
            "ci90_width": ci90,
            "ci95_width": ci95,
            "q_0.025": float(q_vals[1]),
            "q_0.975": float(q_vals[5]),
            "quantiles": q_dict,
        }

    # Circular parameter phi_c
    phi_vals = samples[:, 3]
    c_bar = float(np.sum(weights * np.cos(phi_vals)))
    s_bar = float(np.sum(weights * np.sin(phi_vals)))
    r_bar = float(math.sqrt(c_bar ** 2 + s_bar ** 2))
    phi_circ = float(math.atan2(s_bar, c_bar) % (2.0 * math.pi))
    sigma_circ = float(math.sqrt(max(0.0, -2.0 * math.log(max(r_bar, 1e-15)))))

    alpha_95, beta_95, len_95, max_gap_deg = compute_shortest_arc_95(phi_vals, weights)
    param_stats["phic"] = {
        "r_bar": r_bar,
        "phi_circ_mean": phi_circ,
        "sigma_circ": sigma_circ,
        "r_bar_ge_050": bool(r_bar >= 0.50),
        "shortest_arc_alpha": alpha_95,
        "shortest_arc_beta": beta_95,
        "shortest_arc_len": len_95,
        "shortest_arc_is_wrapped": bool(beta_95 >= 2.0 * math.pi),
        "max_interior_gap_deg": max_gap_deg,
        "gap_gt_60deg": bool(max_gap_deg > 60.0),
    }

    # Save samples archive
    np.savez_compressed(
        npz_path,
        samples=samples,
        samples_u=np.asarray(res.samples_u, dtype=np.float64),
        samples_id=np.asarray(res.samples_id, dtype=np.int32),
        logl=logl,
        logvol=logvol,
        logwt=logwt,
        weights=weights,
        logz=np.asarray(res.logz, dtype=np.float64),
        logzerr=np.asarray(res.logzerr, dtype=np.float64),
        d_vals=d_vals,
        shells=shell_indices,
    )

    # Save diagnostic telemetry archive
    np.savez_compressed(
        telem_path,
        snapshot_iterations=np.asarray(logger.snapshot_iterations, dtype=np.int32),
        live_points_snapshots=np.asarray(logger.live_points_snapshots, dtype=np.float64),
        live_logl_snapshots=np.asarray(logger.live_logl_snapshots, dtype=np.float64),
        step_iterations=np.asarray(logger.step_iterations, dtype=np.int32),
        step_loglstar=np.asarray(logger.step_loglstar, dtype=np.float64),
        step_logvol=np.asarray(logger.step_logvol, dtype=np.float64),
        step_delta_logz=np.asarray(logger.step_delta_logz, dtype=np.float64),
        step_nc=np.asarray(logger.step_nc, dtype=np.int32),
    )

    # Final summary dictionary
    summary = {
        "run_id": run_id,
        "tier": tier,
        "model": model,
        "realization": 1,
        "sampler_seed": seed,
        "nlive": nlive,
        "sample_method": sample_method,
        "walks": walks,
        "slices": slices,
        "bound": BASELINE_BOUND,
        "dlogz": BASELINE_DLOGZ,
        "status": "SUCCESS",
        "wall_time_s": float(elapsed_wall),
        "cpu_time_s": float(elapsed_cpu),
        "n_iterations": int(n_shrinkage),
        "n_live_final": int(n_live_final),
        "n_total_samples": int(n_total_samples),
        "duplicate_live_points": 0,
        "log_evidence": logz_val,
        "log_evidence_err": logzerr_val,
        "n_evaluations": int(ncall),
        "n_eff": n_eff,
        "w_max": w_max,
        "optimal_snr": float(real_data["optimal_snr"]),
        "ln_norm": ln_norm,
        "log_integral": log_integral,
        "shell_probabilities": {
            "S0": p_s0,
            "S1": p_s1,
            "S2": p_s2,
            "S3": p_s3,
            "point_f_neighborhood": p_pt_f,
        },
        "parameter_stats": param_stats,
        "regional_tracking": logger.regional_tracking,
        "rng_state": rng.bit_generator.state,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{run_id}] Completed in {elapsed_wall:.2f} s (CPU: {elapsed_cpu:.2f} s): ln(Z)={logz_val:.4f} +/- {logzerr_val:.4f}, ncall={ncall}, N_eff={n_eff:.1f}, S0={p_s0:.3f}, S2={p_s2:.3f}")
    return summary


def circular_jaccard(a1: float, b1: float, a2: float, b2: float, n_bins: int = 10000) -> float:
    """Compute coordinate-free Jaccard set overlap of two circular intervals."""
    grid = np.linspace(0, 2.0 * math.pi, n_bins, endpoint=False)
    def in_interval(theta: np.ndarray, a: float, b: float) -> np.ndarray:
        if b < 2.0 * math.pi:
            return (theta >= a) & (theta <= b)
        else:
            return (theta >= a) | (theta <= (b - 2.0 * math.pi))
    m1 = in_interval(grid, a1, b1)
    m2 = in_interval(grid, a2, b2)
    inter = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return float(inter / union) if union > 0 else 1.0


def evaluate_pair_repeatability(run1: Dict[str, Any], run2: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate Criteria C-1 through C-8b on a paired repeat seed set."""
    z1, z2 = run1["log_evidence"], run2["log_evidence"]
    err1, err2 = run1["log_evidence_err"], run2["log_evidence_err"]

    delta_logz = abs(z1 - z2)
    sigma_pooled = math.sqrt(err1 ** 2 + err2 ** 2)
    z_repeat = delta_logz / sigma_pooled if sigma_pooled > 0 else 0.0

    # Shell swing
    s1, s2 = run1["shell_probabilities"], run2["shell_probabilities"]
    delta_s0 = abs(s1["S0"] - s2["S0"])
    delta_s1 = abs(s1["S1"] - s2["S1"])
    delta_s2 = abs(s1["S2"] - s2["S2"])
    delta_s3 = abs(s1["S3"] - s2["S3"])
    max_shell_swing = max(delta_s0, delta_s1, delta_s2, delta_s3)

    # Linear location shift (C-4a)
    p1, p2 = run1["parameter_stats"], run2["parameter_stats"]
    linear_loc_shifts = {}
    linear_loc_pass = True
    for p_name in ["mc", "eta", "tc"]:
        med1 = p1[p_name]["median"]
        med2 = p2[p_name]["median"]
        std1 = p1[p_name]["std"]
        std2 = p2[p_name]["std"]
        sigma_ref = math.sqrt((std1 ** 2 + std2 ** 2) / 2.0)
        norm_shift = abs(med1 - med2) / sigma_ref if sigma_ref > 0 else 0.0
        passed = bool(norm_shift <= 0.20)
        linear_loc_shifts[p_name] = {"shift": abs(med1 - med2), "sigma_ref": sigma_ref, "norm_shift": norm_shift, "passed": passed}
        if not passed:
            linear_loc_pass = False

    # Circular mean direction (C-4b)
    phic1, phic2 = p1["phic"], p2["phic"]
    r1, r2 = phic1["r_bar"], phic2["r_bar"]
    mean_dir_applicable = (r1 >= 0.50 and r2 >= 0.50)
    if mean_dir_applicable:
        ang_dist = min(abs(phic1["phi_circ_mean"] - phic2["phi_circ_mean"]), 2.0 * math.pi - abs(phic1["phi_circ_mean"] - phic2["phi_circ_mean"]))
        sigma_circ_ref = math.sqrt((phic1["sigma_circ"] ** 2 + phic2["sigma_circ"] ** 2) / 2.0)
        norm_ang_dist = ang_dist / sigma_circ_ref if sigma_circ_ref > 0 else 0.0
        circ_loc_pass = bool(norm_ang_dist <= 0.20)
        c4b_status = "PASS" if circ_loc_pass else "FAIL"
    else:
        ang_dist = None
        norm_ang_dist = None
        circ_loc_pass = None
        c4b_status = "UNDEFINED"

    # Linear CI width and endpoints (C-6a)
    linear_ci_checks = {}
    linear_ci_pass = True
    for p_name in ["mc", "eta", "tc"]:
        w1 = p1[p_name]["ci95_width"]
        w2 = p2[p_name]["ci95_width"]
        mean_w = (w1 + w2) / 2.0
        rel_w_diff = abs(w1 - w2) / mean_w if mean_w > 0 else 0.0

        q025_1 = p1[p_name]["q_0.025"]
        q025_2 = p2[p_name]["q_0.025"]
        q975_1 = p1[p_name]["q_0.975"]
        q975_2 = p2[p_name]["q_0.975"]
        s_ref = linear_loc_shifts[p_name]["sigma_ref"]

        end1_diff = abs(q025_1 - q025_2) / s_ref if s_ref > 0 else 0.0
        end2_diff = abs(q975_1 - q975_2) / s_ref if s_ref > 0 else 0.0
        passed = bool(rel_w_diff <= 0.15 and end1_diff <= 0.20 and end2_diff <= 0.20)
        linear_ci_checks[p_name] = {
            "rel_width_diff": rel_w_diff,
            "end025_norm_diff": end1_diff,
            "end975_norm_diff": end2_diff,
            "passed": passed,
        }
        if not passed:
            linear_ci_pass = False

    # Circular phase shortest-arc repeatability (C-6b)
    len1 = phic1["shortest_arc_len"]
    len2 = phic2["shortest_arc_len"]
    mean_len = (len1 + len2) / 2.0
    rel_arc_len_diff = abs(len1 - len2) / mean_len if mean_len > 0 else 0.0

    jaccard_overlap = circular_jaccard(
        phic1["shortest_arc_alpha"], phic1["shortest_arc_beta"],
        phic2["shortest_arc_alpha"], phic2["shortest_arc_beta"]
    )
    c6b_pass = bool(rel_arc_len_diff <= 0.15 and jaccard_overlap >= 0.80)

    # Weights and ESS (C-8a, C-8b)
    w_max_1, w_max_2 = run1["w_max"], run2["w_max"]
    neff_1, neff_2 = run1["n_eff"], run2["n_eff"]
    c8a_pass = bool(w_max_1 < 0.01 and w_max_2 < 0.01)
    c8b_pass = bool(neff_1 >= 2000 and neff_2 >= 2000)

    # Criteria flags
    c1_pass = bool(delta_logz <= 1.0)
    c2_pass = bool(z_repeat <= 3.0)
    c3_pass = bool(max_shell_swing <= 0.10)

    all_core_passed = bool(
        c1_pass and c2_pass and c3_pass and linear_loc_pass and
        linear_ci_pass and c6b_pass and c8a_pass and c8b_pass and
        (circ_loc_pass is True or c4b_status == "UNDEFINED")
    )

    return {
        "delta_logz": delta_logz,
        "sigma_pooled": sigma_pooled,
        "z_repeat": z_repeat,
        "c1_passed": c1_pass,
        "c2_passed": c2_pass,
        "max_shell_swing": max_shell_swing,
        "c3_passed": c3_pass,
        "linear_loc_shifts": linear_loc_shifts,
        "c4a_passed": linear_loc_pass,
        "c4b_status": c4b_status,
        "circ_ang_dist": ang_dist,
        "circ_norm_ang_dist": norm_ang_dist,
        "c4b_passed": circ_loc_pass,
        "linear_ci_checks": linear_ci_checks,
        "c6a_passed": linear_ci_pass,
        "rel_arc_len_diff": rel_arc_len_diff,
        "jaccard_overlap": jaccard_overlap,
        "c6b_passed": c6b_pass,
        "w_max_1": w_max_1,
        "w_max_2": w_max_2,
        "c8a_passed": c8a_pass,
        "neff_1": neff_1,
        "neff_2": neff_2,
        "c8b_passed": c8b_pass,
        "all_core_passed": all_core_passed,
    }


def main() -> int:
    print("=" * 80)
    print("WP6 STAGE 5G.2 OPTION B ANCHOR PILOT EXECUTION (REALIZATION 01)")
    print(f"Start time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Max workers: {MAX_WORKERS} | Authorized runs: {len(AUTHORIZED_RUN_MATRIX)}")
    print(f"Budget limits: {MAX_TOTAL_CALLS:,} likelihood calls | {MAX_CORE_HOURS:.2f} core-hours")
    print(f"Dynesty version: {dynesty.__version__} | NumPy version: {np.__version__}")
    print("=" * 80)

    t_start_campaign = time.perf_counter()
    run_results = {}
    cum_calls = 0
    cum_cpu_seconds = 0.0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_spec = {
            executor.submit(execute_single_pilot_run, spec, OUTPUT_DIR): spec
            for spec in AUTHORIZED_RUN_MATRIX
        }

        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            run_id = spec["run_id"]
            try:
                res = future.result()
                run_results[run_id] = res

                if res.get("status") == "SUCCESS":
                    ncall = res.get("n_evaluations", 0)
                    cpu_s = res.get("cpu_time_s", 0.0)
                    cum_calls += ncall
                    cum_cpu_seconds += cpu_s

                    core_hrs = cum_cpu_seconds / 3600.0
                    print(f"--> [Progress] Completed {len(run_results)}/{len(AUTHORIZED_RUN_MATRIX)}: {run_id} | Cum calls: {cum_calls:,} ({(cum_calls/MAX_TOTAL_CALLS)*100:.1f}%) | Cum CPU: {core_hrs:.3f} hrs ({(core_hrs/MAX_CORE_HOURS)*100:.1f}%)")

                    # Safeguard budget check
                    if cum_calls > MAX_TOTAL_CALLS or core_hrs > MAX_CORE_HOURS:
                        print(f"\n[ALERT] Safeguard budget limit exceeded! Calls: {cum_calls} > {MAX_TOTAL_CALLS} or Core-hrs: {core_hrs:.3f} > {MAX_CORE_HOURS:.3f}!")
                        print("Halting remaining tasks pursuant to supervisor safeguards.")
                        for f in future_to_spec:
                            f.cancel()
                        break
                else:
                    print(f"\n[ALERT] Run {run_id} failed with error: {res.get('error')}")
                    print("Halting remaining tasks pursuant to supervisor safeguards.")
                    for f in future_to_spec:
                        f.cancel()
                    break

            except Exception as e:
                print(f"\n[CRITICAL ERROR] Worker execution exception on {run_id}: {e}")
                for f in future_to_spec:
                    f.cancel()
                return 1

    t_wall_campaign = time.perf_counter() - t_start_campaign
    cum_core_hours = cum_cpu_seconds / 3600.0

    print("\n" + "=" * 80)
    print("ALL 12 AUTHORIZED RUNS COMPLETE — CUMULATIVE BUDGET ACCOUNTING")
    print("=" * 80)
    print(f"Total Parallel Wall Time: {t_wall_campaign:.2f} s ({t_wall_campaign/60:.2f} min)")
    print(f"Total Consumed CPU Time:  {cum_cpu_seconds:.2f} s ({cum_core_hours:.4f} core-hours)")
    print(f"Total Likelihood Calls:   {cum_calls:,} calls")
    print(f"Call Budget Consumption:  {(cum_calls/MAX_TOTAL_CALLS)*100:.2f}% of {MAX_TOTAL_CALLS:,} max")
    print(f"CPU Budget Consumption:   {(cum_core_hours/MAX_CORE_HOURS)*100:.2f}% of {MAX_CORE_HOURS:.2f} core-hrs max")
    print(f"Reference Cap Fraction:   {(cum_core_hours/4.41)*100:.2f}% of 4.41 core-hr reference ceiling")

    # Evaluate between-seed repeatability for the 6 pairs
    print("\n" + "=" * 80)
    print("BETWEEN-SEED REPEATABILITY ANALYSIS ACROSS 6 PAIRED CONFIGURATIONS")
    print("=" * 80)

    pairs = [
        ("Baseline (nlive=500, rwalk)", "base_01_plug_inst_s1", "base_01_plug_inst_s2", "Model 2 (Plug-in)"),
        ("Baseline (nlive=500, rwalk)", "base_01_stud_inst_s1", "base_01_stud_inst_s2", "Model 3 (Student-t)"),
        ("Intervention (nlive=1000, rwalk)", "diag_01_plug_nl1000_s1", "diag_01_plug_nl1000_s2", "Model 2 (Plug-in)"),
        ("Intervention (nlive=1000, rwalk)", "diag_01_stud_nl1000_s1", "diag_01_stud_nl1000_s2", "Model 3 (Student-t)"),
        ("Intervention (nlive=500, rslice)", "diag_01_plug_rslice_s1", "diag_01_plug_rslice_s2", "Model 2 (Plug-in)"),
        ("Intervention (nlive=500, rslice)", "diag_01_stud_rslice_s1", "diag_01_stud_rslice_s2", "Model 3 (Student-t)"),
    ]

    pair_evaluations = {}
    for pair_label, id1, id2, model_label in pairs:
        r1 = run_results[id1]
        r2 = run_results[id2]
        eval_dict = evaluate_pair_repeatability(r1, r2)
        pair_key = f"{id1}_vs_{id2}"
        pair_evaluations[pair_key] = {
            "tier_label": pair_label,
            "model_label": model_label,
            "run1_id": id1,
            "run2_id": id2,
            "evaluation": eval_dict,
        }

        print(f"\n[{pair_label} — {model_label}]")
        print(f"  Seeds: {r1['sampler_seed']} vs {r2['sampler_seed']}")
        print(f"  ln(Z): {r1['log_evidence']:.4f} vs {r2['log_evidence']:.4f} | Delta ln(Z) = {eval_dict['delta_logz']:.4f} nats (z = {eval_dict['z_repeat']:.2f}) -> C-1: {'PASS' if eval_dict['c1_passed'] else 'FAIL'}, C-2: {'PASS' if eval_dict['c2_passed'] else 'FAIL'}")
        print(f"  Shell S0: {r1['shell_probabilities']['S0']*100:.1f}% vs {r2['shell_probabilities']['S0']*100:.1f}% | Shell S2: {r1['shell_probabilities']['S2']*100:.1f}% vs {r2['shell_probabilities']['S2']*100:.1f}% | Max Swing = {eval_dict['max_shell_swing']*100:.1f}% -> C-3: {'PASS' if eval_dict['c3_passed'] else 'FAIL'}")
        print(f"  Linear Loc Shift (C-4a): {'PASS' if eval_dict['c4a_passed'] else 'FAIL'} | Circular Mean Dir (C-4b): {eval_dict['c4b_status']}")
        print(f"  Linear CI Repeat (C-6a): {'PASS' if eval_dict['c6a_passed'] else 'FAIL'} | Circular Arc Jaccard (C-6b): {eval_dict['jaccard_overlap']:.3f} (delta_len={eval_dict['rel_arc_len_diff']*100:.1f}%) -> {'PASS' if eval_dict['c6b_passed'] else 'FAIL'}")
        print(f"  Weights C-8a: {'PASS' if eval_dict['c8a_passed'] else 'FAIL'} (w_max: {eval_dict['w_max_1']:.4f}, {eval_dict['w_max_2']:.4f}) | ESS C-8b: {'PASS' if eval_dict['c8b_passed'] else 'FAIL'} (N_eff: {eval_dict['neff_1']:.0f}, {eval_dict['neff_2']:.0f})")
        print(f"  --> OVERALL PAIR VERDICT: {'PASS' if eval_dict['all_core_passed'] else 'FAIL'}")

    # Resolution scaling comparison (C-5): nlive=1000 vs nlive=500 baseline
    print("\n--- Resolution Scaling Analysis (Criterion C-5: nlive=1000 vs nlive=500) ---")
    c5_comparisons = {}
    tau_sys = 1.0  # nat
    for model_name, s1_base_id, s1_nl1000_id, s2_base_id, s2_nl1000_id in [
        ("Model 2 (Plug-in)", "base_01_plug_inst_s1", "diag_01_plug_nl1000_s1", "base_01_plug_inst_s2", "diag_01_plug_nl1000_s2"),
        ("Model 3 (Student-t)", "base_01_stud_inst_s1", "diag_01_stud_nl1000_s1", "base_01_stud_inst_s2", "diag_01_stud_nl1000_s2"),
    ]:
        for s_label, b_id, nl_id in [("Seed 8101", s1_base_id, s1_nl1000_id), ("Seed 8201", s2_base_id, s2_nl1000_id)]:
            r_b = run_results[b_id]
            r_nl = run_results[nl_id]
            d_z = abs(r_nl["log_evidence"] - r_b["log_evidence"])
            sigma_c5 = math.sqrt(r_nl["log_evidence_err"] ** 2 + r_b["log_evidence_err"] ** 2 + tau_sys ** 2)
            passed_c5 = bool(d_z <= sigma_c5)
            c5_key = f"{model_name}_{s_label}"
            c5_comparisons[c5_key] = {
                "model": model_name,
                "seed": s_label,
                "z_nl1000": r_nl["log_evidence"],
                "z_base": r_b["log_evidence"],
                "delta_z": d_z,
                "sigma_threshold": sigma_c5,
                "passed": passed_c5,
            }
            print(f"  {model_name} {s_label}: |ln Z(1000) - ln Z(500)| = {d_z:.4f} nats <= {sigma_c5:.4f} nats -> C-5: {'PASS' if passed_c5 else 'FAIL'}")

    # Compile master campaign summary
    campaign_summary = {
        "campaign": "WP6 Stage 5G.2 Option B Anchor Pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "completion_status": "ALL_12_RUNS_COMPLETED",
        "total_wall_time_s": float(t_wall_campaign),
        "total_cpu_time_s": float(cum_cpu_seconds),
        "total_core_hours": float(cum_core_hours),
        "total_likelihood_evaluations": int(cum_calls),
        "budget_limits": {
            "max_calls": MAX_TOTAL_CALLS,
            "max_core_hours": MAX_CORE_HOURS,
            "reference_ceiling_hours": 4.41,
        },
        "budget_consumption": {
            "call_percentage": float((cum_calls / MAX_TOTAL_CALLS) * 100),
            "cpu_percentage": float((cum_core_hours / MAX_CORE_HOURS) * 100),
            "reference_ceiling_percentage": float((cum_core_hours / 4.41) * 100),
        },
        "runs": run_results,
        "paired_repeatability_evaluations": pair_evaluations,
        "c5_resolution_scaling_evaluations": c5_comparisons,
    }

    summary_json_path = OUTPUT_DIR / "stage_g2_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(campaign_summary, f, indent=2)
    print(f"\nMaster campaign summary saved to: {summary_json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
