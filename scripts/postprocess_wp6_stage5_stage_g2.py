#!/usr/bin/env python3
"""
Post-processing and Master Summary Generator for WP6 Stage 5G.2 Option B Pilot Campaign.

Reads all 12 completed run artifacts from reports/demonstrations/wp6_stage5/stage_g2/,
computes paired repeatability evaluations across all 6 pairs (C-1 to C-8b),
evaluates Criterion C-5 resolution scaling, aggregates budget accounting,
and compiles the official stage_g2_summary.json.
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Ensure workspace root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_wp6_stage5_stage_g2 import (
    evaluate_pair_repeatability,
    OUTPUT_DIR,
    MAX_TOTAL_CALLS,
    MAX_CORE_HOURS,
)


def main() -> int:
    print("=" * 80)
    print("WP6 STAGE 5G.2 OPTION B — POST-PROCESSING & MASTER SUMMARY COMPILER")
    print("=" * 80)

    # 1. Identify and load all 12 run JSON files
    expected_run_ids = [
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

    run_results = {}
    cum_calls = 0
    cum_cpu_seconds = 0.0
    max_wall_time = 0.0

    print(f"Loading {len(expected_run_ids)} run artifacts from: {OUTPUT_DIR}")
    for run_id in expected_run_ids:
        json_path = OUTPUT_DIR / f"{run_id}.json"
        if not json_path.exists():
            print(f"[ERROR] Missing JSON artifact for run {run_id}: {json_path}")
            return 1
        with open(json_path, "r") as f:
            data = json.load(f)
        run_results[run_id] = data

        calls = data["n_evaluations"]
        cpu = data["cpu_time_s"]
        wall = data["wall_time_s"]
        cum_calls += calls
        cum_cpu_seconds += cpu
        if wall > max_wall_time:
            max_wall_time = wall

        # Verify lifecycle accounting
        ndead = data["n_iterations"]
        nlive = data["n_live_final"]
        nsamp = data["n_total_samples"]
        assert nsamp == ndead + nlive, f"Lifecycle sample mismatch for {run_id}: {nsamp} != {ndead} + {nlive}"

        print(f"  Loaded: {run_id:<26} | lnZ={data['log_evidence']:.4f} +/- {data['log_evidence_err']:.4f} | calls={calls:>7} | dead={ndead:>5} live={nlive:>4} total={nsamp:>5} | CPU={cpu:>7.2f}s")

    cum_core_hours = cum_cpu_seconds / 3600.0

    print("\n" + "=" * 80)
    print("CAMPAIGN BUDGET RECONCILIATION")
    print("=" * 80)
    print(f"Total Likelihood Calls:   {cum_calls:,} calls ({(cum_calls / MAX_TOTAL_CALLS) * 100:.2f}% of {MAX_TOTAL_CALLS:,} max)")
    print(f"Total CPU Process Time:   {cum_cpu_seconds:.2f} s ({cum_core_hours:.4f} core-hours)")
    print(f"Planning Cap Reference:   {(cum_core_hours / MAX_CORE_HOURS) * 100:.2f}% of {MAX_CORE_HOURS:.2f} core-hours planning cap")
    print(f"Reference Ceiling:        {(cum_core_hours / 4.41) * 100:.2f}% of 4.41 core-hours reference ceiling")
    print(f"Longest Worker Wall Time: {max_wall_time:.2f} s ({max_wall_time / 60:.2f} min)")

    # 2. Evaluate between-seed repeatability for the 6 pairs
    print("\n" + "=" * 80)
    print("BETWEEN-SEED REPEATABILITY ANALYSIS (CRITERIA C-1 TO C-8b)")
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

    # 3. Resolution scaling comparison (C-5): nlive=1000 vs nlive=500 baseline
    print("\n" + "=" * 80)
    print("RESOLUTION SCALING ANALYSIS (CRITERION C-5: nlive=1000 vs nlive=500)")
    print("=" * 80)
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

    # 4. Compile master campaign summary
    campaign_summary = {
        "campaign": "WP6 Stage 5G.2 Option B Anchor Pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "completion_status": "ALL_12_RUNS_COMPLETED",
        "total_cpu_time_s": float(cum_cpu_seconds),
        "total_core_hours": float(cum_core_hours),
        "max_worker_wall_time_s": float(max_wall_time),
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
    print(f"\n[SUCCESS] Master campaign summary saved to: {summary_json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
