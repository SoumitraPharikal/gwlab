import glob
import json
import os
import numpy as np

def analyze():
    # 1. Stage 5G.1
    g1_files = sorted(glob.glob("reports/demonstrations/wp6_stage5/stage_g1/run_*.json"))
    g1_data = []
    for f in g1_files:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        calls = d["n_evaluations"]
        # In 5G.1, n_dead_points was recorded as n_iterations + nlive (11,451 + 500 = 11,951)
        # So true shrinkage dead points is n_dead_points - 500
        dead_raw = d["n_dead_points"]
        dead_true = dead_raw - 500
        wall = d["wall_time_s"]
        g1_data.append({
            "run_id": d["run_id"],
            "campaign": "stage_5g1",
            "realization": d["realization"],
            "model": d["model"],
            "sampler_seed": d["sampler_seed"],
            "nlive": 500,
            "sample": "rwalk",
            "walks": 50,
            "bound": "multi",
            "n_evaluations": calls,
            "n_dead_points_raw": dead_raw,
            "n_dead_points": dead_true,
            "n_live_final": 500,
            "wall_time_s": wall,
            "cpu_time_s": None, # Unrecorded in 5G.1
            "evals_per_sec": d.get("evals_per_sec"),
            "calls_per_dead": calls / dead_true if dead_true > 0 else np.nan,
            "wall_s_per_call": wall / calls if calls > 0 else np.nan,
            "cpu_s_per_call": None,
            "threads_metadata": "4 workers (default)",
        })

    # 2. Stage 5G.2
    g2_files = sorted(glob.glob("reports/demonstrations/wp6_stage5/stage_g2/base_*.json") +
                      glob.glob("reports/demonstrations/wp6_stage5/stage_g2/diag_*.json"))
    g2_data = []
    for f in g2_files:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        calls = d["n_evaluations"]
        dead = d["n_iterations"]
        wall = d["wall_time_s"]
        cpu = d["cpu_time_s"]
        model_name = "model_2_plug" if "plug" in d["model"] else "model_3_student_t"
        g2_data.append({
            "run_id": d["run_id"],
            "campaign": "stage_5g2",
            "tier": d["tier"],
            "realization": d["realization"],
            "model": model_name,
            "sampler_seed": d["sampler_seed"],
            "nlive": d["nlive"],
            "sample": d["sample_method"],
            "walks": d.get("walks"),
            "slices": d.get("slices"),
            "bound": d["bound"],
            "n_evaluations": calls,
            "n_dead_points": dead,
            "n_live_final": d["n_live_final"],
            "wall_time_s": wall,
            "cpu_time_s": cpu,
            "evals_per_sec": calls / wall if wall > 0 else np.nan,
            "calls_per_dead": calls / dead if dead > 0 else np.nan,
            "wall_s_per_call": wall / calls if calls > 0 else np.nan,
            "cpu_s_per_call": cpu / calls if calls > 0 else np.nan,
            "cpu_to_wall_ratio": cpu / wall if wall > 0 else np.nan,
            "cpu_s_per_dead": cpu / dead if dead > 0 else np.nan,
            "threads_metadata": "6 workers (ProcessPoolExecutor)",
        })

    # Generate Markdown Table
    md_lines = [
        "| Run ID | Campaign | Model | $n_{\\rm live}$ | Sampler | Seed | Likelihood Calls $N_{\\rm call}$ | Process CPU [s] | Worker Wall [s] | CPU s / Call [ms] | Calls / Dead Pt | Dead Points $N_{\\rm dead}$ | Final Live $n_{\\rm live}$ | Concurrency / Metadata |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    for r in g1_data:
        md_lines.append(
            f"| `{r['run_id']}` | 5G.1 | {r['model']} | 500 | `rwalk` | {r['sampler_seed']} | "
            f"{r['n_evaluations']:,} | *Unrecorded* | {r['wall_time_s']:.2f} | *Unrecorded* | "
            f"{r['calls_per_dead']:.2f} | {r['n_dead_points']:,} | {r['n_live_final']} | {r['threads_metadata']} |"
        )
    for r in g2_data:
        cpu_ms = r['cpu_s_per_call'] * 1000
        md_lines.append(
            f"| `{r['run_id']}` | 5G.2 | {r['model']} | {r['nlive']} | `{r['sample']}` | {r['sampler_seed']} | "
            f"{r['n_evaluations']:,} | {r['cpu_time_s']:.2f} | {r['wall_time_s']:.2f} | {cpu_ms:.3f} ms | "
            f"{r['calls_per_dead']:.2f} | {r['n_dead_points']:,} | {r['n_live_final']} | {r['threads_metadata']} |"
        )

    table_md = "\n".join(md_lines)
    with open("reports/demonstrations/wp6_stage5/stage_g2/consolidated_cost_table.md", "w", encoding="utf-8") as fp:
        fp.write(table_md)

    print("Table written to reports/demonstrations/wp6_stage5/stage_g2/consolidated_cost_table.md")

    # Detailed statistics
    cpu_per_call_all = [x["cpu_s_per_call"] for x in g2_data]
    print("\n=== Stage 5G.2 CPU Seconds per Likelihood Call ===")
    print(f"Count: {len(cpu_per_call_all)}")
    print(f"Mean: {np.mean(cpu_per_call_all):.6f} s ({np.mean(cpu_per_call_all)*1000:.3f} ms)")
    print(f"Median: {np.median(cpu_per_call_all):.6f} s ({np.median(cpu_per_call_all)*1000:.3f} ms)")
    print(f"Std: {np.std(cpu_per_call_all, ddof=1):.6f} s ({np.std(cpu_per_call_all, ddof=1)*1000:.3f} ms)")
    print(f"Min: {np.min(cpu_per_call_all):.6f} s ({np.min(cpu_per_call_all)*1000:.3f} ms)")
    print(f"Max: {np.max(cpu_per_call_all):.6f} s ({np.max(cpu_per_call_all)*1000:.3f} ms)")

    # Model 2 vs Model 3
    m2_cpu_per_call = [x["cpu_s_per_call"] for x in g2_data if x["model"] == "model_2_plug"]
    m3_cpu_per_call = [x["cpu_s_per_call"] for x in g2_data if x["model"] == "model_3_student_t"]
    print("\n=== Model 2 vs Model 3 CPU s per Call ===")
    print(f"Model 2 (Plug-in, N={len(m2_cpu_per_call)}): mean={np.mean(m2_cpu_per_call)*1000:.3f} ms, median={np.median(m2_cpu_per_call)*1000:.3f} ms, range=[{np.min(m2_cpu_per_call)*1000:.3f}, {np.max(m2_cpu_per_call)*1000:.3f}] ms")
    print(f"Model 3 (Student-t, N={len(m3_cpu_per_call)}): mean={np.mean(m3_cpu_per_call)*1000:.3f} ms, median={np.median(m3_cpu_per_call)*1000:.3f} ms, range=[{np.min(m3_cpu_per_call)*1000:.3f}, {np.max(m3_cpu_per_call)*1000:.3f}] ms")

    # rwalk vs rslice
    rwalk_cpu_per_call = [x["cpu_s_per_call"] for x in g2_data if x["sample"] == "rwalk"]
    rslice_cpu_per_call = [x["cpu_s_per_call"] for x in g2_data if x["sample"] == "rslice"]
    print("\n=== rwalk vs rslice CPU s per Call ===")
    print(f"rwalk (N={len(rwalk_cpu_per_call)}): mean={np.mean(rwalk_cpu_per_call)*1000:.3f} ms, median={np.median(rwalk_cpu_per_call)*1000:.3f} ms, range=[{np.min(rwalk_cpu_per_call)*1000:.3f}, {np.max(rwalk_cpu_per_call)*1000:.3f}] ms")
    print(f"rslice (N={len(rslice_cpu_per_call)}): mean={np.mean(rslice_cpu_per_call)*1000:.3f} ms, median={np.median(rslice_cpu_per_call)*1000:.3f} ms, range=[{np.min(rslice_cpu_per_call)*1000:.3f}, {np.max(rslice_cpu_per_call)*1000:.3f}] ms")

    # Calls per dead point
    rwalk_calls_per_dead = [x["calls_per_dead"] for x in g2_data if x["sample"] == "rwalk"]
    rslice_calls_per_dead = [x["calls_per_dead"] for x in g2_data if x["sample"] == "rslice"]
    print("\n=== Calls per Dead Point ===")
    print(f"rwalk (N={len(rwalk_calls_per_dead)}): mean={np.mean(rwalk_calls_per_dead):.2f}, median={np.median(rwalk_calls_per_dead):.2f}, range=[{np.min(rwalk_calls_per_dead):.2f}, {np.max(rwalk_calls_per_dead):.2f}]")
    print(f"rslice (N={len(rslice_calls_per_dead)}): mean={np.mean(rslice_calls_per_dead):.2f}, median={np.median(rslice_calls_per_dead):.2f}, range=[{np.min(rslice_calls_per_dead):.2f}, {np.max(rslice_calls_per_dead):.2f}]")

    # nlive=500 vs nlive=1000 for rwalk
    nl500_rwalk = [x for x in g2_data if x["nlive"] == 500 and x["sample"] == "rwalk"]
    nl1000_rwalk = [x for x in g2_data if x["nlive"] == 1000 and x["sample"] == "rwalk"]
    print("\n=== nlive=500 vs nlive=1000 (rwalk) ===")
    print(f"nlive=500 (N={len(nl500_rwalk)}):")
    print(f"  Calls: mean={np.mean([x['n_evaluations'] for x in nl500_rwalk]):.0f}, range=[{min(x['n_evaluations'] for x in nl500_rwalk)}, {max(x['n_evaluations'] for x in nl500_rwalk)}]")
    print(f"  Dead pts: mean={np.mean([x['n_dead_points'] for x in nl500_rwalk]):.0f}, range=[{min(x['n_dead_points'] for x in nl500_rwalk)}, {max(x['n_dead_points'] for x in nl500_rwalk)}]")
    print(f"  Calls/dead: mean={np.mean([x['calls_per_dead'] for x in nl500_rwalk]):.2f}, range=[{min(x['calls_per_dead'] for x in nl500_rwalk):.2f}, {max(x['calls_per_dead'] for x in nl500_rwalk):.2f}]")
    print(f"  CPU s: mean={np.mean([x['cpu_time_s'] for x in nl500_rwalk]):.2f}s, range=[{min(x['cpu_time_s'] for x in nl500_rwalk):.2f}, {max(x['cpu_time_s'] for x in nl500_rwalk):.2f}]")
    print(f"nlive=1000 (N={len(nl1000_rwalk)}):")
    print(f"  Calls: mean={np.mean([x['n_evaluations'] for x in nl1000_rwalk]):.0f}, range=[{min(x['n_evaluations'] for x in nl1000_rwalk)}, {max(x['n_evaluations'] for x in nl1000_rwalk)}]")
    print(f"  Dead pts: mean={np.mean([x['n_dead_points'] for x in nl1000_rwalk]):.0f}, range=[{min(x['n_dead_points'] for x in nl1000_rwalk)}, {max(x['n_dead_points'] for x in nl1000_rwalk)}]")
    print(f"  Calls/dead: mean={np.mean([x['calls_per_dead'] for x in nl1000_rwalk]):.2f}, range=[{min(x['calls_per_dead'] for x in nl1000_rwalk):.2f}, {max(x['calls_per_dead'] for x in nl1000_rwalk):.2f}]")
    print(f"  CPU s: mean={np.mean([x['cpu_time_s'] for x in nl1000_rwalk]):.2f}s, range=[{min(x['cpu_time_s'] for x in nl1000_rwalk):.2f}, {max(x['cpu_time_s'] for x in nl1000_rwalk):.2f}]")

    # Ratio of 1000 to 500
    ratio_calls = np.mean([x['n_evaluations'] for x in nl1000_rwalk]) / np.mean([x['n_evaluations'] for x in nl500_rwalk])
    ratio_dead = np.mean([x['n_dead_points'] for x in nl1000_rwalk]) / np.mean([x['n_dead_points'] for x in nl500_rwalk])
    ratio_cpu = np.mean([x['cpu_time_s'] for x in nl1000_rwalk]) / np.mean([x['cpu_time_s'] for x in nl500_rwalk])
    print(f"\nRatios (nlive=1000 / nlive=500 for rwalk):")
    print(f"  Calls ratio: {ratio_calls:.3f}")
    print(f"  Dead points ratio: {ratio_dead:.3f}")
    print(f"  CPU time ratio: {ratio_cpu:.3f}")

    # Wall time per call across 5G.1
    g1_wall_per_call = [x["wall_s_per_call"] for x in g1_data]
    print("\n=== Stage 5G.1 Worker Wall Seconds per Likelihood Call ===")
    print(f"Count: {len(g1_wall_per_call)}")
    print(f"Mean: {np.mean(g1_wall_per_call)*1000:.3f} ms, median={np.median(g1_wall_per_call)*1000:.3f} ms, range=[{np.min(g1_wall_per_call)*1000:.3f}, {np.max(g1_wall_per_call)*1000:.3f}] ms")

    # 5G.1 evals_per_sec
    g1_evals_sec = [x["evals_per_sec"] for x in g1_data]
    print(f"Evals/sec in 5G.1: mean={np.mean(g1_evals_sec):.1f}, median={np.median(g1_evals_sec):.1f}, range=[{np.min(g1_evals_sec):.1f}, {np.max(g1_evals_sec):.1f}]")

    # Compare Model types in 5G.1
    for m in ["oracle", "plug", "student_t"]:
        sub = [x for x in g1_data if x["model"] == m]
        sub_rates = [x["evals_per_sec"] for x in sub]
        sub_calls = [x["n_evaluations"] for x in sub]
        sub_dead = [x["n_dead_points"] for x in sub]
        sub_ms = [x["wall_s_per_call"] * 1000 for x in sub]
        print(f"5G.1 Model {m} (N={len(sub)}): evals/s={np.mean(sub_rates):.1f}, wall ms/call={np.mean(sub_ms):.3f}, calls={np.mean(sub_calls):.0f}, dead={np.mean(sub_dead):.0f}, calls/dead={np.mean([c/d for c,d in zip(sub_calls, sub_dead)]):.2f}")

    # Save summary json for reference
    out = {
        "stage_g1": g1_data,
        "stage_g2": g2_data,
    }
    with open("reports/demonstrations/wp6_stage5/stage_g2/cost_model_analysis.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print("\nSaved analysis to reports/demonstrations/wp6_stage5/stage_g2/cost_model_analysis.json")

if __name__ == "__main__":
    analyze()
