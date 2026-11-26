# WP6 Stage 5G.2 — Option B Pilot Execution Report
## Sampler Diagnostic Interventions on Realization 01 Anchor Set

**Date:** November 19, 2026  
**Status:** Completed (All 12 Authorized Runs Finished with `SUCCESS`; Master Post-Processing Reconciled)  
**Campaign:** WP6 Stage 5G.2 Option B Anchor Pilot  
**Target:** Realization 01, Models 2 (Plug-in Gaussian) and 3 (Student-$t$ Relative Dispersion), Seeds 8101 & 8201  
**Supervisor Directive:** Formal Authorization Granted under Revision 1.5; Strict Option B Boundary (12 Runs Only)

---

## Executive Summary

Pursuant to supervisor authorization for **Option B** of `reports/WP6_Stage5G2_Sampler_Resolution_Design.md` (Revision 1.5), we executed the authorized 12-run pilot campaign on fixed noise Realization 01. The campaign investigated repeat-seed instability across three paired configurations:
1. **Instrumented Baseline Reruns** ($n_{\rm live}=500$, `bound='multi'`, `sample='rwalk'`, $W=50$, $\text{dlogz}=0.10$).
2. **Intervention 1: Resolution Scaling** ($n_{\rm live}=1000$, `bound='multi'`, `sample='rwalk'`, $W=50$).
3. **Intervention 2: Sampling Method Change** ($n_{\rm live}=500$, `bound='multi'`, `sample='rslice'`, $S=5$).

### Key Empirical Findings

1. **Exact Reproduction of Baseline Pathology:**
   The four instrumented baseline reruns reproduced the historical Stage 5G.1 results down to machine precision ($\Delta\ln\mathcal{Z} = 0.00\times 10^0\,\mathrm{nats}$, identical likelihood call counts). For Model 2, Seed 8101 placed $64.63\%$ of its posterior in Shell $\mathcal{S}_0$ and $3.57\%$ in Shell $\mathcal{S}_2$, whereas Seed 8201 allocated only $2.21\%$ to Shell $\mathcal{S}_0$ and $77.19\%$ to Shell $\mathcal{S}_2$ (the low-$\eta$ region), replicating the $73.63\%$ shell swing and evidence discrepancy $\Delta\ln\mathcal{Z} = 2.9720\,\mathrm{nats}$ ($z = 8.92$).
2. **Regional Occupancy Dynamics and Bounding Geometry:**
   Live-point telemetry recorded at $\Delta i = 500$ iterations tracked regional occupancy using the descriptive partition threshold $\eta = 0.244$ (separating the high-$\eta$ region $\eta > 0.244$ from the low-$\eta$ region $\eta \le 0.244$). In Model 2 Baseline Seed 8201, live points in the high-$\eta$ region decreased from 92 to 6 during iterations 5,000–8,500. This drop was coincident with the multi-ellipsoid bounding region collapsing into a single bounding ellipsoid (`bound_count = 1`) at iteration 7,000—an empirical association consistent with a possible bounding-geometry contribution to regional occupancy changes, though not proof of physical causation. Because Seed 8201 traversed the peak evidence accumulation epoch ($\Delta\ln\mathcal{Z}$ shrinking from 17.4 to 4.3 nats) with $98.8\%$ of its live points in the low-$\eta$ region, $77.19\%$ of its posterior weight was concentrated in Shell $\mathcal{S}_2$.
3. **Impact of Intervention 1 ($n_{\rm live}=1000$, `rwalk`):**
   - **Model 3 (Student-$t$):** Achieved **Criterion C-1 PASS** with evidence discrepancy dropping from $2.48\,\mathrm{nats}$ to $\Delta\ln\mathcal{Z} = \mathbf{0.7443\,\mathrm{nats}}$. However, Criterion C-2 failed ($z_{\rm rep} = 3.09 > 3.0$), and shell swing ($\Delta\mathcal{S}_{\max} = 36.07\%$) and location discrepancies remained elevated. This constitutes a partial improvement in evidence repeatability, but not overall repeatability success.
   - **Model 2 (Plug-in):** Evidence discrepancy remained elevated at $\Delta\ln\mathcal{Z} = 3.1334\,\mathrm{nats}$ ($z = 12.71$), larger than the baseline discrepancy of $2.9720\,\mathrm{nats}$. Although the regional distribution changed (Seed 8201 allocated $76.04\%$ to Shell $\mathcal{S}_0$ and $1.90\%$ to $\mathcal{S}_2$, avoiding the low-$\eta$ concentration of the baseline) and linear parameter medians agreed (Criterion C-4a PASS), shell swing ($12.23\%$, failing C-3), credible-interval endpoints (C-6a FAIL), and circular-arc overlap (C-6b FAIL) did not all pass. This intervention demonstrated altered sampling behaviour with partial location agreement, but left evidence and posterior-shape repeatability unresolved.
   - **Observation of Zero Observed Live Points and Reappearance:** In Model 2 Seed 8101 at $n_{\rm live}=1000$, observed live points in the high-$\eta$ region dropped to zero at iteration 10,500 ($n_{\rm live, high} = 0$). Subsequently, live points reappeared in the high-$\eta$ region at iteration 11,000 ($n_{\rm live, high} = 2$) while still bounded by a single ellipsoid (`bound_count = 1`), followed by a bounding split into 21 ellipsoids at iteration 12,000, eventually capturing 100% of live points by iteration 22,500.
4. **Impact of Intervention 2 ($n_{\rm live}=500$, `rslice`):**
   - **Efficiency:** Required $319,299$ (Seed 8101) and $316,465$ (Seed 8201) calls for Model 2 ($36.22\%$ and $42.99\%$ fewer than paired baseline runs; $39.78\%$ fewer than the Model 2 baseline mean). For Model 3, calls were reduced by $34.02\%$ (Seed 8101) and $49.74\%$ (Seed 8201) relative to paired baselines ($42.36\%$ reduction relative to the Model 3 baseline mean). Pooled across both models, `rslice` averaged $323,339.8$ calls, a $41.12\%$ reduction relative to the pooled baseline mean ($549,149.0$ calls).
   - **Evidence vs. Shape Repeatability:** For Model 2 (Plug-in), evidence discrepancy passed both C-1 and C-2: $\Delta\ln\mathcal{Z} = \mathbf{0.9612\,\mathrm{nats}} \le 1.0\,\mathrm{nat}$ ($z = 2.98 \le 3.0$). However, passing evidence thresholds did not yield posterior-shape or parameter repeatability: shell swing was $55.27\%$, linear location and interval criteria failed, and circular-arc overlap was poor ($J = 0.168$). Passing evidence repeatability alone does not establish posterior-shape or parameter-estimation repeatability.
5. **Campaign Resource Accounting and Budget-Safeguard Clarification:**
   - **Authorization Compliance:** Exactly 12 authorized run specifications were submitted; no additional run specifications were authorized, executed, or reported.
   - **Likelihood Call Cap:** Consumed **7,548,191 evaluations** out of the authorized maximum cap of $9,740,000$ calls ($77.50\%$). The likelihood-call cap was respected.
   - **CPU Consumption vs. Projected Budget:** Summed process CPU time across all 12 runs was **$7,723.50\,\mathrm{s} = 2.1454\,\mathrm{core\text{-}hours}$**, which is approximately $180.3\%$ of the Option B projected upper budget ($1.19\,\mathrm{core\text{-}hours}$), exceeding the projected budget by approximately $80.3\%$. The $4.41\,\mathrm{core\text{-}hour}$ figure served as a computational reference ceiling (against which consumption was $48.65\%$), rather than additional authorization.
   - **Safeguard Implementation Failure:** The runtime budget monitor was embedded in the master thread's `as_completed()` result-collection loop after all 12 run specifications had already been submitted to `ProcessPoolExecutor(max_workers=6)` at startup. Consequently, it did not function as an effective pre-launch gate. When the cumulative CPU threshold was crossed after 7 runs completed, the master called `future.cancel()` and broke the polling loop, but cancellation was ineffective against already-running worker tasks. All 12 runs proceeded to completion on disk.
   - **Summary of Resource Performance:** The likelihood-call cap was respected, but actual CPU consumption exceeded the projected budget because the safeguard failed to provide pre-launch enforcement. The 12 runs themselves were the authorized set, not unauthorized expansions.
   - **Overall Repeatability Verdict:** **None of the six model/configuration pairs passes the complete repeatability diagnostic suite.**

---

## 1. Exact Run Matrix and Completion Status

All 12 runs specified in `reports/WP6_Stage5G2_Sampler_Resolution_Design.md` Revision 1.5, Table 6.1b were executed without alteration or substitution. All 12 completed with `status = "SUCCESS"`.

| Run Index | Run ID | Tier / Configuration | Model | Sampler Seed | $n_{\rm live}$ | Method | Tuning Parameters | Status |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---|:---:|
| 1 | `base_01_plug_inst_s1` | Baseline Rerun | Model 2 (Plug-in) | 8101 | 500 | `rwalk` | `walks=50`, `dlogz=0.10` | **SUCCESS** |
| 2 | `base_01_plug_inst_s2` | Baseline Rerun | Model 2 (Plug-in) | 8201 | 500 | `rwalk` | `walks=50`, `dlogz=0.10` | **SUCCESS** |
| 3 | `base_01_stud_inst_s1` | Baseline Rerun | Model 3 (Student-$t$) | 8101 | 500 | `rwalk` | `walks=50`, `dlogz=0.10` | **SUCCESS** |
| 4 | `base_01_stud_inst_s2` | Baseline Rerun | Model 3 (Student-$t$) | 8201 | 500 | `rwalk` | `walks=50`, `dlogz=0.10` | **SUCCESS** |
| 5 | `diag_01_plug_nl1000_s1` | Intervention 1 | Model 2 (Plug-in) | 8101 | 1000 | `rwalk` | `walks=50`, `dlogz=0.10` | **SUCCESS** |
| 6 | `diag_01_plug_nl1000_s2` | Intervention 1 | Model 2 (Plug-in) | 8201 | 1000 | `rwalk` | `walks=50`, `dlogz=0.10` | **SUCCESS** |
| 7 | `diag_01_plug_rslice_s1` | Intervention 2 | Model 2 (Plug-in) | 8101 | 500 | `rslice` | `slices=5`, `dlogz=0.10` | **SUCCESS** |
| 8 | `diag_01_plug_rslice_s2` | Intervention 2 | Model 2 (Plug-in) | 8201 | 500 | `rslice` | `slices=5`, `dlogz=0.10` | **SUCCESS** |
| 9 | `diag_01_stud_nl1000_s1` | Intervention 1 | Model 3 (Student-$t$) | 8101 | 1000 | `rwalk` | `walks=50`, `dlogz=0.10` | **SUCCESS** |
| 10 | `diag_01_stud_nl1000_s2` | Intervention 1 | Model 3 (Student-$t$) | 8201 | 1000 | `rwalk` | `walks=50`, `dlogz=0.10` | **SUCCESS** |
| 11 | `diag_01_stud_rslice_s1` | Intervention 2 | Model 3 (Student-$t$) | 8101 | 500 | `rslice` | `slices=5`, `dlogz=0.10` | **SUCCESS** |
| 12 | `diag_01_stud_rslice_s2` | Intervention 2 | Model 3 (Student-$t$) | 8201 | 500 | `rslice` | `slices=5`, `dlogz=0.10` | **SUCCESS** |

---

## 2. Per-Run Configuration, Lifecycle Accounting, and Compute Resources

Dynesty 3.1.0 lifecycle accounting was strictly verified for every run:
\[
N_{\rm total\_samples} = N_{\rm iterations} (N_{\rm dead}) + n_{\rm live\_final}
\]
Zero duplicate live points were introduced during the terminal `add_live_points()` phase across all 12 runs.

| Run ID | Model | $n_{\rm live}$ | Method | Seed | Likelihood Calls $N_{\rm call}$ | Shrinkage Dead $N_{\rm dead}$ | Final Live $n_{\rm live}$ | Total Samples $N_{\rm samples}$ | Duplicate Live Pts | Process CPU Time [s] | Worker Wall Time [s] |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `base_01_plug_inst_s1` | Plug-in | 500 | `rwalk` | 8101 | 500,657 | 11,451 | 500 | 11,951 | 0 | 496.88 | 913.19 |
| `base_01_plug_inst_s2` | Plug-in | 500 | `rwalk` | 8201 | 555,124 | 12,569 | 500 | 13,069 | 0 | 560.86 | 956.17 |
| `base_01_stud_inst_s1` | Student-$t$ | 500 | `rwalk` | 8101 | 535,854 | 12,176 | 500 | 12,676 | 0 | 541.73 | 705.66 |
| `base_01_stud_inst_s2` | Student-$t$ | 500 | `rwalk` | 8201 | 604,961 | 13,552 | 500 | 14,052 | 0 | 610.53 | 718.77 |
| `diag_01_plug_nl1000_s1` | Plug-in | 1000 | `rwalk` | 8101 | 1,102,516 | 24,978 | 1000 | 25,978 | 0 | 1,126.27 | 2,085.43 |
| `diag_01_plug_nl1000_s2` | Plug-in | 1000 | `rwalk` | 8201 | 948,049 | 21,845 | 1000 | 22,845 | 0 | 961.78 | 1,796.94 |
| `diag_01_plug_rslice_s1` | Plug-in | 500 | `rslice` | 8101 | 319,299 | 11,079 | 500 | 11,579 | 0 | 312.86 | 580.02 |
| `diag_01_plug_rslice_s2` | Plug-in | 500 | `rslice` | 8201 | 316,465 | 10,630 | 500 | 11,130 | 0 | 301.14 | 562.35 |
| `diag_01_stud_nl1000_s1` | Student-$t$ | 1000 | `rwalk` | 8101 | 1,021,616 | 23,299 | 1000 | 24,299 | 0 | 1,098.73 | 2,035.63 |
| `diag_01_stud_nl1000_s2` | Student-$t$ | 1000 | `rwalk` | 8201 | 986,055 | 22,571 | 1000 | 23,571 | 0 | 1,050.98 | 1,946.44 |
| `diag_01_stud_rslice_s1` | Student-$t$ | 500 | `rslice` | 8101 | 353,544 | 11,986 | 500 | 12,486 | 0 | 359.52 | 667.31 |
| `diag_01_stud_rslice_s2` | Student-$t$ | 500 | `rslice` | 8201 | 304,051 | 10,745 | 500 | 11,245 | 0 | 302.22 | 560.28 |

---

## 3. Evidence Integration and Statistical Error Summary

For all 12 runs on Realization 01, the data log-likelihood normalization constant is identical:
\[
\ln\mathcal{C}_{\rm norm} = 203,711.5082\,\mathrm{nats}
\]
The total log-evidence is computed as $\ln\mathcal{Z} = \ln\mathcal{C}_{\rm norm} + \ln\mathcal{I}$, where $\mathcal{I} = \int \exp(\Delta\ln\mathcal{L})\,\pi(\theta)\,d\theta$.

| Run ID | Model | Configuration | Seed | Total Evidence $\ln\mathcal{Z}$ [nats] | Numerical Error `logzerr` [nats] | Prior Integral $\ln\mathcal{I}$ [nats] | Final $\Delta\ln\mathcal{Z}_{\rm remaining}$ [nats] |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `base_01_plug_inst_s1` | Model 2 | Baseline ($n_{\rm live}=500$, `rwalk`) | 8101 | 201,646.1413 | $\pm 0.2374$ | -2,065.3669 | 0.2523 |
| `base_01_plug_inst_s2` | Model 2 | Baseline ($n_{\rm live}=500$, `rwalk`) | 8201 | 201,643.1693 | $\pm 0.2338$ | -2,068.3390 | 0.1143 |
| `base_01_stud_inst_s1` | Model 3 | Baseline ($n_{\rm live}=500$, `rwalk`) | 8101 | 201,645.3073 | $\pm 0.2367$ | -2,066.2009 | 0.1415 |
| `base_01_stud_inst_s2` | Model 3 | Baseline ($n_{\rm live}=500$, `rwalk`) | 8201 | 201,642.8294 | $\pm 0.2582$ | -2,068.6789 | 0.1110 |
| `diag_01_plug_nl1000_s1` | Model 2 | Intervention 1 ($n_{\rm live}=1000$, `rwalk`) | 8101 | 201,644.0470 | $\pm 0.1805$ | -2,067.4612 | 0.1633 |
| `diag_01_plug_nl1000_s2` | Model 2 | Intervention 1 ($n_{\rm live}=1000$, `rwalk`) | 8201 | 201,647.1804 | $\pm 0.1679$ | -2,064.3278 | 0.1417 |
| `diag_01_plug_rslice_s1` | Model 2 | Intervention 2 ($n_{\rm live}=500$, `rslice`) | 8101 | 201,646.8185 | $\pm 0.2257$ | -2,064.6897 | 0.1169 |
| `diag_01_plug_rslice_s2` | Model 2 | Intervention 2 ($n_{\rm live}=500$, `rslice`) | 8201 | 201,647.7797 | $\pm 0.2299$ | -2,063.7285 | 0.1308 |
| `diag_01_stud_nl1000_s1` | Model 3 | Intervention 1 ($n_{\rm live}=1000$, `rwalk`) | 8101 | 201,646.6070 | $\pm 0.1699$ | -2,064.9012 | 0.1351 |
| `diag_01_stud_nl1000_s2` | Model 3 | Intervention 1 ($n_{\rm live}=1000$, `rwalk`) | 8201 | 201,647.3514 | $\pm 0.1703$ | -2,064.1569 | 0.1075 |
| `diag_01_stud_rslice_s1` | Model 3 | Intervention 2 ($n_{\rm live}=500$, `rslice`) | 8101 | 201,645.9024 | $\pm 0.2345$ | -2,065.6058 | 0.2508 |
| `diag_01_stud_rslice_s2` | Model 3 | Intervention 2 ($n_{\rm live}=500$, `rslice`) | 8201 | 201,648.4393 | $\pm 0.2321$ | -2,063.0689 | 0.1661 |

---

## 4. Posterior ESS, Maximum Weights, and Mismatch Shell Probabilities

The posterior samples were evaluated across waveform-distance mismatch shells relative to the true injection:
- $\mathcal{S}_0: d < 0.05$ (near-injection region)
- $\mathcal{S}_1: 0.05 \le d < 0.10$
- $\mathcal{S}_2: 0.10 \le d < 0.20$ (displaced / low-$\eta$ region)
- $\mathcal{S}_3: d \ge 0.20$ (diffuse tails)

All 12 runs satisfied the weight concentration criteria: $w_{\max} < 0.01$ (Criterion C-8a PASS) and $N_{\rm eff} \ge 2,000$ (Criterion C-8b PASS).

| Run ID | Model | Config | Seed | $N_{\rm eff}$ | $w_{\max}$ | Shell $\mathcal{S}_0$ ($d < 0.05$) | Shell $\mathcal{S}_1$ ($0.05\le d<0.1$) | Shell $\mathcal{S}_2$ ($0.1\le d<0.2$) | Shell $\mathcal{S}_3$ ($d \ge 0.2$) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `base_01_plug_inst_s1` | Plug-in | Baseline | 8101 | 2,867.4 | 0.00050 | **64.63%** | 31.80% | **3.57%** | $1.3\times 10^{-11}\%$ |
| `base_01_plug_inst_s2` | Plug-in | Baseline | 8201 | 3,470.8 | 0.00044 | **2.21%** | 20.60% | **77.19%** | $1.1\times 10^{-7}\%$ |
| `base_01_stud_inst_s1` | Student-$t$ | Baseline | 8101 | 3,100.1 | 0.00047 | **45.42%** | 19.34% | **35.24%** | $1.2\times 10^{-7}\%$ |
| `base_01_stud_inst_s2` | Student-$t$ | Baseline | 8201 | 3,629.8 | 0.00039 | **38.64%** | 30.34% | **31.02%** | $1.0\times 10^{-7}\%$ |
| `diag_01_plug_nl1000_s1` | Plug-in | $n_{\rm live}=1000$ | 8101 | 6,300.7 | 0.00022 | **63.81%** | 32.44% | **3.75%** | $1.3\times 10^{-7}\%$ |
| `diag_01_plug_nl1000_s2` | Plug-in | $n_{\rm live}=1000$ | 8201 | 5,451.6 | 0.00026 | **76.04%** | 22.06% | **1.90%** | $1.2\times 10^{-7}\%$ |
| `diag_01_plug_rslice_s1` | Plug-in | `rslice` | 8101 | 3,171.2 | 0.00046 | **27.57%** | 56.24% | **16.19%** | $7.8\times 10^{-8}\%$ |
| `diag_01_plug_rslice_s2` | Plug-in | `rslice` | 8201 | 2,266.3 | 0.00065 | **82.84%** | 16.73% | **0.43%** | $8.0\times 10^{-8}\%$ |
| `diag_01_stud_nl1000_s1` | Student-$t$ | $n_{\rm live}=1000$ | 8101 | 6,986.0 | 0.00021 | **23.68%** | 33.91% | **42.41%** | $9.9\times 10^{-8}\%$ |
| `diag_01_stud_nl1000_s2` | Student-$t$ | $n_{\rm live}=1000$ | 8201 | 5,985.7 | 0.00022 | **58.12%** | 35.63% | **6.25%** | $1.0\times 10^{-7}\%$ |
| `diag_01_stud_rslice_s1` | Student-$t$ | `rslice` | 8101 | 2,964.4 | 0.00051 | **22.03%** | 57.26% | **20.71%** | $8.2\times 10^{-8}\%$ |
| `diag_01_stud_rslice_s2` | Student-$t$ | `rslice` | 8201 | 2,156.5 | 0.00069 | **75.94%** | 23.95% | **0.11%** | $7.8\times 10^{-8}\%$ |

---

## 5. Between-Seed Repeatability and Diagnostic Criteria Evaluations

Between-seed repeatability was formally evaluated across the six paired repeat configurations using the criteria established in Revision 1.5 of the design document:
- **C-1:** $|\Delta\ln\mathcal{Z}| \le 1.0\,\mathrm{nat}$
- **C-2:** $z_{\rm repeat} = |\Delta\ln\mathcal{Z}| / \sqrt{\sigma_1^2 + \sigma_2^2} \le 3.0$
- **C-3:** Maximum shell probability swing $\max_i |\mathcal{S}_{i,1} - \mathcal{S}_{i,2}| \le 0.10$ ($10\%$)
- **C-4a:** Normalized linear location shift $|\tilde\theta_1 - \tilde\theta_2| / \sigma_{\rm pooled} \le 0.20$ for all $(\mathcal{M}_c, \eta, t_c)$
- **C-4b:** Circular mean direction $|\Delta\bar\phi_{\rm circ}| / \sigma_{\rm circ} \le 0.20$ when $\bar R \ge 0.50$ (declared `UNDEFINED` otherwise)
- **C-5:** Evidence resolution scaling $|\ln\mathcal{Z}(1000) - \ln\mathcal{Z}(500)| \le \sqrt{\sigma_{1000}^2 + \sigma_{500}^2 + \tau_{\rm sys}^2}$ with $\tau_{\rm sys} = 1.0\,\mathrm{nat}$
- **C-6a:** Linear 95% CI relative width difference $\le 15\%$ and endpoint normalized shifts $\le 0.20\sigma$
- **C-6b:** Circular 95% shortest-arc relative length difference $\le 15\%$ and Jaccard overlap $J \ge 0.80$
- **C-8a:** Maximum normalized sample weight $w_{\max} < 0.01$
- **C-8b:** Effective sample size $N_{\rm eff} \ge 2,000$

> [!IMPORTANT]
> **Primary Diagnostic Finding:** None of the six model/configuration pairs passes the complete repeatability diagnostic suite. Individual criterion passes must be strictly distinguished from overall sampler validation.

### Summary of Pairwise Criteria Across All 6 Pairs

| Configuration Pair | Model | $\Delta\ln\mathcal{Z}$ [nats] | $z_{\rm rep}$ | Max Shell Swing | C-1 ($|\Delta\ln Z|\le 1$) | C-2 ($z\le 3$) | C-3 ($\Delta\mathcal{S}\le 10\%$) | C-4a (Linear Loc) | C-4b (Circ Dir) | C-6a (Linear CI) | C-6b (Circ Arc) | C-8a/b (ESS) | Overall Core Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (500, `rwalk`) | Model 2 | 2.9720 | 8.92 | 73.63% | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL ($J=0.17$) | PASS | **FAIL** |
| Baseline (500, `rwalk`) | Model 3 | 2.4779 | 7.07 | 11.08% | FAIL | FAIL | FAIL | FAIL | UNDEF | PASS | FAIL ($J=0.49$) | PASS | **FAIL** |
| Intervention 1 (1000, `rwalk`) | Model 2 | 3.1334 | 12.71 | 12.23% | FAIL | FAIL | FAIL | **PASS** | **PASS** | FAIL | FAIL ($J=0.84$) | PASS | **FAIL** |
| Intervention 1 (1000, `rwalk`) | Model 3 | **0.7443** | 3.09 | 36.07% | **PASS** | FAIL | FAIL | FAIL | UNDEF | FAIL | FAIL ($J=0.71$) | PASS | **FAIL** |
| Intervention 2 (500, `rslice`) | Model 2 | **0.9612** | 2.98 | 55.27% | **PASS** | **PASS** | FAIL | FAIL | UNDEF | FAIL | FAIL ($J=0.17$) | PASS | **FAIL** |
| Intervention 2 (500, `rslice`) | Model 3 | 2.5369 | 7.69 | 53.91% | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL ($J=0.17$) | PASS | **FAIL** |

### Detailed Evaluation of Specific Interventions

1. **Model 2 (Plug-in), $n_{\rm live}=1000$ / `rwalk`:**
   - The evidence discrepancy was $\Delta\ln\mathcal{Z} = 3.1334\,\mathrm{nats}$ ($z = 12.71$), which is larger than the baseline discrepancy ($2.9720\,\mathrm{nats}$).
   - The regional distribution across shells changed substantially compared to the baseline: Seed 8201 allocated $76.04\%$ to Shell $\mathcal{S}_0$ and $1.90\%$ to $\mathcal{S}_2$, avoiding the severe low-$\eta$ trapping seen in baseline Seed 8201.
   - Linear location shifts passed (Criterion C-4a: $|\Delta\mathcal{M}_c|/\sigma = 0.09$, $|\Delta\eta|/\sigma = 0.14$, $|\Delta t_c|/\sigma = 0.10$), and circular mean direction passed (Criterion C-4b: $|\Delta\bar\phi|/\sigma = 0.09$ with $\bar R \ge 0.50$).
   - However, shell swing was $12.23\%$ (failing C-3 $\le 10\%$), linear credible interval endpoints failed (C-6a FAIL due to lower tail differences: $\Delta q_{0.025}/\sigma = 0.84$ on $\mathcal{M}_c$), and circular shortest-arc relative length difference was $16.89\%$ (failing C-6b $\le 15\%$).
   - *Conclusion:* This intervention altered sampling behaviour and produced partial parameter location agreement, but left evidence repeatability and posterior-shape repeatability unresolved. It does not resolve the baseline pathology.
2. **Model 3 (Student-$t$), $n_{\rm live}=1000$ / `rwalk`:**
   - Criterion C-1 passed with evidence discrepancy dropping to $\Delta\ln\mathcal{Z} = 0.7443\,\mathrm{nats} \le 1.0\,\mathrm{nat}$.
   - However, Criterion C-2 failed at $z_{\rm rep} = 3.09 > 3.0$.
   - Shell probabilities differed substantially: Seed 8101 placed $23.68\%$ in $\mathcal{S}_0$ and $42.41\%$ in $\mathcal{S}_2$, while Seed 8201 placed $58.12\%$ in $\mathcal{S}_0$ and $6.25\%$ in $\mathcal{S}_2$ (shell swing $36.07\%$, failing C-3).
   - Linear location shifts failed on $\mathcal{M}_c$ ($0.54\sigma$) and $\eta$ ($1.20\sigma$), and circular mean direction was undefined because $\bar R < 0.50$ for Seed 8101 ($\bar R = 0.2442$).
   - *Conclusion:* This represents a partial improvement in evidence repeatability, but posterior-shape and parameter repeatability remain unresolved.
3. **Model 2 (Plug-in), $n_{\rm live}=500$ / `rslice`:**
   - Evidence criteria C-1 and C-2 both passed: $\Delta\ln\mathcal{Z} = 0.9612\,\mathrm{nats} \le 1.0\,\mathrm{nat}$ and $z_{\rm rep} = 2.98 \le 3.0$.
   - However, shell swing was severe at $55.27\%$ (Seed 8101 had $27.57\%$ in $\mathcal{S}_0$ vs. $82.84\%$ for Seed 8201).
   - Linear location shifts failed ($|\Delta\mathcal{M}_c|/\sigma = 1.05$, $|\Delta\eta|/\sigma = 1.80$), linear interval widths and endpoints failed (C-6a), circular mean direction was undefined for Seed 8101 ($\bar R = 0.4754 < 0.50$), and circular arc Jaccard overlap was very low ($J = 0.168$).
   - *Conclusion:* Passing evidence repeatability thresholds does not establish posterior-shape or parameter-estimation repeatability.

### Parameter Summary Statistics Across All 12 Runs

| Run ID | $\mathcal{M}_c$ Median [$M_\odot$] | $\mathcal{M}_c$ 95% CI [$M_\odot$] | $\eta$ Median | $\eta$ 95% CI | $t_c$ Median [s] | $t_c$ 95% CI [s] | $\bar R$ (Circ Length) | $\bar\phi_{\rm circ}$ [rad] | 95% Shortest Arc [rad] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `base_01_plug_inst_s1` | 28.1965 | [28.053, 28.311] | 0.24809 | [0.2421, 0.2499] | 1.999414 | [1.9983, 2.0006] | 0.6991 | 0.0538 | 3.1115 |
| `base_01_plug_inst_s2` | 28.0945 | [27.978, 28.232] | 0.23813 | [0.2346, 0.2483] | 1.999809 | [1.9987, 2.0010] | 0.5655 | 6.2736 | 4.2323 |
| `diag_01_plug_nl1000_s1` | 28.1756 | [28.016, 28.309] | 0.24760 | [0.2393, 0.2498] | 1.999559 | [1.9983, 2.0005] | 0.6640 | 0.0536 | 3.2912 |
| `diag_01_plug_nl1000_s2` | 28.1824 | [28.078, 28.310] | 0.24790 | [0.2422, 0.2499] | 1.999492 | [1.9984, 2.0011] | 0.7955 | 0.1153 | 2.7785 |
| `diag_01_plug_rslice_s1` | 28.1362 | [27.972, 28.293] | 0.24450 | [0.2351, 0.2497] | 1.999693 | [1.9984, 2.0007] | 0.4754 | 0.1202 | 4.6853 |
| `diag_01_plug_rslice_s2` | 28.2074 | [28.089, 28.281] | 0.24856 | [0.2444, 0.2499] | 1.999379 | [1.9984, 2.0002] | 0.9482 | 0.1292 | 0.7892 |
| `base_01_stud_inst_s1` | 28.0901 | [27.994, 28.288] | 0.24582 | [0.2352, 0.2498] | 1.999858 | [1.9986, 2.0011] | 0.5354 | 0.1706 | 4.1950 |
| `base_01_stud_inst_s2` | 28.1557 | [28.003, 28.298] | 0.24516 | [0.2357, 0.2497] | 1.999566 | [1.9985, 2.0011] | 0.3121 | 6.2736 | 5.1441 |
| `diag_01_stud_nl1000_s1` | 28.1590 | [28.017, 28.309] | 0.24360 | [0.2352, 0.2498] | 1.999459 | [1.9984, 2.0008] | 0.2442 | 0.1788 | 4.6866 |
| `diag_01_stud_nl1000_s2` | 28.2009 | [28.077, 28.310] | 0.24804 | [0.2415, 0.2499] | 1.999376 | [1.9983, 2.0005] | 0.6762 | 0.0899 | 3.3295 |
| `diag_01_stud_rslice_s1` | 28.1514 | [27.973, 28.295] | 0.24450 | [0.2351, 0.2497] | 1.999583 | [1.9984, 2.0008] | 0.5180 | 0.0877 | 4.6214 |
| `diag_01_stud_rslice_s2` | 28.2212 | [28.092, 28.271] | 0.24879 | [0.2451, 0.2499] | 1.999312 | [1.9984, 2.0001] | 0.9758 | 0.1197 | 0.7683 |

---

## 6. Live-Point Regional Occupancy, Temporary Depletion, Reappearance, and Bounding-Region Telemetry

Dynamic live-point telemetry was recorded at intervals of $\Delta i = 500$ iterations. A descriptive regional partition at $\eta = 0.244$ was employed to categorize live points into the high-$\eta$ region ($\eta > 0.244$) and the low-$\eta$ region ($\eta \le 0.244$). This partition is a descriptive spatial tool and does not constitute proof that the likelihood contains two distinct isolated posterior modes.

### Empirical Findings on Regional Live-Point Telemetry

1. **Regional Occupancy Shift in Model 2 Baseline (`base_01_plug_inst_s2`):**
   - At iteration 5,000, 92 live points resided in the high-$\eta$ region ($\eta > 0.244$).
   - At iteration 7,000, the bounding region collapsed into a single ellipsoid (`bound_count = 1`).
   - Between iterations 7,500 and 9,500, live points in the high-$\eta$ region plummeted to 6–9 points (only $1.2\%\text{--}1.8\%$ occupancy).
   - This observed coincidence between `bound_count = 1` and changing regional occupancy is an empirical association consistent with a possible bounding-geometry contribution, but does not prove physical causation.
   - During this iteration window, remaining evidence $\Delta\ln\mathcal{Z}$ shrank from $17.45\,\mathrm{nats}$ to $4.35\,\mathrm{nats}$.
   - Consequently, the sampler allocated $77.19\%$ of its total evidence to the low-$\eta$ region (Shell $\mathcal{S}_2$), leading to the $73.63\%$ shell swing.
2. **Zero Observed Live Points and Reappearance in Intervention 1 (`diag_01_plug_nl1000_s1`):**
   - In Model 2 Seed 8101 with $n_{\rm live}=1000$, observed live points in the high-$\eta$ region reached zero at iteration 10,500:
     \[
     n_{\rm live, high}(10,000) = 3 \quad\longrightarrow\quad n_{\rm live, high}(10,500) = 0 \quad (\text{zero observed live points in high-}\eta\text{ region})
     \]
   - At iteration 10,500, the sampler had a single bounding ellipsoid (`bound_count = 1`) containing 1,000 live points in the low-$\eta$ region.
   - At iteration 11,000, live points reappeared in the high-$\eta$ region ($n_{\rm live, high} = 2$) while bounded by a single ellipsoid (`bound_count = 1`).
   - At iteration 12,000, the bounding region detected multiple clusters and split into 21 ellipsoids (`bound_count = 21`).
   - Following this split, live points in the high-$\eta$ region increased: 18 at iter 12,500; 42 at iter 14,000; 250 at iter 17,000; and 1,000 (complete depletion of the low-$\eta$ region) at iter 22,500.
   - *Diagnostic Implication:* This empirical trajectory demonstrates that live points can become completely depleted in a spatial region and subsequently reappear during single-ellipsoid bounding phases.
3. **Regional Dynamics in Intervention 1 Seed 8201 (`diag_01_plug_nl1000_s2`):**
   - Unlike its baseline counterpart, Seed 8201 with $n_{\rm live}=1000$ never dropped below 21 live points in the high-$\eta$ region.
   - Live points in the high-$\eta$ region expanded steadily: 29 at iter 5,000; 138 at iter 9,000; 465 at iter 11,000; 777 at iter 14,000; and 1,000 at iter 20,000.
   - Consequently, Seed 8201 placed $76.04\%$ in Shell $\mathcal{S}_0$ and only $1.90\%$ in Shell $\mathcal{S}_2$, avoiding the low-$\eta$ concentration of the baseline.

| Run ID | Initial High-$\eta$ Live Pts | Minimum High-$\eta$ Live Pts | Iteration of Minimum | Zero Observed High-$\eta$ Pts? | High-$\eta$ Reappearance Iter? | Final High-$\eta$ Live Pts | Low-$\eta$ Depletion Iter ($n_{\rm live, low}=0$) | Maximum Bounding Objects | Final Bounding Objects |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `base_01_plug_inst_s1` | 25 | 13 | 2,500 | No | N/A | 500 (100%) | 10,500 | 17 | 3 |
| `base_01_plug_inst_s2` | 33 | 6 | 8,500 | No | N/A | 113 (22.6%) | None ($n_{\rm low}=387$) | 17 | 15 |
| `base_01_stud_inst_s1` | 27 | 12 | 2,500 | No | N/A | 500 (100%) | 11,500 | 18 | 4 |
| `base_01_stud_inst_s2` | 31 | 0 | 8,500 | **Yes** | 9,000 | 500 (100%) | 13,000 | 14 | 3 |
| `diag_01_plug_nl1000_s1` | 58 | 0 | 10,500 | **Yes** | 11,000 | 1,000 (100%) | 22,500 | 21 | 3 |
| `diag_01_plug_nl1000_s2` | 58 | 21 | 3,000 | No | N/A | 1,000 (100%) | 19,500 | 22 | 6 |
| `diag_01_stud_nl1000_s1` | 50 | 30 | 4,000 | No | N/A | 1,000 (100%) | 22,000 | 24 | 4 |
| `diag_01_stud_nl1000_s2` | 58 | 28 | 2,000 | No | N/A | 1,000 (100%) | 20,500 | 31 | 3 |
| `diag_01_plug_rslice_s1` | 25 | 16 | 1,000 | No | N/A | 500 (100%) | 10,500 | 12 | 1 |
| `diag_01_plug_rslice_s2` | 33 | 14 | 1,000 | No | N/A | 500 (100%) | 9,500 | 14 | 1 |
| `diag_01_stud_rslice_s1` | 27 | 14 | 2,000 | No | N/A | 494 (98.8%) | None ($n_{\rm low}=6$) | 12 | 2 |
| `diag_01_stud_rslice_s2` | 31 | 14 | 1,000 | No | N/A | 500 (100%) | 8,500 | 7 | 1 |

---

## 7. Evidence Accumulation and Remaining-Evidence Diagnostics

All runs stopped cleanly under the frozen convergence criterion $\text{dlogz} = 0.10$. In nested sampling, remaining evidence is estimated as:
\[
\Delta\ln\mathcal{Z}_{\rm remaining} = \ln\left( \mathcal{Z}_i + \mathcal{L}_{\max} X_i \right) - \ln\mathcal{Z}_i
\]
where $X_i = \exp(-i / n_{\rm live})$.

At the final recorded snapshot before adding live points:
- For all runs, $\Delta\ln\mathcal{Z}_{\rm remaining}$ had dropped to between $0.1075\,\mathrm{nats}$ and $0.2523\,\mathrm{nats}$.
- The contribution from terminal live points ($\sum_j w_j^{\rm final} \approx X_{N_{\rm dead}}$) added between $0.10$ and $0.25$ nats to the accumulated evidence, terminating shrinkage in exact accordance with the stopping criterion.
- Accumulated evidence plateaued smoothly in all 12 runs, confirming that no run stopped prematurely before the posterior mass was integrated.

---

## 8. Instrumentation Overhead and Lifecycle-Integrity Checks

1. **Bitwise Reproduction Verification:**
   Before running the 12-run pilot, a preliminary lifecycle equivalence check was executed comparing uninstrumented baseline sampler execution against the instrumented sampler (`scratch/equivalence_results.json`). The results demonstrated exact bitwise equality:
   \[
   |\ln\mathcal{Z}_{\rm uninst} - \ln\mathcal{Z}_{\rm inst}| = 0.00\times 10^0\,\mathrm{nats}, \quad N_{\rm call, uninst} = N_{\rm call, inst} = 500,657
   \]
2. **Computational Overhead of Hook Callbacks:**
   Telemetry hooks recorded iterations, bounding objects, live point snapshots, and parameter quantiles. Across the 12 runs, the wall-clock time overhead of the hook callbacks was measured at $< 1.5\%$ of total run time. The instrumentation introduced zero memory leaks or process stalls.
3. **Sample Count Integrity:**
   In every run, $N_{\rm total\_samples} = N_{\rm dead} + n_{\rm live}$ was verified with zero duplicate live points.

---

## 9. Actual Cumulative Call Count and CPU Budget Consumption

Execution was managed with a maximum of 6 parallel worker processes using Python's `ProcessPoolExecutor`.

| Metric | Option B Authorized / Projected Budget | Actual Consumed Value | Percentage of Budget | Budget Performance Status |
|:---|:---:|:---:|:---:|:---:|
| **Total Likelihood Evaluations** | $9,740,000$ calls (Authorized Cap) | **7,548,191 calls** | **77.50%** | **Within Authorized Cap** |
| **Total Process CPU Time** | $1.19\,\mathrm{core\text{-}hours}$ ($4,284\,\mathrm{s}$, Projected Budget) | **$2.1454\,\mathrm{core\text{-}hours}$** ($7,723.50\,\mathrm{s}$) | **180.29%** | **Exceeded Projected Budget (~80.3% over)** |
| **Reference Ceiling** | $4.41\,\mathrm{core\text{-}hours}$ ($15,876\,\mathrm{s}$, Reference Ceiling) | $2.1454\,\mathrm{core\text{-}hours}$ | 48.65% | *Reference ceiling only; not additional authorization* |
| **Summed Worker Wall Time** | — | $13,528.19\,\mathrm{s}$ ($3.7578\,\mathrm{hours}$) | — | Summed across 12 worker processes |
| **Longest Single Worker Wall Time** | — | $2,085.43\,\mathrm{s}$ ($34.76\,\mathrm{min}$) | — | `diag_01_plug_nl1000_s1` |
| **Parallel Campaign Wall Time** | — | $\approx 2,720\,\mathrm{s}$ ($\approx 45.3\,\mathrm{min}$) | — | Wall clock across 6 worker slots |

### Detailed Accounting Basis and Verification

1. **How the 7,723.50 s Figure Was Measured:**
   - In `scripts/run_wp6_stage5_stage_g2.py`, each worker process measured process CPU time directly via Python's `time.process_time()`:
     \[
     t_{\rm cpu} = \text{time.process\_time()} - t_{0, \rm cpu}
     \]
   - Each worker saved its measured `cpu_time_s` in its run JSON artifact upon completion.
   - The values from all 12 JSON files sum exactly to:
     \[
     \sum_{k=1}^{12} t_{\rm cpu}^{(k)} = 7,723.50\,\mathrm{s}
     \]
   - Converted to core-hours: $7,723.50\,\mathrm{s} / 3,600\,\mathrm{s/hr} = 2.145417\,\mathrm{core\text{-}hours}$.
2. **Accounting Scope (Process Time vs. Multithreading):**
   - Python's `time.process_time()` measures the combined user and system CPU time of the process. On the Windows host platform, multi-threaded numerical operations in NumPy/BLAS (employed during frequency-domain waveform generation and FFT evaluation) accumulate CPU time across active threads.
   - As a consequence, process CPU time accumulated at an effective rate of $\sim 1.02\,\mathrm{ms}$ per likelihood call, compared to the planning estimate of $\sim 0.44\,\mathrm{ms}$ per call (which had assumed strictly single-threaded execution).
   - In terms of elapsed wall-clock time, the 12 worker processes ran concurrently across 6 physical slots, completing in $\approx 45.3\,\mathrm{minutes}$ of parallel wall time (with the longest single run taking $34.76\,\mathrm{minutes}$).
3. **Distinction Between Authorization Compliance and Resource-Budget Performance:**
   - **Authorized Scope:** Exactly the 12 authorized run specifications were submitted to the execution pool. No additional runs, higher resolutions, alternative realizations, or downstream stages were authorized or attempted.
   - **Likelihood Call Cap:** The primary numerical and physical complexity ceiling of $9.74\times 10^6$ calls was fully respected ($77.50\%$ utilization).
   - **CPU Performance vs. Projected Budget:** Actual CPU process time ($2.1454\,\mathrm{core\text{-}hours}$) exceeded the $1.19\,\mathrm{core\text{-}hour}$ projected budget by approximately $80.3\%$. This discrepancy arose from multi-threaded BLAS operations consuming $\sim 1.02\,\mathrm{ms/call}$ instead of the single-threaded $0.44\,\mathrm{ms/call}$ assumed during planning.
   - **Reference Ceiling:** The separate $4.41\,\mathrm{core\text{-}hour}$ reference ceiling was an external reference ceiling, not an expanded authorization.
   - **Safeguard Enforcement:** As detailed in Section 10, the safeguard implementation failed to provide reliable pre-launch enforcement of the 1.19 core-hour budget because all tasks were dispatched to the worker pool at startup rather than gated in sequence.
   - **Overall Assessment:** The campaign cannot be characterized as fully compliant with resource projections, nor were the completed runs unauthorized expansions; rather, the authorized 12-run pilot was executed, respected the likelihood-call cap, but exceeded its projected CPU budget due to the absence of pre-launch safeguard gating.

---

## 10. Failures, Warnings, Unresolved Diagnostics, and Orchestration Exit Code

### Chronology and Mechanism of the Budget Safeguard Failure

The Option B authorization directive mandated: *"Monitor cumulative likelihood calls and CPU time throughout execution. If the projected budget is exceeded, stop before launching additional runs and report."*

However, the software implementation in `scripts/run_wp6_stage5_stage_g2.py` failed to provide effective pre-launch enforcement:

1. **All 12 Runs Dispatched at Startup:** At $t_0$, the orchestrator submitted all 12 authorized run specifications into `ProcessPoolExecutor(max_workers=6)` simultaneously:
   ```python
   future_to_spec = {
       executor.submit(execute_single_pilot_run, spec, OUTPUT_DIR): spec
       for spec in AUTHORIZED_RUN_MATRIX
   }
   ```
   No pre-launch gating or staged dispatch was implemented.
2. **Post-Hoc Monitoring in Master Collection Loop:** Cumulative resource tracking was performed only as worker processes completed and were yielded by `as_completed(future_to_spec)`.
3. **Exact Chronology of the Threshold Crossing:**
   From the captured stdout log of the execution process (`task-12041.log`), the exact chronology of results yielded to the master thread was:
   - Run 1/12 (`diag_01_plug_rslice_s2`): Cum calls $316,465$ ($3.2\%$), Cum CPU $0.084\,\mathrm{hrs}$ ($7.0\%$).
   - Run 2/12 (`diag_01_plug_rslice_s1`): Cum calls $635,764$ ($6.5\%$), Cum CPU $0.171\,\mathrm{hrs}$ ($14.3\%$).
   - Run 3/12 (`diag_01_stud_rslice_s2`): Cum calls $939,815$ ($9.6\%$), Cum CPU $0.255\,\mathrm{hrs}$ ($21.4\%$).
   - Run 4/12 (`diag_01_stud_rslice_s1`): Cum calls $1,293,359$ ($13.3\%$), Cum CPU $0.354\,\mathrm{hrs}$ ($29.8\%$).
   - Run 5/12 (`diag_01_plug_nl1000_s2`): Cum calls $2,241,408$ ($23.0\%$), Cum CPU $0.622\,\mathrm{hrs}$ ($52.2\%$).
   - Run 6/12 (`diag_01_stud_nl1000_s2`): Cum calls $3,227,463$ ($33.1\%$), Cum CPU $0.913\,\mathrm{hrs}$ ($76.8\%$).
   - Run 7/12 (`diag_01_stud_nl1000_s1`): Cum calls $4,249,079$ ($43.6\%$), Cum CPU $4,387.23\,\mathrm{s} = 1.2187\,\mathrm{core\text{-}hours}$ ($102.41\%$ of the $1.19\,\mathrm{hr}$ budget).
4. **Triggering of the Safeguard:**
   Upon processing the 7th completed run, the condition `core_hrs > MAX_CORE_HOURS` ($1.219 > 1.190$) evaluated to `True`. The master thread printed:
   ```
   [ALERT] Safeguard budget limit exceeded! Calls: 4249079 > 9740000 or Core-hrs: 1.219 > 1.190!
   Halting remaining tasks pursuant to supervisor safeguards.
   ```
5. **Ineffectiveness of Cancellation on Running Tasks:**
   The master script executed:
   ```python
   for f in future_to_spec:
       f.cancel()
   break
   ```
   - *Status of Futures:* Exactly 7 futures had completed. The remaining 5 futures (`base_01_plug_inst_s1`, `base_01_plug_inst_s2`, `base_01_stud_inst_s1`, `base_01_stud_inst_s2`, and `diag_01_plug_nl1000_s1`) were already actively executing across the 6 worker slots.
   - *Return Values of `cancel()`:* The script did not log the boolean return values of `f.cancel()`. However, under Python's `concurrent.futures`, `cancel()` returns `False` for any future that has already started running.
   - *Result:* Because all 12 tasks had already been submitted and the remaining 5 were actively executing in child worker processes, `cancel()` could not terminate them. All 12 worker runs continued executing to their normal numerical termination and wrote their output artifacts to disk.
6. **Master Script `KeyError` and Exit Code 1:**
   Because the master thread broke out of the collection loop after ingesting only 7 results into `run_results`, the subsequent block attempted to compute pairwise repeatability:
   ```python
   r1 = run_results[id1]
   r2 = run_results[id2]
   ```
   This raised an unhandled `KeyError: 'base_01_plug_inst_s1'`, terminating the master process with Python exit code 1.
7. **Resolution:**
   The exit code 1 was strictly a post-processing in-memory dictionary aggregation failure in the master orchestrator, stemming directly from the flawed safeguard implementation. It was not a sampler failure or numerical error. Summary compilation was completed via `scripts/postprocess_wp6_stage5_stage_g2.py` from the verified disk artifacts without launching any new runs or numerical evaluations.

### Verification of `rslice` Efficiency Arithmetic

The claim of evaluation efficiency for `rslice` was recalculated across all defined comparators from the measured likelihood call counts in Table 2:

- **Model 2 (Plug-in Gaussian):**
  - Seed 8101: Baseline consumed $500,657$ calls; `rslice` consumed $319,299$ calls.
    \[
    \frac{500,657 - 319,299}{500,657} = \frac{181,358}{500,657} = 36.22\% \text{ reduction}
    \]
  - Seed 8201: Baseline consumed $555,124$ calls; `rslice` consumed $316,465$ calls.
    \[
    \frac{555,124 - 316,465}{555,124} = \frac{238,659}{555,124} = 42.99\% \text{ reduction}
    \]
  - Model 2 Mean: Baseline averaged $527,890.5$ calls; `rslice` averaged $317,882.0$ calls ($39.78\%$ reduction).
- **Model 3 (Student-$t$ Relative Dispersion):**
  - Seed 8101: Baseline consumed $535,854$ calls; `rslice` consumed $353,544$ calls.
    \[
    \frac{535,854 - 353,544}{535,854} = \frac{182,310}{535,854} = 34.02\% \text{ reduction}
    \]
  - Seed 8201: Baseline consumed $604,961$ calls; `rslice` consumed $304,051$ calls.
    \[
    \frac{604,961 - 304,051}{604,961} = \frac{300,910}{604,961} = 49.74\% \text{ reduction}
    \]
  - Model 3 Mean: Baseline averaged $570,407.5$ calls; `rslice` averaged $328,797.5$ calls ($42.36\%$ reduction).
- **Pooled Across Both Models ($n_{\rm live}=500$):**
  - Pooled Baseline: $2,196,596$ total calls (mean $549,149.0$ calls/run).
  - Pooled `rslice`: $1,293,359$ total calls (mean $323,339.75$ calls/run).
  - Pooled Reduction:
    \[
    \frac{2,196,596 - 1,293,359}{2,196,596} = \frac{903,237}{2,196,596} = 41.12\% \text{ reduction}
    \]
The savings range from $34.02\%$ (Model 3 Seed 8101) to $49.74\%$ (Model 3 Seed 8201), with a model-2 paired average of $39.78\%$, a model-3 paired average of $42.36\%$, and a pooled reduction of $41.12\%$.

---

## 11. Explicit Comparison Against Preserved Stage 5G.1 Baseline

The four instrumented baseline reruns were compared directly against the historical Stage 5G.1 JSON and NPZ artifacts preserved in `reports/demonstrations/wp6_stage5/stage_g1/`:

| Case | Historical Stage 5G.1 Evidence $\ln\mathcal{Z}$ | Stage 5G.2 Baseline Rerun Evidence $\ln\mathcal{Z}$ | Absolute Evidence Difference $|\Delta\ln\mathcal{Z}|$ [nats] | Historical Calls | Rerun Calls | Sample Count $N_{\rm samples}$ | Reproducibility Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Model 2, Seed 8101 | 201,646.1413147189 | 201,646.1413147189 | **$0.00\times 10^0$** | 500,657 | 500,657 | 11,951 | **EXACT BITWISE MATCH** |
| Model 2, Seed 8201 | 201,643.1692960849 | 201,643.1692960849 | **$0.00\times 10^0$** | 555,124 | 555,124 | 13,069 | **EXACT BITWISE MATCH** |
| Model 3, Seed 8101 | 201,645.3073035554 | 201,645.3073035554 | **$0.00\times 10^0$** | 535,854 | 535,854 | 12,676 | **EXACT BITWISE MATCH** |
| Model 3, Seed 8201 | 201,642.8293799185 | 201,642.8293799185 | **$0.00\times 10^0$** | 604,961 | 604,961 | 14,052 | **EXACT BITWISE MATCH** |

All historical Stage 5G.1 artifacts remain unmodified and fully preserved on disk.

---

## 12. Final Git Status and Working Tree Verification

As required by supervisor safeguards, no Git staging, commits, resets, checkouts, stashes, or cleans were performed. The dirty working tree from earlier stages was preserved completely intact.

```
On branch milestone-1.4a
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   pyproject.toml
	modified:   src/gw_inference/__init__.py
	modified:   src/gw_inference/noise.py
	modified:   src/gw_inference/psd.py
	modified:   src/gw_inference/waveform.py

Untracked files:
	reports/demonstrations/wp6_stage5/stage_g2/
	scripts/postprocess_wp6_stage5_stage_g2.py
	scripts/run_wp6_stage5_stage_g2.py
	[Pre-existing untracked demonstration and test scripts preserved]

no changes added to commit (use "git add" and/or "git commit -a")
```

---

## 13. Final Report Interpretation and Supervisor Boundaries

In accordance with strict supervisor instructions, the final findings of this campaign are summarized as follows:

1. **Campaign Completion:** All 12 authorized sampler runs have completed, and all 36 raw artifacts plus the master campaign summary are recorded on disk.
2. **Call Budget Compliance:** The likelihood-call cap was respected ($7,548,191$ calls consumed out of $9,740,000$ authorized, or $77.50\%$).
3. **CPU Budget Accounting and Safeguard Performance:** Summed process CPU time was $2.1454\,\mathrm{core\text{-}hours}$, exceeding the Option B projected budget of $1.19\,\mathrm{core\text{-}hours}$ by approximately $80.3\%$ due to host-level multi-threaded BLAS operations ($\sim 1.02\,\mathrm{ms/call}$ vs. assumed $0.44\,\mathrm{ms/call}$). The safeguard monitor failed to provide pre-launch budget enforcement because all 12 authorized runs were submitted to the pool at launch. The computational reference ceiling of $4.41\,\mathrm{core\text{-}hours}$ was an external reference ceiling, not an expanded authorization.
4. **Repeatability Status:** Evidence agreement improved for some interventions (Model 3 with $n_{\rm live}=1000$ achieved $\Delta\ln\mathcal{Z} = 0.7443\,\mathrm{nats}$; Model 2 with `rslice` achieved $\Delta\ln\mathcal{Z} = 0.9612\,\mathrm{nats}$), but posterior-shape, shell-probability, and parameter-estimation repeatability remain unresolved across all six configurations.
5. **No Overall Diagnostic Pass:** None of the six model/configuration pairs passes the complete repeatability diagnostic suite. Passing evidence repeatability thresholds does not establish posterior-shape or parameter repeatability.
6. **No Downstream Claims:** The pilot does not establish sampler reliability, posterior coverage, or the validity of a production likelihood replacement.
7. **Execution Boundary:** No additional runs or downstream stages are authorized. All work is stopped, awaiting supervisor review.
