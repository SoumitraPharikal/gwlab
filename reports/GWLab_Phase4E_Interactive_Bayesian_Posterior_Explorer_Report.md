# GWLab — Phase 4E: Interactive Bayesian Posterior Explorer Report

**Date:** 2026-11-26  
**Target Route:** `http://localhost:3000/inference`  
**Backend Endpoint:** `GET /api/inference/posterior`  
**Status:** **PASS** (Implemented, Tested, and Validated across backend unit/API tests, TypeScript/ESLint/Next.js production build, and live desktop/mobile browser verification)

---

## 1. Files Inspected and Changed

### Inspected
- `experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz` — Saved Stage 6C Run A NPZ archive
- `experiments/wp6_stage6c_anchor/run_a_anchor_result.json` — Authoritative Stage 6C Run A summary & log-evidence
- `experiments/wp6_stage6c_anchor/stage6c_aggregate_ledger.json` — Computational resource ledger
- `src/gwlab/data/loaders.py` & `src/gwlab/data/schemas.py` — Existing NPZ validation & schema conventions
- `reports/WP6_Stage6C_Run_A_Posterior_Forensic_Analysis.md` & `reports/WP6_Stage6D_Sampler_Repeatability_Design_and_Resource_Review.md` — Authoritative forensic & seed-sensitivity reports
- `src/gwlab/api/service.py`, `src/gwlab/api/app.py`, `frontend/src/lib/api.ts`, `frontend/src/app/inference/page.tsx`

### Changed / Added
- **`src/gwlab/api/service.py`** — Added `_weighted_quantiles(...)` and `get_inference_posterior(...)` loading `run_a_anchor_samples.npz` (`allow_pickle=False`) and `run_a_anchor_result.json` with strict shape, finiteness, and weight-sum (`|sum - 1.0| <= 1e-4`) checks.
- **`src/gwlab/api/app.py`** — Added `GET /api/inference/posterior` with validated query parameters (`param_x`, `param_y`, `bins_1d`, `bins_2d`) and HTTP 400/404/422/500 error handling.
- **`tests/test_gwlab_inference.py`** — Added 3 tests verifying artifact identity, parameter ordering, weighted density normalization, 2D grid shape, deterministic API responses, and missing/corrupt artifact error handling.
- **`frontend/src/lib/api.ts`** — Added `PosteriorParamKey`, `PosteriorExplorerResponseData`, and `fetchInferencePosterior(...)`.
- **`frontend/src/components/PosteriorPlot.tsx`** — Added `Marginal1DPlot` (weighted histogram bars, 90% CI shading, median & injected truth lines) and `Joint2DPlot` (weighted 36×36 2D density heatmap with injected crosshair and hover readout).
- **`frontend/src/app/inference/page.tsx`** — Replaced placeholder with the interactive Bayesian Posterior Explorer (top readouts, axis selectors, 2D joint heatmap, 4× 1D marginal plots, weighted summary table, and provenance/limitations panel).

---

## 2. Artifact Schema & Parameter Order Verified

Direct inspection of `experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz` (`allow_pickle=False`) confirmed:
- **Keys:** `['samples', 'weights', 'posterior_samples']`
- **`samples`:** shape `(11951, 4)`, `dtype=float64` (weighted nested-sampling dead points)
- **`weights`:** shape `(11951,)`, `dtype=float64`, non-uniform (`sum = 0.9999999999999999`)
- **`posterior_samples`:** shape `(11951, 4)`, `dtype=float64`
- **Column Order & Units:**
  1. `mc`: Chirp mass $M_c$ [`M_sun`], prior `[20.0, 40.0]`
  2. `eta`: Symmetric mass ratio $\eta$ [`dimensionless`], prior `[0.10, 0.25]`
  3. `tc`: Coalescence time $t_c$ [`s`], prior `[1.90, 2.10]`
  4. `phic`: Coalescence phase $\phi_c$ [`rad`], prior `[0.0, 2π]`

---

## 3. API Endpoint, Weighting & Summary Methodology

- **Endpoint:** `GET /api/inference/posterior?param_x=mc&param_y=eta&bins_1d=40&bins_2d=36`
- **Weighting Methodology:**
  - All 1D marginal histograms (`np.histogram(..., weights=weights, density=True)`) and 2D joint density grids (`np.histogram2d(..., weights=weights)`) strictly weight each dead point by `weights`.
  - 1D marginal histograms are binned over the high-weight posterior support (`[p0.001, p99.9]` padded by 6% for `mc`, `eta`, `tc`; `[0, 2π]` for `phic`) so narrow peaks are resolved cleanly rather than collapsed into a single prior-wide bin.
  - Summary table quantiles (`median`, `p5`, `p95`, `ci90_width`) use the authoritative values from `run_a_anchor_result.json` (verified consistent with midpoint weighted quantiles on `samples`/`weights`), alongside Kish's effective sample size $N_{\mathrm{eff}} = 1 / \sum w_i^2 \approx 2867.4$ and $\eta \in [0.2485, 0.2500]$ mass concentration (`38.57%`).

---

## 4. Provenance & Explicit Limitations

- **Source Run:** Stage 6C Run A (`run_a_anchor`), `Synthetic Realization 01`, `Model 2 (Plug-in Gaussian Whittle)`, `4D Uniform Prior`, `Dynesty (n_live = 500)`, `Seed 8101`, `11,951` weighted dead points, `487,407` guard-reconciled likelihood calls, $\ln\mathcal{Z} = 201646.1413 \pm 0.2374\,\mathrm{nats}$.
- **Explicit Qualifications Displayed:**
  - Saved synthetic single-realization posterior (`Realization 01`), not a real detector event.
  - Single-seed (`8101`) result; Stage 6D reported between-seed evidence sensitivity against seed `8201` of $|\Delta\ln\mathcal{Z}| = 2.9720\,\mathrm{nats}$.
  - Does not establish full convergence, boundary-artifact freedom, or empirical coverage calibration.
  - Model 3 Student-$t$ and Run B remain frozen/unexecuted.
  - Histogram binning (`40` 1D bins, `36×36` 2D bins) is explicitly identified as a visualization choice over weighted dead points.

---

## 5. Implementation Status & Validation Results

| Component / Check | Status | Details |
| :--- | :--- | :--- |
| **Backend Service & API (`service.py`, `app.py`)** | **Implemented & Tested** | `get_inference_posterior` & `GET /api/inference/posterior` |
| **Interactive Frontend (`PosteriorPlot.tsx`, `inference/page.tsx`)** | **Implemented & Validated** | 4× 1D weighted marginals, interactive 2D joint heatmap, summary table, provenance & limitations |
| **Python Test Suite (`pytest`)** | **Tested (34 passed)** | `tests/test_gwlab_inference.py`, `test_gwlab_data.py`, `test_gwlab_api.py`, `test_gwlab_noise_psd.py`, `test_gwlab_matched_filter.py`, `test_gwlab_physics.py` — 34 passed in 11.15s |
| **TypeScript (`npx tsc --noEmit`)** | **Tested (PASS)** | 0 errors |
| **ESLint (`src/lib/api.ts`, `PosteriorPlot.tsx`, `inference/page.tsx`)** | **Tested (PASS)** | 0 errors, 0 warnings |
| **Next.js Production Build (`npm run build`)** | **Tested (PASS)** | Next.js 16.4.6 Turbopack; all 10 routes prerendered |
| **Live Browser (`1440×900` & `390×844`)** | **Validated (PASS)** | `GET /api/inference/posterior` `200 OK`; all 4 readouts, 4× 1D plots, summary table, `(mc, eta)` $\to$ `(tc, phic)` 2D axis switching, and Stage 6D qualification verified; 0 console errors, 0 HTTP `>= 400` errors, 0 page-level overflow or clipped plot labels |

---

## 6. Known Limitations & Next Recommended Step

- **Known Limitations:**
  - Read-only visualization of `run_a_anchor` only; multi-run comparison (e.g., comparing `seed 8101` vs `seed 8201` in `/experiments`) is deferred to Phase 4F.
  - At `390×844` mobile, the display KaTeX Bayes theorem formula inside `Weighted Posterior Summaries` scrolls horizontally within its own `overflow-x-auto` container without causing page overflow.
- **Recommended Next Step:** **Phase 4F** — Interactive Experiment Explorer (`/experiments`) integrating saved multi-seed comparison artifacts (`base_01_plug_inst_s1` vs `base_01_plug_inst_s2`) and historical campaign status.
