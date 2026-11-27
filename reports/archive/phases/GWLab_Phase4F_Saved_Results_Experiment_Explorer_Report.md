# GWLab Phase 4F — Saved-Results Experiment Explorer Report

**Date:** 2026-11-26  
**Target Route:** `http://localhost:3000/experiments`  
**Backend Endpoint:** `GET /api/experiments/explorer`  
**Status:** **ACCEPTED — Closed** (Passed report-based review on 2026-11-26)

---

## 1. Implemented

- **Backend Read-Only Loader & Endpoint (`Implemented & Tested`):**
  - `src/gwlab/api/service.py`: Added `_load_json_checked`, `_extract_finite_float`, `_extract_optional_int`, and `get_experiments_explorer(...)` with strict path validation, JSON parse checks, finiteness checks, provenance consistency checks (same `realization` and `model`, distinct `sampler_seed`), and `None`-safe optional field handling (never fabricating defaults).
  - `src/gwlab/api/app.py`: Added `GET /api/experiments/explorer` returning HTTP `404` on missing files, `400` on corrupt JSON / non-finite / inconsistent provenance, and `500` on unexpected errors.
- **Typed Frontend API Client & Interactive Explorer (`Implemented & Validated`):**
  - `frontend/src/lib/api.ts`: Added `ExperimentRunSummary`, `CampaignStageStatus`, `CampaignStageItem`, `ExperimentsExplorerResponseData`, and `fetchExperimentsExplorer()`.
  - `frontend/src/app/experiments/page.tsx`: Replaced the placeholder with:
    - **Multi-seed sensitivity cards** (`|Δln Z|`, `σ_Δ`, discrepancy ratio) and **side-by-side Stage 5G.2 comparison table** (`base_01_plug_inst_s1` vs `base_01_plug_inst_s2`) with explicit two-seed sensitivity qualification.
    - **Campaign Timeline & Execution Status** with interactive filter tabs (`All`, `Completed`, `Frozen / Proposed / Unexecuted`) and explicit status badges (`Completed / Historical`, `Unexecuted`, `Proposed`, `Frozen / Specified`).
    - **Archive Provenance & Limitations** section, plus loading and API error states.
- **Automated Tests (`Implemented & Tested`):**
  - `tests/test_gwlab_experiments.py`: Added 3 tests covering valid artifact loading, two-run comparison math, status separation, API `200 OK`, missing files (`FileNotFoundError`), malformed JSON (`ValueError`), missing optional fields (`None` without fabrication), and inconsistent provenance (`ValueError`).

---

## 2. Exact Artifact Sources and Displayed Values

1. **Stage 5G.2 Two-Seed Pair (`reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s1.json` & `base_01_plug_inst_s2.json`):**
   - `base_01_plug_inst_s1` (Seed `8101`, Model 2 Whittle Gaussian, Synthetic Realization 01): `log_evidence = 201646.1413 ± 0.2374 nats`, `500,657` evaluations, `11,951` dead points, `Kish N_eff = 2867.4`, medians `(28.1965 M_sun, 0.2481, 1.9994 s)`.
   - `base_01_plug_inst_s2` (Seed `8201`, Model 2 Whittle Gaussian, Synthetic Realization 01): `log_evidence = 201643.1693 ± 0.2338 nats`, `555,124` evaluations, `13,069` dead points, `Kish N_eff = 3470.8`, medians `(28.0945 M_sun, 0.2381, 1.9998 s)`.
   - Derived comparison: `|Δln Z| = 2.9720 nats` (top card formats `2.972 nats` via `formatFloat`), `σ_Δ = 0.3332 nats`, ratio `8.92`.
2. **Stage 6C Run A (`experiments/wp6_stage6c_anchor/run_a_anchor_result.json`):**
   - `Completed / Historical`: `run_a_anchor`, Seed `8101`, `ln Z = 201646.1413 ± 0.2374 nats`, `11,951` samples, `487,407` guard-reconciled calls.
3. **Stage 6D & Ledger (`experiments/wp6_stage6c_anchor/stage6c_aggregate_ledger.json`):**
   - `Stage 6D`: `Completed / Historical` Pathway A closeout (`487,407` calls used, `712,593` remaining).
   - `Stage 6C Run B`: `Unexecuted` (`run_b_gate.authorized = true` in historical ledger, never executed).
   - `Seed 8301`: `Proposed` diagnostic only (not executed).
   - `Model 3 (Student-t)`: `Frozen / Specified` (not approved for production inference).

---

## 3. Tests, Checks, and Results

- **Backend Suite (`pytest tests/test_gwlab_experiments.py tests/test_gwlab_inference.py tests/test_gwlab_api.py tests/test_gwlab_data.py tests/test_gwlab_noise_psd.py tests/test_gwlab_matched_filter.py tests/test_gwlab_physics.py -q`):** **37 passed** in 20.33s (`Tested`).
- **TypeScript (`npx tsc --noEmit`):** **PASS** (0 errors) (`Tested`).
- **ESLint (`npx eslint src/lib/api.ts src/app/experiments/page.tsx`):** **PASS** (0 errors, 0 warnings) (`Tested`).
- **Next.js Production Build (`npm run build`):** **PASS** (Next.js 16.4.6 Turbopack; all 10 routes prerendered) (`Tested`).

---

## 4. Browser Validation

Verified live on `http://localhost:3000/experiments` at `1440×900` (desktop) and `390×844` (mobile) (`Validated`):
- `GET http://localhost:8000/api/experiments/explorer` returned `200 OK` on both viewports.
- Confirmed top 3 readouts, Stage 5G.2 comparison table, two-seed sensitivity qualification, all 6 campaign stages with status badges, and interactive filter tabs (`All`, `Completed`, `Frozen / Proposed / Unexecuted`).
- **0 console errors**, **0 HTTP `>= 400` failed requests**, **0 page-level horizontal overflow** (`1424.8px` desktop, `375.2px` mobile), and **0 clipped controls or text**.

---

## 5. Known Limitations or Unavailable Data

- `Stage 6C Run B` has no posterior or evidence files because it was never executed.
- `Seed 8301` has no run artifacts because it remains a proposed diagnostic only.
- `formatFloat` trims trailing zeros (`2.972 nats` on the top readout card vs `2.9720 nats` in the qualification text and timeline summary).

---

## 6. Closeout Status

**Accepted and Closed.** Phase 4F passed report-based review and is formally closed with all requirements implemented, tested, and validated, and with zero sampler runs or scientific artifact modifications.
