# GWLab — Phase 4D: Interactive Matched Filtering Implementation Report

**Date:** 2026-11-26  
**Target Route:** `http://localhost:3000/matched-filtering`  
**Backend Endpoint:** `POST http://localhost:8000/api/matched-filter/evaluate`  
**Status:** **PASS** (Implemented, Tested, and Validated across backend unit/API tests, TypeScript/ESLint/Next.js production build, and live desktop/mobile browser verification)

---

## 1. Files Inspected and Changed

### Inspected
- `src/gw_inference/filter.py` — Authoritative WP3 matched-filter engine (`matched_filter`, `matched_filter_core`, `compute_template_norm`, `noise_weighted_inner_product`, `compute_signed_lags`, `MatchedFilterResult`)
- `src/gw_inference/waveform.py` & `src/gw_inference/grid.py` — Frozen TaylorF2 waveform, `apply_coalescence_time_shift`, and `FrequencyGrid` active mask $K_+$
- `src/gw_inference/psd.py` & `src/gw_inference/noise.py` — `analytical_detector_psd`, `PSDEngine`, `generate_colored_noise`
- `scripts/run_wp6_stage5_stage_g1.py` — Canonical Realization 01 injection & noise configuration (`CANONICAL_MC = 28.095555795460438`, `CANONICAL_ETA = 0.24710059171597634`, `CANONICAL_TC = 2.0`, `CANONICAL_DIST_MPC = 410.0`, `seed = 7151`)
- `src/gwlab/api/service.py`, `src/gwlab/api/app.py`, `frontend/src/lib/api.ts`, `frontend/src/app/matched-filtering/page.tsx`

### Changed / Added
- **`src/gwlab/api/service.py`** — Added `compute_matched_filter_response(...)` reusing `FrequencyGrid`, `analytical_detector_psd`, `PSDEngine`, `taylorf2_fd_waveform`, `apply_coalescence_time_shift`, `generate_colored_noise`, and `matched_filter`.
- **`src/gwlab/api/app.py`** — Added `MatchedFilterRequest` (Pydantic validation) and `POST /api/matched-filter/evaluate`.
- **`tests/test_gwlab_matched_filter.py`** — Added 3 tests verifying $\rho_{\mathrm{opt}} = \sigma_h = 45.31529969318099$, deterministic seeding, template-mismatch SNR degradation, active-band endpoints, and HTTP 400/422 validation.
- **`frontend/src/lib/api.ts`** — Added `MatchedFilterRequestParams`, `MatchedFilterResponseData`, and `evaluateMatchedFilter(...)`.
- **`frontend/src/components/MatchedFilterPlot.tsx`** — Added responsive SVG plotting component (`SNR Time Series |ρ(t)|`, `Zoomed Merger Peak` with $|\rho(t)|$ and $\mathrm{Re}[\rho(t)]$, and `Data & Template |d~(f)|` log-log spectrum in `strain/Hz`) with hover crosshair readout and loading/error states.
- **`frontend/src/app/matched-filtering/page.tsx`** — Replaced placeholder with the interactive Matched Filtering workstation (injection controls, template mismatch $\Delta M_c$, active band & PSD proxy selector, deterministic PRNG seed control, distinct `Optimal SNR` and `Recovered Peak SNR` readouts, and provenance notes).

---

## 2. API Endpoint and Response Behavior

- **Endpoint:** `POST /api/matched-filter/evaluate`
- **Request Schema (`MatchedFilterRequest`):**
  - `mc_msun`: `10.0–50.0` $M_\odot$ (default `28.095555795460438`)
  - `eta`: `(0.05, 0.25]` (default `0.24710059171597634`)
  - `distance_mpc`: `50.0–2000.0` Mpc (default `410.0`)
  - `tc_s`: `0.2–3.8` s (default `2.0`)
  - `phic_rad`: `[-2π, 2π]` rad (default `0.0`)
  - `delta_mc_msun`: `-3.0–3.0` $M_\odot$ (default `0.0`)
  - `psd_model`: `"aligo_representative" | "flat"` (default `"aligo_representative"`)
  - `f_min_hz`: `5.0–200.0` Hz (default `20.0`), `f_max_hz`: `50.0–1024.0` Hz (default `512.0`)
  - `seed`: `0–2147483647` (default `7151`)
- **Response Behavior:**
  - Computes matched filter on the `4.0 s @ 2048 Hz` grid (`N = 8192`, `Δf = 0.25 Hz`) over active mask $K_+ \subset [f_{\min}, f_{\max}]$ excluding DC and Nyquist.
  - Uses peak-preserving bucket decimation (`400` buckets) for the full `4.0 s` SNR scan (`snr_time_s`, `snr_abs`, `snr_real`), full-resolution `±0.12 s` zoom (`zoom_time_s`, `zoom_abs`, `zoom_real`), and active-band Fourier amplitudes (`data_amp`, `injection_amp`, `template_amp` in `strain/Hz`).
  - Returns HTTP `400` on invalid domain combinations (`f_min_hz >= f_max_hz`) and HTTP `422` on schema violations.

---

## 3. Exact Matched-Filter Conventions & Provenance

- **Active-Band Inner Product:** $(h \mid h)_{K_+} = 4\,\mathrm{Re}\sum_{k \in K_+} \frac{|\tilde{h}_k|^2}{S_n(f_k)}\,\Delta f$ over $K_+ \subset [20.0, 512.0]\,\mathrm{Hz}$ (`1,969` active bins), excluding DC and Nyquist.
- **Optimal SNR vs Recovered Peak SNR:**
  - **Optimal SNR:** $\rho_{\mathrm{opt}} = \sigma_h = \sqrt{(h \mid h)_{K_+}}$ (noise-free template norm; `45.32` / `45.3153` for the canonical `28.0956 M_sun`, `410 Mpc` template in `aligo_representative`).
  - **Recovered Peak SNR:** $|\rho(t_{\mathrm{peak}})| = \max_{t \in \text{valid}} |z(t)| / \sigma_h$ evaluated on a specific synthetic noisy realization (`44.28` for `seed = 7151`, `44.88` for `seed = 42` under canonical defaults).
- **Provenance Qualification:** All displayed data are synthetic TaylorF2 injections combined with synthetic colored Gaussian noise under the analytical `aligo_representative` PSD proxy (not measured H1 detector telemetry). Archived Realization 01 optimal SNR (`45.3153`) is cited solely as a provenance-specific reference; displayed values are dynamically computed for the active configuration.

---

## 4. Implementation Status & Validation Results

| Component / Check | Status | Details |
| :--- | :--- | :--- |
| **Backend Service & API (`service.py`, `app.py`)** | **Implemented & Tested** | `compute_matched_filter_response` & `POST /api/matched-filter/evaluate` |
| **Interactive Frontend (`MatchedFilterPlot.tsx`, `matched-filtering/page.tsx`)** | **Implemented & Validated** | Full & zoomed SNR time series, `|d~(f)|` vs `|h~(f)|` spectrum, mismatch/band/seed controls, distinct readouts |
| **Python Test Suite (`pytest`)** | **Tested (45 passed)** | `tests/test_gwlab_matched_filter.py`, `test_wp3_filter.py`, `test_gwlab_noise_psd.py`, `test_gwlab_api.py`, `test_gwlab_data.py`, `test_gwlab_physics.py` — 45 passed in 12.60s |
| **TypeScript (`npx tsc --noEmit`)** | **Tested (PASS)** | 0 errors |
| **ESLint (`src/lib/api.ts`, `MatchedFilterPlot.tsx`, `matched-filtering/page.tsx`)** | **Tested (PASS)** | 0 errors, 0 warnings |
| **Next.js Production Build (`npm run build`)** | **Tested (PASS)** | Next.js 16.4.6 Turbopack; all 10 routes prerendered |
| **Live Browser (`1440×900` & `390×844`)** | **Validated (PASS)** | `POST /api/matched-filter/evaluate` `200 OK`; Optimal SNR `45.32` vs Recovered Peak SNR `44.28` (`seed=7151`) and `44.88` (`seed=42`) + `Reset` verified; all 3 plot tabs verified; 0 console errors, 0 HTTP `>= 400` failures, 0 horizontal overflow/clipping |

---

## 5. Known Limitations & Next Recommended Step

- **Known Limitations:**
  - `duration_s` (`4.0 s`), `sampling_rate_hz` (`2048 Hz`), and `compute_chisq=False` (Allen $\chi^2$ veto) are fixed in the UI to keep interactive evaluation latency minimal.
- **Recommended Next Step:** **Phase 4E** — Interactive Bayesian Inference posterior explorer (`/inference`) backed by saved Stage 6C Run A artifacts (`experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz`).
