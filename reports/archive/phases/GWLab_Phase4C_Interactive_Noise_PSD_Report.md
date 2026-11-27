# GWLab — Phase 4C: Interactive Noise & PSD Implementation Report

**Date:** 2026-11-25  
**Target Route:** `http://localhost:3000/noise-and-psd`  
**Backend Endpoint:** `POST http://localhost:8000/api/noise/analyze`  
**Status:** **PASS** (Implemented, Tested, and Validated across backend unit/API tests, TypeScript/ESLint/Next.js production build, and live desktop/mobile browser verification)

---

## 1. Files Inspected and Changed

### Inspected
- `src/gw_inference/psd.py` — Authoritative Welch PSD (`welch_psd`, `tukey_window`, `welch_variance_inflation`, `analytical_detector_psd`)
- `src/gw_inference/noise.py` — Authoritative continuous-Fourier-normalized noise synthesis (`generate_colored_noise`)
- `src/gw_inference/conventions.py` — Authoritative one-sided frequency grid (`build_frequency_grid`)
- `src/gwlab/api/service.py`, `src/gwlab/api/app.py`, `tests/test_gwlab_api.py` — Established FastAPI service patterns
- `frontend/package.json`, `frontend/src/lib/api.ts`, `frontend/src/app/waveforms/page.tsx`, `frontend/src/components/WaveformPlot.tsx` — Existing SVG plotting & workstation layout conventions

### Changed / Added
- **`src/gwlab/api/service.py`** — Added `compute_noise_and_psd_response(...)` wrapping `gw_inference` functions without modifying frozen scientific constants.
- **`src/gwlab/api/app.py`** — Added `NoiseAnalysisRequest` (Pydantic validation) and `POST /api/noise/analyze`.
- **`tests/test_gwlab_noise_psd.py`** — Added 4 dedicated backend unit & API tests covering Stage 5B defaults, PSD/ASD consistency, deterministic PRNG seeding, and invalid-input HTTP 400/422 handling.
- **`frontend/src/lib/api.ts`** — Added `NoiseAnalysisRequestParams`, `NoiseAnalysisResponseData`, and `analyzeNoiseAndPsd(...)`.
- **`frontend/src/components/NoisePsdPlot.tsx`** — Added interactive, responsive SVG plotting component (`ASD (√Sₙ)`, `PSD (Sₙ)`, and `Synthetic Noise n(t)` tabs, log-log axes, hover crosshair readout, loading/error states).
- **`frontend/src/app/noise-and-psd/page.tsx`** — Replaced placeholder with the interactive Noise & PSD workstation (PSD proxy selector, frequency-band sliders, deterministic PRNG seed control, Welch parameters, numerical readouts, and provenance/qualification panel).

---

## 2. API Endpoints and Response Behavior

- **Endpoint:** `POST /api/noise/analyze`
- **Request Schema (`NoiseAnalysisRequest`):**
  - `psd_model`: `"aligo_representative" | "flat"` (default `"aligo_representative"`)
  - `f_min_hz`: `1.0–200.0` Hz (default `20.0`)
  - `f_max_hz`: `50.0–1024.0` Hz (default `512.0`, validated `< sampling_rate_hz / 2`)
  - `sampling_rate_hz`: `512.0–4096.0` Hz (default `2048.0`)
  - `duration_obs_s`: `8.0–128.0` s (default `64.0`)
  - `segment_duration_s`: `1.0–16.0` s (default `4.0`)
  - `overlap_fraction`: `0.0–0.75` (default `0.5`)
  - `tukey_alpha`: `0.0–1.0` (default `0.25`)
  - `seed`: `0–2147483647` (default `7101`)
- **Response Behavior:**
  - Synthesizes deterministic colored Gaussian noise on the full `duration_obs_s * sampling_rate_hz` grid via `generate_colored_noise`.
  - Computes Tukey-windowed Welch PSD via `welch_psd`, analytical PSD proxy via `analytical_detector_psd`, and $\kappa_{\mathrm{overlap}}$ / $\nu_{\mathrm{eff}}$ via `welch_variance_inflation`.
  - Slices to active band `[f_min_hz, f_max_hz]` (excluding DC & Nyquist) and downsamples uniformly (`<= 600` spectral points, `<= 400` time-preview points) for responsive interactive plotting.
  - Returns HTTP `400` on invalid domain combinations (e.g., `f_min_hz >= f_max_hz`, `f_max_hz > Nyquist`) and HTTP `422` on schema violations.

---

## 3. Scientific Definitions and Provenance

- **PSD Provenance:** `aligo_representative` is explicitly labeled as an **Analytical aLIGO-like PSD proxy** — *"Custom, regularized representative noise model for controlled numerical benchmarking; not an official design-sensitivity curve or measured H1 spectrum."* Synthetic colored Gaussian noise realizations are kept strictly distinct from the analytical proxy.
- **Physical Units:** One-sided PSD $S_n(f)$ in `strain²/Hz`, ASD $\sqrt{S_n(f)}$ in `strain/√Hz`, frequency in `Hz`, synthetic strain $n(t)$ dimensionless.
- **Frozen Stage 5B Defaults Verified:**
  - Off-source duration: `64.0 s` (`131,072` samples at `2048 Hz`)
  - Segment length: `4.0 s` (`8,192` samples, $\Delta f = 0.25\,\mathrm{Hz}$), `50%` overlap $\implies K = 31$ segments
  - Tukey window $\alpha = 0.25$, active band `[20.0, 512.0] Hz` (`1,969` active bins)
  - $\kappa_{\mathrm{overlap}} = 1157/837 \approx 1.3823$, $\nu_{\mathrm{eff}} = 2K/\kappa_{\mathrm{overlap}} \approx 44.8522$
  - Explicitly qualified as a **variance-matched effective DOF approximation** for the Welch setup, not an exact predictive or Bayesian DOF.

---

## 4. Implementation Status & Validation Results

| Component / Check | Status | Details |
| :--- | :--- | :--- |
| **Backend Service & API (`service.py`, `app.py`)** | **Implemented & Tested** | `compute_noise_and_psd_response` & `POST /api/noise/analyze` |
| **Interactive Frontend (`NoisePsdPlot.tsx`, `noise-and-psd/page.tsx`)** | **Implemented & Validated** | Log-log ASD/PSD plots, `n(t)` preview, band/seed/Welch controls, provenance card |
| **Python Test Suite (`pytest`)** | **Tested (87 passed)** | `tests/test_gwlab_noise_psd.py`, `test_gwlab_api.py`, `test_gwlab_data.py`, `test_gwlab_physics.py`, `test_wp5_noise_psd.py`, `test_wp6_stage5_welch.py` — 87 passed in 7.68s |
| **TypeScript (`npx tsc --noEmit`)** | **Tested (PASS)** | 0 errors |
| **ESLint (`src/lib/api.ts`, `NoisePsdPlot.tsx`, `noise-and-psd/page.tsx`)** | **Tested (PASS)** | 0 errors, 0 warnings |
| **Next.js Production Build (`npm run build`)** | **Tested (PASS)** | Next.js 16.4.6 Turbopack; all 10 routes prerendered |
| **Live Browser (`1440×900` & `390×844`)** | **Validated (PASS)** | `POST /api/noise/analyze` `200 OK`, all 3 tabs verified (`ASD`, `PSD`, `Synthetic Noise n(t)`), seed `42` & `flat` model updates + `Reset` verified (`K=31`, `κ=1.3823`, `ν_eff=44.85`, `1969` bins), 0 console errors, 0 HTTP `>= 400` errors, 0 horizontal overflow/clipping |

---

## 5. Known Limitations & Next Recommended Step

- **Known Limitations:**
  - `sampling_rate_hz` (`2048 Hz`) and `duration_obs_s` (`64.0 s`) are fixed to the frozen Stage 5B baseline in the UI controls to keep API latency instantaneous (~5 ms).
  - Rapidly dragging sliders can cancel superseded in-flight requests (`status: 0` abort) while the latest request resolves cleanly (`200 OK`).
- **Recommended Next Step:** **Phase 4D** — Interactive Matched Filtering implementation (`/matched-filtering`), exposing complex SNR time series $\rho(t)$, phase maximization, and binned Allen $\chi^2$ diagnostics backed by `src/gw_inference/filter.py`.
