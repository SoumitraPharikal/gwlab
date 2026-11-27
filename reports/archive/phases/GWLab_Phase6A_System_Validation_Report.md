# GWLab Phase 6A — System Validation Report (Next.js + FastAPI Stack)

**Date:** 2026-11-26  
**Architecture:** Next.js `16.4.6` Frontend + FastAPI Backend (`src/gwlab/api/`)  
**Status:** **PASS — Ready for Phase 6B** (with 1 pre-existing scientific-engine mock test failure documented below)

---

## 1. Implemented

- **Frontend/Backend Contract & Error-Handling Integration Suite (`tests/test_gwlab_integration.py`):**
  - Added `11` focused integration tests validating exact JSON schemas against `frontend/src/lib/api.ts` for all `8` FastAPI endpoints (`/api/status`, `/api/benchmarks/overview`, `/api/waveforms/generate`, `/api/waveforms/chirp-time`, `/api/noise/analyze`, `/api/matched-filter/evaluate`, `/api/inference/posterior`, `/api/experiments/explorer`).
  - Added scientific invariant assertions: `asd_analytic == sqrt(psd_analytic)` for both `aligo_representative` and `flat` PSD models, matched vs mismatched chirp-mass SNR ordering (`44.28` vs lower mismatched peak under identical PSD), posterior weight unity normalization, and `p5 <= median <= p95` quantile ordering.
  - Added HTTP `400`, `404`, `422`, and `500` error-handling tests covering FastAPI route handlers and `service.py` input validation.
- **Playwright E2E Regression Suite (`frontend/scripts/verify_phase6a.mjs` & `frontend/package.json` `"test:e2e"`):**
  - Extended the existing Playwright setup to test all `7` routes (`/`, `/waveforms`, `/noise-and-psd`, `/matched-filtering`, `/inference`, `/experiments`, `/methods`) across `3` viewports (`1440×900` desktop, `768×1024` tablet, `390×844` mobile), verifying zero console/page errors, zero failed `/api/` requests, zero horizontal overflow, required scientific labels/units, and live interactive controls.

---

## 2. Tested

- **Full Repository Baseline (`python -m pytest -q`):**
  - Before Phase 6A additions: `505 passed, 1 failed` in `109.14s` (`tests/test_wp6_sampler_foundation.py::TestNestedSamplerAdapter::test_graceful_handling_when_dynesty_unavailable` failed because `dynesty` is installed and `adapter.run()` executed instead of raising `ImportError`).
- **GWLab Suite + Coverage (`uv run --with coverage --with pytest` over `tests/test_gwlab_*.py`):**
  - `59 passed, 0 failed` in `18.63s` (including all `11` new tests in `tests/test_gwlab_integration.py`).
- **Frontend TypeScript & Lint (`frontend/`):**
  - `npx tsc --noEmit`: passed with `0` errors.
  - `npm run lint`: passed with `0` errors (`7` pre-existing `@typescript-eslint/no-unused-vars` warnings in `page.tsx`, `waveforms/page.tsx`, and `WaveformPlot.tsx`).

---

## 3. Validated (Playwright Route × Viewport Matrix)

Executed live via `npm run test:e2e` against `http://localhost:3000` and `http://127.0.0.1:8000`:
- **Matrix:** `7` routes × `3` viewports (`1440×900`, `768×1024`, `390×844`) = `21` checks.
- **Result:** **21/21 passed** (`0` console errors, `0` page errors, `0` failed API requests, `0` horizontal overflows, `0` missing scientific labels).
- **Interactive Controls Verified at `1440×900`:**
  - `/waveforms`: tab switching (`Fourier Phase`, `Chirp Time`, `Quadratures`).
  - `/noise-and-psd`: `Flat` PSD model toggle.
  - `/matched-filtering`: `Spectrum` view toggle.
  - `/inference`: `param_y` selector update (`tc`).
  - `/experiments`: `Completed` and `Frozen / Proposed / Unexecuted` campaign filter tabs.

---

## 4. Actual Coverage & Target Assessment (`src/gwlab/`)

Measured via `coverage.py` across `tests/test_gwlab_*.py`:
- **Total (`src/gwlab/` non-Streamlit-page modules):** **`86%` (`788 / 911` statements covered, `123` missed)** — **meets the `>85%` target**.
- **Key Modules:**
  - `src/gwlab/api/app.py`: **`100%`** (`95/95`)
  - `src/gwlab/api/service.py`: **`86%`** (`323/375`)
  - `src/gwlab/data/schemas.py`: **`99%`** (`86/87`)
  - `src/gwlab/physics.py`: **`99%`** (`68/69`)
  - `src/gwlab/data/provenance.py`: **`92%`** (`23/25`)
  - `src/gwlab/data/loaders.py`: **`89%`** (`109/122`)
  - `src/gwlab/format.py`: **`83%`** (`45/54`)
  - `src/gwlab/config.py`: **`100%`** (`13/13`)
- **Weak / Legacy Modules:**
  - Legacy Streamlit UI files remain low/unimported (`src/gwlab/app.py`: `0%`, `src/gwlab/components.py`: `29%`, `src/gwlab/theme.py`: `42%`, and `src/gwlab/pages/*.py` unused by Next.js + FastAPI).

---

## 5. Remaining Failures or Limitations

1. **Pre-existing Scientific Engine Test Failure (Outside Phase 6A Scope):**
   - `tests/test_wp6_sampler_foundation.py::TestNestedSamplerAdapter::test_graceful_handling_when_dynesty_unavailable` failed in the full `506`-test run (`DID NOT RAISE ImportError` when `dynesty` is installed in the environment). Left untouched to respect frozen `src/gw_inference/` boundaries.
2. **ESLint Warnings:** `7` unused-variable warnings remain in `frontend/src/app/page.tsx`, `frontend/src/app/waveforms/page.tsx`, and `frontend/src/components/WaveformPlot.tsx`.

---

## 6. Phase 6B Readiness

**Ready for Phase 6B (Deployment Packaging, Air-Gapped Font Bundling & Documentation).** End-to-end backend/API contracts, `>85%` `src/gwlab/` coverage (`86%`), TypeScript/ESLint checks, and 21-cell Playwright regression are all verified and green.
