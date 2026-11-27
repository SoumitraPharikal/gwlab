# GWLab Architecture Documentation

> **Historical Note:** [GWLab_Phase0_Architecture_and_Repository_Audit.md](../reports/GWLab_Phase0_Architecture_and_Repository_Audit.md) describes the original Streamlit multipage architecture. That frontend was replaced in **Phase 4A** by the Next.js + FastAPI architecture documented below. Legacy Streamlit modules (`src/gwlab/app.py`, `src/gwlab/pages/*.py`) remain in the repository for reference only and are not part of the production Next.js + FastAPI runtime.

---

## 1. High-Level System Architecture

```text
+-----------------------------------+       HTTP/JSON        +----------------------------------+
|     Next.js 16.4.6 Frontend       | <--------------------> |        FastAPI Backend           |
|  (frontend/src/app/, lib/api.ts)  |   NEXT_PUBLIC_API_URL  |  (src/gwlab/api/app.py,          |
|  - React 19 + Tailwind + KaTeX    |                        |   service.py, config.py)         |
|  - output: "standalone"           |                        +----------------+-----------------+
+-----------------------------------+                                         |
                                                                              | calls
                                                             +----------------v-----------------+
                                                             |   gw_inference Scientific Core   |
                                                             |   (src/gw_inference/)            |
                                                             |   - waveform, psd, noise,        |
                                                             |     filter, grid, conventions    |
                                                             +----------------+-----------------+
                                                                              |
                                                             +----------------v-----------------+
                                                             |   Read-Only Saved Artifacts      |
                                                             |   (GWLAB_DATA_ROOT)              |
                                                             |   - experiments/wp6_stage6c_*    |
                                                             |   - reports/demonstrations/...   |
                                                             +----------------------------------+
```

---

## 2. Frontend Layer (`frontend/`)
- **Framework:** Next.js `16.4.6` (App Router, Turbopack, `output: "standalone"` in `frontend/next.config.ts`), React `19.3.0`, TypeScript, Tailwind CSS, KaTeX.
- **Routes:**
  - `/` & `/overview` (`src/app/page.tsx`, `src/app/overview/page.tsx`)
  - `/waveforms` (`src/app/waveforms/page.tsx`)
  - `/noise-and-psd` (`src/app/noise-and-psd/page.tsx`)
  - `/matched-filtering` (`src/app/matched-filtering/page.tsx`)
  - `/inference` (`src/app/inference/page.tsx`)
  - `/experiments` (`src/app/experiments/page.tsx`)
  - `/methods` (`src/app/methods/page.tsx`)
- **Typed API Client (`frontend/src/lib/api.ts`):** Centralizes all fetch calls and TypeScript interfaces using `process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"`.
- **Fonts (`frontend/src/app/layout.tsx`):** Uses `next/font/google` (`Newsreader`, `IBM_Plex_Sans`, `IBM_Plex_Mono`). Fonts are downloaded at build time (`next build`) and bundled into `.next/static/media/*.woff2` for self-hosted runtime serving.

---

## 3. FastAPI Backend & Service Layer (`src/gwlab/api/`)
- **API Endpoints (`src/gwlab/api/app.py`):**
  - `GET /api/status` — Health check and frozen detector/band metadata.
  - `GET /api/benchmarks/overview` — Stage 6C anchor, Stage 5G.2 seed comparison, Stage 6D ledger, provenance matrix.
  - `POST /api/waveforms/generate` — Deterministic TaylorF2 `3.5PN` strain, phase, and binary mechanics.
  - `POST /api/waveforms/chirp-time` — Newtonian and `3.5PN` chirp-time trajectories.
  - `POST /api/noise/analyze` — Analytical PSD/ASD proxy (`aligo_representative` or `flat`), synthetic colored Gaussian noise, and Welch PSD/ASD.
  - `POST /api/matched-filter/evaluate` — Complex matched-filter SNR time series $\rho(t)$ on synthetic injection + noise.
  - `GET /api/inference/posterior` — Weighted 1D marginal and 2D joint histograms from Stage 6C Run A samples.
  - `GET /api/experiments/explorer` — Stage 5G.2 two-seed comparison and campaign execution status.
- **Configuration (`src/gwlab/config.py`):** `get_data_root()` and `get_api_settings()` resolve `GWLAB_DATA_ROOT`, `GWLAB_API_HOST`, `GWLAB_API_PORT`, `GWLAB_CORS_ORIGINS`, and `GWLAB_ENV` with safe local defaults.
- **Path Safety (`src/gwlab/api/service.py`):** `resolve_safe_path()` enforces relative paths within `get_data_root()`, blocking absolute paths and directory traversal, and returns `404`/`400` on missing/corrupt artifacts.

---

## 4. Scientific Engine Boundaries (`src/gw_inference/`)
- The presentation API reuses the frozen `gw_inference` modules (`constants.py`, `conventions.py`, `grid.py`, `psd.py`, `noise.py`, `waveform.py`, `filter.py`) without duplicating physics or altering likelihood/PSD normalizations.
- **No Sampler Execution:** `gw_inference.sampler` is never invoked by `src/gwlab/api/`.

---

## 5. Saved Artifact & Provenance Flow
All posterior and experiment-archive views read directly from frozen repository files under `GWLAB_DATA_ROOT`:
- `experiments/wp6_stage6c_anchor/run_a_anchor_result.json`
- `experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz`
- `experiments/wp6_stage6c_anchor/stage6c_aggregate_ledger.json`
- `reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s1.json`
- `reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s2.json`

---

## 6. Deployment & Validation Workflow
- **Local Launcher:** `scripts/launch_gwlab.py` and `scripts/launch_gwlab.ps1`.
- **Container Packaging:** `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`, `.dockerignore`, `.env.example`.
- **Testing:**
  - Full repository suite: `python -m pytest -q` (`517 passed`).
  - GWLab-specific suite: `tests/test_gwlab_*.py` (`62 passed`, `86%` coverage on `src/gwlab/`).
  - Frontend static & build checks: `npx tsc --noEmit`, `npm run lint`, `npm run build`.
  - Playwright E2E regression: `cd frontend && npm run test:e2e` (`21/21` route × viewport checks).
