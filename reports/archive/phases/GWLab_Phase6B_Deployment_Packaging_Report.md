# GWLab Phase 6B — Deployment Packaging Report

**Date:** 2026-11-26  
**Architecture:** Next.js `16.4.6` (`output: "standalone"`) + FastAPI (`uvicorn`)  
**Status:** **PASS (Non-Docker Verified; Docker/Compose Config Ready, Docker CLI Unavailable in Host Environment)**

---

## 1. Implemented

- **Centralized Runtime Configuration & Artifact Root (`src/gwlab/config.py`, `src/gwlab/api/service.py`, `src/gwlab/api/app.py`, `.env.example`):**
  - Added `get_data_root()` and `get_api_settings()` in `src/gwlab/config.py` supporting `GWLAB_ENV`, `GWLAB_API_HOST`, `GWLAB_API_PORT`, `GWLAB_DATA_ROOT`, and `GWLAB_CORS_ORIGINS` with safe local defaults.
  - Updated `resolve_safe_path()` in `src/gwlab/api/service.py` to resolve read-only scientific artifacts relative to `get_data_root()` while preserving path-traversal and absolute-path rejection.
  - Updated `CORSMiddleware` in `src/gwlab/api/app.py` to use `get_api_settings().cors_origins`.
  - Added `api` optional dependency group (`fastapi`, `uvicorn[standard]`, `pydantic`, `scipy`) to `pyproject.toml`.
  - Enabled `output: "standalone"` in `frontend/next.config.ts` for minimal multi-stage production container packaging.
- **Docker & Compose Packaging (`Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`, `.dockerignore`):**
  - `Dockerfile.backend`: `python:3.13-slim`, installs `.[api]`, runs non-root `uvicorn gwlab.api.app:app --host 0.0.0.0 --port 8000` with `HEALTHCHECK` on `/api/status`.
  - `Dockerfile.frontend`: multi-stage `node:20-alpine` build + minimal non-root `node server.js` runner on port `3000` with `HEALTHCHECK`.
  - `docker-compose.yml`: orchestrates `backend` and `frontend` (`depends_on: backend: condition: service_healthy`), mounting only the required read-only scientific artifact directories (`./experiments/wp6_stage6c_anchor` and `./reports/demonstrations/wp6_stage5/stage_g2`) into `/data` (`:ro`) with `GWLAB_DATA_ROOT=/data`.
- **Cross-Platform & Windows Local Launcher (`scripts/launch_gwlab.py`, `scripts/launch_gwlab.ps1`):**
  - Validates Python dependencies, Node/npm, required read-only artifacts, `.next/` production build (or `--dev`), and port availability (`8000`, `3000`) before starting `uvicorn` and `next start`.
  - Fails fast on missing dependencies/artifacts or already-bound ports (prevents silent duplicate servers) and never triggers sampler execution.
- **Configuration Tests (`tests/test_gwlab_config.py`):**
  - Added `3` tests verifying default settings, `GWLAB_DATA_ROOT` / `GWLAB_CORS_ORIGINS` overrides, path-traversal protection, and invalid `GWLAB_API_PORT` rejection.

---

## 2. Configuration & Startup Instructions

Copy `.env.example` to `.env` if customizing ports, origins, or artifact root:
- `GWLAB_ENV` (`production` / `development`)
- `GWLAB_API_HOST` (default `127.0.0.1` locally, `0.0.0.0` in Docker)
- `GWLAB_API_PORT` (default `8000`)
- `GWLAB_DATA_ROOT` (defaults to repository root locally; `/data` in Docker)
- `GWLAB_CORS_ORIGINS` (comma-separated allowed browser origins)
- `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`; inlined at `next build` time)
- `GWLAB_FRONTEND_PORT` (default `3000`)

**Local Production Startup:**
1. Build frontend once: `cd frontend && npm run build`
2. Preflight check: `python scripts/launch_gwlab.py --check-only`
3. Launch (Windows PowerShell): `.\scripts\launch_gwlab.ps1` (or cross-platform: `python scripts/launch_gwlab.py`)

**Docker Compose Startup (on a host with Docker):**
```bash
docker compose up --build
```

---

## 3. Docker / Compose Status

- **Files Created:** `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`, `.dockerignore`, `.env.example`.
- **Execution Status:** **Not executed in this environment** because `docker` / `docker compose` is not installed on the host (`docker: The term 'docker' is not recognized`). Container build and `docker compose up` must be verified on a Docker-enabled host.

---

## 4. Launcher Status

- **Preflight & Duplicate-Port Guard Validated:**
  - `python scripts/launch_gwlab.py --check-only --api-port 8099 --frontend-port 3099` → **Passed** (`GWLab preflight passed: dependencies, artifacts, build, and ports are ready.`).
  - `python scripts/launch_gwlab.py --check-only` (with FastAPI already bound on `8000`) → **Correctly rejected** (`Port 8000 is already in use (cannot start duplicate FastAPI server).`).

---

## 5. Offline-Font Status

- **Runtime (Offline):** Verified `frontend/.next/static/media/` contains `50` bundled `.woff2` files (Next.js self-hosted `Newsreader`, `IBM_Plex_Sans`, `IBM_Plex_Mono` + KaTeX fonts). Production runtime makes **zero external font requests**.
- **Build Time:** `frontend/src/app/layout.tsx` uses `next/font/google`, which downloads Google Fonts during `npm run build` / `docker build`. A completely air-gapped *build* environment without cached `.next` or pre-bundled local `.woff2` assets would require switching to `next/font/local` with verified local font files.

---

## 6. Tests & Smoke Checks Executed

- **GWLab Backend & Config Tests (`tests/test_gwlab_*.py`):** **`62 passed, 0 failed`** in `9.92s` (includes `3` new tests in `tests/test_gwlab_config.py`).
- **Frontend TypeScript / ESLint / Production Build (`frontend/`):**
  - `npx tsc --noEmit`: `0` errors.
  - `npm run lint`: `0` errors (`7` pre-existing unused-var warnings).
  - `npm run build`: succeeded (`output: "standalone"` verified via `frontend/.next/standalone/server.js`).
- **Live HTTP Smoke Check (`http://127.0.0.1:8000` & `http://127.0.0.1:3000`):**
  - `/api/status`, `/`, `/waveforms`, `/noise-and-psd`, `/matched-filtering`, `/inference`, `/experiments`, `/methods` all returned HTTP `200`.

---

## 7. Known Limitations

1. Docker/Compose build and runtime health checks were not executed locally because Docker is not installed on this machine.
2. `NEXT_PUBLIC_API_URL` is inlined by Next.js at build time (`Dockerfile.frontend` passes it as a build `ARG`).
3. `next/font/google` requires internet access at build time (`next build`), though production serving is fully self-hosted.

---

## 8. Phase 6C Readiness

**Ready for Phase 6C (Final Documentation & Release Preparation)** once you review Phase 6B and authorize final documentation.
