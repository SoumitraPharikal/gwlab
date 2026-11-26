# GWLab User Manual

## 1. Scientific Scope
GWLab is an interactive research workbench for gravitational-wave data analysis under controlled synthetic benchmarks:
- **Deterministic Signal & Filter Evaluation:** TaylorF2 (`3.5PN` phase, Newtonian amplitude) waveforms, synthetic colored Gaussian noise + Welch PSD estimation, and frequency-domain matched filtering.
- **Read-Only Posterior & Campaign Archives:** Interactive inspection of saved nested-sampling posteriors (Stage 6C Run A) and multi-seed evidence sensitivity comparisons (Stage 5G.2).
- **Strict Execution Boundary:** The web application never launches `dynesty` or recomputes Bayesian evidence.

---

## 2. Requirements & Installation

- **Python:** `>= 3.10`
- **Node.js:** `20+` with `npm`
- **Python packages:**
  ```bash
  pip install -e ".[api]"
  ```
  *(Optional for full scientific-engine test suite: `pip install -e ".[api,test,sampling]"`)*
- **Frontend packages:**
  ```bash
  cd frontend
  npm ci
  ```

---

## 3. Startup Modes

### A. Production Standalone Startup via Local Launcher (Recommended)
1. Build the Next.js standalone bundle once:
   ```bash
   cd frontend && npm run build && cd ..
   ```
2. Run preflight checks (dependencies, read-only artifacts, `.next/` build, and free ports):
   ```bash
   python scripts/launch_gwlab.py --check-only
   ```
3. Start both FastAPI (`127.0.0.1:8000`) and Next.js (`localhost:3000`):
   ```bash
   python scripts/launch_gwlab.py
   ```
   Or on Windows PowerShell:
   ```powershell
   .\scripts\launch_gwlab.ps1
   ```

### B. Local Development Startup
Pass `--dev` to the launcher to run Next.js in hot-reload mode (`npx next dev`):
```bash
python scripts/launch_gwlab.py --dev
```
Or start each service manually in separate terminals:
```bash
# Terminal 1 — FastAPI backend
python -m uvicorn gwlab.api.app:app --host 127.0.0.1 --port 8000

# Terminal 2 — Next.js dev frontend
cd frontend
npm run dev -- -p 3000
```

### C. Docker Compose Startup *(Configured, Not Yet Locally Executed)*
`Dockerfile.backend`, `Dockerfile.frontend`, and `docker-compose.yml` are configured for multi-container production deployment:
```bash
docker compose up --build
```
- Mounts only the required read-only artifact directories (`./experiments/wp6_stage6c_anchor` and `./reports/demonstrations/wp6_stage5/stage_g2`) into `/data` (`:ro`) with `GWLAB_DATA_ROOT=/data`.
- **Note:** Docker CLI was unavailable in the local validation environment, so `docker compose up --build` has not yet been executed locally.

---

## 4. Environment Variables & Artifact Configuration

See [`.env.example`](../.env.example) for the complete template:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `GWLAB_ENV` | `development` | Runtime environment mode (`development` or `production`) |
| `GWLAB_API_HOST` | `127.0.0.1` | FastAPI bind host (`0.0.0.0` inside Docker) |
| `GWLAB_API_PORT` | `8000` | FastAPI bind port (`1–65535`) |
| `GWLAB_DATA_ROOT` | Repository root | Root directory containing read-only `experiments/` and `reports/` artifacts |
| `GWLAB_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000,http://localhost:8502,http://127.0.0.1:8502` | Comma-separated allowed CORS origins for FastAPI |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI URL used by the browser (inlined at `next build` time) |
| `GWLAB_FRONTEND_PORT` | `3000` | Frontend port used by the local launcher / Docker |

---

## 5. Route-by-Route Usage Guide

- **Overview (`/` and `/overview`):** Mission summary, Stage 6C Run A anchor benchmark ($\ln Z = 201646.1413 \pm 0.2374\,\text{nats}$), Stage 5G.2 two-seed repeatability card (`8.92` ratio with qualification), historical Stage 6D computational ledger, and provenance matrix.
- **Waveforms (`/waveforms`):** Interactive TaylorF2 (`3.5PN`) waveform generator with sliders for $\mathcal{M}_c$, $\eta$, $d_L$, $t_c$, $\phi_c$, binary mechanics readouts ($m_1, m_2, M, q, f_{\text{ISCO}}$), and tabs for Strain Amplitude, Fourier Phase, Chirp Time, and Quadratures.
- **Noise & PSD (`/noise-and-psd`):** Compares the analytical PSD/ASD proxy (`aligo_representative` or `flat`) against deterministic synthetic colored Gaussian noise and Tukey-windowed Welch PSD estimates, with overlap/inflation diagnostics ($\kappa$, $\nu_{\text{eff}}$).
- **Matched Filtering (`/matched-filtering`):** Evaluates complex matched-filter SNR $\rho(t)$ on synthetic injection + colored noise, allowing chirp-mass template mismatch ($\Delta \mathcal{M}_c$), frequency-band adjustment, and spectrum vs time-domain inspection.
- **Inference (`/inference`):** Interactive explorer for saved Stage 6C Run A weighted posterior samples (`11,951` samples), displaying 1D marginal densities, 2D joint credible contours, and the upper-1% $\eta$-prior-range boundary mass (`38.57%`).
- **Experiments (`/experiments`):** Read-only comparison of the Stage 5G.2 two-seed pair (`base_01_plug_inst_s1` vs `base_01_plug_inst_s2`) and filterable campaign timeline distinguishing completed, unexecuted (`Run B`), proposed (`Seed 8301`), and frozen (`Model 3`) stages.
- **Methods (`/methods`):** Frozen mathematical conventions, Whittle likelihood formulation, prior bounds, and scientific limitations.

---

## 6. Troubleshooting

- **`Port 8000 is already in use` / `Port 3000 is already in use`:** Another server is already running. Stop the existing process or pass custom ports: `python scripts/launch_gwlab.py --api-port 8001 --frontend-port 3001`.
- **`Artifact not found` (`404` on `/api/inference/posterior` or `/api/experiments/explorer`):** Verify `GWLAB_DATA_ROOT` points to the directory containing `experiments/wp6_stage6c_anchor/` and `reports/demonstrations/wp6_stage5/stage_g2/`.
- **Frontend calls wrong API host after changing `NEXT_PUBLIC_API_URL`:** `NEXT_PUBLIC_API_URL` is inlined by Next.js during `npm run build`. Rebuild the frontend after changing it.
- **Browser CORS error:** Add the frontend origin to `GWLAB_CORS_ORIGINS` (comma-separated) and restart FastAPI.
