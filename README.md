# GWLab — Gravitational-Wave Matched Filtering & Bayesian Inference Laboratory

**Author:** SOUMITRA PHARIKAL *(Independent Scientific Computing Project)*

**GWLab** is a scientific web laboratory for exploring frequency-domain compact binary coalescence (CBC) waveforms, synthetic colored detector noise and Welch PSD estimation, matched filtering, and saved Bayesian nested-sampling parameter-estimation results.

The application runs on a **Next.js `16.4.6` frontend** (`frontend/`) backed by a **FastAPI service** (`src/gwlab/api/`) over the deterministic **`gw_inference` scientific engine** (`src/gw_inference/`).

> **Important Note on Historical Docs:** The original Streamlit-era architecture in [GWLab_Phase0_Architecture_and_Repository_Audit.md](reports/GWLab_Phase0_Architecture_and_Repository_Audit.md) was superseded by the Next.js + FastAPI migration in [GWLab_Phase4A_Frontend_Migration_Report.md](reports/GWLab_Phase4A_Frontend_Migration_Report.md). Historical reports under `reports/` are preserved as point-in-time records.

---

## Documentation

- **[User Manual](docs/USER_MANUAL.md)** — Installation, environment configuration, local/Docker startup, route-by-route usage guide, and troubleshooting.
- **[Architecture Documentation](docs/ARCHITECTURE.md)** — Frontend/backend boundaries, FastAPI endpoints, `gw_inference` scientific engine, artifact provenance, and testing workflow.
- **Release & Validation Reports:**
  - [Phase 6A System Validation Report](reports/GWLab_Phase6A_System_Validation_Report.md)
  - [Phase 6B Deployment Packaging Report](reports/GWLab_Phase6B_Deployment_Packaging_Report.md)
  - [Phase 6C Final Documentation & Release Readiness Report](reports/GWLab_Phase6C_Final_Documentation_and_Release_Readiness_Report.md)

---

## Quick Start (Local)

### Prerequisites
- Python `>= 3.10` (tested on Python `3.13.5`)
- Node.js `20+` and `npm`

### 1. Install dependencies & build frontend
```bash
pip install -e ".[api]"
cd frontend && npm ci && npm run build && cd ..
```

### 2. Verify environment & launch
```bash
python scripts/launch_gwlab.py --check-only
python scripts/launch_gwlab.py
```
On Windows PowerShell:
```powershell
.\scripts\launch_gwlab.ps1
```
- **Frontend:** `http://localhost:3000`
- **FastAPI Backend:** `http://127.0.0.1:8000` (OpenAPI docs at `http://127.0.0.1:8000/docs`)

### Docker Compose (Configured, Not Locally Executed)
```bash
docker compose up --build
```
*(Dockerfiles and `docker-compose.yml` are configured and reviewed, but Docker CLI was not available in the local validation environment.)*

---

## Scientific Scope & Limitations

- **Analytical PSD Proxy:** The default `aligo_representative` model is a custom, regularized analytical aLIGO-like PSD proxy for controlled numerical benchmarking; it is **not** an official LIGO design-sensitivity curve or measured H1 spectrum.
- **Synthetic Data:** `/noise-and-psd` and `/matched-filtering` evaluate deterministic synthetic realizations (colored Gaussian noise + TaylorF2 3.5PN injection) rather than live or open-data strain.
- **Likelihood & Band:** Benchmarks and interactive evaluations use Model 2 (Whittle Gaussian likelihood) over the active frequency band `[20, 512] Hz` (`4 s` duration, `2048 Hz` sampling rate).
- **Read-Only Posterior & Experiment Archives:** `/inference` displays saved weighted posterior samples from single-seed **Stage 6C Run A** (`seed 8101`). `/experiments` compares the saved **Stage 5G.2** two-seed sensitivity pair (`base_01_plug_inst_s1` vs `base_01_plug_inst_s2`), which is a sensitivity diagnostic (`|\Delta \ln Z| = 2.9720`, ratio `8.92`), **not** a convergence or frequentist coverage proof.
- **Unexecuted / Frozen Campaigns:** `Stage 6C Run B` and `Seed 8301` remain unexecuted/proposed; `Model 3` (Student-t) remains frozen/specified and is not approved as production inference.
- **No Live Sampling:** The web interface never launches nested samplers or recomputes Bayesian evidence.

---

## External Data Validation (GW150914)

While the `517/517` repository test suite and interactive web application operate exclusively on synthetic data and saved benchmark archives, GWLab's conditioning, off-source Welch PSD, and `gw_inference.filter.matched_filter` pipeline were separately validated against **public GWOSC detector strain** for **GW150914** (`GWTC-1-confident`, GPS `1126259462.4`, 32 s `4096 Hz` H1 and L1 HDF5 files):

- **Reproducible script:**
  ```bash
  python scripts/validate_gw150914_gwosc.py
  ```
- **Method:** DC removal, `15 Hz` zero-phase Butterworth high-pass, polyphase anti-alias downsampling (`4096 -> 2048 Hz`), off-source Welch PSD excluding the event window, and a **fixed, non-optimized inspiral template applied to a known event** (TaylorF2 $m_1 = 35\,M_\odot$, $m_2 = 30\,M_\odot$, $f_{\mathrm{ISCO}} = 67.65\,\mathrm{Hz}$) over `[20, 512] Hz`. Full Bayesian parameter-estimation or population generalization validation on real detector data has not been performed.
- **Observed Results:** `H1` peak SNR `8.94` at `+0.0718 s` (`GPS 1126259462.4718`) and `L1` peak SNR `5.55` at `+0.0610 s` (`GPS 1126259462.4610`). Because TaylorF2 terminates at $f_{\mathrm{ISCO}}$ and omits merger/ringdown, it recovers only the inspiral fraction of GW150914's full IMR signal power.
- **Data Acknowledgement:** This validation uses public data from the Gravitational Wave Open Science Center ([gwosc.org](https://gwosc.org)), a service of LIGO Laboratory, the LIGO Scientific Collaboration, and the Virgo Collaboration.
