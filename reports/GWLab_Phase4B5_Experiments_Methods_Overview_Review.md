# GWLab Phase 4B.5 — Experiments, Methods & Overview UI Review

**Date:** 2026-11-25  
**Target Routes:** `/experiments`, `/methods`, `/` (Overview Computational Ledger mobile header)  
**Status:** **PASS** (All scientific, provenance, ESLint, and mobile clipping issues resolved and verified)

---

## 1. Files Inspected and Changed

### Inspected
- `frontend/src/app/experiments/page.tsx` — Experiments registry placeholder
- `frontend/src/app/methods/page.tsx` — Methods & provenance documentation page
- `frontend/src/app/page.tsx` & `frontend/src/app/overview/page.tsx` — Overview dashboard & Computational Ledger card
- `frontend/src/components/ScientificPanel.tsx` — Shared scientific panel header/badge/subtitle layout
- `src/gw_inference/constants.py` & `src/gw_inference/waveform.py` — Frozen SI constants (`M_SUN_SI`, `C_SI`, `G_SI`, `MPC_SI`, `C_ISCO_SI = 4397.17`) and `apply_coalescence_time_shift` sign convention

### Changed
- **`frontend/src/components/ScientificPanel.tsx`** — Fixed mobile header wrapping/clipping (`min-w-0 max-w-full`, `flex-wrap` on title/badge row, `break-words [overflow-wrap:anywhere]` on subtitle).
- **`frontend/src/app/experiments/page.tsx`** — Distinguished completed historical campaigns (Stage 5G.2 2-seed pair, Stage 6C Run A, Stage 6D closeout) from unexecuted/frozen proposals (Stage 6C Run B, Seed 8301, Model 3 Student-$t$), removed unused `FolderGit2` import.
- **`frontend/src/app/methods/page.tsx`** — Corrected `C_ISCO` to `4397.17 M_sun·Hz` and `M_sun` to `1.9884099 × 10³⁰ kg`, fixed redundant heading (`Stationary-Phase Approximation Approximant` $\to$ `TaylorF2 Stationary-Phase Approximant`), added KaTeX equations, added the frozen PSD/Matched-Filtering/Model 2 Likelihood/4D Prior conventions panel, and removed unused `BookOpen` import.

---

## 2. Scientific & Provenance Findings and Exact Corrections

| Part | Finding / Defect | Exact Correction | Status |
| :--- | :--- | :--- | :--- |
| **Part A (`/experiments`)** | Lumped `Model 1 Zero Noise, Model 2 Whittle, Model 3 Student-t` together, called Stage 6C Run A "verified" without qualification, omitted Stage 5G.2 between-seed vs within-run distinction, and had unused `FolderGit2` import | Separated **Completed Historical Campaigns** (`base_01_plug_inst_s1`/`s2` between-seed sensitivity $|\Delta\ln\mathcal{Z}| = 2.9720\,\mathrm{nats}$ vs $\sigma_\Delta \approx 0.333\,\mathrm{nats}$, `run_a_anchor` `11,951` samples / `487,407` calls, Stage 6D ledger closeout) from **Unexecuted / Frozen Proposals** (Run B unexecuted, Seed 8301 proposed only, Model 3 Student-$t$ frozen/not approved for production); removed `FolderGit2`. | **Implemented & Validated** |
| **Part B (`/methods`)** | Listed `4397.0` instead of frozen `C_ISCO_SI = 4397.17`, redundant heading `"Stationary-Phase Approximation Approximant"`, omitted frozen PSD/Matched-Filtering/Likelihood/Prior conventions, and had unused `BookOpen` import | Corrected `C_ISCO = 4397.17 M_sun·Hz` and `M_sun = 1.9884099 × 10³⁰ kg`, fixed heading, added KaTeX formulas for TaylorF2 amplitude/ISCO/time-shift and matched filter/optimal SNR, documented frozen Model 2 Whittle likelihood & 4D uniform prior vs frozen Model 3 Student-$t$ and unexecuted Run B; removed `BookOpen`. | **Implemented & Validated** |
| **Part C (`/` Overview Mobile)** | On `390×844`, the Computational Ledger header (`362.54px`) overflowed the `341.60px` card by `36.9px`, clipping the `Budget ceiling: 1,200,000 calls` badge and subtitle | Added `min-w-0 max-w-full`, `flex-wrap` on the title+badge row, and `break-words [overflow-wrap:anywhere]` on the subtitle in `ScientificPanel.tsx`. Preserved `1440×900` inline desktop layout. | **Implemented & Validated** |

---

## 3. Validation Results

- **TypeScript (`npx tsc --noEmit`):** **PASS** (0 errors).
- **ESLint (`npx eslint src/app/experiments/page.tsx src/app/methods/page.tsx src/components/ScientificPanel.tsx`):** **PASS** (0 errors, 0 warnings).
- **Production Build (`npm run build`):** **PASS** (Next.js 16.4.6 Turbopack; all 10 routes prerendered).
- **Live Browser Verification (`http://localhost:3000`):**
  - **`/experiments` (`1440×900` & `390×844`):** HTTP 200, 0 console errors, 0 failed requests (`0 / 50`), 0 horizontal overflow/clipping.
  - **`/methods` (`1440×900` & `390×844`):** HTTP 200, 0 console errors, 0 failed requests (`0 / 50`), 0 horizontal overflow/clipping across all 5 KaTeX blocks.
  - **`/` Overview (`1440×900` & `390×844`):** HTTP 200, 0 console errors, 0 failed requests (`0 / 50`), 0 horizontal overflow; mobile Computational Ledger header title (`267.59px`), stacked badge (`222.20px`), and wrapped subtitle (`309.60px`) fit cleanly within the `341.60px` card with zero clipping; desktop `1440×900` inline header preserved.

---

## 4. Remaining Limitations

1. **Interactive Modules:** `/noise-and-psd` (Phase 4B), `/matched-filtering` (Phase 4C), `/inference` (Phase 4D), and `/experiments` (Phase 4E) remain structured scientific placeholders ahead of interactive plotting and backend API endpoints.

---

## 5. Recommended Next Phase

- **Interactive Phase 4B (Noise & PSD):** Implement the interactive analytical PSD vs Welch periodogram explorer and colored-noise visualization backed by `src/gw_inference/psd.py` and `src/gw_inference/noise.py`.
