# GWLab — Phase 4A: Frontend Migration Foundation Report

**Document ID:** `reports/GWLab_Phase4A_Frontend_Migration_Report.md`  
**Date:** November 19, 2026  
**Status:** Completed and Verified  
**Supervisor / Reviewer:** Project Owner & ChatGPT Independent Scientific Review  
**Author:** GWLab Software Architecture & Scientific Visualization Team  

---

## 1. Executive Summary

Following the Phase 3B design system and framework feasibility audit, Streamlit was evaluated as insufficient for the long-term presentation layer of GWLab due to its reliance on brittle CSS hacks, fragile DOM class selectors (`st-key-*`, `.stApp`), unpredictable iframe rendering, and rigid layout constraints.

In Phase 4A, GWLab has successfully migrated its presentation layer to the modern, high-fidelity stack established by the **Kerr Geodesics Lab**:
- **Framework:** Next.js 16.4.6 (App Router) with React 19.3.0 and Turbopack.
- **Styling & Tokens:** Tailwind CSS v4.3.3 with inline theme configuration, CSS custom variables, and the editorial color system (Parchment light, Obsidian dark, Terracotta accent).
- **Typography:** Google variable fonts `Newsreader` (editorial serif), `IBM Plex Sans` (clean sans), and `IBM Plex Mono` (tabular scientific numerals).
- **API Boundary:** Thin, stateless Python FastAPI service (`src/gwlab/api/app.py`) running on `http://localhost:8000`, cleanly separating scientific computation from presentation without duplicating physics in JavaScript.
- **Visual Center:** Pure SVG reactive scientific plotting with zero external heavyweight charting bundles, delivering sub-millisecond render latency, interactive hover crosshairs, hairline grid styling, and complete theme integration.

All 7 visitor-facing routes have been established, with **Overview** and **Waveforms** fully migrated and verified. The test suite passed with 100% compliance across all 493 unit/regression tests, and automated visual testing via Playwright confirmed **0 uncaught console errors** and **0 horizontal overflows** across desktop (1440×900), tablet (1024×768), and mobile (390×844) viewports in both Light and Dark themes.

---

## 2. Kerr Geodesics Lab Architecture Audit

A rigorous inspection of the reference Kerr Geodesics Lab repository (`C:\Users\Soumitra\Desktop\Kerr-Geodesic-Integrator\frontend`) was conducted prior to implementation.

### 2.1 Exact Package Versions and Build Tooling
From `Kerr-Geodesic-Integrator/frontend/package.json`:
- `next`: `16.4.6`
- `react`: `19.3.0`, `react-dom`: `19.3.0`
- `tailwindcss`: `4.3.3`, `@tailwindcss/turbopack`: `4.3.3`
- `katex`: `^0.19.1`, `@types/katex`: `^0.16.8`
- `lucide-react`: `^1.75.0`
- `typescript`: `^5`, `eslint`: `^9`, `eslint-config-next`: `16.4.6`
- **Build Engine:** Next.js Turbopack (`next build` executes static prerendering across all 10 routes in ~1.9s, with full TypeScript typechecking in ~7.3s).

### 2.2 Design System Tokens & Color Palette
Kerr's color palette is an editorial, publication-grade scientific design system rather than a generic dark/light dashboard:
- **Light Theme (Editorial Warm Parchment):**
  - `--background`: `#FAF9F6` (warm off-white parchment)
  - `--surface`: `#FFFFFF`
  - `--surface-subtle`: `#F3EFEA`
  - `--border`: `#DEDAD2`
  - `--border-subtle`: `#E8E4DC`
  - `--foreground`: `#20201E`
  - `--foreground-secondary`: `#55534F`
  - `--accent`: `#B94A24` (burnt orange / terracotta)
- **Dark Theme (Technical Deep Obsidian):**
  - `--background`: `#0E1013` (deep slate obsidian)
  - `--surface`: `#15181C`
  - `--surface-subtle`: `#191C21`
  - `--border`: `#262930`
  - `--border-subtle`: `#1E2127`
  - `--foreground`: `#ECEFF1`
  - `--foreground-secondary`: `#B0B5BF`
  - `--accent`: `#EA580C` (vibrant technical orange)

### 2.3 Typography & Formatting Conventions
- **Headings & Brand:** `Newsreader` variable serif font (`--font-newsreader`), normal weight, tracking-tight.
- **UI Prose & Controls:** `IBM Plex Sans` (`--font-ibm-plex-sans`), 400/500/600 weight.
- **Numerals & Code:** `IBM Plex Mono` (`--font-ibm-plex-mono`) with `tabular-nums` enforced globally via `.tabular-nums, .font-mono, td.numeric { font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }`.
- **Scientific Formatting Utility (`src/lib/format.ts`):**
  - True typographic minus sign: `\u2212` (`−`), never an ASCII hyphen-minus `-`.
  - Scientific notation: multiplication cross `\u00D7` (`×`) with Unicode superscripts (`10⁻²¹`), avoiding raw `e-21`.
  - Physical unit separation: Thin space `\u2009` between numerical value and symbol (e.g., `28.0956 M_sun`, `410.0 Mpc`).

---

## 3. System Architecture and Service Boundary

```
[ Visitor Browser (Desktop / Tablet / Mobile) ]
                      │
                      ▼
[ Next.js 16.4.6 App Router (Port 3000) ]
   ├── src/app/layout.tsx (Theme, Google Fonts, Global CSS)
   ├── src/app/page.tsx (Overview Dashboard)
   ├── src/app/waveforms/page.tsx (Interactive Signal Lab)
   ├── src/components/WaveformPlot.tsx (Reactive SVG Plot)
   ├── src/context/ThemeContext.tsx (Light/Dark Persistence)
   └── src/lib/api.ts (Typed HTTP Fetchers)
                      │
                      ▼ (REST JSON API / CORS Enabled)
[ Python FastAPI Service (Port 8000) ]
   ├── src/gwlab/api/app.py (FastAPI Routes)
   └── src/gwlab/api/service.py (Artifact Loader & Dispatcher)
                      │
                      ▼ (Zero Side Effects / Pure Functions)
[ Underlying Scientific Engine ]
   ├── src/gwlab/physics.py (TaylorF2 3.5PN, Chirp Time)
   ├── experiments/ (Audited Stage 6C JSON & Aggregate Ledger)
   └── reports/demonstrations/ (Historical Stage 5G Seed Comparisons)
```

### 3.1 Strict Safety & Governance Boundaries
1. **Scientific Engine Untouched:** `src/gw_inference/` was not modified in any way.
2. **Zero Likelihood Calls / Zero Samplers:** No Dynesty, no MCMC samplers, and no likelihood evaluators are accessible from the presentation layer. The API serves precomputed audited artifacts and evaluates only pure, deterministic frequency-domain waveforms.
3. **No Scientific Duplication:** Post-Newtonian phase and amplitude calculations are executed strictly in Python via `gwlab.physics.generate_taylorf2_strain` and `compute_chirp_time_series`. No formulas are reimplemented or approximated in TypeScript.
4. **Independent Service Lifecycles:** The FastAPI engine runs independently on port 8000, allowing the Next.js frontend on port 3000 to be hot-reloaded or statically served without interfering with Python processes.

---

## 4. Route-by-Route Migration Status

| Route | Destination Name | Status | Key Features & Implementation Notes |
| :--- | :--- | :--- | :--- |
| `/` or `/overview` | **Overview** | **Migrated & Complete** | Editorial narrative, Stage 6C Run A anchor benchmark card, independent-seed repeatability card, Stage 6D closeout ledger, provenance audit matrix. |
| `/waveforms` | **Waveforms** | **Migrated & Complete** | Deterministic TaylorF2 3.5PN synthesis, interactive controls ($\mathcal{M}_c, \eta, d_L, t_c, \phi_c$), benchmark reset button, derived mechanics readouts, dominant SVG plot (>65% width) with 4 tabs, KaTeX mathematical formulation. |
| `/noise-and-psd` | **Noise and PSD** | **Placeholder** | Page header, module overview, H1 design sensitivity specs, Tukey windowing parameters, $\nu_{\text{eff}} \approx 44.85$, link back to Waveforms. |
| `/matched-filtering` | **Matched filtering** | **Placeholder** | Page header, optimal Wiener filter formulation, inner product equation, injection SNR benchmark $\rho_{\text{opt}} \approx 25.14$, link back to Waveforms. |
| `/inference` | **Inference** | **Placeholder** | Page header, nested sampling overview, Stage 6C posterior artifact reference (11,951 samples), $\eta$-boundary mass accumulation note, link back to Overview. |
| `/experiments` | **Experiments** | **Placeholder** | Page header, campaign archive overview (Stage 5G, 6C, 6D), multi-seed comparison matrix scope, link back to Overview. |
| `/methods` | **Methods and provenance** | **Placeholder** | Detailed documentation of SI unit standards, TaylorF2 approximations, ISCO cutoff frequency, and scientific governance principles. |

---

## 5. Detailed Walkthrough of Migrated Pages

### 5.1 Overview Page (`/`)
- **Restrained Research Narrative:** Cleanly explains GWLab's purpose as an interactive research instrument for gravitational-wave signal analysis and parameter estimation under LIGO Hanford (H1) design sensitivity.
- **Reference Benchmark Card (Run A Anchor):**
  - Displays authoritative values with tabular numerals: $\ln Z = 201646.1413 \pm 0.2374\,\text{nats}$, $500,657\,\text{calls (raw)}$, $\text{Wall: } 515.0\,\text{s } (514.95\,\text{s})$, $\text{CPU: } 473.16\,\text{s}$.
  - Median and 90% credible intervals for $\mathcal{M}_c, \eta, t_c, \phi_c$.
  - Explicit reporting of the 38.57% posterior mass accumulation in the upper 1% equal-mass boundary $[\eta \ge 0.2485]$.
- **Independent-Seed Repeatability Card (Seed 8101 vs Seed 8201):**
  - Side-by-side run metrics for `base_01_s1` vs `base_01_s2`.
  - Evidence difference $|\Delta \ln Z| = 2.9720\,\text{nats}$, quadrature combined uncertainty $\sigma_\Delta = 0.3332\,\text{nats}$.
  - Ratio prominently highlighted: **8.92**.
  - **Required Scientific Qualification Callout:**
    > *"Observed evidence discrepancy / quadrature-combined reported internal uncertainty = 8.92. This ratio reflects internal sampler reporting under 1 degree of freedom; it is not a calibrated frequentist significance and does not establish population-level seed variance."*
- **Computational Ledger at Stage 6D Closeout:**
  - Displays remaining recorded budget at Stage 6D closeout ($487,407\,\text{calls used}$ / $712,593\,\text{calls reserve}$, $40.6\%$ consumed out of $1,200,000$ cap).
  - Explicit disclaimer: *"Historical ledger snapshot, not live resource telemetry and not authorization to execute experiments."*
- **Authoritative Data Provenance Matrix:**
  - Audited table listing artifact classification, repository path, scientific scope, and verification status.

### 5.2 Waveforms Page (`/waveforms`)
- **Interactive Controls Panel:**
  - Compact numeric sliders with instant reactive updates:
    - Chirp Mass $\mathcal{M}_c$: $10.0$ to $50.0\,M_\odot$ (default $28.0956\,M_\odot$).
    - Symmetric Mass Ratio $\eta$: $0.05$ to $0.25$ (default $0.2471$).
    - Luminosity Distance $d_L$: $50.0$ to $1500.0\,\text{Mpc}$ (default $410.0\,\text{Mpc}$).
    - Arrival Shift $t_c$: $-0.1$ to $+0.1\,\text{s}$ (default $0.000\,\text{s}$).
    - Coalescence Phase $\phi_c$: $0$ to $2\pi\,\text{rad}$ (default $0.00\,\text{rad}$).
  - **Reset Button:** Restores exact Stage 6C injection benchmark parameters in one click.
- **Derived Binary Mechanics Readout:**
  - Dynamic calculation of primary mass $m_1$, secondary mass $m_2$, total mass $M$, mass ratio $q = m_1 / m_2$, and Schwarzschild ISCO frequency $f_{\text{ISCO}} = 4397.0 / M\,\text{Hz}$.
- **Dominant Visual Center (>65% Width):**
  - Positioned above the fold at 1440×900 desktop resolution.
  - Interactive SVG waveform canvas featuring 4 tabs:
    1. **Strain Amplitude:** Log-log frequency-domain spectrum showing the characteristic $f^{-7/6}$ inspiral power-law scaling up to $f_{\text{ISCO}}$.
    2. **Fourier Phase:** Continuous unwrapped 3.5PN post-Newtonian phase trajectory $\Psi(f)$.
    3. **Chirp Time $\tau(f)$:** Dual trace comparing Newtonian $\tau_0(f)$ (dashed) against 3.5PN time to merger (solid terracotta).
    4. **Quadratures:** Simultaneous display of real and imaginary strain components $\text{Re}[\tilde{h}(f)]$ and $\text{Im}[\tilde{h}(f)]$.
  - **Interactive Hover Crosshair:** Tracks cursor along the frequency grid and renders a clean readout pill displaying exact frequency $f$ and corresponding observable.
- **Mathematical Formulation & 3.5PN Expansion:**
  - Beautiful KaTeX-rendered equations displaying:
    $$\tilde{h}(f) = \mathcal{A}(f)\,e^{-i\Psi(f)}$$
    $$\mathcal{A}(f) = \sqrt{\frac{5}{24}}\,\frac{G^{5/6}\mathcal{M}_c^{5/6}\,c^{-3/2}}{\pi^{2/3}\,d_L}\,f^{-7/6}$$
    $$\tau(f) = \frac{5}{256}\left(\frac{G\mathcal{M}_c}{c^3}\right)^{-5/3}(\pi f)^{-8/3}$$
    $$\Psi(f) = 2\pi f t_c - \phi_c - \frac{\pi}{4} + \frac{3}{128\eta v^5}\sum_{k=0}^7 \alpha_k(\eta)\,v^k, \quad v = (\pi G M f / c^3)^{1/3}$$

---

## 6. Verification and Visual Testing Results

### 6.1 Backend API Unit Testing
The FastAPI backend service was verified with a comprehensive unit test suite in `tests/test_gwlab_api.py`:
- `test_api_status`: Validates online health, detector configuration (H1), and frequency bounds ($20$–$512\,\text{Hz}$).
- `test_api_benchmarks_overview`: Validates numerical fidelity of Stage 6C Anchor results, resource ledger, seed repeatability ratio ($8.92$), and provenance records.
- `test_api_waveform_generate`: Validates deterministic TaylorF2 3.5PN array generation, coordinate conversion ($m_1 \ge m_2$), and ISCO cutoff enforcement.
- `test_api_chirp_time`: Validates Newtonian and 3.5PN time-to-merger curves.
- **Result:** **4 Passed out of 4 (100%)**.

### 6.2 Full Repository Regression Testing
Running `pytest` across the entire repository confirmed **493 passing tests** with 0 regressions against existing Work Package deliverables.

### 6.3 Automated Visual & Playwright Testing
An automated Playwright script (`frontend/scripts/verify_visuals.mjs`) was executed against the live application.
- **Viewports Tested:**
  1. Desktop: $1440 \times 900\,\text{px}$
  2. Tablet: $1024 \times 768\,\text{px}$
  3. Mobile: $390 \times 844\,\text{px}$ (iPhone 12/13/14 profile)
- **Themes Tested:** Light Mode (`data-theme="light"`) and Dark Mode (`data-theme="dark"`).
- **Screenshots Captured:** **45 total screenshots** saved to `reports/screenshots/phase4a/`.
- **Horizontal Overflow Check:** `document.documentElement.scrollWidth <= window.innerWidth` verified on every single page and viewport. **Zero overflows detected**.
- **Console Error Audit:** `page.on('console')` and `page.on('pageerror')` monitored continuously. **Zero uncaught errors detected**.

---

## 7. Execution and Local Preview Instructions

Both servers are currently configured and running in background daemon processes:

### 7.1 URLs
- **Next.js Production Frontend:** [`http://localhost:3000`](http://localhost:3000)
- **FastAPI Scientific Service:** [`http://localhost:8000`](http://localhost:8000) (Swagger docs at [`http://localhost:8000/docs`](http://localhost:8000/docs))
- **Streamlit Legacy Fallback:** [`http://localhost:8502`](http://localhost:8502) (preserved intact as required)

### 7.2 Manual Start Commands (if restarting session)
1. **Start the FastAPI Backend:**
   ```bash
   python -m uvicorn gwlab.api.app:app --host 127.0.0.1 --port 8000
   ```
2. **Start the Next.js Frontend:**
   ```bash
   cd frontend
   npx next start -p 3000
   ```
   *(Or for live development mode: `npm run dev -- -p 3000`)*

---

## 8. Conclusion and Sign-off Recommendation

Phase 4A has delivered a robust, publication-grade scientific frontend architecture for GWLab matching the Kerr Geodesics Lab design standard. The separation of concerns between Next.js and FastAPI provides optimal responsiveness, clean code structure, and zero risk to the frozen scientific core.

Phase 4A is ready for independent scientific review and formal closeout sign-off.
