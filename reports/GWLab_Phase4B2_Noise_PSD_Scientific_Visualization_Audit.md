# GWLab Phase 4B.2 — Noise & PSD Scientific Visualization Audit Report

**Auditor:** Scientific Visualization Auditor & Implementation Agent  
**Date:** 2026-11-25  
**Target URL:** `http://localhost:3000/noise-and-psd`  
**Recommendation:** **PASS WITH QUALIFICATIONS**  
**Audit Scope:** Read-only scientific visualization and layout audit of the existing Noise & PSD page. Zero source modifications, zero sampler executions, zero working-tree modifications.

---

## 1. Exact Files Inspected

### Frontend Source & Shell Architecture
1. `frontend/src/app/noise-and-psd/page.tsx` — Current Next.js App Router implementation of the Noise & PSD page.
2. `frontend/src/components/Navbar.tsx` — Top application chrome, navigation tabs, breakpoint visibility rules, and theme toggles.
3. `frontend/src/components/PageHeader.tsx` — Page title, badge metadata, subtitle, and description formatting.
4. `frontend/src/components/ScientificPanel.tsx` — Card container and panel hierarchy.
5. `frontend/src/app/globals.css` — Global CSS variables, design tokens (`--surface-subtle`, `--accent`, `--border`), typography.

### Scientific Engine & Ground-Truth Reference Artifacts
6. `src/gw_inference/psd.py` — Authoritative Welch PSD implementation: discrete Tukey windowing, window power normalization ($S_2$), window overlap correlation ($r_\ell$), multi-lag variance inflation ($\kappa_{\text{overlap}}$), and analytical detector model proxy (`aligo_representative`).
7. `src/gw_inference/noise.py` — Continuous Fourier-normalized colored noise synthesis (`generate_colored_noise`) and white noise (`generate_white_noise`).
8. `src/gwlab/config.py` — Analysis bandwidth ($[20.0, 512.0]\,\text{Hz}$), sampling rate ($2048\,\text{Hz}$), and nominal observing run strings.
9. `src/gwlab/pages/3_noise_and_psd.py` — Historical Phase 3 Streamlit reference implementation.
10. `scripts/run_wp6_stage5_stage_b.py` — Authoritative Stage 5B Welch calibration campaign script establishing $T_{\text{obs}} = 64.0\,\text{s}$, $T_{\text{seg}} = 4.0\,\text{s}$, $K = 31$, $r_1 = 4/9$, $\kappa = 1157/837$, and $\nu_{\text{eff}} \approx 44.8522$.
11. `reports/WP5_Implementation_Report.md` & `reports/WP6_Stage5G5_Scientific_Synthesis_and_Decision_Report.md` — Authoritative records establishing Gaussian Whittle likelihood with frozen off-source Welch PSD and documented rejection/freezing of Student-$t$ likelihood.

### Visual & Runtime Audit Artifacts
12. `reports/screenshots/phase4b2/noise_psd_desktop_1440x900_light.png`
13. `reports/screenshots/phase4b2/noise_psd_desktop_1440x900_dark.png`
14. `reports/screenshots/phase4b2/noise_psd_mobile_390x844_light.png`
15. `reports/screenshots/phase4b2/noise_psd_mobile_390x844_dark.png`
16. `reports/screenshots/phase4b2/audit_summary.json`
17. Browser host evidence log: `C:/Users/Soumitra/.gemini/antigravity/brain/f07d6c26-1866-4b32-8c0a-f82af82eed2f/browser_evidence/0cb40bd8-652a-45c5-b501-589366a570a1/final_step_24.png`

---

## 2. Runtime and Browser Verification Results

Browser inspection was conducted live on `http://localhost:3000/noise-and-psd` via automated Playwright test harness and the interactive Chromium browser agent:

| Verification Dimension | Desktop (1440×900) | Mobile (390×844) | Status | Observations |
| :--- | :--- | :--- | :--- | :--- |
| **HTTP Status & Hydration** | 200 OK | 200 OK | **PASS** | Page hydrates cleanly; App Router SSR/client hydration succeeded. |
| **Console Errors** | 0 errors | 0 errors | **PASS** | Zero browser console errors or unhandled exceptions. |
| **Page Errors / Crashes** | None | None | **PASS** | Zero Next.js runtime error boundaries triggered. |
| **Network Requests** | All 200 OK | All 200 OK | **PASS** | Static assets, chunks, and CSS bundle loaded without 4xx/5xx failures. |
| **Horizontal Overflow** | Scroll: 1440px / Client: 1440px | Scroll: 390px / Client: 390px | **PASS** | Zero horizontal overflow; `scrollWidth == clientWidth` across both themes. |
| **Theme Toggle Interaction** | Verified | Verified | **PASS** | Clicking theme toggle cleanly updates `data-theme` between `light` and `dark`. |
| **Navigation Link (`/waveforms`)** | Verified | Verified | **PASS** | Link to `/waveforms` navigates cleanly to the Waveforms laboratory. |
| **Interactive Controls / Plots** | 0 `<canvas>`, 0 sliders | 0 `<canvas>`, 0 sliders | **NOTE** | Page is currently a structured informative placeholder module. |

---

## 3. Scientific Correctness and Traceability Findings

### 3.1 Detector Sensitivity & Proxy Qualification (Issue 1.1)
- **Current Display:** The page header states *"Exploration of colored Gaussian detector noise for LIGO Hanford (H1)..."*, the specification box lists *"Detector: LIGO Hanford (H1) Design Sensitivity"*, and the footer reads *"LIGO Hanford (H1) design sensitivity · Realization 01"*.
- **Ground Truth:** In the authoritative scientific engine (`src/gw_inference/psd.py`, lines 1130–1138), the model is titled `aligo_representative`:
  > *"This model is a custom, regularized, hand-constructed representative detector-noise proxy designed for controlled numerical benchmarking. It is NOT an official Advanced LIGO design-sensitivity curve, measured detector PSD, or standard reference formula."*
- **Audit Assessment:** Calling this curve "LIGO Hanford (H1) Design Sensitivity" without explicitly labeling it as an **analytical model proxy** implies real detector telemetry or an official LALSimulation instrument curve. GWLab's strict scientific integrity guidelines require that synthetic analytical proxies must not be presented as empirical detector data.

### 3.2 Effective Degrees of Freedom Context & Mathematical Grounding (Issue 1.2)
- **Current Display:** `• Effective Welch Degrees of Freedom: ν_eff ≈ 44.85`
- **Ground Truth:** In `scripts/run_wp6_stage5_stage_b.py` (lines 69–90), $\nu_{\text{eff}} \approx 44.8522$ is mathematically derived from a specific observation campaign:
  - Total observation duration: $T_{\text{obs}} = 64.0\,\text{s}$ ($N = 131,072$ samples at $f_s = 2048\,\text{Hz}$)
  - Segment length: $T_{\text{seg}} = 4.0\,\text{s}$ ($M = 8192$ samples)
  - Overlap: $50\%$ ($H = 4096$ samples)
  - Number of segments: $K = (N - \text{overlap}) / H = 31$
  - Adjacent-segment correlation for discrete Tukey ($\alpha = 0.25$): $r_1 = 4/9$
  - All-lags variance inflation factor: $\kappa_{\text{overlap}} = \frac{1157}{837} \approx 1.3823$
  - Effective degrees of freedom:
    $$\nu_{\text{eff}} = \frac{2 K}{\kappa_{\text{overlap}}} = \frac{2 \times 31}{1157 / 837} = \frac{51894}{1157} \approx 44.8522$$
- **Audit Assessment:** Presenting $\nu_{\text{eff}} \approx 44.85$ directly beneath `Welch Segment Duration: 4.0 s` without stating $T_{\text{obs}} = 64.0\,\text{s}$ ($K = 31$ segments) is misleading. A user might assume that 44.85 degrees of freedom belong to a single 4.0 s segment (which has only 2 degrees of freedom per frequency bin), rather than being the cumulative degrees of freedom of 31 overlapping segments.

### 3.3 Physical Units for PSD & ASD (Issue 1.3)
- **Current Display:** The text introduces "Power Spectral Density (PSD)" and "analytic PSD curves", but does not provide physical units.
- **Ground Truth:** 
  - One-sided Power Spectral Density: $S_n(f)$ has units of $\mathbf{\text{strain}^2/\text{Hz}}$ (or $\text{Hz}^{-1}$).
  - Amplitude Spectral Density: $\sqrt{S_n(f)}$ has units of $\mathbf{\text{strain}/\sqrt{\text{Hz}}}$ (or $\text{Hz}^{-1/2}$).
- **Audit Assessment:** In Phase 4B.1, Fourier-domain strain units were rigorously standardized to $\text{strain}/\text{Hz}$ (or $\text{s}$). The Noise & PSD page should explicitly display the units for both $S_n(f)$ ($\text{strain}^2/\text{Hz}$) and $\sqrt{S_n(f)}$ ($\text{strain}/\sqrt{\text{Hz}}$) to avoid confusion.

### 3.4 Student-$t$ Likelihood Representation
- **Audit Assessment:** The current page **completely omits** any mention of the Student-$t$ likelihood.
- **Scientific Conformance:** This is strictly correct. Stage 5E/5E2/5F/5G5 reports established that Student-$t$ PSD uncertainty marginalization is not approved for production inference and was formally frozen in favor of standard Gaussian Whittle likelihood with frozen off-source Welch estimation. The page correctly avoids implying that Student-$t$ is an accepted or approved inference model.

### 3.5 Matched Filtering Metric Characterization
- **Current Display:** *"noise Power Spectral Density (PSD) estimation is critical for calibrating the inner product in matched filtering..."*
- **Audit Assessment:** In statistical signal processing, $S_n(f)$ defines the noise-weighted metric tensor of the Hilbert space:
  $$\langle a \mid b \rangle = 4 \operatorname{Re} \int_{f_{\text{low}}}^{f_{\text{high}}} \frac{\tilde{a}(f)\,\tilde{b}^*(f)}{S_n(f)}\,df$$
  Describing this as "calibrating the inner product" is slightly colloquial; "defining the noise-weighted inner product" or "specifying the metric of the inner product" is more mathematically precise.

---

## 4. Desktop and Mobile Visual Findings

### Desktop (1440×900)
- **Visual Balance:** Clean, editorial design following the Kerr Geodesics Lab typography hierarchy (Newsreader serif heading, Inter/sans body, IBM Plex Mono specifications).
- **Navbar:** Clean active indicator beneath `Noise and PSD`. Status badge `Reference archive` is green and positioned correctly.
- **Spacing:** Margin between PageHeader and ScientificPanel is generous ($32\,\text{px}$) and consistent with other modules.
- **Typography & Hierarchy:** Clear separation between primary header, badge (`Upcoming Module · Phase 4B`), and informative card.

### Mobile (390×844)
- **Responsive Stacking:** Header, card, and footer wrap cleanly with zero horizontal scrollbar or element clipping.
- **Text Wrapping in Code Box:** Inside the monospace box, lines such as `• Detector: LIGO Hanford (H1) Design Sensitivity` wrap cleanly onto two lines. Padding and font size ($12\,\text{px}$) are legible.
- **Navbar Collapsing:** Collapses cleanly into brand logo, mobile theme toggle, and hamburger menu.

### Intermediate Breakpoint Observation (768px – 1023px)
- **Duplicate Moon Theme Toggle Icons:** In `frontend/src/components/Navbar.tsx`:
  - Line 103 desktop toggle: `hidden md:flex`
  - Line 147 mobile toggle: `flex lg:hidden`
  - Between $768\,\text{px}$ (`md`) and $1023\,\text{px}$ (`lg`), both classes evaluate to visible, rendering two identical theme-toggle buttons side by side in the header. (Minor responsive defect; does not affect the audited 1440×900 or 390×844 viewports).

---

## 5. Prioritized Issues

| ID | Priority | Category | Issue Description | Location |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | **Medium** | Scientific Traceability | **Unqualified Analytical Proxy Labeled as "Design Sensitivity"**<br>The page labels the detector curve as "LIGO Hanford (H1) Design Sensitivity" without stating it is a synthetic analytical proxy (`aligo_representative`), violating the guideline against implying real instrument telemetry. | `page.tsx` lines 21, 36 |
| **1.2** | **Medium** | Scientific Traceability | **Ungrounded Degrees of Freedom ($\nu_{\text{eff}} \approx 44.85$) Lacks Observation Duration & Segment Count**<br>$\nu_{\text{eff}}$ is presented directly under a 4.0 s segment duration without identifying that it requires $K = 31$ segments across a $64.0\,\text{s}$ observation with $\kappa_{\text{overlap}} \approx 1.3823$. | `page.tsx` line 40 |
| **1.3** | **Minor** | Scientific Definition | **Omission of Physical Units for PSD & ASD**<br>The text discusses PSD without stating units ($\text{strain}^2/\text{Hz}$ and $\text{strain}/\sqrt{\text{Hz}}$). | `page.tsx` lines 31–33 |
| **2.1** | **Minor** | Typography | **Monospace Greek Nu (`ν_eff`) Renders Indistinguishably from `V_eff`**<br>In standard monospace fonts, the Greek small letter nu (`ν`) looks identical to uppercase `V` or Latin `v`, causing ambiguity with voltage or volume. | `page.tsx` line 40 |
| **2.2** | **Minor** | Frontend Responsive | **Dual Theme-Toggle Icons at Intermediate Tablet Breakpoints (768px–1023px)**<br>`hidden md:flex` and `flex lg:hidden` overlap between 768px and 1023px in `Navbar.tsx`. | `Navbar.tsx` lines 103, 147 |

---

## 6. Audit Recommendation

### Final Recommendation: **PASS WITH QUALIFICATIONS**

- **Runtime & Interaction:** **PASS** (Zero crashes, zero errors, clean hydration, zero horizontal overflow at 1440×900 and 390×844).
- **Visual Presentation:** **PASS** (Follows approved Kerr design system, high readability).
- **Scientific Meaning & Traceability:** **QUALIFIED** (Page makes no false empirical claims and correctly omits Student-$t$, but requires precise qualification of the analytical detector proxy, explicit PSD/ASD physical units, and full provenance for $\nu_{\text{eff}} \approx 44.85$).

---

## 7. Proposed Smallest Targeted Corrections (For Future Phase 4B Implementation)

*Note: Per audit constraints, these are proposals only. No application source code has been modified in this task.*

1. **Explicit Proxy Labeling in `page.tsx`:**
   ```tsx
   // Replace:
   <div>&bull; Detector: LIGO Hanford (H1) Design Sensitivity</div>
   // With:
   <div>&bull; Detector Sensitivity: Analytical aLIGO Proxy (Synthetic H1 Model)</div>
   ```

2. **Grounding $\nu_{\text{eff}}$ in Observation Campaign Parameters:**
   ```tsx
   // Replace:
   <div>&bull; Welch Segment Duration: 4.0 s (50% segment overlap)</div>
   <div>&bull; Effective Welch Degrees of Freedom: ν_eff ≈ 44.85</div>
   // With:
   <div>&bull; Welch Segment Duration: 4.0 s (50% overlap, K = 31 segments across 64.0 s observation)</div>
   <div>&bull; Effective Degrees of Freedom: ν_eff = 2K / κ_overlap ≈ 44.85 (κ_overlap ≈ 1.3823)</div>
   ```

3. **Defining Physical Units:**
   ```tsx
   // In introductory paragraph:
   "In gravitational-wave data analysis, noise Power Spectral Density S_n(f) [strain²/Hz] " +
   "and Amplitude Spectral Density √S_n(f) [strain/√Hz] define the noise-weighted inner product..."
   ```

4. **Correcting Breakpoint Overlap in `Navbar.tsx`:**
   ```tsx
   // In Navbar.tsx, adjust desktop theme toggle from:
   <div className="hidden md:flex items-center gap-2">
   // To:
   <div className="hidden lg:flex items-center gap-2">
   ```
