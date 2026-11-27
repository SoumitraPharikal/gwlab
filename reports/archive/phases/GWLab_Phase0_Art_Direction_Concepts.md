# GWLab — Phase 0: Art Direction and Three Visual Concepts

**Author:** Senior Art Director & Frontend Design Architect  
**Audience:** Project Owner & Independent Scientific Reviewer (ChatGPT)  
**Date:** 2026-11-19  
**Status:** Reviewable Design Milestone (Targeted Scientific & Design Corrections Applied)  
**Repository Scope:** Deterministic static review concepts (`concepts/`), visual evidence capture (`reports/screenshots/concepts/`), zero sampler runs, zero modifications to `src/gw_inference/` or saved experiment artifacts.

---

## 1. Executive Summary

This milestone establishes three genuinely distinct visual directions for **GWLab** (Gravitational-Wave Analysis Laboratory), an interactive scientific portfolio and research application for exploring gravitational waveforms, detector noise, matched filtering, and Bayesian inference.

Following targeted scientific and design review feedback, all three static prototypes have been refined to ensure absolute mathematical rigor, physical accuracy, and visual honesty:
1. **Neutral Scholarly Citation:** The fabricated arXiv identifier in Concept A has been replaced with a neutral, publication-grade technical report folio label (`GWLab Monograph Series · Technical Report 2026-01`).
2. **Explicit Synthetic Configuration:** All control-room status indicators in Concept B have been explicitly labeled as an *illustrative synthetic model configuration* (`Synthetic Realization 01`), eliminating any implication of live detector telemetry. Fictional LEDs and fake alarms have been removed. Monospace typography is strictly restricted to numerical values, parameters, code identifiers, and tabular telemetry, while interface narrative and buttons use `IBM Plex Sans`. Single UI accent discipline (`#C44D18` light / `#FF6A28` dark) is maintained, with Channel 2 cyan designated as a non-accent data-series color.
3. **Relativistic ISCO Qualification:** The merger description in Concept C has been scientifically corrected: $r = 6M$ ($R = 6GM/c^2$) is strictly the test-particle Innermost Stable Circular Orbit in Schwarzschild spacetime. In comparable-mass binaries ($m_1 \sim m_2$), dynamical finite-mass interactions and spacetime curvature shift the plunge; $r=6M$ is not a universal binary merger threshold.
4. **Model Boundary Distinction:** Across all concepts, the frequency-domain TaylorF2 approximant is explicitly distinguished from illustrative merger-ringdown depictions. TaylorF2 is strictly a stationary-phase inspiral model up to the heuristic cutoff $f_{\text{ISCO}}$; it does not compute plunge, merger, or ringdown.
5. **Exact Numerical Parameter Derivations & SI Amplitude:** Physical equations, dependencies, and units connecting $(\mathcal{M}_c, \eta, d_L) \to (M, m_1, m_2, q, f_{\text{ISCO}}, R_{\text{ISCO}}, \tau_0, |\tilde{h}(f)|)$ are verified and documented. The frequency-domain strain amplitude $|\tilde{h}(f)| = \sqrt{5/24}\,\pi^{-2/3}\,(G\mathcal{M}_c/c^3)^{5/6}\,(c/d_L)\,f^{-7/6}$ includes the exact SI factor $c/d_L$, yielding Fourier strain units of seconds ($\text{strain}/\text{Hz}$ or $\text{s}$), not $\text{Hz}^{-1/2}$.

An automated Playwright visual audit across three viewports (1440×900 desktop, 1024×768 tablet, 390×844 mobile) and two themes verified **50 captured screenshots**, confirming **0 console/page errors** and **0 horizontal overflows**. Two concise contact sheets have been generated for rapid visual inspection.

---

## 2. Kerr Geodesics Lab Foundation Audit & Package Verification

The production frontend repository of **Kerr Geodesics Lab** (`C:\Users\Soumitra\Desktop\Kerr-Geodesic-Integrator\frontend`) was directly inspected to verify architectural compatibility, framework versions, and styling foundations:

### Inspected Files & Manifest Evidence
- **`package.json` & `package-lock.json`:**
  - `next`: `16.4.6` (App Router, Turbopack)
  - `react`: `19.3.0`
  - `react-dom`: `19.3.0`
  - `tailwindcss`: `^4.3.3` (Tailwind v4 CSS `@theme` tokens)
  - `katex`: `^0.19.1` (SSR and client-side mathematical typesetting)
  - `lucide-react`: `^1.75.0` (Iconography)
- **`src/app/layout.tsx`:**
  - Font configurations: Google Fonts `Newsreader` (optical sizes 6–72pt), `IBM Plex Sans` (300, 400, 500, 600), and `IBM Plex Mono` (400, 500, 600).
  - Enforces `data-theme="light"` or `data-theme="dark"` attribute on `<html>`.
- **`src/app/globals.css`:**
  - CSS custom properties (`--background`, `--surface`, `--border`, `--foreground`, `--accent`).
  - Strict tabular numeral enforcement: `.tabular-nums { font-variant-numeric: tabular-nums; }`.
  - Accessible reduced-motion media query clamping animation and transition durations to $\le 0.01\,\text{ms}$.
- **`src/components/ui/plot.tsx` & `src/lib/formatting.ts`:**
  - Sub-millisecond responsive pure SVG plotting using `viewBox` without heavy canvas dependencies.
  - Mathematical typography conventions: true Unicode minus (`\u2212`, `−`), thin mathematical spaces (`\u2009`).

---

## 3. Authoritative Scientific Assets, Derivations & Provenance

### 3.1 Numerical Parameter Relationships & Exact Equations
All numerical values in GWLab are strictly derived from fundamental general relativistic and post-Newtonian equations:

1. **Total Mass ($M$):**
   Derived from Chirp Mass $\mathcal{M}_c$ and Symmetric Mass Ratio $\eta$:
   $$M = \mathcal{M}_c\,\eta^{-3/5}$$
   For the canonical Stage 6C benchmark injection ($\mathcal{M}_c = 28.0956\,M_\odot, \eta = 0.2471$):
   $$M = \frac{28.095556}{(0.2471006)^{0.6}} = 65.0000\,M_\odot$$

2. **Component Masses ($m_1, m_2$) and Mass Ratio ($q$):**
   Derived via the quadratic relation $m^2 - M m + \eta M^2 = 0$:
   $$m_1 = \frac{M}{2}\left(1 + \sqrt{1 - 4\eta}\right) = \frac{65.0}{2}\left(1 + \sqrt{1 - 4(0.2471006)}\right) = 36.0000\,M_\odot$$
   $$m_2 = \frac{M}{2}\left(1 - \sqrt{1 - 4\eta}\right) = \frac{65.0}{2}\left(1 - \sqrt{1 - 4(0.2471006)}\right) = 29.0000\,M_\odot$$
   $$q = \frac{m_1}{m_2} = \frac{36.0}{29.0} \approx 1.2414$$

3. **Schwarzschild ISCO Radius ($R_{\text{ISCO}}$):**
   Test-particle circular orbit stability limit in Schwarzschild geometry:
   $$R_{\text{ISCO}} = \frac{6 G M}{c^2} = 6 \times \left(1.476625\,\frac{\text{km}}{M_\odot}\right) \times M \approx 8.85975 \times M\,\text{km}$$
   For $M = 65.0\,M_\odot$:
   $$R_{\text{ISCO}} = 575.88\,\text{km} \approx 575.9\,\text{km}$$
   *Physical Qualification:* This radius strictly applies to a test particle orbiting a static Schwarzschild black hole. In comparable-mass coalescences ($q \sim 1$), dynamical finite-mass interactions, mutual tidal forces, and spacetime self-force significantly shift the plunge transition; $r = 6M$ is not a universal binary merger threshold.

4. **Heuristic ISCO Cutoff Frequency ($f_{\text{ISCO}}$):**
   Gravitational-wave frequency at the test-particle Schwarzschild ISCO ($f_{\text{GW}} = 2 f_{\text{orb}}$):
   $$f_{\text{ISCO}} = \frac{c^3}{6^{3/2} \pi G M} = \frac{C_{\text{ISCO}}}{M / M_\odot} \approx \frac{4397.17}{M / M_\odot}\,\text{Hz}$$
   For $M = 65.0\,M_\odot$:
   $$f_{\text{ISCO}} = \frac{4397.17}{65.0} = 67.6488\,\text{Hz} \approx 67.65\,\text{Hz}$$
   *Model Role:* Serves as the heuristic termination frequency for the post-Newtonian TaylorF2 inspiral series.

5. **Frequency-Domain TaylorF2 Strain Amplitude ($|\tilde{h}(f)|$):**
   The Fourier-domain strain amplitude under the stationary-phase approximation in SI units is:
   $$|\tilde{h}(f)| = \sqrt{\frac{5}{24}}\,\pi^{-2/3}\,\left(\frac{G\,\mathcal{M}_c}{c^3}\right)^{5/6}\,\frac{c}{d_L}\,f^{-7/6} = A_0\,f^{-7/6}$$
   where the constant amplitude prefactor $A_0$ is:
   $$A_0 = \sqrt{\frac{5}{24}}\,\pi^{-2/3}\,\left(\frac{G\,\mathcal{M}_c}{c^3}\right)^{5/6}\,\frac{c}{d_L}$$
   - **Dimensional Analysis (SI Units):**
     - Chirp mass time scale: $[G \mathcal{M}_c / c^3] = \text{s}$, hence $[(G \mathcal{M}_c / c^3)^{5/6}] = \text{s}^{5/6}$.
     - Distance rate factor: $[c / d_L] = (\text{m/s}) / \text{m} = \text{s}^{-1}$.
     - Prefactor dimension: $[A_0] = \text{s}^{5/6} \times \text{s}^{-1} = \text{s}^{-1/6}$.
     - Frequency power law: $[f^{-7/6}] = \text{Hz}^{-7/6} = \text{s}^{7/6}$.
     - Total Fourier strain dimension: $[|\tilde{h}(f)|] = [A_0 \cdot f^{-7/6}] = \text{s}^{-1/6} \cdot \text{s}^{7/6} = \text{s} = \text{strain}/\text{Hz} = \text{Hz}^{-1}$.
     - *Distinction from ASD:* $\text{Hz}^{-1/2}$ (or $\text{strain}/\sqrt{\text{Hz}}$) is strictly the physical unit of the detector Amplitude Spectral Density (ASD, $\sqrt{S_n(f)}$), whereas the Fourier-transformed continuous strain $\tilde{h}(f) = \int_{-\infty}^{\infty} h(t) e^{-i 2\pi f t} dt$ has units of seconds ($\text{strain}\cdot\text{s} = \text{strain}/\text{Hz}$).
   - **Numerical Benchmark Values:**
     - For canonical Stage 6C / Realization 01 injection parameters ($\mathcal{M}_c = 28.095556\,M_\odot$, $d_L = 410.0\,\text{Mpc} \approx 1.26513 \times 10^{25}\,\text{m}$):
       $$A_0 = \sqrt{\frac{5}{24}}\,\pi^{-2/3}\,\left(1.38319 \times 10^{-4}\,\text{s}\right)^{5/6}\,\frac{2.99792 \times 10^8\,\text{m/s}}{1.26513 \times 10^{25}\,\text{m}} \approx 3.0681 \times 10^{-21}\,\text{s}^{-1/6}$$
     - At $f = 20\,\text{Hz}$:
       $$|\tilde{h}(20\,\text{Hz})| = 3.0681 \times 10^{-21} \times (20)^{-7/6} \approx 9.311 \times 10^{-23}\,\text{s}\quad (\text{strain}/\text{Hz})$$

6. **Newtonian Coalescence Chirp Time ($\tau_0(f)$):**
   Leading-order time remaining before coalescence:
   $$\tau_0(f) = t_c - t(f) = \frac{5}{256}\,\left(\frac{G\,\mathcal{M}_c}{c^3}\right)^{-5/3}\,(\pi f)^{-8/3}$$
   At $f = 20\,\text{Hz}$ for $\mathcal{M}_c = 28.0956\,M_\odot$: $\tau_0 \approx 0.81\,\text{s}$.

### 3.2 Model Boundary & Scope Qualification
- **TaylorF2 Scope:** Strictly a frequency-domain stationary-phase approximation (SPA) to the 3.5PN post-Newtonian inspiral phase:
  $$\tilde{h}(f) = \mathcal{A}(f)\,\exp\left(-i\,\Psi_{3.5\text{PN}}(f)\right)\,\Theta(f_{\text{ISCO}} - f)$$
  TaylorF2 models point-mass inspiral up to $f_{\text{ISCO}}$. It does **not** compute the non-linear plunge, event-horizon merger, or quasi-normal mode ringdown.
- **Illustrative Time-Domain Depictions:** Any time-domain waveforms displaying merger and ringdown (e.g., Channel 2 in Concept B or the four regimes in Concept C) are explicitly labeled as *illustrative phenomenological surrogates* provided for qualitative context.

### 3.3 Authoritative Stage 6C Anchor Telemetry (`run_a_anchor_result.json`)
- Marginal log-evidence: $\ln\mathcal{Z} = 201646.1413 \pm 0.2374\,\text{nats}$ (Stage 6C Run A Model 2 Gaussian likelihood).
- 2-seed discrepancy ratio: $8.92\,\sigma$ ($|\Delta\ln\mathcal{Z}| = 2.9720\,\text{nats} / \sigma_\Delta = 0.3332\,\text{nats}$, seeds 8101 vs 8201).
- Resource ledger snapshot: $487,407$ likelihood calls consumed of $1,200,000$ hard cap ($40.6\%$ consumed, $712,593$ reserve; soft watchdogs $515.0\,\text{s}$ wall, $473.2\,\text{s}$ CPU).
- Equal-mass boundary density: $38.57\%$ posterior mass concentrated in top $1\%$ interval $\eta \in [0.2485, 0.2500]$.

---

## 4. Concept A: Scientific Journal (Corrected)

### 4.1 Aesthetic Metaphor & Narrative
Evokes classical scholarly treatises (*Philosophical Transactions of the Royal Society*, *Living Reviews in Relativity*). Paced for deep reading with formal plate figures, derivations, and margin notes.

### 4.2 Corrections Applied
- **Removed Fabricated Folio:** Removed `arXiv:gr-qc/GWLab-2026` and `ISSN 2026-6C4A`. Replaced with neutral publication folio:
  `GWLab Monograph Series · Technical Report 2026-01 (Document Ref: GWLAB-2026-TR01)`.
- **Component Mass Accuracy:** Corrected component masses in the parameter bounds table to exact values: $m_1 = 36.00\,M_\odot, m_2 = 29.00\,M_\odot, M = 65.00\,M_\odot, q = 1.241$.
- **Explicit Model Scope:** Added explicit figure captions noting that TaylorF2 models inspiral only, with abrupt heuristic termination at $f_{\text{ISCO}} = 67.65\,\text{Hz}$.

### 4.3 Color & Typography Tokens
- *Light Mode:* Warm Cream (`#FBF9F5`), Scholarly Ink (`#1A1D20`), Venetian Oxblood (`#8C2D19`).
- *Dark Mode:* Archival Deep Gray (`#111215`), Fine Off-White (`#E5E3DC`), Warm Copper (`#D85838`).
- *Typography:* `Newsreader` for body narrative and derivations; `IBM Plex Mono` for tabular figures and formulas.

### 4.4 Deliverables
- **Hero Page:** [concept-a/index.html](../../../concepts/concept-a/index.html)
- **Waveforms Page:** [concept-a/waveforms.html](../../../concepts/concept-a/waveforms.html)

---

## 5. Concept B: Instrument Laboratory (Corrected)

### 5.1 Aesthetic Metaphor & Architecture
Evokes an operational experimental physics facility (Caltech 40-meter prototype, LIGO control consoles). Features rack bays, calibrated reticle grids, and parameter potentiometers.

### 5.2 Corrections Applied
- **Eliminated Fictional Styling:** Removed pulsing status LEDs (`.led-dot`), fake alarm badges (`alarm-ok`, `alarm-warn`), and any claim of "LIVE" telemetry.
- **Explicit Synthetic Configuration:** Top bus explicitly labeled:
  `CONFIGURATION: SYNTHETIC REALIZATION 01 // 4.0s DURATION // fs = 2048 Hz // BAND: 20–512 Hz (ILLUSTRATIVE MODEL PARAMETERS)`.
  Telemetry bays labeled: `HISTORICAL BENCHMARK LEDGER (RECORDED AT STAGE 6D CLOSEOUT)`.
- **Single UI Accent Maintained:** Primary UI accent is strictly Cadmium Orange (`#C44D18` light / `#FF6A28` dark) for all interactive buttons, focus outlines, and active tabs. Channel 2 cyan (`#0D74CE` light / `#38BDF8` dark) is formally documented as a **non-accent data-series color** for the second trace channel.
- **Selective Typography Discipline:** Interface text, button labels, and explanatory prose now use `IBM Plex Sans`. Monospace (`IBM Plex Mono`) is strictly reserved for numerical values, technical identifiers, formulas, and channel tags.
- **TaylorF2 vs Time-Domain Distinction:** Channel 2 header and caption explicitly state that TaylorF2 is strictly frequency-domain inspiral, and that the time-domain trace is an illustrative surrogate.
- **Verified Derived Equations:** Parameter summary card updated with exact formulas, units, and Schwarzschild ISCO test-particle qualification.

### 5.3 Deliverables
- **Hero Page:** [concept-b/index.html](../../../concepts/concept-b/index.html)
- **Waveforms Page:** [concept-b/waveforms.html](../../../concepts/concept-b/waveforms.html)

---

## 6. Concept C: Contemporary Physics Atlas (Corrected)

### 6.1 Aesthetic Metaphor & Architecture
Draws inspiration from contemporary geometric gravitation monographs (Misner-Thorne-Wheeler, Kip Thorne). Expansive geometric canvas, quadrupole wavefront spirals, and multi-scale regime ribbons.

### 6.2 Corrections Applied
- **Corrected Merger Threshold Description:** Corrected Regime 03 description: replaced the statement that binary merger occurs at $r < 6M$ with a proper relativistic qualification:
  *\(r = 6M\) marks the test-particle Schwarzschild ISCO. In comparable-mass coalescences, strong mutual dynamical spacetime deformations alter the plunge; \(r = 6M\) is not a universal binary merger threshold. Merger is modeled by Numerical Relativity.*
- **Model Boundaries Stated:** Explicitly noted that TaylorF2 covers Regimes 01 and 02 only (inspiral), and does not model Regime 03 (merger) or Regime 04 (ringdown).
- **Plate II.B Trajectory Caption:** Updated Plate II.B caption to label $r = 6M$ as a test-particle heuristic cutoff rather than a physical boundary of binary coalescence.
- **Verified Parameter Derivations:** Updated geometric readout card to display $R_{\text{ISCO}} = 575.9\,\text{km}$ ($8.8596 \times 65.0\,\text{km}$) and $f_{\text{ISCO}} = 67.65\,\text{Hz}$ ($4397.17 / 65.0$).

### 6.3 Deliverables
- **Hero Page:** [concept-c/index.html](../../../concepts/concept-c/index.html)
- **Waveforms Page:** [concept-c/waveforms.html](../../../concepts/concept-c/waveforms.html)

---

## 7. Cross-Concept Architectural Comparison Matrix

| Evaluation Dimension | Concept A: Scientific Journal | Concept B: Instrument Laboratory | Concept C: Contemporary Physics Atlas |
| :--- | :--- | :--- | :--- |
| **Aesthetic Metaphor** | Classical scholarly treatise / Living Reviews publication | Caltech / Hanford observatory control console | Geometric gravitation atlas / contemporary physics monograph |
| **Layout Architecture** | Asymmetric 2-column (7:5) grid with academic margin notes | Modular rack bays with diagnostic configuration bus | Expansive full-width canvas with 4-regime ribbon |
| **Primary Palette (Light)** | Warm Cream (`#FBF9F5`) / Venetian Oxblood (`#8C2D19`) | Brushed Aluminum (`#EEF0F3`) / Cadmium Orange (`#C44D18`) | Celestial Bone (`#F7F5F0`) / Stellar Terracotta (`#B34726`) |
| **Primary Palette (Dark)** | Archival Black (`#111215`) / Warm Copper (`#D85838`) | Obsidian Void (`#0A0C0F`) / Radiant Orange (`#FF6A28`) | Deep Cosmic Void (`#0B0D13`) / Radiant Terracotta (`#E65E2E`) |
| **UI Accent Discipline** | Single oxblood/copper accent token | Single cadmium/radiant orange accent; non-accent data trace | Single stellar terracotta accent; non-accent cosmic blue trace |
| **Typographic Roles** | `Newsreader` (Narrative/Titles) + `IBM Plex Mono` (Notes) | `IBM Plex Sans` (UI/Buttons) + `IBM Plex Mono` (Telemetry) | `Newsreader` (Titles) + `IBM Plex Sans` (UI) + `Mono` (Coords) |
| **Waveform Presentation** | Engraved Plate figures with mathematical derivations | Dual CRT oscilloscope with phosphor beam & reticle | Radiating quadrupole spirals & multi-scale atlas |
| **Information Density** | Medium-High (Paced, linear reading flow) | Very High (Compact telemetry, dials, diagnostic bus) | Medium (Contemplative, spatial, geometric) |
| **Interactive Character** | Plate figure inspection & mathematical expanders | Stepped potentiometers & dynamic CRT beam adjustment | Multi-scale regime exploration & orbital tracking |
| **Model Qualifications** | Explicit Plate model scope notes & citation block | Prominent theoretical model disclaimer badge | Plate caption theoretical disclaimer & geometric unit labels |

---

## 8. Automated Visual Verification Evidence & Screenshot Audit

### 8.1 Precise Accounting of the 50 Screenshots
The test runner executes an exhaustive matrix:
- **8 Pages Audited:**
  1. Portal Index (`/index.html`)
  2. Side-by-Side Comparison Sheet (`/comparison.html`)
  3. Concept A Hero (`/concept-a/index.html`)
  4. Concept A Waveforms (`/concept-a/waveforms.html`)
  5. Concept B Hero (`/concept-b/index.html`)
  6. Concept B Waveforms (`/concept-b/waveforms.html`)
  7. Concept C Hero (`/concept-c/index.html`)
  8. Concept C Waveforms (`/concept-c/waveforms.html`)
- **3 Viewports:** Desktop ($1440 \times 900$), Tablet ($1024 \times 768$), Mobile ($390 \times 844$).
- **2 Themes:** Light (`data-theme="light"`) and Dark (`data-theme="dark"`).
- **Matrix Calculation:**
  $$8\text{ pages} \times 3\text{ viewports} \times 2\text{ themes} = 48\text{ screenshots}$$
- **Dynamic Interactive Verification Tests (+2 Screenshots):**
  1. *Concept B Interactive Slider*: $\mathcal{M}_c$ adjusted to $32.50\,M_\odot$; captured `concept_b_waveforms_interactive_mc325_dark.png` verifying dynamic reticle motion.
  2. *Concept C Interactive Slider*: $\mathcal{M}_c$ adjusted to $30.00\,M_\odot$; captured `concept_c_waveforms_interactive_mc300_light.png` verifying metric updates.
- **Total:** $48 + 2 = \mathbf{50}\text{ full-resolution screenshots}$.

### 8.2 Quality Gate Summary
- **Console Errors:** **0 errors**.
- **Page Exceptions:** **0 errors**.
- **Horizontal Overflows:** **0 overflows** across all viewports and themes.
- **Interactive State Tests:** **2 / 2 passed**.

---

## 9. Visual Contact Sheets & Direct Local URLs

### 9.1 Direct Browser URLs (Local HTTP Server Port 3005)
A background static HTTP server is running and serving the `concepts/` directory:
- **Master Comparison Sheet:** [http://localhost:3005/comparison.html](http://localhost:3005/comparison.html)
- **Master Portal Index:** [http://localhost:3005/index.html](http://localhost:3005/index.html)
- **Concept A (Hero):** [http://localhost:3005/concept-a/index.html](http://localhost:3005/concept-a/index.html)
- **Concept A (Waveforms):** [http://localhost:3005/concept-a/waveforms.html](http://localhost:3005/concept-a/waveforms.html)
- **Concept B (Hero):** [http://localhost:3005/concept-b/index.html](http://localhost:3005/concept-b/index.html)
- **Concept B (Waveforms):** [http://localhost:3005/concept-b/waveforms.html](http://localhost:3005/concept-b/waveforms.html)
- **Concept C (Hero):** [http://localhost:3005/concept-c/index.html](http://localhost:3005/concept-c/index.html)
- **Concept C (Waveforms):** [http://localhost:3005/concept-c/waveforms.html](http://localhost:3005/concept-c/waveforms.html)

### 9.2 Concise Contact Sheets
Two high-density visual contact sheets have been generated at $2240 \times 586\,\text{px}$:
1. **Home Concepts Contact Sheet:** [contact_sheet_home.png](../../screenshots/concepts/contact_sheet_home.png)  
   *(Compares Concept A, Concept B, and Concept C heroes side-by-side at 1440×900 desktop resolution)*
2. **Waveforms Concepts Contact Sheet:** [contact_sheet_waveforms.png](../../screenshots/concepts/contact_sheet_waveforms.png)  
   *(Compares Concept A, Concept B, and Concept C waveform inner pages side-by-side at 1440×900 desktop resolution)*

---

## 10. Confirmation of Exact Files Modified in this Turn

The following files were modified to apply the targeted corrections:
1. `concepts/concept-a/index.html` (Replaced fabricated folio with neutral technical report label).
2. `concepts/concept-a/waveforms.html` (Replaced fabricated folio, corrected exact component masses $m_1 = 36, m_2 = 29, M = 65, q = 1.241$).
3. `concepts/concept-b/index.html` (Removed LEDs, fake alarms, and "LIVE" claims; labeled synthetic configuration; enforced `IBM Plex Sans` on UI buttons; maintained single UI accent).
4. `concepts/concept-b/waveforms.html` (Labeled Channel 2 cyan as non-accent data trace; distinguished TaylorF2 from illustrative time domain; added physical equations & derivations; qualified Schwarzschild ISCO; enforced `IBM Plex Sans`).
5. `concepts/concept-c/index.html` (Corrected Regime 03 merger description: $r=6M$ is test-particle ISCO, not universal binary merger threshold; distinguished TaylorF2 scope).
6. `concepts/concept-c/waveforms.html` (Corrected Plate II.B caption; added derived geometric equations; qualified Schwarzschild ISCO; used exact $C_{\text{ISCO}} = 4397.17$).
7. `concepts/comparison.html` (Updated comparison matrix and layout descriptions).
8. `frontend/public/concepts/*` (Synced all updated prototype files).
9. `scripts/generate_contact_sheets.py` (Script to generate side-by-side contact sheets).
10. `reports/screenshots/concepts/` (Regenerated 50 full-resolution screenshots + 2 contact sheets).
11. `reports/GWLab_Phase0_Art_Direction_Concepts.md` (Updated report with complete evidence and corrections).

---

### Implementation Status Labels Summary
- **Implemented**: Targeted scientific and design corrections across all 3 visual concepts, comparison sheet, and contact sheet generator.
- **Tested**: Playwright Chromium test suite covering 8 pages $\times$ 3 viewports $\times$ 2 themes + 2 dynamic slider tests (50 total screenshots).
- **Validated**: 0 console errors, 0 horizontal overflows, verified parameter derivations and units, single UI accent discipline, contact sheets generated.
- **Proposed**: Concept selection by Project Owner and Reviewer to proceed into Phase 1 application component construction.
