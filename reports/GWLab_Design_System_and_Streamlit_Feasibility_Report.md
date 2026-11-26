# GWLab — Phase 3B: Design System and Streamlit Feasibility Gate Report

**Project:** Gravitational-Wave Matched Filtering and Bayesian Inference Pipeline  
**Application:** GWLab (Interactive Research Laboratory)  
**Deliverable:** Stage 3B Design System Implementation and Framework Feasibility Gate  
**Review Status:** Submitted for Independent Scientific and Architectural Review  
**Date:** November 2026  

---

## 1. Executive Summary

This report documents the implementation of the Phase 3B design system for **GWLab** and provides an evidence-based feasibility evaluation of Streamlit as the permanent application framework.

Following the design direction established by the reference project **Kerr Geodesics Lab** (`C:\Users\Soumitra\Desktop\Kerr-Geodesic-Integrator`), GWLab was transformed from an industrial, IBM Carbon-styled prototype into an editorial scientific research instrument. The default presentation is now a warm, publication-grade light paper palette (`#FAF9F6`), paired with a faithful dark inversion (`#0E1013`), terracotta accent typography (`#B94A24` light, `#EA580C` dark), serif headings (`Newsreader`), monospace tabular figures (`IBM Plex Mono`), and true typographical formatting (`\u2212`, `\u2009`, unicode superscripts).

Global application chrome was purged of emojis, loud caution banners, and development toolbars (`toolbarMode = "minimal"` in `.streamlit/config.toml`). All seven visitor-facing destinations were renamed to sentence-case titles (`Overview`, `Waveforms`, `Noise and PSD`, `Matched filtering`, `Inference`, `Experiments`, `Methods and provenance`). Representative pages (`Overview` and `Waveforms`) were redesigned to ensure that the primary analytical visualization occupies over 60% of the visual field above the fold at a standard 1440&times;900 desktop viewport.

### Framework Feasibility Verdict: CONDITIONAL GO for Internal Research / RECOMMENDED PIVOT to Next.js for Portfolio Showcase

| Evaluation Dimension | Verdict | Summary |
| :--- | :---: | :--- |
| **Visual Craft & Aesthetics** | **PASS (8.5/10)** | Streamlit can achieve a convincing editorial paper look via `.streamlit/config.toml` and targeted CSS injection. |
| **Above-the-Fold Density** | **PASS (8.0/10)** | Two-column layout (`[1.1, 2.9]`) places full interactive waveform plots and controls within 900px viewport without scrolling. |
| **Responsive Adaptation** | **PASS (8.0/10)** | Zero horizontal overflow at 390px, 1024px, and 1440px. Auto-collapsing sidebar provides clean mobile view. |
| **Accessibility & Contrast** | **PASS (10/10)** | Programmatic WCAG 2.1 AA compliance (all body text &ge; 5.5:1, headings &ge; 4.9:1, tabular figures &ge; 15:1). |
| **DOM Maintenance & Upgrades** | **RISK (5.5/10)** | Deep styling of widget internals requires targeting unstable `data-testid` attributes subject to breakage on minor version bumps. |
| **Reactivity & Client State** | **ACCEPTABLE (7.0/10)** | Script re-run model is fast for analytical waveforms (~35 ms) but induces iframe re-draw flashes on theme toggles. |

**Final Recommendation:** Maintain Streamlit through Phase 3 and Phase 4 for rapid internal algorithm exploration and signal analysis. However, for a permanent, public-facing portfolio instrument matching the exact interactive fidelity of Kerr Geodesics Lab (smooth slider dragging, sub-pixel widget layout, zero re-run flicker, client-side WebAssembly/Wasm corner plots), execute an architectural migration to Next.js/React after Phase 4.

---

## 2. Visual Design System Translation

### 2.1 Design Token Architecture

The design system directly translates the token hierarchy of Kerr Geodesics Lab (`frontend/src/globals.css`) into Python dictionary structures (`src/gwlab/theme.py`) and native Streamlit configuration (`.streamlit/config.toml`).

```
Kerr Geodesics Lab (CSS Variables)    -->    GWLab Design Tokens (Python)
----------------------------------           ----------------------------
--background: #FAF9F6 (Paper)                THEME_TOKENS["light"]["bg_primary"]
--surface:    #FFFFFF (Card)                 THEME_TOKENS["light"]["bg_surface"]
--surface-hover: #F5F3ED                     THEME_TOKENS["light"]["bg_secondary"]
--border:     #DEDAD2 (Hairline)             THEME_TOKENS["light"]["border"]
--text:       #20201E (High Contrast)        THEME_TOKENS["light"]["text_primary"]
--text-secondary: #55534F                    THEME_TOKENS["light"]["text_secondary"]
--muted:      #68655F                        THEME_TOKENS["light"]["text_muted"]
--accent:     #B94A24 (Terracotta)           THEME_TOKENS["light"]["accent"]
--accent-subtle: #FBF4EE                     THEME_TOKENS["light"]["accent_subtle"]
--accent-border: #E3A894                     THEME_TOKENS["light"]["accent_border"]
```

### 2.2 Color Tokens and Programmatic WCAG 2.1 AA Contrast Ratios

All color combinations were verified against WCAG 2.1 Level AA standards (minimum 4.5:1 for normal text, 3.0:1 for large text/headings).

| Color Role | Light Theme (Default) | Dark Theme (Inversion) | Contrast on Background (Light) | Contrast on Background (Dark) | WCAG Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Background** | `#FAF9F6` | `#0E1013` | N/A (Canvas) | N/A (Canvas) | Reference |
| **Surface / Card Background** | `#FFFFFF` | `#15181C` | N/A (Surface) | N/A (Surface) | Reference |
| **Primary Body Text** | `#20201E` | `#ECEFF1` | **15.82 : 1** | **16.14 : 1** | **AAA Pass** |
| **Secondary Editorial Text** | `#55534F` | `#B0B5BF` | **7.24 : 1** | **9.12 : 1** | **AAA Pass** |
| **Muted Caption Text** | `#68655F` | `#9CA3AF` | **5.51 : 1** | **7.43 : 1** | **AA Pass** |
| **Terracotta Accent** | `#B94A24` | `#EA580C` | **4.93 : 1** | **5.36 : 1** | **AA Pass** |
| **Hairline Borders** | `#DEDAD2` | `#262930` | 1.35 : 1 (Non-text) | 1.42 : 1 (Non-text) | Structural |

### 2.3 Typography & Font Hierarchy

1. **Editorial Titles & Headings:** `Newsreader`, Georgia, Cambria, serif (`font-weight: 500`, `letter-spacing: -0.015em`). Provides a dignified, academic monograph tone distinct from SaaS dashboards.
2. **Controls & Body Copy:** `IBM Plex Sans`, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif (`font-weight: 400`, `line-height: 1.55`).
3. **Data Figures, Tables & Code:** `IBM Plex Mono`, ui-monospace, SFMono-Regular, Consolas, monospace (`font-variant-numeric: tabular-nums`). Guarantees vertical alignment of decimals and numerical parameters.

### 2.4 Typographical Scientific Formatter (`src/gwlab/format.py`)

A standalone formatting module was implemented to enforce editorial typesetting standards across all numerical outputs:
- **True Minus Sign:** Replaces standard ASCII hyphen-minus (`-`, U+002D) with the true typographical mathematical minus (`\u2212`, `−`), eliminating unsightly hyphen artifacts in negative values.
- **Unicode Superscript Exponents:** Scientific notation formats exponents as true superscripts (e.g., $1.234 \times 10^{\u221219}$ via `1.234 × 10⁻¹⁹`) rather than raw machine notation (`1.234e-19`).
- **Thin Space Unit Separation:** Values and units are coupled with a thin space (`\u2009`, U+2009, e.g., `28.10 M_sun`, `515.0 s`, `201646.1413 ± 0.2374 nats`).
- **Handling of Singularities:** Robust formatting for `NaN`, `+inf` ($+\infty$), `-inf` ($\u2212\infty$), and exact zero.

---

## 3. Global Chrome and Information Architecture

### 3.1 Streamlit Native Theming (`.streamlit/config.toml`)

To eliminate default Streamlit styling before DOM render and prevent bright flashes or unwanted red accenting, `.streamlit/config.toml` was configured:

```toml
[client]
toolbarMode = "minimal"
showErrorDetails = false

[theme]
base = "light"
primaryColor = "#B94A24"
backgroundColor = "#FAF9F6"
secondaryBackgroundColor = "#F5F3ED"
textColor = "#20201E"
font = "sans serif"

[browser]
gatherUsageStats = false
```

Setting `toolbarMode = "minimal"` permanently suppresses the "Deploy" button and hamburger menu chrome, presenting GWLab as a standalone scientific workstation.

### 3.2 Purge of Decorative Clutter

1. **Emoji Removal:** All emojis were removed from page titles, navigation sections, cards, badges, and buttons.
2. **Safety Banner Restraint:** The repetitive, high-saturation yellow safety banner was removed from every page. It was replaced by a quiet, restrained archive pill (`Reference archive`) in the persistent header and a subtle, one-line caption where appropriate.
3. **Sentence-Case Destination Hierarchy:**
   The visitor-facing destinations were renamed to clean sentence-case titles in `src/gwlab/config.py` and `src/gwlab/app.py`:
   1. `Overview` (System dashboard & authoritative benchmarks)
   2. `Waveforms` (TaylorF2 3.5PN inspection & chirp dynamics)
   3. `Noise and PSD` (Spectral estimation & detector sensitivity)
   4. `Matched filtering` (Complex SNR time series & chi-squared veto)
   5. `Inference` (Stage 6C posterior forensics & circular phase)
   6. `Experiments` (Multi-seed repeatability & model comparison)
   7. `Methods and provenance` (Mathematical formulations & audit archive)

---

## 4. Representative Pages Redesign

### 4.1 Overview (`src/gwlab/pages/1_overview.py`)

- **Executive Context:** Clean narrative establishing GWLab's purpose: deterministic examination of verified gravitational-wave inference results.
- **De-emphasized Jargon:** Replaced internal project-management labels ("Pathway A", "Milestone Governance") with clear scientific classifications ("Reference parameter estimation benchmark", "Independent-seed repeatability baseline").
- **Card Layout:** Two equal-width cards comparing Stage 6C Run A anchor results with the historical independent-seed comparison (Seed 8101 vs Seed 8201).
- **Tabular Figures:** Numbers formatted with `src/gwlab/format.py` using tabular numerals, true minus signs, and thin spaces.
- **Computational Ledger:** Replaced loud red/green metric cards with a restrained typographic readout and thin terracotta progress indicator.
- **Traceability Matrix:** Full provenance audit table mapping each result to its repository JSON/NPZ artifact and validation status.

### 4.2 Waveforms (`src/gwlab/pages/2_signal_lab.py`)

- **Visual Center Dominance (>60%):** Configured two-column layout `st.columns([1.1, 2.9])`. The visualization panel occupies 72.5% of horizontal layout space.
- **Zero-Scroll Viewport Alignment:** Above-the-fold chart height is locked to 410px. At 1440&times;900 desktop resolution, the entire waveform chart, ISCO cutoff line, analysis band shading, and tabs fit within the upper viewport without scrolling.
- **Compact Numerical Controls:** Controls use `st.number_input` with exact step sizes, formats, and parameter ranges, supplemented by an expander for grid parameters ($T$, $f_s$, PN order).
- **Mathematical Rigor:** Analytical formulas are typeset in LaTeX via `st.latex(...)` at the top of each tab:
  - Amplitude: $|\tilde{h}(f)| = \mathcal{A} f^{-7/6} \Theta(f_{\mathrm{ISCO}} - f)$
  - Phase: $\Psi(f) = 2\pi f t_c - \phi_c - \frac{\pi}{4} + \frac{3}{128\eta}(\pi M f)^{-5/3} \sum_{i=0}^7 \alpha_i (\pi M f)^{i/3}$
  - Chirp Time: $\tau(f) = -\frac{1}{2\pi} \frac{\mathrm{d}\Psi_{\mathrm{int}}}{\mathrm{d}f} \approx \frac{5}{256} \mathcal{M}_c^{-5/3} (\pi f)^{-8/3}$
  - Quadratures: $\tilde{h}(f) = \mathrm{Re}[\tilde{h}(f)] + i\,\mathrm{Im}[\tilde{h}(f)]$
- **Shared Plotly Template:** Plot background matches panel surface (`#FFFFFF` light, `#15181C` dark) with hairline gridlines (`#E8E4DC` / `#1E2127`) and terracotta trace line (`#B94A24` / `#EA580C`). Logarithmic frequency axes display clean decade ticks (`20`, `50`, `100`, `200`, `500`, `1k`).

---

## 5. Streamlit Feasibility Evaluation

The Phase 3B implementation tested whether Streamlit can support a research-grade scientific laboratory without brittle DOM hacks across seven explicit criteria.

### Criterion 1: Visual Art Direction & Craft
- **Feasibility:** **FEASIBLE WITH RESTRAINT**
- **Assessment:** Streamlit's native theme configuration (`config.toml`) handles base colors, primary accents, and font families cleanly. Paired with modular CSS injection in `inject_custom_css()`, GWLab convincingly achieves the paper-like editorial aesthetic of Kerr Geodesics Lab.
- **Limitation:** Streamlit does not provide native APIs to style widget internals (e.g., number input stepper buttons, slider tracks, or tab border highlights). Styling them requires targeting CSS class patterns (`div[data-testid="stNumberInput"]`) or emotion-cache hashes (`.st-emotion-cache-...`), which are private implementation details.

### Criterion 2: Layout Control & Vertical Rhythm
- **Feasibility:** **PARTIALLY FEASIBLE**
- **Assessment:** Using custom column proportions (`st.columns([1.1, 2.9])`) successfully allocates over 70% of screen width to visualization.
- **Limitation:** Streamlit enforces rigid vertical gaps (default 1rem between block elements). Compacting the layout to guarantee above-the-fold visibility required overriding `.block-container` top padding (`1.8rem`) and setting explicit Plotly heights (`410px`). Streamlit lacks flexbox alignment controls for widgets inside columns.

### Criterion 3: Responsive Behavior & Multi-Device Degradation
- **Feasibility:** **FEASIBLE**
- **Assessment:** Tested across 1440&times;900 (desktop), 1024&times;768 (tablet), and 390&times;844 (mobile). All viewports demonstrated **zero horizontal overflow** (`scrollWidth == clientWidth`).
- **Limitation:** On mobile devices (390px), `initial_sidebar_state="expanded"` completely covers the screen with an overlay drawer. Setting `initial_sidebar_state="auto"` resolved this, collapsing the sidebar by default on mobile screens. However, Streamlit columns collapse linearly into a single vertical stack, which places controls above charts on mobile, requiring users to scroll down to view waveforms.

### Criterion 4: Performance, Reactivity, and Re-run Model
- **Feasibility:** **FEASIBLE FOR ANALYTICAL WAVEFORMS / QUESTIONABLE FOR LARGE DATASETS**
- **Assessment:** For TaylorF2 3.5PN strain synthesis, evaluating 8,193 frequency bins in NumPy and rendering Plotly takes **32–45 ms** per parameter update. This is imperceptible to users on typical desktop hardware.
- **Limitation:** Because Streamlit executes top-to-bottom on every widget interaction, every parameter adjustment re-runs the script. For Phase 4 (matched filtering across millions of samples) and Phase 5 (corner plot generation across 11,951 samples), this execution model will introduce noticeable latency (300–800 ms) unless heavy caching (`@st.cache_data`) is used. Furthermore, Plotly charts re-mount on script re-run, causing a subtle flicker.

### Criterion 5: Plotting & Data Visualization Fidelity
- **Feasibility:** **EXCELLENT**
- **Assessment:** Plotly graph objects integrated through `st.plotly_chart` render publication-grade scientific figures. The centralized `get_plotly_layout()` helper ensures that axis grids, tick typography, hovertemplates, and ISCO cutoff lines match the global editorial design tokens.
- **Limitation:** Hover tooltips in Plotly operate in client-side canvas/SVG; they cannot easily trigger cross-filtering or coordinated multi-view selections across separate Streamlit components without complex custom component extensions (`streamlit-plotly-events`).

### Criterion 6: Accessibility, Typography, and Contrast
- **Feasibility:** **EXCELLENT (FULLY COMPLIANT)**
- **Assessment:** All text colors exceed WCAG 2.1 Level AA requirements. Light theme body text achieves **15.82:1**; dark theme achieves **16.14:1**. Tabular figures are styled with `font-variant-numeric: tabular-nums` to ensure exact column alignment.
- **Limitation:** Keyboard focus rings in Streamlit depend on browser defaults and custom CSS overrides.

### Criterion 7: Maintenance Burden & Framework Fragility
- **Feasibility:** **MODERATE RISK**
- **Assessment:** The code is modularly structured (`format.py`, `theme.py`, `components.py`, `physics.py`). Zero third-party hack plugins were installed.
- **Limitation:** Streamlit minor version upgrades frequently alter internal DOM hierarchies (e.g., changing `data-testid` names or emotion-cache wrapper divs). Future Streamlit updates could silently disrupt custom hairline borders, definition lists, or compact padding.

---

## 6. Architecture Comparison Matrix: Streamlit vs. Next.js / React

To inform the Phase 3/4 roadmap and portfolio transition, the engineering trade-offs between continuing in Streamlit versus porting to a Next.js/React architecture (identical to Kerr Geodesics Lab) were evaluated:

| Architectural Requirement | Streamlit (Current GWLab) | Next.js + React + Tailwind + Lucide (Kerr Architecture) |
| :--- | :--- | :--- |
| **Development Velocity** | **Very High**: Pure Python, zero API scaffolding, immediate data layer binding. | **Moderate**: Requires TypeScript frontend + Python FastAPI/Wasm backend. |
| **Visual Art Direction** | **Good**: Achieved via CSS overrides and `config.toml`, but brittle under the hood. | **Flawless**: Pixel-perfect control over every DOM node, transition, and border via Tailwind. |
| **Interaction Latency** | **30–100 ms**: Server roundtrip over WebSocket on every slider/input change. | **< 16 ms (60 FPS)**: Instant client-side state updates; sliders glide smoothly. |
| **Plotly / Canvas Integration** | **Good**: Re-mounts on server re-run; slight flicker on theme toggle. | **Seamless**: Client-side canvas/WebGL without full-page re-mounts; zero flicker. |
| **Mobile UX** | **Acceptable**: Linear column collapse; sidebar overlay modal. | **Optimized**: Responsive drawer/bottom-sheet navigation, pinch-zoom, and responsive chart resizing. |
| **Corner Plot / Multi-View** | **Slow**: Static image generation or heavy Plotly scatter matrix re-render. | **Instant**: Canvas-based interactive cross-filtering with density contours. |
| **Maintenance & Portability** | **Tied to Streamlit**: Upgrades may break DOM selectors. | **Standard Web Platform**: Durable, framework-standard CSS and TypeScript components. |

---

## 7. Gate Decision and Roadmap Recommendation

### Gate Decision: CONDITIONAL GO (Phases 3–4) & SCHEDULED PORTFOLIO PIVOT (Phase 5+)

1. **Immediate Execution (Phases 3 & 4): CONDITIONAL GO on Streamlit**
   - **Rationale:** The scientific workflows for Phase 3 (Noise and PSD estimation) and Phase 4 (Matched filtering SNR time series) require tight coupling to the established Python engine (`src/gw_inference/psd.py`, `src/gw_inference/filter.py`). Building and validating these analytical tools in Streamlit maximizes velocity while maintaining full scientific rigor.
   - **Condition:** Enforce strict adherence to Phase 3B design tokens (`theme.py`), typography, and formatting standards (`format.py`). Forbid unvetted CSS hacks.

2. **Portfolio & Publication Showcase: SCHEDULED PIVOT to Next.js / React**
   - **Rationale:** To achieve a public-facing research portfolio piece that matches the world-class visual craft of Kerr Geodesics Lab, a pure React/Tailwind frontend backed by a lightweight Python/FastAPI execution service or client-side WebAssembly will be required for Phase 5 (Bayesian Posterior Forensics) and Phase 6 (Cross-Run Experiment Workbench).
   - **Action Item:** Upon completion of Phase 4 matched-filtering verification, evaluate migrating GWLab into a unified monorepo frontend sharing Kerr Geodesics Lab's design tokens and UI component library.

---

## 8. Verification and Visual Evidence Archive

### 8.1 Automated Regression Test Suite

All unit, regression, and safety tests pass cleanly:

```
tests/test_gwlab_shell.py::test_gwlab_package_metadata PASSED            [  3%]
tests/test_gwlab_shell.py::test_gwlab_config_and_page_registry PASSED    [  6%]
tests/test_gwlab_shell.py::test_gwlab_safety_boundary_no_dynesty PASSED  [  9%]
tests/test_gwlab_shell.py::test_gwlab_theme_tokens_structure PASSED      [ 12%]
tests/test_gwlab_shell.py::test_no_sampler_launch_controls_in_pages PASSED [ 16%]
tests/test_gwlab_shell.py::test_app_entrypoint_exists PASSED             [ 19%]
tests/test_gwlab_data.py::test_resolve_safe_artifact_path_valid PASSED   [ 22%]
tests/test_gwlab_data.py::test_resolve_safe_artifact_path_rejections PASSED [ 25%]
tests/test_gwlab_data.py::test_load_stage6c_result_authoritative_values PASSED [ 29%]
tests/test_gwlab_data.py::test_load_aggregate_ledger_authoritative_values PASSED [ 32%]
tests/test_gwlab_data.py::test_load_posterior_samples_authoritative_arrays PASSED [ 35%]
tests/test_gwlab_data.py::test_load_seed_comparison_pair_authoritative_values PASSED [ 38%]
tests/test_gwlab_data.py::test_loader_rejects_malformed_json PASSED      [ 41%]
tests/test_gwlab_data.py::test_loader_rejects_missing_json_keys PASSED   [ 45%]
tests/test_gwlab_data.py::test_loader_rejects_malformed_weights PASSED   [ 48%]
tests/test_gwlab_data.py::test_loader_rejects_non_finite_arrays PASSED   [ 51%]
tests/test_gwlab_physics.py::TestPhysicalParameterMappings::test_stage6c_injection_mass_conversion PASSED [ 54%]
tests/test_gwlab_physics.py::TestPhysicalParameterMappings::test_equal_mass_boundary_eta_025 PASSED [ 58%]
tests/test_gwlab_physics.py::TestPhysicalParameterMappings::test_invalid_parameters_rejected PASSED [ 61%]
tests/test_gwlab_physics.py::TestTaylorF2StrainGeneration::test_deterministic_waveform_output PASSED [ 64%]
tests/test_gwlab_physics.py::TestTaylorF2StrainGeneration::test_frequency_grid_and_array_shapes PASSED [ 67%]
tests/test_gwlab_physics.py::TestTaylorF2StrainGeneration::test_finiteness_over_valid_range PASSED [ 70%]
tests/test_gwlab_physics.py::TestTaylorF2StrainGeneration::test_isco_cutoff_enforcement PASSED [ 74%]
tests/test_gwlab_physics.py::TestTaylorF2StrainGeneration::test_chirp_time_series_evaluation PASSED [ 77%]
tests/test_gwlab_physics.py::TestSafetyBoundaries::test_no_dynesty_in_modules PASSED [ 80%]
tests/test_gwlab_physics.py::TestSafetyBoundaries::test_signal_lab_source_syntax_and_no_samplers PASSED [ 83%]
tests/test_gwlab_format.py::test_to_superscript PASSED                   [ 87%]
tests/test_gwlab_format.py::test_format_float PASSED                     [ 90%]
tests/test_gwlab_format.py::test_format_scientific PASSED                [ 93%]
tests/test_gwlab_format.py::test_format_with_unit PASSED                 [ 96%]
tests/test_gwlab_format.py::test_format_uncertainty PASSED               [100%]

============================= 31 passed in 5.96s ==============================
```

- **Safety Invariant:** `dynesty` and `gw_inference.sampler` are verified absent from `sys.modules` across all GWLab execution paths.
- **Numerical Determinism:** Waveform strain generation produces bitwise identical complex arrays across repeated evaluations.

### 8.2 Visual Screenshot Manifest

All twelve verification screenshots were captured via headless Playwright automation and archived in `reports/figures/`:

| Target Page | Viewport | Theme | Screenshot File | Observations |
| :--- | :---: | :---: | :--- | :--- |
| **Overview** | 1440 &times; 900 | Light | [gwlab_phase3b_1440x900_light.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_1440x900_light.png) | Paper baseline `#FAF9F6`, Newsreader serif title, two-column benchmark cards, tabular numbers. |
| **Overview** | 1440 &times; 900 | Dark | [gwlab_phase3b_1440x900_dark.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_1440x900_dark.png) | Dark inversion `#0E1013`, crisp borders `#262930`, terracotta accent `#EA580C`. |
| **Overview** | 1024 &times; 768 | Light | [gwlab_phase3b_1024x768_light.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_1024x768_light.png) | Tablet portrait mode; cards maintain structure; zero horizontal overflow. |
| **Overview** | 1024 &times; 768 | Dark | [gwlab_phase3b_1024x768_dark.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_1024x768_dark.png) | High-contrast dark theme on tablet; legible metrics. |
| **Overview** | 390 &times; 844 | Light | [gwlab_phase3b_390x844_light.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_390x844_light.png) | Mobile viewport; zero horizontal scroll (`scrollWidth == 390`); clean column collapse. |
| **Overview** | 390 &times; 844 | Dark | [gwlab_phase3b_390x844_dark.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_390x844_dark.png) | Mobile dark theme; auto-collapsed sidebar. |
| **Waveforms** | 1440 &times; 900 | Light | [gwlab_phase3b_waveforms_1440x900_light.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_waveforms_1440x900_light.png) | **Above-the-fold visual center (>70% width)**; waveform plot, ISCO line, and band visible without scroll. |
| **Waveforms** | 1440 &times; 900 | Dark | [gwlab_phase3b_waveforms_1440x900_dark.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_waveforms_1440x900_dark.png) | Dark theme waveform plot; dark surface `#15181C`; bright terracotta trace line. |
| **Waveforms** | 1024 &times; 768 | Light | [gwlab_phase3b_waveforms_1024x768_light.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_waveforms_1024x768_light.png) | Tablet layout; responsive Plotly chart reflow. |
| **Waveforms** | 1024 &times; 768 | Dark | [gwlab_phase3b_waveforms_1024x768_dark.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_waveforms_1024x768_dark.png) | Tablet dark layout; LaTeX math rendered crisply. |
| **Waveforms** | 390 &times; 844 | Light | [gwlab_phase3b_waveforms_390x844_light.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_waveforms_390x844_light.png) | Mobile layout; controls stack cleanly above the waveform plot. |
| **Waveforms** | 390 &times; 844 | Dark | [gwlab_phase3b_waveforms_390x844_dark.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/figures/gwlab_phase3b_waveforms_390x844_dark.png) | Mobile dark layout; zero horizontal overflow. |

---

## 9. Next Steps (Phase 3C / Phase 4 Planning)

1. **Phase 3C: Noise and PSD Interactive Laboratory**
   - Ingest analytical aLIGO Zero-Detuned High Power sensitivity curve.
   - Implement interactive Welch periodogram inspector with segment length ($T_{\mathrm{seg}}$), overlap percentage (0% to 75%), and window tapering (Tukey $\alpha$).
   - Display empirical noise realization alongside Welch PSD and compute variance inflation factors ($\nu_{\mathrm{eff}}$).
2. **Phase 4: Matched Filtering Laboratory**
   - Implement complex SNR time-series synthesis $z(t)$ via IFFT.
   - Interactive time-lag recovery ($\hat{t}_c$, $\hat{\phi}_c$) and template normalization.
   - Implement chi-squared veto ($\chi^2_r$) frequency banding visualization.
3. **Streamlit Maintenance Protocol:**
   - Preserve `.streamlit/config.toml` as single source of truth for base theming.
   - Keep `src/gwlab/format.py` as mandatory formatter for all future analytical readouts.
   - Leave the Streamlit server running on `http://localhost:8502` for interactive user inspection.
