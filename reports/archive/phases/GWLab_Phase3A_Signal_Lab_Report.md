# GWLab — Phase 3A: Signal Lab Implementation & Phase 2 Corrections Report

**Document Version:** 1.0.0-PROD  
**Date:** 2026-11-19  
**Role:** Implementation Engineer  
**Project:** Gravitational-Wave Matched-Filtering & Bayesian Inference (`gw-inference` / `gwlab`)  
**Target Environment:** Python 3.13.5 (`win_amd64`), Streamlit 1.67.0, Plotly 7.1.0, Playwright  
**Status:** PHASE 3A COMPLETE & VERIFIED • ZERO SAMPLER RUNS • PURE DETERMINISTIC EVALUATION  
**Deliverable Path:** `reports/GWLab_Phase3A_Signal_Lab_Report.md`

---

## 1. Executive Summary

Phase 3A of **GWLab** has been successfully implemented, validated, and verified. Following acceptance of Phase 2, this milestone completes the targeted documentation and UI corrections for Phase 2 and delivers **Signal Lab** (`src/gwlab/pages/2_signal_lab.py`), a deterministic, interactive scientific workstation for frequency-domain gravitational waveform inspection and post-Newtonian chirp dynamics.

The primary achievements of this release include:
1. **Targeted Phase 2 Corrections:**
   - Updated the Overview dashboard and report to display **"Remaining recorded budget at Stage 6D closeout"** with an explicit disclaimer that this is a historical ledger snapshot, not live resource telemetry and not authorization to execute experiments.
   - Corrected the Stage 6C Run A timing display to authoritative ledger values: **Wall: 514.95 s (515.0 s)** and **CPU: 473.16 s**.
   - Standardized the $8.92$ metric label across UI cards, tables, and reports to: **"Observed evidence discrepancy / quadrature-combined reported internal uncertainty"**, emphasizing that it does not represent a calibrated frequentist significance or population-level variance estimate.
   - Narrowed the artifact preservation statement to confirm that **no historical artifacts were intentionally modified**.
2. **Reuse of Core TaylorF2 Waveform Engine:**
   - Reused the pure, deterministic TaylorF2 3.5PN implementation in `src/gw_inference/waveform.py` without code duplication or alterations.
   - Verified that the core engine is pure, thread-safe, SI-consistent, and operates with zero dependencies on likelihood, noise, PSD, or nested sampling machinery.
3. **Physical Parameter Mapping & Boundary Defense (`src/gwlab/physics.py`):**
   - Implemented exact, invertible conversions between $(\mathcal{M}_c, \eta)$ and component masses $(m_1, m_2, M)$ with explicit SI unit handling.
   - Enforced numerical stability at the equal-mass boundary $\eta = 0.25$, guarding the radicand $\max(0, 1 - 4\eta)$ against floating-point roundoff errors.
4. **Interactive Signal Lab Workstation (`src/gwlab/pages/2_signal_lab.py`):**
   - Implemented responsive parameter controls for $\mathcal{M}_c \in [10, 60]\,M_\odot$, $\eta \in [0.10, 0.25]$, $d_L \in [50, 2000]\,\text{Mpc}$, $\phi_c \in [0, 2\pi]\,\text{rad}$, and $t_c \in [0, 4]\,\text{s}$.
   - Provided one-click reset to the verified **Stage 6C Run A Injection Benchmark** ($\mathcal{M}_c = 28.095556\,M_\odot, \eta = 0.247101, d_L = 410.0\,\text{Mpc}, t_c = 2.00\,\text{s}, \phi_c = 0.00\,\text{rad}$), with explicit labels distinguishing injection benchmarks from user selections.
   - Delivered four tabbed Plotly scientific visualizations:
     * **Strain Amplitude $|\tilde{h}(f)|$:** Log-log frequency spectrum with $f^{-7/6}$ power-law decay, active band shading $[20, 512]\,\text{Hz}$, and Schwarzschild ISCO cutoff line.
     * **Fourier Phase $\Psi(f)$:** Unwrapped continuous phase trajectory, wrapped principal value $[-\pi, \pi]$, and stacked comparison illustrating the $-2\pi t_c$ time-shift slope.
     * **Inspiral Chirp Time $\tau(f)$:** Physical time-frequency diagnostic comparing leading Newtonian $\tau_N(f) \propto f^{-8/3}$ to full 3.5PN inspiral duration $\tau_{3.5\rm PN}(f) = -\frac{1}{2\pi} \frac{d\Psi_{\rm int}}{df}$.
     * **Strain Quadratures:** Re/Im interference fringes over $[25, 60]\,\text{Hz}$.
5. **Rigorous Verification & Testing:**
   - Authored 10 dedicated physics and safety tests (`tests/test_gwlab_physics.py`), passing in $3.01\,\text{s}$.
   - Executed combined GWLab test suite (26 tests passing in $7.07\,\text{s}$).
   - Executed full scientific regression suite (165 tests passing in $9.21\,\text{s}$).
   - Conducted automated headless browser verification with Playwright (Microsoft Edge / Chromium), verifying DOM rendering, Plotly chart generation, tab switching, and parameter reset with **0 browser console errors**.
6. **Strict Execution Safety Boundary:**
   - Zero likelihood evaluations, zero noise simulations, zero PSD estimations, zero matched filtering calculations, and zero nested sampler (Dynesty) runs were executed.
   - `src/gw_inference/` and historical artifacts remain completely unmodified.

---

## 2. Phase 2 Corrections Applied

In accordance with project audit instructions, four targeted corrections were applied to `src/gwlab/pages/1_overview.py` and `reports/GWLab_Phase2_Overview_and_Data_Ingestion_Report.md`:

### 2.1 Recorded Budget Nomenclature
- **Previous Text:** "Preserved Budget Reserve" / "exact live remaining budget".
- **Corrected Text:**
  ```
  Remaining recorded budget at Stage 6D closeout: 712,593 Calls
  (CPU: 1,326.8 s • Wall: 2,485.1 s)
  Historical ledger snapshot; not live resource telemetry and not authorization to execute experiments.
  ```
- **Rationale:** Clarifies that the recorded reserve is a frozen ledger snapshot from the Stage 6D closeout gate, preventing any misinterpretation as live background telemetry or authorization to run new sampling experiments.

### 2.2 Authoritative Run A Wall Time
- **Previous Value:** $514.8\,\text{s}$ (derived from single un-reconciled JSON key `wall_time_s = 514.834`).
- **Corrected Values:**
  * **Wall Time:** $\mathbf{514.95\,\text{s}}$ (or $\mathbf{515.0\,\text{s}}$ at one decimal place).
  * **CPU Time:** $\mathbf{473.16\,\text{s}}$.
- **Rationale:** Matches the authoritative aggregate resource ledger (`stage6c_aggregate_ledger.json` keys `cumulative_wall_s = 514.9468...` and `cumulative_cpu_s = 473.15625`), which reconciled worker execution time with supervisor watchdog telemetry.

### 2.3 Statistical Labeling of the 8.92 Metric
- **Previous Text:** "Discrepancy Ratio ($z_{\rm rep}$)" / unqualified z-score connotations.
- **Corrected Label:**
  $$\mathbf{Observed\ evidence\ discrepancy\ /\ quadrature\text{-}combined\ reported\ internal\ uncertainty = 8.92}$$
- **Mandatory Caveat Displayed:**
  The metric value $8.92$ is explicitly documented as the observed difference $|\Delta\ln\mathcal{Z}| = 2.9720\,\text{nats}$ divided by the quadrature-combined reported internal uncertainties $\sigma_\Delta = 0.3332\,\text{nats}$ across two independent-seed runs under the frozen physical inference problem (Realization 01, Model 2 Whittle likelihood). It is **not** a calibrated frequentist significance, does **not** estimate a population-level seed variance (which has only 1 degree of freedom), and does **not** establish general sampler failure.

### 2.4 Artifact Integrity Statement
- **Previous Claim:** "All historical experiment JSON files, NPZ arrays, and ledger files ... remain bit-for-bit identical."
- **Corrected Statement:** Replaced with the narrower, verified claim:
  **"All historical experiment JSON files, NPZ arrays, and ledger files were accessed strictly in read-only mode (`r` or `allow_pickle=False` NPZ read); no historical artifacts were intentionally modified."**

---

## 3. Existing Waveform Engine Inspection & Reuse

### 3.1 Core Waveform Implementation (`src/gw_inference/waveform.py`)
Rather than duplicating mathematical models in the UI layer, Signal Lab reuses the existing, audited waveform engine:
- **Primary Waveform Function:** `taylorf2_fd_waveform(freqs, m1_kg, m2_kg, distance_m, phi_c=0.0, f_end=None, pn_order=3.5) -> np.ndarray`
- **Time Translation:** `apply_coalescence_time_shift(waveform, freqs, t_c) -> np.ndarray`
- **Mass Conversion:** `component_masses_to_chirp_mass(m1_kg, m2_kg) -> Tuple[float, float, float]`
- **ISCO Cutoff:** `schwarzschild_isco_frequency(m_total_msun) -> float`
- **Termination Frequency:** `effective_termination_frequency(f_end, f_isco) -> float`
- **Amplitude Prefactor:** `taylorf2_amplitude_prefactor(chirp_mass_kg, distance_m) -> float`
- **Chirp Time Diagnostics:** `newtonian_chirp_time(f, chirp_mass_kg)` and `taylorf2_35pn_chirp_time(f, m1_kg, m2_kg)`

### 3.2 Pure & Deterministic Execution Verification
Inspection confirmed:
1. **Purity:** Functions depend exclusively on input arguments and physical constants defined in `gw_inference.constants`. They possess no internal mutable state, no caching side-effects, and no random number generation.
2. **Deterministic Output:** Identical inputs produce bit-for-bit identical complex arrays (`np.testing.assert_array_equal` verified in unit tests).
3. **Zero Likelihood / Sampler Coupling:** The module contains zero imports of `dynesty`, `gw_inference.sampler`, `gw_inference.likelihood`, `gw_inference.noise`, or `gw_inference.psd`.

---

## 4. Scientific Conventions & Physical Parameter Mapping

### 4.1 Post-Newtonian Waveform Conventions
- **Approximation:** Stationary Phase Approximation (SPA) restricted to the leading-order Newtonian quadrupole amplitude with 3.5PN post-Newtonian Fourier phase.
- **Fourier Phase Convention:**
  $$\tilde{h}(f) = A f^{-7/6} \exp\left[-i\left(\Psi_{\rm TF2}(f) + 2\pi f t_c\right)\right] \quad \text{for } 0 < f \le f_{\rm term}$$
  $$\tilde{h}(f) = 0 \quad \text{for } f > f_{\rm term}$$
- **Intrinsic Phase Expansion:**
  $$\Psi_{\rm int}(f) = \frac{3}{128 \eta v^5} \sum_{i=0}^{7} \alpha_i(v) v^i$$
  where $v = \left(\frac{\pi G M f}{c^3}\right)^{1/3}$ and coefficients $\alpha_0$ through $\alpha_7$ follow the frozen Group C specification.
- **ISCO Cutoff:** Evaluated using the Schwarzschild innermost stable circular orbit:
  $$f_{\rm ISCO} = \frac{c^3}{6^{3/2} \pi G M} = \frac{C_{\rm ISCO}}{M / M_\odot} = \frac{4397.17}{M / M_\odot}\,\text{Hz}$$
  All bins satisfying $f_k \le f_{\rm ISCO}$ are retained; all bins $f_k > f_{\rm ISCO}$ are set to identically $0.0 + 0.0j$ (inclusive cutoff convention).

### 4.2 Exact Physical Parameter Mapping (`src/gwlab/physics.py`)
To map user inputs $(\mathcal{M}_c, \eta)$ to physical component masses $(m_1, m_2)$:
$$M = m_1 + m_2 = \mathcal{M}_c \eta^{-3/5}$$
$$m_{1,2} = \frac{M}{2} \left(1 \pm \sqrt{\max(0, 1 - 4\eta)}\right) \quad \text{with } m_1 \ge m_2$$

**Boundary Defense & Numerical Stability:**
- At the equal-mass boundary $\eta = 0.25$, floating-point evaluation can yield $1 - 4\eta = -2.22 \times 10^{-16}$. Clamping via $\max(0.0, 1.0 - 4.0\eta)$ ensures the radicand never becomes negative, preventing `ValueError: math domain error`.
- At $\eta = 0.25$, $m_1 = m_2 = M / 2$ exactly.
- Inverse consistency is verified to machine precision:
  $$\eta = \frac{m_1 m_2}{(m_1 + m_2)^2}, \quad \mathcal{M}_c = (m_1 + m_2) \eta^{3/5}$$

---

## 5. Signal Lab Interactive Workstation Implementation

Signal Lab is implemented in `src/gwlab/pages/2_signal_lab.py` using a two-column layout:

### 5.1 Interactive Parameter Controls (Left Column)
- **Chirp Mass $\mathcal{M}_c$:** Slider $[10.0, 60.0]\,M_\odot$, step $0.05\,M_\odot$, default $28.095556\,M_\odot$.
- **Symmetric Mass Ratio $\eta$:** Slider $[0.10, 0.25]$, step $0.001$, default $0.247101$.
- **Luminosity Distance $d_L$:** Slider $[50.0, 2000.0]\,\text{Mpc}$, step $10.0\,\text{Mpc}$, default $410.0\,\text{Mpc}$.
- **Reference Phase $\phi_c$:** Slider $[0.0, 2\pi]\,\text{rad}$, step $0.05\,\text{rad}$, default $0.00\,\text{rad}$.
- **Coalescence Time $t_c$:** Slider $[0.0, 4.0]\,\text{s}$, step $0.01\,\text{s}$, default $2.00\,\text{s}$.
- **Benchmark Reset Button:** One-click button resetting all parameters to the exact Stage 6C injection.
- **Analysis Grid Options (Expander):** Duration $T \in \{2, 4, 8, 16\}\,\text{s}$, sampling rate $f_s \in \{1024, 2048, 4096\}\,\text{Hz}$, PN order (3.5PN vs 3.0PN), and Stage 6C active band $[20, 512]\,\text{Hz}$ overlay toggle.

### 5.2 Derived Physical Quantities Table
Displays real-time computed values with explicit units and mathematical definitions:
- Primary Mass $m_1 = 36.0000\,M_\odot$
- Secondary Mass $m_2 = 29.0000\,M_\odot$
- Total Mass $M = 65.0000\,M_\odot$
- Mass Ratio $q = m_1 / m_2 = 1.2414$
- ISCO Cutoff Frequency $f_{\rm ISCO} = 67.65\,\text{Hz}$
- Effective Termination $f_{\rm term} = 67.65\,\text{Hz}$
- SI Amplitude Prefactor $A = 1.5462 \times 10^{-20}\,\text{s}^{-1/6}$
- Frequency Resolution $\Delta f = 0.25\,\text{Hz}$ ($N=8192, f_s=2048\,\text{Hz}$)

### 5.3 Tabbed Plotly Scientific Visualizations (Right Column)
1. **Strain Amplitude $|\tilde{h}(f)|$:**
   - Plots $|\tilde{h}(f)|$ vs $f$ on a log-log scale.
   - Overlays vertical dashed line for $f_{\rm ISCO}$ ($67.65\,\text{Hz}$).
   - Overlays shaded green band for active analysis band $[20, 512]\,\text{Hz}$.
   - Shows power-law decay and inclusive cutoff at $67.5\,\text{Hz}$ (last bin $\le f_{\rm ISCO}$).
2. **Fourier Phase $\Psi(f)$:**
   - Radio toggle: Unwrapped Phase (Continuous), Wrapped Phase ($[-\pi, \pi]$), or Both (Stacked).
   - Demonstrates the monotonic phase accumulation and the steep linear time-shift slope $-2\pi t_c = -12.57\,\text{rad/Hz}$ induced by $t_c = 2.0\,\text{s}$.
3. **Inspiral Chirp Time $\tau(f)$:**
   - Evaluates physical time remaining to coalescence:
     $$\tau(f) = t_c - t(f) = -\frac{1}{2\pi} \frac{d\Psi_{\rm int}}{df}$$
   - Compares the leading Newtonian curve $\tau_N(f) \propto f^{-8/3}$ to full 3.5PN post-Newtonian inspiral time evaluated via high-precision central differences.
   - At $f = 20\,\text{Hz}$, displays remaining duration $\tau_{3.5\rm PN} = 0.147\,\text{s}$ (Newtonian: $0.158\,\text{s}$).
4. **Quadratures $\text{Re}[\tilde{h}(f)]$ and $\text{Im}[\tilde{h}(f)]$:**
   - Plots cosine and sine quadratures over $[25, 60]\,\text{Hz}$, showing interferometric fringe oscillations.

---

## 6. Verification & Test Evidence

### 6.1 GWLab Physics & Safety Test Suite (`tests/test_gwlab_physics.py`)
A comprehensive test suite was implemented covering all required physics and safety criteria:

| Test Name | Verification Focus | Result |
| :--- | :--- | :---: |
| `test_stage6c_injection_mass_conversion` | $(\mathcal{M}_c, \eta) \leftrightarrow (m_1, m_2)$ round-trip inverse consistency | **PASSED** |
| `test_equal_mass_boundary_eta_025` | Exact $\eta = 0.25$ handling without float error ($m_1 = m_2 = 15.0\,M_\odot$) | **PASSED** |
| `test_invalid_parameters_rejected` | Rejection of $\mathcal{M}_c \le 0$, $\eta > 0.25$, $\eta \le 0$, NaN inputs | **PASSED** |
| `test_deterministic_waveform_output` | Bit-for-bit identical complex arrays for identical inputs | **PASSED** |
| `test_frequency_grid_and_array_shapes` | $N=8192, f_s=2048\,\text{Hz} \implies$ grid length $4097, \Delta f = 0.25\,\text{Hz}$ | **PASSED** |
| `test_finiteness_over_valid_range` | All amplitude and phase values strictly finite across band | **PASSED** |
| `test_isco_cutoff_enforcement` | All bins $f_k > f_{\rm term}$ identically $0.0 + 0.0j$; DC bin is $0$ | **PASSED** |
| `test_chirp_time_series_evaluation` | $\tau_N > 0, \tau_{3.5\rm PN} > 0$, strictly monotonic decrease with $f$ | **PASSED** |
| `test_no_dynesty_in_modules` | Confirms `dynesty` is not present in `sys.modules` | **PASSED** |
| `test_signal_lab_source_syntax_and_no_samplers` | AST parse check; prohibits sampler, noise, and PSD imports | **PASSED** |

**Execution Result:**
```
pytest tests/test_gwlab_physics.py -v
============================= 10 passed in 3.01s =============================
```

### 6.2 Combined GWLab Shell, Data, and Physics Suites
```
pytest tests/test_gwlab_data.py tests/test_gwlab_shell.py tests/test_gwlab_physics.py -v
============================= 26 passed in 7.07s =============================
```

### 6.3 Core Waveform Engine Regression Suite
```
pytest tests/test_wp2_waveform.py -v
============================= 15 passed in 2.82s =============================
```

### 6.4 Full Project Regression Suite
```
pytest tests/test_conventions.py tests/test_wp0_parseval.py tests/test_wp1_grid_and_psd.py \
       tests/test_wp2_waveform.py tests/test_wp3_filter.py tests/test_gwlab_shell.py \
       tests/test_gwlab_data.py tests/test_gwlab_physics.py -q
============================= 165 passed in 9.21s =============================
```
**Outcome:** Zero failures, zero regressions across all 165 unit and integration tests.

### 6.5 End-to-End Headless Browser Verification (Playwright)
The automated browser verification script ([test_gwlab_phase3a_browser.py](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/experiments/gwlab_phase2_verification/test_gwlab_phase3a_browser.py)) was executed against the running application on port 8502 using Microsoft Edge (Chromium):
- **DOM Assertions Verified:**
  * Title: `"GWLab — Gravitational-Wave Analysis Laboratory"`
  * Page Header: `"Signal Lab — Waveform Inspection & Chirp Dynamics"`
  * Parameter Controls Badge: `"TaylorF2 3.5PN"`
  * Derived Primary Mass: `"36.0000 M_sun"`
  * Derived Secondary Mass: `"29.0000 M_sun"`
  * Derived Total Mass: `"65.0000 M_sun"`
  * Derived ISCO Frequency: `"67.65 Hz"`
  * Rendered Plotly Chart Containers: 4 elements present in DOM
  * Tab switching: Successfully clicked Fourier Phase and Chirp Time tabs
  * Parameter Reset Button: Clicked and reactive state verified
- **Console Errors:** Exactly **0 errors** observed.
- **Captured Workstation Screenshots:**
  * Desktop ($1440 \times 900$): [gwlab_phase3a_signal_lab_desktop.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/screenshots/phases/gwlab_phase3a_signal_lab_desktop.png) ($126.9\,\text{KB}$)
  * Tablet ($1024 \times 768$): [gwlab_phase3a_signal_lab_tablet.png](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/screenshots/phases/gwlab_phase3a_signal_lab_tablet.png) ($102.6\,\text{KB}$)

---

## 7. Execution and Safety Boundary Compliance

Signal Lab was designed and validated under strict non-execution boundaries:
- **Prohibited Computations Verified Absent:**
  * Zero likelihood evaluations (neither `WhittleLikelihood` nor `StudentTLikelihood` imported or called).
  * Zero matched filtering or SNR time series computations.
  * Zero noise realizations generated.
  * Zero PSD estimation algorithms executed.
  * Zero nested sampler runs (Dynesty never loaded or invoked).
  * Zero experiment files created or mutated.
- **Source Code Integrity:** Production code in `src/gw_inference/` remains completely unmodified.
- **Historical Artifact Integrity:** Primary experiment artifacts in `experiments/` and `reports/demonstrations/` were not modified.
- **Git Repository Integrity:** No git commands (`git add`, `git commit`, `git checkout`, `git stash`, `git reset`) were executed. The working tree remains in its exact required state.

---

## 8. Implementation Status Matrix

| Component | Status | Verification Evidence | Notes |
| :--- | :---: | :---: | :--- |
| Phase 2 Budget Labeling | **Validated** | `1_overview.py` & Phase 2 report | "Remaining recorded budget at Stage 6D closeout". |
| Phase 2 Wall Timing Display | **Validated** | `1_overview.py` & Phase 2 report | Authoritative $514.95\,\text{s}$ ($515.0\,\text{s}$) wall, $473.16\,\text{s}$ CPU. |
| Phase 2 Discrepancy Metric Label | **Validated** | `1_overview.py` & Phase 2 report | "Observed evidence discrepancy / reported uncertainty = 8.92". |
| Phase 2 Artifact Integrity Statement | **Validated** | Phase 2 report | Narrowed to "no artifacts intentionally modified". |
| Physical Parameter Mapping (`physics.py`) | **Validated** | `test_gwlab_physics.py` (3 tests) | Invertible $(\mathcal{M}_c, \eta) \leftrightarrow (m_1, m_2)$, equal-mass $\eta=0.25$ guard. |
| Waveform Generation Engine (`physics.py`) | **Validated** | `test_gwlab_physics.py` (5 tests) | Pure evaluation of `gw_inference.waveform`, inclusive ISCO cutoff. |
| Signal Lab Workstation UI (`2_signal_lab.py`) | **Tested** | Playwright browser automation | Full controls, Plotly amplitude/phase/chirp tabs, Carbon styling. |
| Non-Execution Safety Boundary | **Validated** | `test_gwlab_physics.py` (2 tests) | Zero dynesty/sampler/likelihood/noise imports in modules or UI. |
| Full Scientific Regression Suite | **Validated** | `pytest` (165 passed in 9.21s) | Complete regression safety across WP0–WP3 and GWLab. |
| Browser & Screenshot Verification | **Tested** | Playwright (Edge) | 0 console errors; desktop and tablet screenshots captured. |

---

## 9. Known Limitations & Recommendations for Phase 3B

### 9.1 Known Limitations of Signal Lab
1. **Model Scope:** Restricted to nonspinning, quasi-circular compact binary inspiral. Spin-orbit coupling, spin-precession, and eccentricity are outside the model scope.
2. **Termination Behavior:** The waveform terminates abruptly at the Schwarzschild ISCO frequency. It does not contain phenomenological merger-ringdown extensions (IMRPhenom).
3. **Observation Frame:** Frequency grid and waveform parameters represent detector-frame quantities.

### 9.2 Recommendation for Phase 3B (Noise & PSD)
With the Signal Lab operational, the recommended scope for **Phase 3B: Noise & PSD (`src/gwlab/pages/3_noise_and_psd.py`)** is:
1. **Analytical vs Empirical PSD Inspection:** Ingest and compare analytical aLIGO Zero-Detuned High-Power PSD curves against Welch periodogram PSD estimates from saved off-source data.
2. **Effective Degrees of Freedom Visualization:** Provide an interactive demonstration of Welch segment averaging and the $\nu_{\rm eff} \approx 44.85$ variance-matched DoF qualification for colored noise.
3. **Preserve Read-Only Governance:** Ingest saved PSD arrays without generating new stochastic noise realizations or re-running Welch PSD estimation on the fly.
