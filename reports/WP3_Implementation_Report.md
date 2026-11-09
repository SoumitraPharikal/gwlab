# Work Package 3 (WP3): Implementation and Verification Report
## Matched Filtering, SNR, Phase Maximization, and Diagnostics

**Project:** Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals  
**Role:** Antigravity / Gemini 3.8 Flash High — Scientific Implementation Agent  
**Independent Scientific Reviewer:** ChatGPT  
**Project Lead:** User  
**Branch:** `milestone-1.4a`  
**Approved Baseline Commit:** `ff91df71d3c57a64313a3537f523808f789f59d3`  
**Execution Mode:** IMPLEMENTATION AUTHORIZED — STRICT SCOPE CONTROL  
**Date:** 2026-11-09  

---

## 1. Executive Summary

This report documents the completed implementation and verification of **Work Package 3 (WP3): Matched Filtering, SNR, Phase Maximization, and Detection Statistics**.

WP3 synthesizes the discrete Fourier conventions established in WP0, the frequency grids, active observation masks $\mathcal{K}_+$, and PSD interpolation of WP1, and the 3.5PN nonspinning TaylorF2 waveform generator of WP2 into a fast, scientifically verified matched-filtering engine.

The core implementation adheres strictly to the approved specification in `reports/WP3_Mathematical_Specification_and_PreImplementation_Audit.md`:
1. **Authoritative WP1 Observation Grid:** The direct finite frequency sum on the frozen WP1 grid is established and verified as the authoritative reference detection statistic $(d|h)_{\mathcal{K}_+}(t)$.
2. **Complex Matched Filtering via $N$-Point IFFT:** The fast circular matched-filter correlation is implemented via positive-frequency complex embedding and $N$-point IFFT with exact $4\Delta t$ continuous-transform scaling, retaining both physical quadratures without duplicate transforms.
3. **Phase Maximization:** Coalescence-phase maximization over $\phi_c \in (-\pi, \pi]$ is evaluated analytically from the two orthogonal quadratures: $\rho_{\mathrm{max}}(t) = |z(t)| / \sigma_h$, with best-fit phase $\hat{\phi}_c(t) = \operatorname{atan2}(\operatorname{Im} z(t), \operatorname{Re} z(t))$.
4. **Allen Time-Frequency $\chi^2$ Diagnostic:** Implemented with adaptive sub-band merging for degenerate bins ($\sigma_{h, l}^2 \le \epsilon_{\mathrm{tol}} \sigma_h^2$). When $q < 2$, $\chi^2$ is recognized as mathematically undefined and returns `None`, avoiding synthetic zero-division artifacts. In the exceptional case $q < p$, the retained-band $\chi^2$ diagnostic and full-support SNR form a documented, controlled approximation.
5. **Reweighted SNR:** Evaluates the standard piecewise Allen (2005) statistic $\tilde{\rho}(t)$, ensuring clean injections with $\chi_r^2 \le 1$ suffer zero artificial SNR penalty.
6. **Full Validation Matrix (M1–M12):** All 12 test scenarios plus inner-product property tests and clean/mismatched $\chi^2$ tests pass with zero failures. Combined with WP0–WP2 regression tests, **214 tests pass** across the entire test suite.

---

## 2. Files Created and Modified

- **Created Source File:** [`src/gw_inference/filter.py`](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/src/gw_inference/filter.py) (517 lines)
  - Implements `noise_weighted_inner_product`, `compute_template_norm`, `direct_matched_filter_at_lag`, `matched_filter_core`, `compute_signed_lags`, `compute_chisq_binned`, `compute_reweighted_snr`, `matched_filter`, and the immutable `MatchedFilterResult` dataclass.
- **Created Test File:** [`tests/test_wp3_filter.py`](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/tests/test_wp3_filter.py) (542 lines)
  - Implements complete test coverage for M1–M12 and ancillary diagnostics (14 tests).
- **Modified Source File:** [`src/gw_inference/__init__.py`](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/src/gw_inference/__init__.py)
  - Exposes the approved WP3 functions and classes in the public namespace.
- **Modified Documentation File:** [`reports/WP3_Mathematical_Specification_and_PreImplementation_Audit.md`](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/WP3_Mathematical_Specification_and_PreImplementation_Audit.md)
  - Updated in previous review step to finalize statistical statements on Allen $\chi^2$ null and non-central conditions.

---

## 3. Implemented Mathematical Equations

### 3.1 Noise-Weighted Inner Product (`noise_weighted_inner_product`)
$$(a|b)_{\mathcal{K}_+} \equiv 4 \Delta f \operatorname{Re} \sum_{k \in \mathcal{K}_+} \frac{\tilde{a}_k \tilde{b}_k^*}{S_n(f_k)}$$
- Domain: Discrete one-sided active bins $\mathcal{K}_+ = \{k : f_{\mathrm{start}} \le f_k \le f_{\mathrm{end}}, 1 \le k \le \lfloor(N-1)/2\rfloor\}$.
- DC ($k = 0$) and even-$N$ Nyquist ($k = N/2$) are strictly excluded.
- Symmetry: $(a|b) = (b|a)$. Linearity: $(a|\alpha b) = \alpha (a|b)$.

### 3.2 Template Norm and Optimal SNR (`compute_template_norm`)
$$\sigma_h^2 \equiv (h|h)_{\mathcal{K}_+} = 4 \Delta f \sum_{k \in \mathcal{K}_+} \frac{|\tilde{h}(f_k)|^2}{S_n(f_k)}, \qquad \sigma_h = \sqrt{\sigma_h^2}, \qquad \rho_{\mathrm{opt}} \equiv \sigma_h$$

### 3.3 Authoritative Direct Frequency-Sum Reference (`direct_matched_filter_at_lag`)
$$z(\tau) \equiv 4 \Delta f \sum_{k \in \mathcal{K}_+} \frac{\tilde{d}(f_k) \tilde{h}^*(f_k)}{S_n(f_k)} e^{i 2\pi f_k \tau}$$
- Evaluates the exact Fourier sum at any arbitrary continuous or discrete lag $\tau \in \mathbb{R}$ on the frozen WP1 frequency grid.

### 3.4 Fast Analytic $N$-Point IFFT (`matched_filter_core`)
Let $Z \in \mathbb{C}^N$ be an $N$-point complex frequency array initialized to zero:
$$Z_k = \begin{cases} 4 \Delta f \frac{\tilde{d}(f_k) \tilde{h}^*(f_k)}{S_n(f_k)}, & k \in \mathcal{K}_+ \\ 0, & \text{otherwise} \end{cases}$$
$$z(t_m) = N \cdot \operatorname{ifft}(Z, n=N)_m = 4 \Delta t \cdot \operatorname{ifft}\left( \left\{ \frac{D_k H_k^*}{S_n(f_k)} \right\}_{k \in \mathcal{K}_+} \right)_m$$
where $t_m = m \Delta t$ ($m = 0, \dots, N-1$).

### 3.5 Signed Time-Lag Coordinates (`compute_signed_lags`)
$$\tau_m = \begin{cases} m \Delta t, & 0 \le m < \lfloor(N+1)/2\rfloor \\ (m - N)\Delta t, & \lfloor(N+1)/2\rfloor \le m < N \end{cases}$$
- For even $N$: covers $[-T/2, T/2)$ with $\tau_0 = 0$ and $\tau_{N/2} = -T/2$.
- For odd $N$: covers $[-(N-1)\Delta t / 2, (N-1)\Delta t / 2]$.

### 3.6 Coalescence Phase Maximization (`matched_filter`)
Decomposing into in-phase $x(t) = \operatorname{Re}[z(t)]$ and quadrature $y(t) = \operatorname{Im}[z(t)]$:
$$\rho(t) \equiv \frac{z(t)}{\sigma_h} = \frac{x(t) + i y(t)}{\sigma_h}$$
$$\rho_{\mathrm{max}}(t) = |\rho(t)| = \frac{\sqrt{x(t)^2 + y(t)^2}}{\sigma_h}$$
$$\hat{\phi}_c(t) = \operatorname{atan2}(y(t), x(t)) \in (-\pi, \pi]$$

### 3.7 Binned Allen $\chi^2$ Diagnostic (`compute_chisq_binned`)
Dividing the active band $\mathcal{K}_+$ into $p \ge 2$ nominal sub-bands partitioned by cumulative template power. Sub-bands with $\sigma_{h, l}^2 \le \text{power\_tol} \times \sigma_h^2$ ($\text{power\_tol} = 10^{-12}$) are omitted.
Let $q$ denote the number of retained active sub-bands ($2 \le q \le p$).
The retained sub-band template powers are $\tilde{\sigma}_{h, l}^2 > 0$ ($l = 1, \dots, q$), with normalized retained weights:
$$\tilde{w}_l = \frac{\tilde{\sigma}_{h, l}^2}{\sum_{m=1}^q \tilde{\sigma}_{h, m}^2}, \qquad \sum_{l=1}^q \tilde{w}_l = 1$$
The total retained-band matched filter is:
$$z_{\mathrm{total}}(t) = \sum_{l=1}^q z_l(t) = 4 \Delta f \sum_{k \in \mathcal{K}_{\mathrm{ret}}} \frac{\tilde{d}(f_k)\tilde{h}^*(f_k)}{S_n(f_k)} e^{i 2\pi f_k t}, \qquad \mathcal{K}_{\mathrm{ret}} = \bigcup_{l=1}^q \mathcal{K}_l$$
The projection residual vector on $\mathcal{K}_{\mathrm{ret}}$ is:
$$\Delta z_l(t) = z_l(t) - \tilde{w}_l z_{\mathrm{total}}(t), \qquad \sum_{l=1}^q \Delta z_l(t) = 0$$
$$\chi^2(t) = \sum_{l=1}^q \frac{|\Delta z_l(t)|^2}{\tilde{\sigma}_{h, l}^2}, \qquad \nu = 2q - 2$$
When $q < 2$, $\chi^2$ is mathematically unavailable and returns `(None, None)`.

### 3.8 Consistent Reweighted SNR (`compute_reweighted_snr`)
$$\chi_r^2(t) \equiv \frac{\chi^2(t)}{2q - 2}$$
$$\tilde{\rho}(t) = \begin{cases} \rho_{\mathrm{max}}(t), & \chi_r^2(t) \le 1 \\ \frac{\rho_{\mathrm{max}}(t)}{\left[ \frac{1 + (\chi_r^2(t))^3}{2} \right]^{1/6}}, & \chi_r^2(t) > 1 \end{cases}$$
When $\chi^2$ is unavailable ($q < 2$), $\tilde{\rho}(t) = \rho_{\mathrm{max}}(t)$ unconditionally.

### 3.9 Conditional Sub-Band Support and Reweighted SNR Approximation
We explicitly distinguish the two operational frequency-support regimes:
1. **The Ordinary Case ($q = p$):**
   When template power is distributed across all $p$ sub-bands such that each nominal sub-band satisfies $\sigma_{h, l}^2 > \text{power\_tol} \times \sigma_h^2$, no nominal sub-bands are omitted:
   $$\mathcal{K}_{\mathrm{ret}} = \mathcal{K}_+.$$
   In this case, the Allen $\chi^2$ diagnostic and the matched-filter SNR $\rho_{\mathrm{max}}(t)$ share the identical, full WP1 active frequency support $\mathcal{K}_+$. The construction is exact, symmetric, and mathematically consistent.
2. **The Exceptional Case ($q < p$):**
   When one or more nominal sub-bands have template power at or below the relative threshold $\text{power\_tol} \times \sigma_h^2$ (e.g., empty or unpopulated frequency bins at band edges), those sub-bands are omitted from the $\chi^2$ calculation to prevent zero-division and matrix singularity.
   In this exceptional case:
   - **Internal Consistency of Retained Projection:** The retained-band Allen projection $\Delta z_l = z_l - \tilde{w}_l z_{\mathrm{total}}$ is internally consistent on $\mathcal{K}_{\mathrm{ret}} \subsetneq \mathcal{K}_+$, satisfying the zero-sum projection constraint $\sum_{l=1}^q \Delta z_l(t) = 0$ and yielding a rank-$(q-1)$ complex projection with $\nu = 2q - 2$ degrees of freedom under standard circular-Gaussian noise assumptions.
   - **Mixed Support in Reweighted SNR:** The reweighted SNR $\tilde{\rho}(t)$ combines the full-support matched-filter SNR $\rho_{\mathrm{max}}(t)$ (computed over the full active mask $\mathcal{K}_+$) with the reduced-support diagnostic $\chi^2(t)$ (computed over $\mathcal{K}_{\mathrm{ret}}$).
   - **Controlled Approximation:** This construction is a **controlled approximation**, not an exact same-support Allen reweighted-SNR construction.
   - **Template-Power Bound vs. Data-Dependent Contributions:** The threshold $\text{power\_tol} = 10^{-12}$ bounds the omitted template power:
     $$\sigma_{h, \mathrm{omitted}}^2 = \sum_{j \notin \text{retained}} \sigma_{h, j}^2 \le (p - q) \times 10^{-12} \sigma_h^2 \le p \times 10^{-12} \sigma_h^2 \implies \sigma_{h, \mathrm{omitted}} \le \sqrt{p} \times 10^{-6} \sigma_h.$$
     By the Cauchy–Schwarz inequality, the general data-dependent matched-filter output from the omitted bins is bounded by:
     $$|z_{\mathrm{omitted}}(t)| \le \sigma_{d, \mathrm{omitted}} \cdot \sigma_{h, \mathrm{omitted}} \le \sigma_{d, \mathrm{omitted}} \left(\sqrt{p} \times 10^{-6} \sigma_h\right),$$
     where $\sigma_{d, \mathrm{omitted}}^2 \equiv 4 \Delta f \sum_{k \in \mathcal{K}_{\mathrm{omitted}}} \frac{|\tilde{d}(f_k)|^2}{S_n(f_k)}$ is the noise-weighted data energy in the omitted bins.
   - **Omitted-Band Data Energy:**
     - For **noise-only Gaussian data** under the continuous-scaling normalization, $\mathbb{E}[\sigma_{d, \mathrm{omitted}}^2] = 2 N_{\mathrm{omitted}}$. In an illustrative case with $N_{\mathrm{omitted}} \le 10^4$ and $p=16$, the RMS-scale energy is $\sigma_{d, \mathrm{omitted}} \sim \sqrt{2 \times 10^4} \approx 141 \lesssim 150$. At this RMS scale, the omitted SNR contribution is $|z_{\mathrm{omitted}}|/\sigma_h \lesssim 4 \times 10^{-6} \times 150 \approx 6 \times 10^{-4}$, which is negligible. However, this estimate is strictly an illustrative RMS-scale estimate for the stated noise-only Gaussian example, **not** a deterministic upper bound.
     - For **signal-containing data**, the relation $\mathbb{E}[\sigma_{d, \mathrm{omitted}}^2] = 2 N_{\mathrm{omitted}}$ does not hold unchanged; an injected or astrophysical signal contributes additional deterministic noise-weighted energy to $\sigma_{d, \mathrm{omitted}}^2$.
     - For **non-Gaussian glitches, spectral lines, or narrow-band artifacts**, the data energy in the omitted bins can be arbitrarily large ($\sigma_{d, \mathrm{omitted}}^2 \gg 10^{12}$). **No universal guarantee is made for glitches, spectral lines, or arbitrary data with unusually large energy in omitted bins.**
   - **Architectural Rationale:** The implementation retains the full-support matched-filter SNR $\rho_{\mathrm{max}}(t)$ rather than changing it as a function of the $\chi^2$ binning parameters (`chisq_bins`, `power_tol`). The matched-filter SNR is the primary detection statistic established on the physical grid $\mathcal{K}_+$; dynamically pruning the matched-filter SNR to match downstream diagnostic binning would make the core detection statistic depend on arbitrary diagnostic settings.

---

## 4. API Documentation and Input/Output Conventions

### `MatchedFilterResult` (Immutable Dataclass)
Container storing:
- `snr_time_series` (`np.ndarray`, complex128, shape `(N,)`): Complex SNR $\rho(t) = z(t) / \sigma_h$.
- `snr_max` (`np.ndarray`, float64, shape `(N,)`): Phase-maximized SNR $|\rho(t)|$.
- `phase_max` (`np.ndarray`, float64, shape `(N,)`): Best-fit coalescence phase in radians.
- `signed_lags` (`np.ndarray`, float64, shape `(N,)`): Physical signed lags in seconds.
- `circular_lags` (`np.ndarray`, float64, shape `(N,)`): Circular lags $[0, T)$ in seconds.
- `valid_mask` (`np.ndarray`, bool, shape `(N,)`): Boolean array of uncorrupted interior lags.
- `sigma_h` (`float`): Active-band template norm.
- `optimal_snr` (`float`): Optimal SNR $\rho_{\mathrm{opt}} = \sigma_h$.
- `peak_snr` (`float`): Maximum phase-maximized SNR over valid lags.
- `peak_time` (`float`): Signed time lag in seconds at peak SNR.
- `peak_phase` (`float`): Coalescence phase in radians at peak SNR.
- `chisq` (`Optional[np.ndarray]`, float64, shape `(N,)`): Allen $\chi^2$ time series (or None).
- `chisq_dof` (`Optional[int]`): Degrees of freedom $2q - 2$ (or None).
- `reweighted_snr` (`Optional[np.ndarray]`, float64, shape `(N,)`): Reweighted SNR time series.

### Key Public Functions in `gw_inference`
1. `noise_weighted_inner_product(a_fd, b_fd, psd, delta_f, active_indices=None) -> float`
2. `compute_template_norm(template_fd, inv_psd_active, active_indices, delta_f) -> float`
3. `direct_matched_filter_at_lag(data_fd, template_fd, inv_psd_active, active_indices, freqs, delta_f, time_lag) -> complex`
4. `matched_filter_core(data_fd, template_fd, inv_psd_active, active_indices, n_samples, delta_f) -> Tuple[np.ndarray, float]`
5. `compute_signed_lags(n_samples, delta_t) -> np.ndarray`
6. `compute_chisq_binned(data_fd, template_fd, inv_psd_active, active_indices, n_samples, delta_f, n_bins=16, power_tol=1e-12) -> Tuple[Optional[np.ndarray], Optional[int]]`
7. `compute_reweighted_snr(snr_max, chisq, chisq_dof) -> np.ndarray`
8. `matched_filter(data, template_fd, grid, psd_engine, mask, f_start, f_0, tau_chirp=None, tau_filter_diag=0.2, compute_chisq=False, chisq_bins=16) -> MatchedFilterResult`

---

## 5. Fourier and Lag Normalization Conventions

1. **Continuous-Scaled Transform:**
   Data in time domain $d_j$ is converted via continuous Fourier transform scaling:
   $$\tilde{d}(f_k) = \Delta t \cdot \operatorname{rfft}(d)_k$$
   Inverting from frequency to time domain preserves exact amplitudes via:
   $$d(t_j) = \frac{1}{\Delta t} \operatorname{irfft}(\tilde{d}, n=N)_j$$
2. **Lag Sign Convention:**
   Fourier time-shift property:
   $$\mathcal{F}\{h(t - t_c)\} = \tilde{h}(f) e^{-i 2\pi f t_c}$$
   Matched filter correlation:
   $$z(t) = 4 \Delta f \sum_{k \in \mathcal{K}_+} \frac{\tilde{d}(f_k) \tilde{h}^*(f_k)}{S_n(f_k)} e^{+i 2\pi f_k t}$$
   Because the kernel uses $+i 2\pi f t$, evaluation via inverse discrete Fourier transform (`ifft`) correctly places a signal arriving at lag $+t_c$ at index $m = t_c / \Delta t$.
3. **Statistical Scaling:**
   Under colored Gaussian noise with $\mathbb{E}[|\tilde{d}(f_k)|^2] = S_n(f_k) / (2 \Delta f)$:
   - $\operatorname{Var}(\operatorname{Re} z) = \operatorname{Var}(\operatorname{Im} z) = \sigma_h^2$
   - $\mathbb{E}[|z|^2] = 2 \sigma_h^2$
   - Real and imaginary parts of $\rho(t) = z(t)/\sigma_h$ are independent standard normal variables $\mathcal{N}(0, 1)$.
   - $|\rho(t)| \sim \operatorname{Rayleigh}(1)$.

---

## 6. Direct-Sum Reference Methodology

To ensure numerical independence and eliminate reliance on FFT implementation artifacts:
1. `direct_matched_filter_at_lag` directly evaluates the finite frequency sum $\sum_{k \in \mathcal{K}_+}$ without zero-padding, interpolation, or FFT wrappers.
2. The fast $N$-point IFFT implementation `matched_filter_core` is verified against `direct_matched_filter_at_lag` at $\tau = 0$, $\tau = +10\Delta t$, and $\tau = -10\Delta t$ (Test M1), and across an interior lag grid (Test M11).
3. The observed relative discrepancies are $< 10^{-14}$ across all tests, confirming that the $N$-point analytic IFFT evaluates the exact discrete Fourier sum at observation samples down to float64 machine precision.

---

## 7. Test Matrix Mapping (M1–M12)

```
+----+----------------------------------+-----------------------------------------------------------+----------+
| ID | Specification Description        | Test Function in tests/test_wp3_filter.py                 | Status   |
+----+----------------------------------+-----------------------------------------------------------+----------+
| -- | Inner Product Properties         | test_noise_weighted_inner_product_properties              | PASSED   |
| M1 | Direct Sum vs. IFFT Equivalence  | test_m1_direct_sum_vs_ifft_equivalence                   | PASSED   |
| M2 | Zero-Lag Time-Domain Recovery    | test_m2_zero_lag_injection_recovery                       | PASSED   |
| M3 | Cyclic & Linear Time Shifts      | test_m3_cyclic_and_linear_time_shifts                     | PASSED   |
| M4 | Coalescence Phase Sweep          | test_m4_coalescence_phase_sweep                           | PASSED   |
| M5 | Template Normalization Invariance| test_m5_template_normalization_invariance                 | PASSED   |
| M6 | Odd/Even-N Grid Parity & Lags    | test_m6_odd_and_even_grid_parity                          | PASSED   |
| M7 | Gaussian Noise Ensemble (6 tests)| test_m7_gaussian_noise_ensemble_statistics                | PASSED   |
| M8 | Parameter Mismatch & Norms       | test_m8_parameter_mismatch_with_separate_norms            | PASSED   |
| M9 | PSD Scaling Homogeneity          | test_m9_psd_scaling_homogeneity                           | PASSED   |
| M10| Malformed Inputs & Degenerate Chi| test_m10_malformed_input_suite_and_degenerate_chisq       | PASSED   |
| M11| Direct-Sum Interior Benchmark    | test_m11_direct_sum_vs_circular_interior_benchmark        | PASSED   |
| M12| Active Mask Invariance Across M  | test_m12_active_mask_invariance                           | PASSED   |
| -- | Allen Chi^2 Clean vs Mismatch    | test_allen_chisq_on_clean_and_mismatched_signals          | PASSED   |
+----+----------------------------------+-----------------------------------------------------------+----------+
```

---

## 8. Test Results and Numerical Comparisons

### 8.1 WP3 Test Suite Execution
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference
collected 14 items

tests/test_wp3_filter.py::test_noise_weighted_inner_product_properties PASSED [  7%]
tests/test_wp3_filter.py::test_m1_direct_sum_vs_ifft_equivalence PASSED  [ 14%]
tests/test_wp3_filter.py::test_m2_zero_lag_injection_recovery PASSED     [ 21%]
tests/test_wp3_filter.py::test_m3_cyclic_and_linear_time_shifts PASSED   [ 28%]
tests/test_wp3_filter.py::test_m4_coalescence_phase_sweep PASSED         [ 35%]
tests/test_wp3_filter.py::test_m5_template_normalization_invariance PASSED [ 42%]
tests/test_wp3_filter.py::test_m6_odd_and_even_grid_parity PASSED        [ 50%]
tests/test_wp3_filter.py::test_m7_gaussian_noise_ensemble_statistics PASSED [ 57%]
tests/test_wp3_filter.py::test_m8_parameter_mismatch_with_separate_norms PASSED [ 64%]
tests/test_wp3_filter.py::test_m9_psd_scaling_homogeneity PASSED         [ 71%]
tests/test_wp3_filter.py::test_m10_malformed_input_suite_and_degenerate_chisq PASSED [ 78%]
tests/test_wp3_filter.py::test_m11_direct_sum_vs_circular_interior_benchmark PASSED [ 85%]
tests/test_wp3_filter.py::test_m12_active_mask_invariance PASSED         [ 92%]
tests/test_wp3_filter.py::test_allen_chisq_on_clean_and_mismatched_signals PASSED [100%]

============================= 14 passed in 1.77s ==============================
```

### 8.2 Test M7 Ensemble Statistics Verification
For $M = 10,000$ independent noise realizations passing through the matched filter core, all sample statistics fall strictly within their theoretical 99% marginal acceptance intervals:
- **Real Quadrature Mean:** $-0.005364 \in [-0.0258, +0.0258]$ (**PASSED**)
- **Imaginary Quadrature Mean:** $-0.003685 \in [-0.0258, +0.0258]$ (**PASSED**)
- **Real Quadrature Variance:** $0.987454 \in [0.9639, 1.0368]$ (**PASSED**)
- **Imaginary Quadrature Variance:** $1.005517 \in [0.9639, 1.0368]$ (**PASSED**)
- **Real-Imag Covariance:** $+0.001792 \in [-0.0258, +0.0258]$ (**PASSED**)
- **Rayleigh Magnitude Mean:** $1.250288 \in [1.2364, 1.2702]$ (**PASSED**)

#### 8.2.1 Statistical Justification, Berry–Esseen Bound, and Coverage Properties
The six sample statistics in Test M7 verify that the normalized complex matched-filter output $\rho = z / \sigma_h = x + i y$ conforms to circular complex Gaussian noise under the null hypothesis:
$x, y \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 1)$, $\operatorname{Cov}(x, y) = 0$, and $R = |\rho| = \sqrt{x^2 + y^2} \sim \operatorname{Rayleigh}(1)$.

1. **Rayleigh Moments and Correction of Earlier Audit Claim:**
   For $R \sim \operatorname{Rayleigh}(1)$:
   - Mean: $\mu_R = \mathbb{E}[R] = \sqrt{\pi / 2} \approx 1.253314$
   - Variance: $\sigma_R^2 = \operatorname{Var}(R) = 2 - \pi / 2 \approx 0.429204 \implies \sigma_R \approx 0.655136$, $\sigma_R^3 \approx 0.281206$
   - Signed third central moment: $\mu_3 = \mathbb{E}[(R - \mu_R)^3] = \sqrt{\pi/2}(\pi - 3) \approx 0.17746$
   - Absolute third central moment (Lyapunov moment):
     $$\mathbb{E}[|R - \mu_R|^3] \approx 0.45544$$
   *Correction Note:* An earlier draft of the audit assessment cited a uniform Berry–Esseen CDF error bound below $0.003$. That calculation was incorrect because it evaluated the signed third central moment $\mu_3 \approx 0.17746$ instead of the absolute third central moment $\mathbb{E}[|R - \mu_R|^3] \approx 0.45544$.

2. **Standardized Berry–Esseen Upper Bound:**
   Applying the Berry–Esseen theorem for i.i.d. random variables with $M = 10{,}000$ and the standard Shevtsova (2011) universal constant $C_0 = 0.4748$:
   $$\sup_{x \in \mathbb{R}} |F_M(x) - \Phi(x)| \le \frac{C_0 \, \mathbb{E}[|R - \mu_R|^3]}{\sigma_R^3 \sqrt{M}} = \frac{0.4748 \times 0.45544}{0.281206 \times 100} \approx 0.00769 \lesssim \mathbf{0.0077}$$

3. **CLT-Based 99% Regression Acceptance Interval:**
   The standard error of the sample mean $\bar{R} = \frac{1}{M}\sum_{i=1}^M R_i$ is $\sigma_{\bar{R}} = \sigma_R / \sqrt{M} \approx 0.00655136$. The CLT-based 99% two-sided marginal acceptance interval is:
   $$\mu_R \pm z_{0.005} \sigma_{\bar{R}} = 1.253314 \pm 2.575829 \times 0.00655136 = [1.236439, 1.270189] \approx [\mathbf{1.2364}, \mathbf{1.2702}]$$
   The existing CLT-based 99% Rayleigh sample-mean interval $[1.2364, 1.2702]$ remains a reasonable large-sample regression interval for $M = 10{,}000$.

4. **Edgeworth Endpoint Deviation Heuristic vs. Rigorous Coverage:**
   By the one-term Edgeworth expansion for smooth densities, the asymptotic deviation at the standardized threshold $x = z_{0.005} \approx 2.5758$ is governed by skewness $\gamma_1 = \mu_3 / \sigma_R^3 \approx 0.63107$:
   $$|F_M(x) - \Phi(x)| \approx \left| -\frac{\gamma_1}{6\sqrt{M}} (x^2 - 1) \phi(x) \right| \approx \frac{0.63107}{600} \times 5.635 \times 0.01449 \approx \mathbf{8.6 \times 10^{-5}}$$
   This endpoint deviation estimate of approximately $8.6 \times 10^{-5}$ is an asymptotic estimate, not a rigorous finite-sample coverage guarantee. We do not claim a guaranteed actual coverage exceeding $98.98\%$.

5. **Marginal vs. Joint Multi-Statistic Coverage:**
   The six acceptance criteria in Test M7 are constructed as marginal 99% intervals ($\alpha = 0.01$), and statistical independence between the six sample statistics is not assumed. Under the conservative Bonferroni inequality, which holds without requiring independence, the joint family-wise type-I error rate across all six simultaneous checks is bounded by $\alpha_{\mathrm{joint}} \le 6 \times 0.01 = 0.06$, guaranteeing a joint family-wise confidence of at least $94\%$. The distinction between marginal 99% testing intervals and joint multi-statistic coverage across the six checks is preserved.

### 8.3 Full Regression Results (All Packages)
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference
collected 214 items

tests\test_conventions.py .............................................. [ 21%]
........................................                                 [ 40%]
tests\test_noise.py .............                                        [ 46%]
tests\test_psd.py ...................................................... [ 71%]
....                                                                     [ 73%]
tests\test_wp0_conventions.py ....                                       [ 75%]
tests\test_wp0_parseval.py ........                                      [ 78%]
tests\test_wp1_grid_and_psd.py ................                          [ 86%]
tests\test_wp2_waveform.py ...............                               [ 93%]
tests\test_wp3_filter.py ..............                                  [100%]

============================= 214 passed in 6.86s =============================
```

---

## 9. Independent Numerical Comparisons and Errors

1. **Direct Sum vs. $N$-Point IFFT (M1, M11):**
   $$\max_\tau \frac{|z_{\mathrm{fft}}(\tau) - z_{\mathrm{direct}}(\tau)|}{|z_{\mathrm{direct}}(\tau)|} = 3.22 \times 10^{-15} < 1.0 \times 10^{-12}$$
2. **Zero-Lag Injection Reconstruction (M2):**
   $$\frac{|\rho_{\mathrm{max}}(0) - \rho_{\mathrm{opt}}|}{\rho_{\mathrm{opt}}} = 1.11 \times 10^{-16} < 1.0 \times 10^{-12}$$
   $$|\hat{\phi}_c(0)| = 0.00 \text{ rad} < 1.0 \times 10^{-12} \text{ rad}$$
3. **Template Normalization Invariance (M5):**
   $$\max_\alpha \frac{\max_t |\rho_\alpha(t) - \rho_{\mathrm{orig}}(t)|}{\max_t \rho_{\mathrm{orig}}(t)} = 2.13 \times 10^{-14} < 5.0 \times 10^{-14}$$
4. **PSD Scaling Invariance (M9):**
   $$\frac{|\sigma_h(c S_n) - \sigma_h(S_n)/\sqrt{c}|}{\sigma_h} < 1.0 \times 10^{-15}$$
   $$\frac{\max_t |\chi^2(c S_n) - \chi^2(S_n)/c|}{\max_t \chi^2} = 1.58 \times 10^{-14} < 5.0 \times 10^{-14}$$

---

## 10. Known Limitations, Approximations, and Deferred Scope

1. **Conditional Allen $\chi^2$ Frequency Support Approximation ($q < p$):**
   - **Ordinary Case ($q = p$):** When template power spans all $p$ nominal sub-bands, $\mathcal{K}_{\mathrm{ret}} = \mathcal{K}_+$. Both the Allen $\chi^2$ diagnostic and the matched-filter SNR $\rho_{\mathrm{max}}(t)$ share the identical full WP1 active frequency support.
   - **Exceptional Case ($q < p$):** When one or more low-power nominal sub-bands have $\sigma_{h, l}^2 \le 10^{-12} \sigma_h^2$, they are omitted from the $\chi^2$ diagnostic, while `snr_max` remains computed over the full active mask $\mathcal{K}_+$.
   - **Internal Consistency:** The retained-band Allen projection is internally consistent and has strictly $\nu = 2q - 2$ degrees of freedom under the specified circular-Gaussian assumptions on $\mathcal{K}_{\mathrm{ret}}$.
   - **Controlled Approximation:** In this exceptional case, the reweighted-SNR construction combines a full-support SNR with a reduced-support $\chi^2$ diagnostic. This is a **controlled approximation**, not an exact same-support Allen reweighted-SNR construction.
   - **Template Power vs. Data-Dependent Contributions:** The relative threshold $\text{power\_tol} = 10^{-12}$ bounds omitted template power ($\sigma_{h, \mathrm{omitted}} \le \sqrt{p} \times 10^{-6}\sigma_h$), not arbitrary data-dependent contributions.
   - **Cauchy–Schwarz Bound:** The data-dependent matched-filter output from omitted bins is bounded by $|z_{\mathrm{omitted}}(t)| \le \sigma_{d, \mathrm{omitted}} \cdot \sigma_{h, \mathrm{omitted}}$.
   - **Data Energy Scaling:** For noise-only Gaussian data under the continuous-scaling normalization, $\mathbb{E}[\sigma_{d, \mathrm{omitted}}^2] = 2 N_{\mathrm{omitted}}$, yielding an illustrative RMS-scale estimate $\sigma_{d, \mathrm{omitted}} \lesssim 150$ for the example grid ($N_{\mathrm{omitted}} \le 10^4$). This is an illustrative RMS-scale estimate, **not** a deterministic upper bound. Furthermore, this relation does not hold unchanged for signal-containing data, where an injected or astrophysical signal contributes additional noise-weighted energy.
   - **No Universal Guarantee for Anomalous Data:** No universal guarantee is made for glitches, spectral lines, or arbitrary data with unusually large energy in omitted bins ($\sigma_{d, \mathrm{omitted}}^2 \gg 10^{12}$).
   - **Filter Independence:** The implementation deliberately retains the full-support matched-filter SNR rather than changing it as a function of $\chi^2$ binning parameters (`chisq_bins`, `power_tol`).
2. **Circular Boundary Certification:** As established in the approved specification, circular correlation on the WP1 grid introduces non-compact filter leakage because $1/S_n(f)$ contains continuous analytical roll-offs governed by $f_0$. A certified zero-leakage boundary is not mathematically proven. The `valid_mask` heuristic with $\tau_{\mathrm{filter, diag}} = 0.2\text{ s}$ is an operational diagnostic and is not certified as an absolute boundary guarantee.
3. **Padded Correlation Grids:** Padded correlation grids ($N_{\mathrm{pad}} > N$) alter the discrete frequency spacing $\Delta f_{\mathrm{pad}} < \Delta f$ and evaluate a different discrete inner product. Equivalence to the WP1 inner product has not been established. Padded grid implementations remain explicitly deferred.
4. **Glitch Populations and False-Alarm Rates:** The Allen $\chi^2$ distribution is not modeled as a central or non-central $\chi^2$ law for non-Gaussian glitches. Empirical glitch distributions, time-slide backgrounds, and FAR estimation are deferred to Stage 2.
5. **Decision D8 ($f_0$ Low-Frequency Scale):** The physical value of $f_0$ remains open at the governance level. In compliance with WP1 contracts, all public signatures require $f_0$ as an explicit, mandatory floating-point argument without default.

---

## 11. Complete Item Classification Table

```
+---------------------------------------------------------------+----------------------------------------------------------+
| Item Description                                              | Exact Status Classification                              |
+---------------------------------------------------------------+----------------------------------------------------------+
| Noise-weighted inner product (a|b)_{K_+}                      | IMPLEMENTED, TESTED, NUMERICALLY VALID                   |
| Template norm sigma_h and optimal SNR                         | IMPLEMENTED, TESTED, NUMERICALLY VALID                   |
| Direct frequency-sum reference matched filter                 | IMPLEMENTED, TESTED, NUMERICALLY VALID                   |
| Fast analytic N-point IFFT matched-filter core                | IMPLEMENTED, TESTED, NUMERICALLY VALID                   |
| Signed and circular time-lag coordinate generator             | IMPLEMENTED, TESTED, NUMERICALLY VALID                   |
| Coalescence-phase maximization and angle extraction           | IMPLEMENTED, TESTED, NUMERICALLY VALID                   |
| Binned Allen chi^2 with degenerate sub-band handling          | IMPLEMENTED, TESTED, NUMERICALLY VALID (Approximation documented)|
| Piecewise reweighted SNR statistic (Allen 2005)               | IMPLEMENTED, TESTED, NUMERICALLY VALID (Approximation documented)|
| High-level matched_filter entry point & MatchedFilterResult   | IMPLEMENTED, TESTED, NUMERICALLY VALID                   |
| Validation matrix M1–M12                                      | IMPLEMENTED, TESTED, NUMERICALLY VALID                   |
| Full WP0–WP2 regression suite (200 tests)                      | TESTED, NUMERICALLY VALIDATED                            |
| Decision D8 (f_0 low-frequency PSD continuation scale)        | UNRESOLVED (Mandatory explicit arg)                      |
| Circular correlation boundary certification                   | DEFERRED                                                 |
| Padded-grid correlation equivalence                           | DEFERRED                                                 |
| Whittle likelihood for MCMC parameter estimation              | DEFERRED (WP4)                                           |
| Time-slide false-alarm rate estimation and bank placement     | DEFERRED (Stage 2)                                       |
+---------------------------------------------------------------+----------------------------------------------------------+
```

---

## 12. Reproducible Commands to Run the Tests

```powershell
# Set Python path to include source directory
$env:PYTHONPATH='src'

# Run the 14 dedicated WP3 matched-filtering tests with verbose output
pytest tests/test_wp3_filter.py -v

# Run the complete test suite (214 tests across WP0, WP1, WP2, WP3)
pytest -q
```

---

## 13. Git Status and Milestone Governance Audit

- **Active Branch:** `milestone-1.4a`
- **Current HEAD Commit:** `ff91df71d3c57a64313a3537f523808f789f59d3`
- **Working Tree Status (`git status --short`):**
  ```text
   M src/gw_inference/__init__.py
  ?? reports/WP3_Implementation_Report.md
  ?? reports/WP3_Mathematical_Specification_and_PreImplementation_Audit.md
  ?? src/gw_inference/filter.py
  ?? tests/test_wp3_filter.py
  ```
- **Git Commit Confirmation:** **NO COMMIT HAS BEEN CREATED.** The working tree remains uncommitted and completely reviewable for independent scientific audit by ChatGPT.
- **Scope Compliance:** WP4 (Whittle likelihood) and WP5 (MCMC inference) have **NOT** been implemented. Zero stubs or downstream files have been created.
