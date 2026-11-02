# Stage 1.2 Implementation and Revision Report
## Welch Power Spectral Density Estimation

**Project:** Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals  
**Stage:** 1.2 — Welch Power Spectral Density Estimation (Final Revision)  
**Author:** Scientific Software Developer (AI Assistant)  
**Supervisor / Reviewer:** Independent Scientific Reviewer (ChatGPT)  
**Date:** 2026-11-02  
**Repository State:** Master branch, clean working directory, 154/154 tests passing  

---

### 1. Executive Summary of Corrections

Following the independent review of commit `d56c99a`, three final targeted scientific corrections were implemented without modifying Stage 1.1 files or expanding beyond Stage 1.2 scope:

1. **Correction 1 — Paired Bootstrap Across Monte Carlo Realizations:**
   Replaced the decoupled resampling with a true paired bootstrap:
   ```python
   idx = boot_rng.choice(n_realizations, size=n_realizations, replace=True)
   boot_rho = _calc_mean_r(x_bins[idx], y_bins[idx])
   ```
   Both segment-0 and segment-1 periodogram arrays share the exact same resampled realization indices, preserving realization-level pairing. Resampling occurs across independent Monte Carlo trials ($R=300$), not across frequency bins. Increased bootstrap replicates to $B = 1000$. The bootstrap standard error is evaluated as the sample standard deviation with `ddof=1`. The report and docstrings explicitly clarify that the bootstrap estimates the sampling uncertainty of the realization-level correlation statistic across independent realizations, and does not establish that the theoretical correlation model is universally valid.

2. **Correction 2 — Clarified and Disentangled Median-Welch Validation:**
   Disentangled the theoretical order-statistics theorem from window-tapering approximations:
   - **Part 3A (Exact Theoretical Baseline):** Added a dedicated experiment with independent Gaussian white noise, zero overlap, rectangular window (`window="rect"`), interior positive frequencies ($0 < k < M/2$), $K=15$ segments, and $R=300$ realizations. Because the rectangular window on white noise has identically vanishing interior pseudo-covariance ($P_{11}(k) \equiv 0$), the Fourier coefficients are strictly proper complex Gaussian and their periodograms are identically, independently exponentially distributed. This provides a mathematically exact validation of the $C(15) = 0.725372$ theorem ($z_M = -1.404$, PASS).
   - **Part 3B (Tukey-Window Robustness Diagnostic):** Retained the Tukey-window experiment as a separate numerical robustness diagnostic, clearly labeled as evaluating empirical accuracy under mild window-induced pseudo-covariance ($z_M = +0.009$).
   - **Conceptual Distinction:** Formally distinguished 1) independence between non-overlapping segments, 2) marginal exponential distribution of individual periodograms, and 3) the mathematical assumptions of $C(K)$. Clarified that non-overlap alone does *not* guarantee all three.

3. **Correction 3 — Direct Evaluation of Discrete Pseudo-Covariance Window Sums:**
   Removed the inaccurate claim that pseudo-covariance suppression was demonstrated by $z_Q$ or $z_\rho$. Implemented `discrete_pseudo_covariance_ratios` in `src/gw_inference/psd.py` to evaluate the exact discrete window sums derived in Stage 0.6 Section 3:
   $$C_{11} = \sum_{j=0}^{M-1} w_j^2 = M S_2, \qquad C_{12}(k) = e^{-2\pi i H k / M} \sum_{l=0}^{M-H-1} w_l w_{l+H}$$
   $$P_{11}(k) = \sum_{j=0}^{M-1} w_j^2 e^{-4\pi i j k / M}, \qquad P_{12}(k) = e^{-2\pi i H k / M} \sum_{l=0}^{M-H-1} w_l w_{l+H} e^{-4\pi i l k / M}$$
   Evaluated the complete Isserlis/Wick periodogram correlation:
   $$\rho_P(k) = \frac{|C_{12}(k)|^2 + |P_{12}(k)|^2}{C_{11}^2 + |P_{11}(k)|^2}$$
   Demonstrated numerically that:
   - At DC ($k=0$) and Nyquist ($k=M/2$), $|P_{11}|/C_{11} = 1.0$ and $|P_{12}|/|C_{12}| = 1.0$ (proper-complex assumption fails completely, though $\rho_P \equiv r_w^2$ identically).
   - In transition bins ($k=1..4$), pseudo-covariance causes a $+5.6\%$ deviation in $\rho_P(1)$.
   - In the analysis band ($k \in [5, M/2-5]$), $|P_{11}|/C_{11} < 0.012$ (mean $1.76 \times 10^{-5}$) and $|\rho_{\mathrm{full}} - r_w^2|/r_w^2 < 7.81 \times 10^{-5}$ ($< 0.008\%$).
   Explicitly distinguished theoretical proper-complex assumptions, exact discrete expressions, direct numerical measurements, and Monte Carlo empirical correlation evidence.

---

### 2. Status of Stage 0 Mathematical Audit Documents

All Stage 0 mathematical consistency audit documents were located and inspected in the workspace root:
- `Stage_0_2_Mathematical_Consistency_Audit.md` (46,558 B)
- `Stage_0_3_Final_Mathematical_Consistency_Audit.md` (56,755 B)
- `Stage_0_4_Final_Scientific_Corrections.md` (45,770 B)
- `Stage_0_5_Final_Approval_Readiness.md` (46,717 B)
- `Stage_0_6_Final_Consistency_and_Approval.md` (49,339 B)

The mathematical conventions verified include:
- Discrete Tukey window midpoint formula: $M_{\mathrm{taper}} = \lfloor \alpha M / 2 \rfloor$.
- Normalization factor $S_2 = 1 - 5\alpha/8 = 27/32 = 0.84375$.
- Coherent gain $S_1 = 1 - \alpha/2 = 7/8 = 0.875$.
- 50% overlap correlation $r_w = 4/9 \approx 0.444444$ and $r_1^2 = 16/81 \approx 0.197531$.
- Median bias correction factor $C(15) = 0.725372$, $C(255) = 0.695104$.
- Frequency grid endpoints and factor 2 for interior bins, factor 1 for DC/Nyquist.

No mathematical contradictions between Stage 0.6 and the implementation exist.

---

### 3. Corrected Mathematical Formulation

#### 3.1 Discrete Sampling and Frequency Grid
For discrete real time series $x_0, x_1, \dots, x_{N-1}$ sampled at interval $\Delta t = 1/f_s$:
$$f_k = \frac{k}{M \Delta t}, \qquad k = 0, 1, \dots, \frac{M}{2}, \qquad \Delta f = \frac{1}{M \Delta t}$$

#### 3.2 Segment Extraction and Unambiguous Windowing
For segment $m \in \{0, \dots, K-1\}$ with hop length $H = M - N_{\mathrm{overlap}}$:
$$x_j^{(m)} = x_{j + m H} \, w_j, \qquad j = 0, \dots, M-1$$
The discrete real FFT is:
$$X_k^{(m)} = \operatorname{rfft}(x^{(m)})_k = \sum_{j=0}^{M-1} x_j^{(m)} e^{-i 2\pi j k / M}$$
The windowed one-sided periodogram is:
$$\widehat{S}_m(f_k) = \begin{cases}
\frac{\Delta t}{M S_2} |X_k^{(m)}|^2, & k = 0, \, M/2 \\[6pt]
\frac{2\Delta t}{M S_2} |X_k^{(m)}|^2, & 1 \le k < M/2
\end{cases}$$
The window $w_j$ is applied exactly once to each raw data slice prior to the FFT.

#### 3.3 Discrete Tukey Window Formulation
With taper length $M_{\mathrm{taper}} = \lfloor \alpha M / 2 \rfloor$:
$$w_j = \begin{cases}
\frac{1}{2}\left[ 1 - \cos\left( \frac{\pi (j + 0.5)}{M_{\mathrm{taper}}} \right) \right], & 0 \le j < M_{\mathrm{taper}} \\[6pt]
1.0, & M_{\mathrm{taper}} \le j < M - M_{\mathrm{taper}} \\[6pt]
\frac{1}{2}\left[ 1 - \cos\left( \frac{\pi (M - 1 - j + 0.5)}{M_{\mathrm{taper}}} \right) \right], & M - M_{\mathrm{taper}} \le j < M
\end{cases}$$
Exact discrete energy sum:
$$S_2 = \frac{1}{M} \sum_{j=0}^{M-1} w_j^2, \qquad S_1 = \frac{1}{M} \sum_{j=0}^{M-1} w_j$$

#### 3.4 Generalized All-Lags Overlap Covariance
For hop $H$, segments share time samples across lags $\ell \ge 1$ where $\ell H < M$. The normalized window correlation at lag $\ell$ is:
$$r_\ell = \frac{\sum_{j=0}^{M - \ell H - 1} w_j w_{j + \ell H}}{\sum_{j=0}^{M-1} w_j^2}$$
The maximum shared lag is $\ell_{\max} = \min\left( \left\lfloor \frac{M - 1}{H} \right\rfloor, \, K - 1 \right)$.
Under the proper-complex Gaussian approximation, the Welch mean variance inflation factor is:
$$\kappa_{\mathrm{overlap}} = 1 + 2 \sum_{\ell=1}^{\ell_{\max}} \left( 1 - \frac{\ell}{K} \right) r_\ell^2$$
- Zero overlap ($H = M$): $\kappa = 1.0$.
- Single segment ($K = 1$): $\kappa = 1.0$.
- $50\%$ overlap ($H = M/2$): $\kappa = 1 + 2\frac{K-1}{K}r_1^2$ (reproduces adjacent-only result).
- $75\%$ overlap ($M=256, K=13$): $\kappa_{\mathrm{all}} = 2.3810$ (vs adjacent-only $\kappa_{\mathrm{adj}} = 2.0130$).

#### 3.5 Exact Discrete Pseudo-Covariance and Isserlis/Wick Correlation
For windowed segments of discrete Gaussian white noise:
$$C_{11} = \sum_{j=0}^{M-1} w_j^2 = M S_2, \qquad C_{12}(k) = e^{-2\pi i H k / M} \sum_{l=0}^{M-H-1} w_l w_{l+H}$$
$$P_{11}(k) = \sum_{j=0}^{M-1} w_j^2 e^{-4\pi i j k / M}, \qquad P_{12}(k) = e^{-2\pi i H k / M} \sum_{l=0}^{M-H-1} w_l w_{l+H} e^{-4\pi i l k / M}$$
By Isserlis' theorem for complex Gaussians:
$$\rho_P(k) = \frac{|C_{12}(k)|^2 + |P_{12}(k)|^2}{C_{11}^2 + |P_{11}(k)|^2}$$
When $P_{11}(k) \approx 0$ and $P_{12}(k) \approx 0$, this reduces to $\rho_P^{\mathrm{approx}} = r_w^2$.

---

### 4. Revised Median API and Independence Assumptions

The function `welch_psd` in `src/gw_inference/psd.py` enforces:
```python
def welch_psd(
    x: np.ndarray,
    delta_t: float,
    segment_length: int,
    overlap_samples: Optional[int] = None,
    window: Optional[Union[np.ndarray, str]] = "tukey",
    method: Literal["mean", "median"] = "mean",
    bias_correct: bool = False,
    assume_independent_exponential_periodograms: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
```

#### Behavioral Rules and Constraints:
1. **Raw Sample Median (`bias_correct=False`):** Fully available for any overlap (0%, 50%, 75%). No assertions required.
2. **Bias-Corrected Median (`bias_correct=True`):**
   - **Strictly rejected for overlapping segments:** If `overlap_samples > 0`, raises `ValueError`.
   - **Gated by caller assertion:** Requires `assume_independent_exponential_periodograms=True`. Zero overlap alone does *not* authorize bias correction because coloured noise can retain inter-segment dependence across disjoint slices, and window tapering can perturb the marginal exponential distribution.

---

### 5. Statistically Calibrated Monte Carlo Methodology

The numerical validation script `scripts/validate_welch_psd.py` adheres to strict statistical testing standards:

#### 5.1 Realization-Level Band Statistic
For each Monte Carlo trial $r = 1, \dots, R$:
1. Generate independent white noise $x^{(r)} \sim \mathcal{N}(0, \sigma^2)$.
2. Estimate Welch PSD $\widehat{S}^{(r)}(f_k)$.
3. Compute realization-level interior band average:
   $$Q_r \equiv \frac{1}{|B|} \sum_{k \in B} \widehat{S}^{(r)}(f_k), \qquad B = \{ k : 10 \le k \le M/2 - 10 \}$$
4. Compute realization-level endpoints: $D_r \equiv \widehat{S}^{(r)}(0)$ and $Y_r \equiv \widehat{S}^{(r)}(f_{\mathrm{Nyq}})$.

#### 5.2 Standardized Deviations ($z$-Scores)
- Ensemble mean: $\bar{Q} = \frac{1}{R} \sum_{r=1}^R Q_r$
- Sample standard deviation: $s_Q = \sqrt{\frac{1}{R-1}\sum_{r=1}^R (Q_r - \bar{Q})^2}$
- Standard error of ensemble mean: $\operatorname{SE}(\bar{Q}) = \frac{s_Q}{\sqrt{R}}$
- Standardized deviation: $z_Q = \frac{\bar{Q} - Q_{\mathrm{expected}}}{\operatorname{SE}(\bar{Q})}$
- Threshold: $|z| < 5.0$.

#### 5.3 Paired Bootstrap for Adjacent Correlation (Correction 1)
For adjacent segments across $R = 300$ realizations:
```python
boot_rng = np.random.default_rng(seed + 999)
boot_rhos = np.empty(n_bootstrap, dtype=np.float64)
for b in range(n_bootstrap):
    idx = boot_rng.choice(n_realizations, size=n_realizations, replace=True)
    boot_rhos[b] = _calc_mean_r(x_bins[idx], y_bins[idx])
se_rho_p = float(np.std(boot_rhos, ddof=1))
z_rho_p = float((measured_rho_p - r1_squared) / se_rho_p)
```
- Replicates: $B = 1000$.
- Preserves the paired realization structure $(x^{(r)}, y^{(r)})$.
- The bootstrap estimates sampling uncertainty of the correlation statistic across independent realizations; it does not establish that the theoretical correlation model is universally valid.

#### 5.4 Exact Rectangular Median Baseline vs Tukey Diagnostic (Correction 2)
- **Part 3A (Rectangular Baseline):** $K=15$ independent segments, zero overlap, rectangular window ($P_{11}(k) \equiv 0$). Tests exact $C(K)$ independent-exponential theorem.
- **Part 3B (Tukey Diagnostic):** $K=15$ segments, zero overlap, Tukey window ($\alpha=0.25$). Numerical robustness diagnostic under mild window-induced pseudo-covariance.

---

### 6. Numerical Validation Results

Execution of `python scripts/validate_welch_psd.py --n-realizations 300 --seed 42`:

```text
================================================================================
STAGE 1.2: STATISTICALLY CALIBRATED WELCH PSD VALIDATION
================================================================================

--- Part 1: White Noise Welch Mean Estimation (Calibrated Statistics) ---
Sampling frequency (f_s):           4096.0 Hz
Sampling interval (delta_t):         2.441406e-04 s
Total duration:                      8.00 s (32768 samples)
Segment duration:                    1.00 s (4096 samples)
Overlap fraction:                    50.0% (2048 samples)
Hop length:                          2048 samples
Number of valid segments (K):        15
Number of shared lags (ell_max):     1
Tukey window alpha:                  0.25
Discrete window power (S_2):         0.843750
Adjacent correlation (r_1):          0.444444
All-lags variance inflation (kappa): 1.3687
Number of Monte Carlo trials (R):    300
Interior expected PSD:               4.882812e-04 strain^2/Hz
Interior ensemble band mean Q_bar:   4.880679e-04 strain^2/Hz
Realization band std dev (s_Q):      3.865193e-06 strain^2/Hz
Ensemble band SE (SE(Q_bar)):        2.231570e-07 strain^2/Hz
Single-bin theoretical SE:           8.515728e-06 strain^2/Hz
Interior standardized deviation z_Q: -0.956
DC expected PSD:                     2.441406e-04 strain^2/Hz
DC ensemble mean:                    2.421378e-04 strain^2/Hz (SE: 6.745622e-06, z = -0.297)
Nyquist expected PSD:                2.441406e-04 strain^2/Hz
Nyquist ensemble mean:               2.431392e-04 strain^2/Hz (SE: 6.143074e-06, z = -0.163)
Empirical adj-seg corr (rho_P):      0.1969 (pred: r_1^2 = 0.1975)
Paired bootstrap replicates (B):     1000
Paired bootstrap SE (SE(rho_P)):     1.404661e-03
Correlation standardized z-score:    -0.439
Mean Parseval relative error:        7.35e-17

--- Part 2: Exact Discrete Pseudo-Covariance Window Sums (Correction 3) ---
Discrete window total energy (C_11): 3456.0000
Discrete window overlap prod (C_12): 1536.0000
Proper-complex prediction (r_w^2):   0.197531
DC (k=0):      |P_11|/C_11 = 1.0000e+00, |P_12|/|C_12| = 1.0000e+00, rho_full = 0.197531
Bin k=1:       |P_11|/C_11 = 1.5090e-01, |P_12|/|C_12| = 2.8294e-01, rho_full = 0.208594 (rel diff: +5.6009e-02)
Bin k=2:       |P_11|/C_11 = 7.4074e-02, |P_12|/|C_12| = 1.6667e-01, rho_full = 0.201910 (rel diff: +2.2169e-02)
Bin k=5:       |P_11|/C_11 = 1.1976e-02, |P_12|/|C_12| = 8.0841e-03, rho_full = 0.197515 (rel diff: -7.8071e-05)
Nyquist (k=M/2): |P_11|/C_11 = 1.0000e+00, |P_12|/|C_12| = 1.0000e+00, rho_full = 0.197531
Analysis band [5, M/2-5] Max |P_11|/C_11:   1.1976e-02 (Mean: 1.7621e-05)
Analysis band [5, M/2-5] Max |P_12|/|C_12|: 8.0841e-03 (Mean: 1.3876e-05)
Analysis band Max |rho_full - r_w^2|/r_w^2:  7.8071e-05 (Mean: 7.7495e-08)

--- Part 3A: Median-Welch Exact Theoretical Baseline (Rectangular Window) ---
Number of independent segments (K):  15
Stage 0.6 odd bias factor C(15):     0.725372
Expected uncorrected raw median:     3.541855e-04 strain^2/Hz
Measured uncorrected raw median:     3.538620e-04 strain^2/Hz
Corrected median PSD ensemble mean:  4.878353e-04 strain^2/Hz
Corrected median ensemble SE:        3.177379e-07 strain^2/Hz
Exact baseline median z-score:       -1.404

--- Part 3B: Median-Welch Robustness Diagnostic (Tukey Window, alpha=0.25) ---
Number of independent segments (K):  15
Expected uncorrected raw median:     3.541855e-04 strain^2/Hz
Measured uncorrected raw median:     3.541877e-04 strain^2/Hz
Corrected median PSD ensemble mean:  4.882843e-04 strain^2/Hz
Corrected median ensemble SE:        3.391336e-07 strain^2/Hz
Tukey diagnostic median z-score:     +0.009

--- Part 4: Coloured Noise AR(1) Broad Spectral Shape (Diagnostic) ---
AR(1) parameter phi:                 0.65
Mean relative error across band:     1.553% (diagnostic spectral agreement)
Max relative error across band:      6.882% (reflects window spectral smoothing)

================================================================================
STATISTICALLY CALIBRATED ACCEPTANCE CRITERIA CHECK
================================================================================
Criteria 1: Interior band mean z-score |z_Q| < 5.0:            PASS (z = -0.956)
Criteria 2: DC bin z-score |z_DC| < 5.0:                       PASS (z = -0.297)
Criteria 3: Nyquist bin z-score |z_Nyq| < 5.0:                 PASS (z = -0.163)
Criteria 4: Parseval energy conservation error < 1e-10:        PASS (7.35e-17)
Criteria 5: Paired bootstrap correlation |z_rho| < 5.0:        PASS (z = -0.439, B = 1000)
Criteria 6: Rectangular median exact baseline |z_M| < 5.0:     PASS (z = -1.404)
Criteria 7: Analysis band pseudo-covariance rel diff < 1e-3:   PASS (7.81e-05)
Diagnostic: Tukey median robustness diagnostic z-score:        +0.009
Diagnostic: AR(1) coloured-noise band mean error:             1.553% (max: 6.882%)

Overall Validation Status: ALL CRITERIA PASSED
```

---

### 7. Full Test Suite Results

The complete test suite runs and passes in 2.02 seconds:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.6.2.post1, zarr-3.2.1
collected 154 items

tests/test_conventions.py ..................................................... [ 34%]
...................................                                             [ 57%]
tests/test_noise.py .............                                               [ 65%]
tests/test_psd.py ............................................................. [100%]

============================= 154 passed in 2.02s =============================
```

#### Test Suite Breakdown (154 tests total):
- `tests/test_conventions.py` (86 tests): Fourier conventions, scaling, grid, Parseval, folding.
- `tests/test_noise.py` (13 tests): White-noise generation, seed reproducibility, sample statistics.
- `tests/test_psd.py` (55 tests):
  - Frequency grid properties and Stage 1.1 consistency: 2 tests
  - Window normalization, symmetry, short segments, multi-lag correlations: 6 tests
  - Single-segment consistency, single-window application proof, Parseval conservation: 4 tests
  - White-noise ensemble PSD expectation: 1 test
  - Segment extraction, overlap bounds, trailing sample discarding: 6 tests
  - Coloured-noise AR(1) spectral shape recovery: 1 test
  - Generalized all-lags variance inflation factor: 5 tests
  - Exact discrete pseudo-covariance window sums (Correction 3): 4 tests
  - Redesigned median estimator, raw median, exact rectangular baseline, assertion gating: 9 tests
  - Input rejection (complex, NaN, Inf, invalid dimensions, invalid parameter combinations): 7 tests
  - Validation script integration smoke tests: 5 tests

---

### 8. Files Created and Modified

1. **Modified in this revision:**
   - [`src/gw_inference/psd.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/src/gw_inference/psd.py):
     - Implemented `discrete_pseudo_covariance_ratios` evaluating exact discrete window sums.
     - Preserved all-lags variance inflation and median API gating.
   - [`src/gw_inference/__init__.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/src/gw_inference/__init__.py):
     - Exported `discrete_pseudo_covariance_ratios`.
   - [`scripts/validate_welch_psd.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/scripts/validate_welch_psd.py):
     - Implemented paired bootstrap ($B=1000$) across Monte Carlo realizations.
     - Added direct numerical evaluation of discrete pseudo-covariance sums (Part 2).
     - Disentangled exact rectangular baseline ($K=15, z_M = -1.404$) from Tukey robustness diagnostic ($z_M = +0.009$).
   - [`tests/test_psd.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/tests/test_psd.py):
     - Added `TestPseudoCovariance` with 4 dedicated tests.
     - Added `test_exact_rectangular_white_noise_median`.
     - Updated integration smoke tests (total 154 tests).
   - [`README.md`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/README.md):
     - Updated test counts and feature checklists.
   - [`reports/Stage_1_2_Implementation_Report.md`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/Stage_1_2_Implementation_Report.md):
     - Updated documentation reflecting the three targeted corrections.

---

### 9. Remaining Limitations and Scope Boundaries

1. **Proper-Complex Gaussian Approximation:** The generalized all-lags expression $\kappa_{\mathrm{overlap}} = 1 + 2 \sum_{\ell=1}^{\ell_{\max}} (1 - \ell/K) r_\ell^2$ is conditional on the proper-complex Gaussian approximation. Direct numerical evaluation of the discrete window sums confirms that for the discrete Tukey window ($\alpha=0.25$), pseudo-covariance is suppressed across the analysis band ($k \in [5, M/2-5]$), introducing a relative error on $\rho_P$ of less than $0.008\%$. However, at DC ($k=0$), Nyquist ($k=M/2$), and transition bins ($k=1..4$), pseudo-covariance is significant ($|P_{11}|/C_{11} \ge 1.2\%$) and the proper-complex approximation breaks down.
2. **Median Variance Under Overlap:** While raw median estimation is robust and fully supported for overlapping segments, no simple universal analytical variance inflation factor exists for median order statistics under correlated Gaussian processes.
3. **Window Spectral Leakage Near Steep Gradients:** Scalar division by $S_2$ assumes locally flat spectra across the window kernel's main lobe. Low-frequency seismic drift ($\propto f^{-4}$) requires the time-domain Butterworth bandpass conditioning scheduled for Stage 1.3.

---

### 10. Confirmations

I formally confirm that:
- Stage 1.1 source files (`conventions.py`, `noise.py`, `test_conventions.py`, `test_noise.py`, `validate_white_noise_psd.py`) remain completely unchanged.
- No functionality from Stage 1.3 or later has been implemented (no Butterworth bandpass filtering, no whitening, no matched filtering, no waveform models, no Bayesian sampling).

---

*Stage 1.2 (final revision) is hereby submitted to the independent scientific reviewer (ChatGPT) for formal evaluation.*
