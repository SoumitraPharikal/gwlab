# Stage 1.1 Technical Implementation Report (Revised)
## Discrete Fourier Conventions, Scaling, and Synthetic Time-Series Generation

**Project Title:** Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals  
**Stage:** 1.1 — Discrete Fourier Conventions and Synthetic Time-Series Generation  
**Revision:** Post-Scientific Review Corrections (Correction A: Input Validation; Correction B: DC/Nyquist Uncertainty)  
**Date:** 2026-11-02  
**Implementation Status:** Revisions implemented, verified (99 unit tests passing, Monte Carlo script validated), and submitted for independent scientific review.  
**Execution Environment:** Windows 11 host Python 3.13.5 (64-bit), NumPy 2.1.2, pytest 9.2.0 (per user authorization).  

---

### A. Summary of Corrections Implemented

Following independent scientific review feedback on the initial Stage 1.1 submission, two targeted mathematical and numerical corrections were executed without expanding project scope or modifying Stage 0 conventions:

#### 1. Correction A: Consistent Real-Valued Input Validation
- **Defect Identified:** Previously, functions (`discrete_parseval_time_energy`, `one_sided_periodogram`, `validate_parseval_identity`) cast inputs via `np.asarray(x, dtype=np.float64)`. In NumPy, casting complex arrays to `np.float64` silently drops the imaginary part with only a warning, instead of rejecting invalid complex data. Furthermore, in `validate_parseval_identity`, calculating `delta_f = 1.0 / (n_samples * delta_t)` prior to length checks risked `ZeroDivisionError` on empty arrays.
- **Resolution:** Implemented a unified private validation helper `_validate_real_time_series(x, delta_t) -> tuple[np.ndarray, float]`:
  - Inspects array dtype *before* any type conversion.
  - Explicitly rejects complex-valued arrays (`TypeError` / `ValueError`) with descriptive message: `x must be real-valued, got complex dtype ...`.
  - Strictly rejects non-finite values (`NaN`, `+Inf`, `-Inf`).
  - Rejects empty arrays (`x.size == 0`) with `ValueError: x must be a non-empty array`.
  - Rejects invalid dimensionality (`x.ndim != 1`).
  - Enforces the even-length requirement (`x.size % 2 == 0`).
  - Validates `delta_t` as positive and finite (`math.isfinite(delta_t) and delta_t > 0`).
  - Safely casts valid real integer or float inputs to `np.float64` without data loss.
  - In `validate_parseval_identity`, input validation executes strictly before `delta_f` computation, preventing any division-by-zero.
- **Coverage:** Applied across all 4 time-series functions (`continuous_rfft`, `discrete_parseval_time_energy`, `validate_parseval_identity`, `one_sided_periodogram`). Also validated `discrete_parseval_freq_energy` (checking dimensionality, size $\ge 2$, finiteness, and valid $\Delta t, \Delta f$).

#### 2. Correction B: Correct DC/Nyquist Ensemble Uncertainty
- **Defect Identified:** In `tests/test_noise.py` and `tests/test_conventions.py`, explanatory comments and tolerance estimates previously modeled the relative standard error of the ensemble mean at DC and Nyquist as $\mathrm{SE} = S / \sqrt{R}$. This was incorrect because Fourier coefficients at DC and Nyquist are purely real Gaussians, not complex Gaussians.
- **Mathematical Resolution:** Derived and documented the exact variance and standard error:
  - For white Gaussian noise $n_j \sim \mathcal{N}(0, \sigma^2)$, the coefficients $X_0 = \sum_j n_j$ and $X_{N/2} = \sum_j (-1)^j n_j$ are real zero-mean normal random variables with variance $N\sigma^2$.
  - Their squared magnitudes follow a scaled 1-degree-of-freedom chi-squared distribution: $|X|^2 \sim N\sigma^2 \chi^2_1$.
  - Because $\operatorname{Var}(\chi^2_1) = 2$, the variance of the squared real coefficient is:
    $$\operatorname{Var}(|X|^2) = 2 (N\sigma^2)^2 = 2 (\mathbb{E}[|X|^2])^2$$
  - Consequently, for the periodogram estimator $\widehat{S}_{\mathrm{DC}} = \frac{\Delta t}{N} |X_0|^2$ with expectation $S_{\mathrm{DC}} = \sigma^2 \Delta t$:
    $$\operatorname{Var}(\widehat{S}_{\mathrm{DC}}) = 2 S_{\mathrm{DC}}^2, \qquad \operatorname{Var}(\widehat{S}_{\mathrm{Nyquist}}) = 2 S_{\mathrm{Nyquist}}^2$$
  - In contrast, interior bins ($1 \le k < N/2$) are complex with two independent quadratures: $|X_k|^2 \sim \frac{N\sigma^2}{2} \chi^2_2$ (exponential), giving $\operatorname{Var}(\widehat{S}_{\mathrm{interior}}) = S_{\mathrm{interior}}^2$.
  - Therefore, the standard error of the ensemble mean $\overline{S} = \frac{1}{R} \sum_{r=1}^R \widehat{S}^{(r)}$ over $R$ realizations is:
    $$\operatorname{SE}(\overline{S}_{\mathrm{DC}}) = \frac{\sqrt{2} S_{\mathrm{DC}}}{\sqrt{R}}, \qquad \operatorname{SE}(\overline{S}_{\mathrm{Nyquist}}) = \frac{\sqrt{2} S_{\mathrm{Nyquist}}}{\sqrt{R}}$$
  - For $R = 200$ (unit test mini-ensemble): $\mathrm{SE}_{\mathrm{rel}} = \sqrt{2 / 200} = 10.0\%$ (not $7.07\%$). A $3.5\sigma$ acceptance tolerance is $35.0\%$.
  - For $R = 1000$ (Monte Carlo script): $\mathrm{SE}_{\mathrm{rel}} = \sqrt{2 / 1000} \approx 4.472\%$. A $3.5\sigma$ acceptance tolerance is $15.65\%$.
  - Updated all test comments, analytical tolerances, and script acceptance thresholds accordingly, while preserving the empirical sample standard error calculation from realization arrays.

#### 3. Additional Edge-Case Validation in Monte Carlo Script
- Added parameter guards in `scripts/validate_white_noise_psd.py`:
  - Rejects $n_{\mathrm{realizations}} \le 1$ (`ValueError`) preventing division by zero in sample variance ($R - 1$).
  - Rejects $n_{\mathrm{samples}} \le 0$ or odd $n_{\mathrm{samples}}$.
  - Rejects $f_s \le 0$ or non-finite $f_s$.
  - Rejects $\sigma < 0$ or non-finite $\sigma$.

---

### B. Mathematical Convention Audit

All numerical implementations strictly reflect the approved Stage 0 conventions:

#### 1. Continuous Fourier Transform
$$\tilde{x}(f) = \int_{-\infty}^\infty x(t) e^{-2\pi i f t} \, dt$$

#### 2. Discrete Sampling Parameters
- Sampling frequency: $f_s = 4096.0 \, \mathrm{Hz}$
- Sampling interval: $\Delta t = \frac{1}{f_s} = 2.44140625 \times 10^{-4} \, \mathrm{s}$
- Segment length: $N = 32{,}768$ samples (even integer)
- Duration: $T = N \Delta t = 8.0 \, \mathrm{s}$
- Frequency resolution: $\Delta f = \frac{1}{T} = \frac{f_s}{N} = 0.125 \, \mathrm{Hz}$

#### 3. Frequency-Grid Construction
$$f_k = k \Delta f = \frac{k}{N \Delta t}, \qquad k = 0, 1, \dots, \frac{N}{2}$$
- DC bin ($k = 0$): $f_0 = 0.0 \, \mathrm{Hz}$
- Nyquist bin ($k = N/2 = 16{,}384$): $f_{N/2} = \frac{f_s}{2} = 2048.0 \, \mathrm{Hz}$
- Length: $N/2 + 1 = 16{,}385$ bins.

#### 4. Real FFT Scaling
$$X_k = \sum_{j=0}^{N-1} x_j e^{-2\pi i j k / N}, \qquad k = 0, 1, \dots, \frac{N}{2}$$
$$\tilde{x}(f_k) \simeq \Delta t \, X_k = \frac{1}{f_s} X_k$$
No extra $1/N$ factor is applied to the forward transform.

#### 5. Discrete Parseval Energy Balance
$$\Delta t \sum_{j=0}^{N-1} |x_j|^2 = \Delta f \, \Delta t^2 \left[ |X_0|^2 + 2 \sum_{k=1}^{N/2-1} |X_k|^2 + |X_{N/2}|^2 \right]$$
- Interior bins ($1 \le k < N/2$): Folded weight $2$ (Hermitian symmetry).
- DC ($k = 0$) and Nyquist ($k = N/2$): Weight $1$ (non-degenerate modes).

#### 6. One-Sided Periodogram Normalization
$$\widehat{S}_n(f_k) = \begin{cases} \displaystyle \frac{\Delta t}{N} |X_k|^2, & k = 0, \, \frac{N}{2} \\[8pt] \displaystyle \frac{2\Delta t}{N} |X_k|^2, & 1 \le k < \frac{N}{2} \end{cases}$$
Integral property: $\sum_{k=0}^{N/2} \widehat{S}_n(f_k) \Delta f = \frac{1}{N} \sum_{j=0}^{N-1} |x_j|^2$.

#### 7. Theoretical White-Noise PSD and Second-Order Statistics
For discrete Gaussian white noise $n_j \sim \mathcal{N}(0, \sigma^2)$:
$$\mathbb{E}[\widehat{S}_n(f_k)] = \frac{2\sigma^2}{f_s} \quad \left(1 \le k < \frac{N}{2}\right), \qquad \mathbb{E}[\widehat{S}_n(0)] = \mathbb{E}\left[\widehat{S}_n\left(\frac{f_s}{2}\right)\right] = \frac{\sigma^2}{f_s}$$
$$\operatorname{Var}(\widehat{S}_n(f_k)) = S_{\mathrm{interior}}^2 \quad \left(1 \le k < \frac{N}{2}\right), \qquad \operatorname{Var}(\widehat{S}_n(0)) = 2 S_{\mathrm{DC}}^2, \qquad \operatorname{Var}\left(\widehat{S}_n\left(\frac{f_s}{2}\right)\right) = 2 S_{\mathrm{Nyq}}^2$$

---

### C. Test Report and Numerical Results

#### 1. Pytest Test Summary
The complete expanded test suite was executed on the Windows 11 host environment using Python 3.13.5 and pytest 9.2.0:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference
configfile: pyproject.toml
plugins: anyio-4.6.2.post1, zarr-3.2.1
collected 99 items

tests/test_conventions.py::TestFrequencyGrid::test_baseline_grid_properties PASSED [  1%]
tests/test_conventions.py::TestFrequencyGrid::test_grid_length_and_endpoints[10] PASSED [  2%]
tests/test_conventions.py::TestFrequencyGrid::test_grid_length_and_endpoints[64] PASSED [  3%]
tests/test_conventions.py::TestFrequencyGrid::test_grid_length_and_endpoints[1024] PASSED [  4%]
tests/test_conventions.py::TestFrequencyGrid::test_grid_length_and_endpoints[8192] PASSED [  5%]
tests/test_conventions.py::TestFrequencyGrid::test_invalid_parameters PASSED [  6%]
tests/test_conventions.py::TestContinuousRFFTScaling::test_impulse_response PASSED [  7%]
tests/test_conventions.py::TestContinuousRFFTScaling::test_unnormalised_rfft_scaling_factor PASSED [  8%]
tests/test_conventions.py::TestRealValuedInputValidation::test_complex_valued_input_rejection[continuous_rfft] PASSED [  9%]
tests/test_conventions.py::TestRealValuedInputValidation::test_complex_valued_input_rejection[discrete_parseval_time_energy] PASSED [ 10%]
tests/test_conventions.py::TestRealValuedInputValidation::test_complex_valued_input_rejection[validate_parseval_identity] PASSED [ 11%]
tests/test_conventions.py::TestRealValuedInputValidation::test_complex_valued_input_rejection[one_sided_periodogram] PASSED [ 12%]
tests/test_conventions.py::TestRealValuedInputValidation::test_nan_rejection[continuous_rfft] PASSED [ 13%]
tests/test_conventions.py::TestRealValuedInputValidation::test_nan_rejection[discrete_parseval_time_energy] PASSED [ 14%]
tests/test_conventions.py::TestRealValuedInputValidation::test_nan_rejection[validate_parseval_identity] PASSED [ 15%]
tests/test_conventions.py::TestRealValuedInputValidation::test_nan_rejection[one_sided_periodogram] PASSED [ 16%]
tests/test_conventions.py::TestRealValuedInputValidation::test_positive_infinity_rejection[continuous_rfft] PASSED [ 17%]
tests/test_conventions.py::TestRealValuedInputValidation::test_positive_infinity_rejection[discrete_parseval_time_energy] PASSED [ 18%]
tests/test_conventions.py::TestRealValuedInputValidation::test_positive_infinity_rejection[validate_parseval_identity] PASSED [ 19%]
tests/test_conventions.py::TestRealValuedInputValidation::test_positive_infinity_rejection[one_sided_periodogram] PASSED [ 20%]
tests/test_conventions.py::TestRealValuedInputValidation::test_negative_infinity_rejection[continuous_rfft] PASSED [ 21%]
tests/test_conventions.py::TestRealValuedInputValidation::test_negative_infinity_rejection[discrete_parseval_time_energy] PASSED [ 22%]
tests/test_conventions.py::TestRealValuedInputValidation::test_negative_infinity_rejection[validate_parseval_identity] PASSED [ 23%]
tests/test_conventions.py::TestRealValuedInputValidation::test_negative_infinity_rejection[one_sided_periodogram] PASSED [ 24%]
tests/test_conventions.py::TestRealValuedInputValidation::test_empty_array_rejection[continuous_rfft] PASSED [ 25%]
tests/test_conventions.py::TestRealValuedInputValidation::test_empty_array_rejection[discrete_parseval_time_energy] PASSED [ 26%]
tests/test_conventions.py::TestRealValuedInputValidation::test_empty_array_rejection[validate_parseval_identity] PASSED [ 27%]
tests/test_conventions.py::TestRealValuedInputValidation::test_empty_array_rejection[one_sided_periodogram] PASSED [ 28%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_dimensionality_rejection[continuous_rfft] PASSED [ 29%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_dimensionality_rejection[discrete_parseval_time_energy] PASSED [ 30%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_dimensionality_rejection[validate_parseval_identity] PASSED [ 31%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_dimensionality_rejection[one_sided_periodogram] PASSED [ 32%]
tests/test_conventions.py::TestRealValuedInputValidation::test_odd_length_rejection[continuous_rfft] PASSED [ 33%]
tests/test_conventions.py::TestRealValuedInputValidation::test_odd_length_rejection[discrete_parseval_time_energy] PASSED [ 34%]
tests/test_conventions.py::TestRealValuedInputValidation::test_odd_length_rejection[validate_parseval_identity] PASSED [ 35%]
tests/test_conventions.py::TestRealValuedInputValidation::test_odd_length_rejection[one_sided_periodogram] PASSED [ 36%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[continuous_rfft--0.01] PASSED [ 37%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[continuous_rfft-0.0] PASSED [ 38%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[continuous_rfft-nan] PASSED [ 39%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[continuous_rfft-inf] PASSED [ 40%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[discrete_parseval_time_energy--0.01] PASSED [ 41%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[discrete_parseval_time_energy-0.0] PASSED [ 42%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[discrete_parseval_time_energy-nan] PASSED [ 43%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[discrete_parseval_time_energy-inf] PASSED [ 44%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[validate_parseval_identity--0.01] PASSED [ 45%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[validate_parseval_identity-0.0] PASSED [ 46%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[validate_parseval_identity-nan] PASSED [ 47%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[validate_parseval_identity-inf] PASSED [ 48%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[one_sided_periodogram--0.01] PASSED [ 49%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[one_sided_periodogram-0.0] PASSED [ 50%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[one_sided_periodogram-nan] PASSED [ 51%]
tests/test_conventions.py::TestRealValuedInputValidation::test_invalid_sampling_interval_rejection[one_sided_periodogram-inf] PASSED [ 52%]
tests/test_conventions.py::TestRealValuedInputValidation::test_valid_zero_valued_input[continuous_rfft] PASSED [ 53%]
tests/test_conventions.py::TestRealValuedInputValidation::test_valid_zero_valued_input[discrete_parseval_time_energy] PASSED [ 54%]
tests/test_conventions.py::TestRealValuedInputValidation::test_valid_zero_valued_input[validate_parseval_identity] PASSED [ 55%]
tests/test_conventions.py::TestRealValuedInputValidation::test_valid_zero_valued_input[one_sided_periodogram] PASSED [ 56%]
tests/test_conventions.py::TestRealValuedInputValidation::test_valid_integer_input[continuous_rfft] PASSED [ 57%]
tests/test_conventions.py::TestRealValuedInputValidation::test_valid_integer_input[discrete_parseval_time_energy] PASSED [ 58%]
tests/test_conventions.py::TestRealValuedInputValidation::test_valid_integer_input[validate_parseval_identity] PASSED [ 59%]
tests/test_conventions.py::TestRealValuedInputValidation::test_valid_integer_input[one_sided_periodogram] PASSED [ 60%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[unit_impulse_origin] PASSED [ 61%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[shifted_impulse] PASSED [ 62%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[pure_dc] PASSED [ 63%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[pure_nyquist] PASSED [ 64%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[single_tone] PASSED [ 65%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[multitone] PASSED [ 66%]
tests/test_conventions.py::TestParsevalIdentity::test_random_gaussian_signals[128] PASSED [ 67%]
tests/test_conventions.py::TestParsevalIdentity::test_random_gaussian_signals[1024] PASSED [ 68%]
tests/test_conventions.py::TestParsevalIdentity::test_random_gaussian_signals[8192] PASSED [ 69%]
tests/test_conventions.py::TestParsevalIdentity::test_random_gaussian_signals[32768] PASSED [ 70%]
tests/test_conventions.py::TestParsevalIdentity::test_dc_bin_weighting PASSED [ 71%]
tests/test_conventions.py::TestParsevalIdentity::test_nyquist_bin_weighting PASSED [ 72%]
tests/test_conventions.py::TestParsevalIdentity::test_interior_bin_weighting PASSED [ 73%]
tests/test_conventions.py::TestOneSidedPeriodogram::test_periodogram_integral_equals_time_mean_square PASSED [ 74%]
tests/test_conventions.py::TestOneSidedPeriodogram::test_expected_white_noise_psd_formula PASSED [ 75%]
tests/test_conventions.py::TestStage11MandatorySummary::test_frequency_grid_construction PASSED [ 76%]
tests/test_conventions.py::TestStage11MandatorySummary::test_fft_scaling PASSED [ 77%]
tests/test_conventions.py::TestStage11MandatorySummary::test_parseval_identity_tolerance PASSED [ 78%]
tests/test_conventions.py::TestStage11MandatorySummary::test_dc_weighting_not_doubled PASSED [ 79%]
tests/test_conventions.py::TestStage11MandatorySummary::test_nyquist_weighting_not_doubled PASSED [ 80%]
tests/test_conventions.py::TestStage11MandatorySummary::test_interior_bin_weighting_doubled PASSED [ 81%]
tests/test_conventions.py::TestStage11MandatorySummary::test_periodogram_units_and_scaling PASSED [ 82%]
tests/test_conventions.py::TestStage11MandatorySummary::test_noise_reproducibility PASSED [ 83%]
tests/test_conventions.py::TestStage11MandatorySummary::test_noise_input_validation PASSED [ 84%]
tests/test_conventions.py::TestStage11MandatorySummary::test_white_noise_ensemble_psd_interior PASSED [ 85%]
tests/test_conventions.py::TestStage11MandatorySummary::test_dc_nyquist_ensemble_psd_half_level PASSED [ 86%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_sample_count_and_dtype PASSED [ 87%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_output_finite PASSED  [ 88%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_reproducibility_identical_seeds PASSED [ 89%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_different_seeds_produce_different_realizations PASSED [ 90%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_generator_instance_support PASSED [ 91%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_zero_sigma PASSED     [ 92%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_sample_statistics_convergence PASSED [ 93%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_input_validation PASSED [ 94%]
tests/test_noise.py::TestEnsemblePeriodogramMini::test_ensemble_periodogram_expectation PASSED [ 95%]
tests/test_noise.py::TestValidationScriptEdgeCases::test_invalid_n_realizations PASSED [ 96%]
tests/test_noise.py::TestValidationScriptEdgeCases::test_invalid_n_samples PASSED [ 97%]
tests/test_noise.py::TestValidationScriptEdgeCases::test_invalid_f_s PASSED [ 98%]
tests/test_noise.py::TestValidationScriptEdgeCases::test_invalid_sigma PASSED [100%]

============================= 99 passed in 0.75s ==============================
```

#### 2. Mandatory Validation Matrix

| Test Name (Prompt §7) | Expected Analytical Property | Numerical Test Case | Actual Result | Status |
|---|---|---|---|---|
| **Frequency-grid construction** | $N/2 + 1$ bins, $f_0 = 0$, $f_{N/2} = f_s/2$, $\Delta f = 1/T$ | $N = 32{,}768, f_s = 4096 \, \mathrm{Hz}$ | $16{,}385$ bins, $0.0$ to $2048.0 \, \mathrm{Hz}$, $\Delta f = 0.125 \, \mathrm{Hz}$ | **PASS** |
| **FFT scaling** | $\tilde{x}_k = \Delta t X_k$ | Delta function and random Gaussian signals | Matches raw `rfft` $\times \Delta t$ within $1.11 \times 10^{-16}$ | **PASS** |
| **Parseval identity** | Relative error $< 10^{-12}$ | Deterministic & random arrays ($N \le 32{,}768$) | Relative errors: $0.0$ to $4.44 \times 10^{-16} \ll 10^{-12}$ | **PASS** |
| **DC weighting** | Weight $1$ (not doubled) | Pure DC signal $x_j = C$ | Doubling DC produces $100\%$ error; weight $1$ error $< 10^{-14}$ | **PASS** |
| **Nyquist weighting** | Weight $1$ (not doubled) | Pure Nyquist tone $x_j = (-1)^j C$ | Doubling Nyquist produces $100\%$ error; weight $1$ error $< 10^{-14}$ | **PASS** |
| **Interior-bin weighting** | Weight $2$ (folded) | Pure interior sinusoid at $k = 15$ | Undoubling produces $50\%$ error; weight $2$ error $< 10^{-14}$ | **PASS** |
| **Periodogram units and scaling** | $\sum \widehat{S}_n \Delta f = \frac{1}{N} \sum x_j^2$ | $N = 4096, \sigma = 2.5$ random Gaussian | Discrepancy: $1.11 \times 10^{-16}$ | **PASS** |
| **Noise reproducibility** | Bitwise identity for identical seeds | Seed $777, N = 1024$ | Bitwise array equality confirmed (`assert_array_equal`) | **PASS** |
| **Noise input validation** | Raises ValueError / TypeError | $N \le 0, \sigma < 0, \mathrm{NaN}, \mathrm{Inf}$, invalid types | Descriptive exceptions raised for all invalid inputs | **PASS** |
| **Time-series input validation (Correction A)** | Rejects complex, NaN, $\pm\mathrm{Inf}$, empty, odd, 2D | Tested across all 4 time-series functions | 40 parameterized tests; all invalid inputs rejected | **PASS** |
| **White-noise ensemble PSD (interior)** | $\langle \widehat{S}_n \rangle \to 2\sigma^2 / f_s$ | $1000$ realizations, $N = 32{,}768, \sigma = 1.0$ | Measured: $4.8818 \times 10^{-4}$, expected: $4.8828 \times 10^{-4}$ (rel error: $-0.0207\%$) | **PASS** |
| **DC/Nyquist ensemble PSD (Correction B)** | $\langle \widehat{S}_n \rangle \to \sigma^2 / f_s$ with $\mathrm{SE} = \frac{\sqrt{2}S}{\sqrt{R}}$ | $1000$ realizations, $N = 32{,}768, \sigma = 1.0$ | DC: $-1.57\%$ ($-0.37\sigma$), Nyquist: $-3.11\%$ ($-0.68\sigma$) | **PASS** |

#### 3. 1000-Realization Monte Carlo Simulation Results
Executed via `python scripts/validate_white_noise_psd.py --realizations 1000 --seed 20261102` (runtime: $1.20 \, \mathrm{s}$):

```text
==============================================================================
STAGE 1.1: WHITE-NOISE PERIODOGRAM ENSEMBLE VALIDATION
==============================================================================
Sampling frequency (f_s) : 4096.0 Hz (Delta t = 2.44140625e-04 s)
Sample count (N)        : 32768 (T = 8.00 s)
Frequency resolution    : 0.1250 Hz (Nyquist = 2048.0 Hz)
Total frequency bins    : 16385 (Interior bins: 16383)
Noise standard dev      : 1.0000 (Variance = 1.0000)
Ensemble realizations   : 1000
Random seed             : 20261102
Numerical precision     : IEEE 754 float64 (double precision)
------------------------------------------------------------------------------
  Processed  200 / 1000 realizations (0.23 s)...
  Processed  400 / 1000 realizations (0.47 s)...
  Processed  600 / 1000 realizations (0.70 s)...
  Processed  800 / 1000 realizations (0.89 s)...
  Processed 1000 / 1000 realizations (1.20 s)...
------------------------------------------------------------------------------
RESULTS SUMMARY:
------------------------------------------------------------------------------
1. Interior Frequency Bins (0.125 Hz to 2047.875 Hz, 16383 bins):
   - Theoretical expectation S_0 = 2 sigma^2 / f_s : 4.88281250e-04 [signal]^2/Hz
   - Measured ensemble grand mean                  : 4.88180023e-04 [signal]^2/Hz
   - Empirical SEM of grand mean                   : 1.20646103e-07 [signal]^2/Hz
   - Relative deviation                            : -0.0207%
   - Standardized residual (z-score)               : -0.84 sigma

2. DC Frequency Bin (f = 0.0 Hz, k = 0):
   - Theoretical expectation S_DC = sigma^2 / f_s  : 2.44140625e-04 [signal]^2/Hz
   - Theoretical SE = sqrt(2) S_DC / sqrt(R)       : 1.09183007e-05 (4.47%)
   - Measured ensemble mean                        : 2.40301002e-04 [signal]^2/Hz
   - Empirical SEM of mean                         : 1.05040570e-05 [signal]^2/Hz
   - Relative deviation                            : -1.5727%
   - Standardized residual (empirical z-score)     : -0.37 sigma

3. Nyquist Frequency Bin (f = 2048.0 Hz, k = 16384):
   - Theoretical expectation S_Nyq = sigma^2 / f_s : 2.44140625e-04 [signal]^2/Hz
   - Theoretical SE = sqrt(2) S_Nyq / sqrt(R)      : 1.09183007e-05 (4.47%)
   - Measured ensemble mean                        : 2.36543770e-04 [signal]^2/Hz
   - Empirical SEM of mean                         : 1.11578786e-05 [signal]^2/Hz
   - Relative deviation                            : -3.1117%
   - Standardized residual (empirical z-score)     : -0.68 sigma

4. Parseval Power Conservation:
   - Mean frequency-domain integrated PSD          : 9.99791270e-01
   - Mean time-domain mean-square power            : 9.99791270e-01
   - Discrepancy                                   : 1.11e-16

Total execution time: 1.20 seconds.
Validation status: PASSED
==============================================================================
```

---

### D. Scientific Limitations

Stage 1.1 establishes the foundational discrete Fourier grid, scaling, and white-noise generator. The following physical and numerical domains remain **unvalidated and strictly outside this stage**:

1. **Coloured-Noise Generation:** Noise generation is strictly limited to white Gaussian noise. Coloring noise via analytical curves (e.g. aLIGO ZDHP) is deferred to Stage 1.2.
2. **Windowed PSD Normalization:** No window functions (Tukey $\alpha = 0.25$ or Hann) were applied. Window power loss correction ($S_2$) and Equivalent Noise Bandwidth ($\mathrm{ENBW}$) belong to Stage 1.2.
3. **Welch Estimation & Bias Correction:** No segment partitioning, segment averaging, or finite-sample median correction factors ($C(K)$) have been implemented.
4. **Filtered-Noise Normalization:** No Butterworth digital filtering has been implemented; the $|H(f)|^4 S_n(f)$ filtered residual scaling is not tested.
5. **Overlap Covariance & Pseudo-Covariance:** Adjacent-segment correlations $C_{12}(k)$ and pseudo-covariances $P_{11}(k), P_{12}(k)$ have not been computed.
6. **Detector Real-Noise Features:** Non-stationary noise, glitches, violin resonances, and real GWOSC data ingestion are outside Stage 1.1.

---

### E. Stage Boundary Confirmation

- **Stage 1.1 Status:** Revised, verified, and complete.
- **Stage 1.2 Status:** **Not started.** No PSD estimation, windowing, or filtering code exists.
- **Detector Data:** Zero gravitational-wave detector data have been downloaded or ingested.
- **Waveforms:** No waveform approximants (`TaylorF2`, `IMRPhenomD`) have been implemented.
- **Inference:** No matched filtering, inner products, likelihood calculations, or Bayesian samplers have been written.
- **Environment:** Implemented using standard Python with NumPy and pytest; no unapproved packages were installed.

---

### F. Independent Scientific Review Required

The Stage 1.1 revision and test results are complete and submitted to the independent reviewer (ChatGPT) for formal evaluation.

**Paths for Reviewer Inspection:**
- Core Conventions Module: [`src/gw_inference/conventions.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/src/gw_inference/conventions.py)
- Noise Generator Module: [`src/gw_inference/noise.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/src/gw_inference/noise.py)
- Conventions Unit Tests: [`tests/test_conventions.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/tests/test_conventions.py)
- Noise Unit Tests: [`tests/test_noise.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/tests/test_noise.py)
- Monte Carlo Script: [`scripts/validate_white_noise_psd.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/scripts/validate_white_noise_psd.py)
- Full Implementation Report: [`reports/Stage_1_1_Implementation_Report.md`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/reports/Stage_1_1_Implementation_Report.md)
- Project Configuration: [`pyproject.toml`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/pyproject.toml)
- Repository Documentation: [`README.md`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/README.md)

**Awaiting independent scientific reviewer evaluation and authorization before proceeding to Stage 1.2.**
