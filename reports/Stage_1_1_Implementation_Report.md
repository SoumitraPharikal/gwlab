# Stage 1.1 Technical Implementation Report
## Discrete Fourier Conventions, Scaling, and Synthetic Time-Series Generation

**Project Title:** Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals  
**Stage:** 1.1 — Discrete Fourier Conventions and Synthetic Time-Series Generation  
**Date:** 2026-11-02  
**Implementation Status:** Complete and awaiting independent reviewer evaluation.  
**Execution Environment:** Windows 11 host Python 3.13.5 (64-bit), NumPy 2.1.2, pytest 9.2.0.  

---

### A. Implementation Summary

The following files were created in the repository root `C:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference`:

1. `pyproject.toml`: Modern packaging specification defining project metadata, build backend, dependencies (`numpy >= 1.24.0`, `pytest >= 7.0.0`), and pytest configuration (`pythonpath = ["src"]`, `testpaths = ["tests"]`).
2. `.gitignore`: Standard Python and testing ignore file excluding bytecode, `.pytest_cache`, coverage files, and environment directories.
3. `src/gw_inference/__init__.py`: Package entrypoint cleanly exporting the core public API.
4. `src/gw_inference/conventions.py`: Core mathematical module implementing frequency-grid generation, continuous FFT scaling ($\tilde{x}(f_k) = \Delta t X_k$), time- and frequency-domain Parseval energy calculations, Parseval verification, one-sided periodogram calculation, and theoretical white-noise PSD formulas.
5. `src/gw_inference/noise.py`: Generator module implementing `generate_white_noise` using `numpy.random.default_rng`, fully avoiding global random state, validating inputs, and supporting explicit seeds and generator instances.
6. `tests/test_conventions.py`: Comprehensive test suite containing 34 automated unit tests covering frequency-grid construction, continuous FFT scaling, deterministic and random Parseval identity balances, DC/Nyquist/interior weighting, periodogram energy integrals, and a 1-to-1 mapping with the Stage 1.1 mandatory summary matrix.
7. `tests/test_noise.py`: Test suite containing 10 unit tests covering sample dimensions, data type, finiteness, bitwise reproducibility across identical seeds, divergence across different seeds, sample statistical convergence (mean and variance), input validation errors, and a fast automated ensemble periodogram test.
8. `scripts/validate_white_noise_psd.py`: Executable standalone CLI script executing the full 1000-realization Monte Carlo simulation at the baseline resolution ($N = 32{,}768, f_s = 4096 \, \mathrm{Hz}$), reporting ensemble averages, standard errors, z-scores, and Parseval energy conservation.
9. `README.md`: Comprehensive public-facing repository documentation summarizing research goals, Stage 1.1 scope, mathematical conventions, directory structure, and instructions to reproduce tests.
10. `reports/Stage_1_1_Implementation_Report.md`: This comprehensive technical audit report.

No historical Stage 0 audit documents were overwritten or modified.

---

### B. Mathematical Convention Audit

All numerical implementations strictly reflect the approved Stage 0 conventions:

#### 1. Continuous Fourier Transform
$$\tilde{x}(f) = \int_{-\infty}^\infty x(t) e^{-2\pi i f t} dt$$

#### 2. Discrete Sampling Parameters
- Sampling frequency: $f_s = 4096.0 \, \mathrm{Hz}$
- Sampling interval: $\Delta t = 1 / f_s = 2.44140625 \times 10^{-4} \, \mathrm{s}$
- Sample count: $N = 32{,}768$ (even integer)
- Total duration: $T = N \Delta t = 8.0 \, \mathrm{s}$
- Frequency resolution: $\Delta f = 1 / T = f_s / N = 0.125 \, \mathrm{Hz}$

#### 3. Frequency Grid Construction
For an even-length real time series of $N$ samples, the one-sided physical frequency grid consists of $N/2 + 1 = 16{,}385$ discrete bins:
$$f_k = k \Delta f = \frac{k}{N \Delta t}, \qquad k = 0, 1, \dots, \frac{N}{2}$$
- DC bin ($k = 0$): $f_0 = 0.0 \, \mathrm{Hz}$.
- Nyquist bin ($k = N/2 = 16{,}384$): $f_{N/2} = f_s / 2 = 2048.0 \, \mathrm{Hz}$.
- Implementation: `gw_inference.conventions.build_frequency_grid(n_samples, delta_t)`.

#### 4. Unnormalised Real FFT Scaling
The forward unnormalised discrete Fourier transform computed by `numpy.fft.rfft` on real sequence $x_j$ is:
$$X_k = \sum_{j=0}^{N-1} x_j e^{-2\pi i j k / N}, \qquad k = 0, 1, \dots, \frac{N}{2}$$
The continuous-time Fourier transform approximation is:
$$\tilde{x}(f_k) \simeq \Delta t \, X_k = \frac{1}{f_s} X_k$$
No extra $1/N$ factor is applied to the forward transform.  
Implementation: `gw_inference.conventions.continuous_rfft(x, delta_t)`.

#### 5. Discrete Parseval Energy Balance
For real-valued time series $x_j$, energy conservation between time and frequency domains is formulated exactly as:
$$\Delta t \sum_{j=0}^{N-1} |x_j|^2 = \Delta f \, \Delta t^2 \left[ |X_0|^2 + 2 \sum_{k=1}^{N/2-1} |X_k|^2 + |X_{N/2}|^2 \right]$$
**Weight justification:**
- Interior bins ($1 \le k < N/2$) carry weight $2$ because positive frequency $f_k$ and negative frequency $-f_k$ have equal squared magnitudes $|X_k|^2 = |X_{-k}|^2$ under Hermitian symmetry $X_{-k} = X_k^*$.
- DC ($k = 0$) and Nyquist ($k = N/2$) are self-conjugate and non-degenerate in the two-sided spectrum; they carry weight $1$.
Implementation: `gw_inference.conventions.validate_parseval_identity(x, delta_t)`.

#### 6. One-Sided Periodogram Normalization
For an unwindowed discrete time series $x_j$, the one-sided power spectral density estimator is:
$$\widehat{S}_n(f_k) = \begin{cases} \displaystyle \frac{\Delta t}{N} |X_k|^2, & k = 0, \, \frac{N}{2} \\[8pt] \displaystyle \frac{2 \Delta t}{N} |X_k|^2, & 1 \le k < \frac{N}{2} \end{cases}$$
This has physical units of $[\mathrm{signal}]^2 / \mathrm{Hz}$ and satisfies the continuous Parseval average power relation:
$$\sum_{k=0}^{N/2} \widehat{S}_n(f_k) \Delta f = \frac{1}{N} \sum_{j=0}^{N-1} |x_j|^2$$
Implementation: `gw_inference.conventions.one_sided_periodogram(x, delta_t)`.

#### 7. Expected White-Noise PSD
For discrete zero-mean Gaussian white noise $n_j \sim \mathcal{N}(0, \sigma^2)$:
$$\mathbb{E}[n_j n_l] = \sigma^2 \delta_{j, l} \quad \Longrightarrow \quad \mathbb{E}[|X_k|^2] = \sigma^2 N \quad (\forall k)$$
Substituting into the one-sided periodogram normalization gives the exact analytical expectations:
$$\mathbb{E}[\widehat{S}_n(f_k)] = \frac{2 \sigma^2 \Delta t}{N} N = 2 \sigma^2 \Delta t = \frac{2 \sigma^2}{f_s} \quad \left(1 \le k < \frac{N}{2}\right)$$
$$\mathbb{E}[\widehat{S}_n(0)] = \mathbb{E}\left[\widehat{S}_n\left(\frac{f_s}{2}\right)\right] = \frac{\sigma^2 \Delta t}{N} N = \sigma^2 \Delta t = \frac{\sigma^2}{f_s} \quad \left(k = 0, \, \frac{N}{2}\right)$$
Implementation: `gw_inference.conventions.expected_white_noise_psd(sigma, delta_t, bin_type)`.

---

### C. Test Report and Numerical Results

#### 1. Pytest Test Suite Results
The full test suite was executed on the Windows 11 host environment using Python 3.13.5 and pytest 9.2.0:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference
configfile: pyproject.toml
plugins: anyio-4.6.2.post1, zarr-3.2.1
collected 44 items

tests/test_conventions.py::TestFrequencyGrid::test_baseline_grid_properties PASSED [  2%]
tests/test_conventions.py::TestFrequencyGrid::test_grid_length_and_endpoints[10] PASSED [  4%]
tests/test_conventions.py::TestFrequencyGrid::test_grid_length_and_endpoints[64] PASSED [  6%]
tests/test_conventions.py::TestFrequencyGrid::test_grid_length_and_endpoints[1024] PASSED [  9%]
tests/test_conventions.py::TestFrequencyGrid::test_grid_length_and_endpoints[8192] PASSED [ 11%]
tests/test_conventions.py::TestFrequencyGrid::test_invalid_parameters PASSED [ 13%]
tests/test_conventions.py::TestContinuousRFFTScaling::test_impulse_response PASSED [ 15%]
tests/test_conventions.py::TestContinuousRFFTScaling::test_unnormalised_rfft_scaling_factor PASSED [ 18%]
tests/test_conventions.py::TestContinuousRFFTScaling::test_input_validation PASSED [ 20%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[unit_impulse_origin] PASSED [ 22%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[shifted_impulse] PASSED [ 25%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[pure_dc] PASSED [ 27%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[pure_nyquist] PASSED [ 29%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[single_tone] PASSED [ 31%]
tests/test_conventions.py::TestParsevalIdentity::test_deterministic_signals[multitone] PASSED [ 34%]
tests/test_conventions.py::TestParsevalIdentity::test_random_gaussian_signals[128] PASSED [ 36%]
tests/test_conventions.py::TestParsevalIdentity::test_random_gaussian_signals[1024] PASSED [ 38%]
tests/test_conventions.py::TestParsevalIdentity::test_random_gaussian_signals[8192] PASSED [ 40%]
tests/test_conventions.py::TestParsevalIdentity::test_random_gaussian_signals[32768] PASSED [ 43%]
tests/test_conventions.py::TestParsevalIdentity::test_dc_bin_weighting PASSED [ 45%]
tests/test_conventions.py::TestParsevalIdentity::test_nyquist_bin_weighting PASSED [ 47%]
tests/test_conventions.py::TestParsevalIdentity::test_interior_bin_weighting PASSED [ 50%]
tests/test_conventions.py::TestOneSidedPeriodogram::test_periodogram_integral_equals_time_mean_square PASSED [ 52%]
tests/test_conventions.py::TestOneSidedPeriodogram::test_expected_white_noise_psd_formula PASSED [ 54%]
tests/test_conventions.py::TestStage11MandatorySummary::test_frequency_grid_construction PASSED [ 56%]
tests/test_conventions.py::TestStage11MandatorySummary::test_fft_scaling PASSED [ 59%]
tests/test_conventions.py::TestStage11MandatorySummary::test_parseval_identity_tolerance PASSED [ 61%]
tests/test_conventions.py::TestStage11MandatorySummary::test_dc_weighting_not_doubled PASSED [ 63%]
tests/test_conventions.py::TestStage11MandatorySummary::test_nyquist_weighting_not_doubled PASSED [ 65%]
tests/test_conventions.py::TestStage11MandatorySummary::test_interior_bin_weighting_doubled PASSED [ 68%]
tests/test_conventions.py::TestStage11MandatorySummary::test_periodogram_units_and_scaling PASSED [ 70%]
tests/test_conventions.py::TestStage11MandatorySummary::test_noise_reproducibility PASSED [ 72%]
tests/test_conventions.py::TestStage11MandatorySummary::test_noise_input_validation PASSED [ 75%]
tests/test_conventions.py::TestStage11MandatorySummary::test_white_noise_ensemble_psd_interior PASSED [ 77%]
tests/test_conventions.py::TestStage11MandatorySummary::test_dc_nyquist_ensemble_psd_half_level PASSED [ 79%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_sample_count_and_dtype PASSED [ 81%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_output_finite PASSED  [ 84%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_reproducibility_identical_seeds PASSED [ 86%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_different_seeds_produce_different_realizations PASSED [ 88%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_generator_instance_support PASSED [ 90%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_zero_sigma PASSED     [ 93%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_sample_statistics_convergence PASSED [ 95%]
tests/test_noise.py::TestWhiteNoiseGenerator::test_input_validation PASSED [ 97%]
tests/test_noise.py::TestEnsemblePeriodogramMini::test_ensemble_periodogram_expectation PASSED [100%]

============================= 44 passed in 0.44s ==============================
```

#### 2. Mandatory Test Compliance Matrix

| Mandatory Test Category (Prompt §7) | Expected Analytical Property | Numerical Test Case | Actual Result | Status |
|---|---|---|---|---|
| **Frequency-grid construction** | $N/2 + 1$ bins, $f_0 = 0$, $f_{N/2} = f_s/2$, $\Delta f = 1/T$ | $N = 32{,}768, f_s = 4096 \, \mathrm{Hz}$ | $16{,}385$ bins, $0.0 \, \mathrm{Hz}$ to $2048.0 \, \mathrm{Hz}$, $\Delta f = 0.125 \, \mathrm{Hz}$ | **PASS** |
| **FFT scaling** | $\tilde{x}_k = \Delta t X_k$ | Delta function and random Gaussian signals | Matches raw rfft times $\Delta t$ within $1.11 \times 10^{-16}$ | **PASS** |
| **Parseval identity** | Relative error $< 10^{-12}$ | Baseline $N = 32{,}768$, random & deterministic | Relative errors: $0.0$ to $4.44 \times 10^{-16} \ll 10^{-12}$ | **PASS** |
| **DC weighting** | Weight $1$ (not doubled) | Pure DC signal $x_j = C$ | Doubling DC produces $100\%$ relative error; weight $1$ gives error $< 10^{-14}$ | **PASS** |
| **Nyquist weighting** | Weight $1$ (not doubled) | Pure Nyquist tone $x_j = (-1)^j C$ | Doubling Nyquist produces $100\%$ relative error; weight $1$ gives error $< 10^{-14}$ | **PASS** |
| **Interior-bin weighting** | Weight $2$ (folded) | Pure interior sinusoid at $k = 15$ | Omitting factor of 2 produces $50\%$ error; weight $2$ gives error $< 10^{-14}$ | **PASS** |
| **Periodogram scaling** | Total energy matches time mean-square | $N = 4096, \sigma = 2.5$ random Gaussian | $|\sum \widehat{S}_n \Delta f - \langle x^2 \rangle| / \langle x^2 \rangle = 1.11 \times 10^{-16}$ | **PASS** |
| **Noise reproducibility** | Bitwise identity for identical seeds | Seed $777, N = 1024$ | Bitwise array equality verified (`assert_array_equal`) | **PASS** |
| **Noise input validation** | Raises ValueError / TypeError | $N \le 0, \sigma < 0, \mathrm{NaN}, \mathrm{Inf}$, wrong types | All invalid inputs trigger appropriate descriptive errors | **PASS** |
| **White-noise ensemble PSD (interior)** | $\langle \widehat{S}_n \rangle \to 2\sigma^2 / f_s$ | 1000 realizations, $N = 32{,}768, \sigma = 1.0$ | Measured: $4.8818 \times 10^{-4}$, expected: $4.8828 \times 10^{-4}$ (rel error: $-0.0207\%$) | **PASS** |
| **DC/Nyquist ensemble PSD** | $\langle \widehat{S}_n \rangle \to \sigma^2 / f_s$ | 1000 realizations, $N = 32{,}768, \sigma = 1.0$ | DC: $-1.57\%$ ($-0.37\sigma$), Nyquist: $-3.11\%$ ($-0.68\sigma$) | **PASS** |

#### 3. Full 1000-Realization Monte Carlo Simulation Results

The standalone verification experiment `scripts/validate_white_noise_psd.py` was executed with parameters:
- $N_{\mathrm{realizations}} = 1000$
- $N = 32{,}768$ samples per realization
- $f_s = 4096.0 \, \mathrm{Hz}$ ($\Delta t = 2.44140625 \times 10^{-4} \, \mathrm{s}$, $\Delta f = 0.125 \, \mathrm{Hz}$)
- Noise standard deviation $\sigma = 1.0000$ ($\sigma^2 = 1.0000 \, [\mathrm{signal}]^2$)
- Random seed: $20261102$
- Numerical precision: 64-bit IEEE 754 floating point (`numpy.float64`)
- Wall-clock runtime: $1.58 \, \mathrm{s}$

**Numerical Outcomes:**
1. **Interior Frequency Bins ($16{,}383$ bins from $0.125 \, \mathrm{Hz}$ to $2047.875 \, \mathrm{Hz}$):**
   - Theoretical Expectation: $S_0 = 2\sigma^2 / f_s = 4.88281250 \times 10^{-4} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Measured Ensemble Grand Mean: $\langle \bar{S}_{\mathrm{interior}} \rangle = 4.88180023 \times 10^{-4} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Standard Error of Grand Mean: $\mathrm{SEM} = 1.20646103 \times 10^{-7} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Relative Deviation: $-0.0207\%$ ($-2.07 \times 10^{-4}$)
   - Standardized Z-Score: $Z = -0.84 \, \sigma$ (statistically consistent with zero bias under standard normal distribution)

2. **DC Bin ($k = 0, f = 0.0 \, \mathrm{Hz}$):**
   - Theoretical Expectation: $S_{\mathrm{DC}} = \sigma^2 / f_s = 2.44140625 \times 10^{-4} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Measured Ensemble Mean: $\bar{S}(0) = 2.40301002 \times 10^{-4} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Standard Error of Single Bin: $\mathrm{SEM} = 1.05040570 \times 10^{-5} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Relative Deviation: $-1.5727\%$
   - Standardized Z-Score: $Z = -0.37 \, \sigma$

3. **Nyquist Bin ($k = 16{,}384, f = 2048.0 \, \mathrm{Hz}$):**
   - Theoretical Expectation: $S_{\mathrm{Nyq}} = \sigma^2 / f_s = 2.44140625 \times 10^{-4} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Measured Ensemble Mean: $\bar{S}(N/2) = 2.36543770 \times 10^{-4} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Standard Error of Single Bin: $\mathrm{SEM} = 1.11578786 \times 10^{-5} \, [\mathrm{signal}]^2 / \mathrm{Hz}$
   - Relative Deviation: $-3.1117\%$
   - Standardized Z-Score: $Z = -0.68 \, \sigma$

4. **Parseval Energy Conservation Across Ensemble:**
   - Ensemble-averaged integrated PSD: $\sum_{k=0}^{N/2} \bar{S}(f_k) \Delta f = 0.999791270$
   - Ensemble-averaged time-domain power: $\frac{1}{N} \sum_{j=0}^{N-1} \overline{n_j^2} = 0.999791270$
   - Numerical Discrepancy: $1.11 \times 10^{-16}$ (machine epsilon level)

---

### D. Scientific Limitations

Stage 1.1 establishes the foundational mathematical coordinate system and noise scaling. To maintain strict scientific transparency, the following physical and data-analysis components are **explicitly unvalidated and excluded from this stage**:

1. **Coloured Noise and Spectral Slopes:** The generator is strictly limited to white Gaussian noise. Coloured noise generation via analytical sensitivity curves (e.g. aLIGO ZDHP) is reserved for Stage 1.2.
2. **Windowing and Spectral Leakage:** No tapering windows (Tukey $\alpha = 0.25$ or Hann) were applied. Window power normalization $S_2$ and equivalent noise bandwidth adjustments are deferred to Stage 1.2.
3. **Welch Segment Averaging and Bias Correction:** No Welch segment partitioning, overlapping segment averaging, or finite-sample median correction factors ($C(K)$) have been implemented.
4. **Filtered Noise Residuals:** No digital filtering (Butterworth `filtfilt`) has been implemented; the $|H(f)|^4$ filtered PSD response is not tested here.
5. **Segment Overlap Covariance and Pseudo-Covariance:** The cross-covariance $C_{12}(k)$ and pseudo-covariances $P_{11}(k), P_{12}(k)$ derived in Stage 0.6 have not been computed.
6. **Detector Real-Data Characteristics:** Non-stationarities, non-Gaussian glitches, line features (violin modes, $60 \, \mathrm{Hz}$ harmonics), and real GWOSC data ingestion are outside Stage 1.1 scope.

---

### E. Stage Boundary Confirmation

I confirm the following strict execution boundaries:

- **Stage 1.1 Status:** Fully implemented, verified, and completed.
- **Stage 1.2 Status:** **Not started.** No PSD estimation or filtering code has been written.
- **Detector Data:** No strain data from GWOSC or local cache have been downloaded or ingested.
- **Waveforms:** No gravitational-wave approximants (`TaylorF2`, `IMRPhenomD`) have been written.
- **Matched Filtering & Bayesian Inference:** No SNR time series, Whittle likelihood evaluations, phase marginalization algorithms, or nested samplers have been created.
- **Environment Integrity:** Host Windows 11 environment Python 3.13.5 with pre-installed NumPy 2.1.2 and pytest 9.2.0 was utilized per user approval without modifying local packages.

---

### F. Independent Scientific Review Required

The Stage 1.1 implementation is submitted to the independent scientific reviewer (ChatGPT) for evaluation. Implementation of Stage 1.2 remains suspended pending formal reviewer authorization.

#### Artifacts Available for Reviewer Inspection:
- Core Module: [`src/gw_inference/conventions.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/src/gw_inference/conventions.py)
- Noise Module: [`src/gw_inference/noise.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/src/gw_inference/noise.py)
- Conventions Test Suite: [`tests/test_conventions.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/tests/test_conventions.py)
- Noise Test Suite: [`tests/test_noise.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/tests/test_noise.py)
- Monte Carlo Script: [`scripts/validate_white_noise_psd.py`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/scripts/validate_white_noise_psd.py)
- Repository Documentation: [`README.md`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/README.md)
- Build Config: [`pyproject.toml`](file:///C:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/pyproject.toml)
