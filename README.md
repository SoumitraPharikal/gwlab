# Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing: pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC.svg)](https://docs.pytest.org/)

A rigorous, reproducible computational gravitational-wave data analysis research repository investigating the effects of noise characterization, power spectral density (PSD) estimation, and waveform-model choices on compact binary coalescence (CBC) parameter estimation.

---

## 1. Project Overview

This repository develops from first principles:
- **Part A:** Gravitational-wave matched filtering and detection statistics.
- **Part B:** Bayesian parameter estimation, likelihood marginalization, and posterior inference robustness.

The project is executed in strictly supervised stages, with formal independent scientific review gating each implementation phase.

---

## 2. Current Implementation Status: Stage 1.2

**Stage 1.1:** Formally APPROVED by independent scientific reviewer.
**Stage 1.2: Welch Power Spectral Density Estimation** is implemented, validated, and awaiting formal review.

### Scope of Stage 1.2
- [x] Discrete symmetric Tukey window generation with exact half-sample midpoint cosine arguments ($\alpha = 0.25$, $M = 32{,}768$, $M_{\mathrm{taper}} = 4096$).
- [x] Exact discrete window power normalization factor $S_2 = \frac{1}{M}\sum w_j^2$ and coherent gain $S_1 = \frac{1}{M}\sum w_j$.
- [x] Configurable segment extraction with non-truncated slicing, trailing data safety, and hop length $H = M - N_{\mathrm{overlap}}$.
- [x] Single-segment windowed one-sided periodogram calculation matching Stage 1.1 when unwindowed ($w_j \equiv 1, S_2 = 1$).
- [x] Standard Welch arithmetic mean PSD estimator with generalized all-lags variance inflation factor $\kappa_{\mathrm{overlap}} = 1 + 2 \sum_{\ell=1}^{\ell_{\max}} (1 - \ell/K) r_\ell^2$.
- [x] Exact discrete pseudo-covariance sums evaluation (`discrete_pseudo_covariance_ratios`) quantifying proper-complex limits and full Isserlis/Wick periodogram correlation.
- [x] Raw sample median Welch PSD estimation available across arbitrary overlap configurations.
- [x] Finite-sample bias-corrected Welch median PSD estimator ($C(K)$ for odd/even $K$) gated behind explicit caller assertion of independent exponential periodograms (`assume_independent_exponential_periodograms=True`) and strictly rejected for overlapping segments.
- [x] Automated test suite (154 tests total: 99 Stage 1.1 tests + 55 Stage 1.2 tests, 100% passing).
- [x] Statistically calibrated numerical validation script (`scripts/validate_welch_psd.py`) using realization-level band statistics, paired bootstrap correlation uncertainty ($B=1000$), exact rectangular median baseline, and diagnostic coloured-noise AR(1) comparison.
- [x] Technical implementation report (`reports/Stage_1_2_Implementation_Report.md`).

### Strict Execution Boundary
The following belong to subsequent stages and remain **strictly excluded** from Stage 1.2:
- Butterworth bandpass filtering, `filtfilt`, and edge-transient trimming (Stage 1.3).
- Frequency-domain noise whitening and normalized residuals (Stage 1.3).
- Noise-weighted inner products, optimal SNR, and matched filtering (Stage 1.4).
- Waveform generation (`TaylorF2`, `IMRPhenomD`) (Stage 1.5).
- Whittle likelihood evaluation and phase marginalization (Stage 1.6).
- MCMC / Nested sampling parameter estimation (Stage 1.7+).
- GWOSC strain data download or real detector data analysis (Stage 1.8+).

---

## 3. Mathematical Foundations and Conventions

### 3.1 Sampling and Grid Parameters
- Sampling frequency: $f_s = 4096 \, \mathrm{Hz}$ ($\Delta t = 1 / f_s = 2.44140625 \times 10^{-4} \, \mathrm{s}$)
- Segment duration: $T = 8.0 \, \mathrm{s}$ ($N = 32{,}768$ samples)
- Frequency resolution: $\Delta f = f_s / N = 1 / T = 0.125 \, \mathrm{Hz}$
- One-sided frequency grid: $f_k = k \Delta f$ for $k = 0, 1, \dots, N/2$ ($16{,}385$ bins from $0 \, \mathrm{Hz}$ DC to $2048 \, \mathrm{Hz}$ Nyquist)

### 3.2 Continuous-to-Discrete Fourier Transform Scaling
$$\tilde{x}(f) = \int_{-\infty}^\infty x(t) e^{-2\pi i f t} dt \quad \Longrightarrow \quad \tilde{x}(f_k) \simeq \Delta t \, X_k$$
where $X_k = \sum_{j=0}^{N-1} x_j e^{-2\pi i j k / N}$ is the unnormalised NumPy real FFT (`numpy.fft.rfft`).

### 3.3 Discrete Parseval Identity
$$\Delta t \sum_{j=0}^{N-1} |x_j|^2 = \Delta f \, \Delta t^2 \left[ |X_0|^2 + 2 \sum_{k=1}^{N/2-1} |X_k|^2 + |X_{N/2}|^2 \right]$$
* DC ($k=0$) and Nyquist ($k=N/2$) bins have non-degenerate weight $1$.
* Interior positive-frequency bins ($1 \le k < N/2$) have weight $2$ due to Hermitian symmetry folding ($+f_k$ and $-f_k$).

### 3.4 One-Sided Periodogram Normalization
$$\widehat{S}_n(f_k) = \begin{cases} \frac{\Delta t}{N} |X_k|^2, & k = 0, \, N/2 \\[6pt] \frac{2\Delta t}{N} |X_k|^2, & 1 \le k < N/2 \end{cases}$$
For discrete white Gaussian noise $n_j \sim \mathcal{N}(0, \sigma^2)$:
$$\mathbb{E}[\widehat{S}_n(f_k)] = \frac{2\sigma^2}{f_s} \quad (1 \le k < N/2), \qquad \mathbb{E}[\widehat{S}_n(0)] = \mathbb{E}[\widehat{S}_n(f_{\mathrm{Nyq}})] = \frac{\sigma^2}{f_s}$$

---

## 4. Repository Structure

```text
GW-matched-filtering-and-baysian-interference/
├── README.md                                       <- Project overview and documentation
├── pyproject.toml                                  <- Build configuration and test dependencies
├── .gitignore                                      <- Git ignore rules
│
├── src/
│   └── gw_inference/
│       ├── __init__.py                             <- Package exports
│       ├── conventions.py                          <- FFT scaling, frequency grid, Parseval, periodogram
│       ├── noise.py                                <- Reproducible Gaussian white-noise generator
│       └── psd.py                                  <- Welch PSD estimation (mean & median), Tukey windowing
│
├── tests/
│   ├── test_conventions.py                         <- 86 unit tests for Fourier conventions & Parseval
│   ├── test_noise.py                               <- 13 unit tests for noise generator & statistics
│   └── test_psd.py                                 <- 55 unit tests for Welch PSD, Tukey, and median
│
├── scripts/
│   ├── validate_white_noise_psd.py                 <- 1000-realization Monte Carlo validation script
│   └── validate_welch_psd.py                       <- Stage 1.2 Welch PSD numerical validation script
│
├── reports/
│   ├── Stage_1_1_Implementation_Report.md         <- Stage 1.1 technical report
│   └── Stage_1_2_Implementation_Report.md         <- Stage 1.2 technical report
│
└── Stage_0_*                                       <- Historical Stage 0 scientific audit documents
```

---

## 5. Getting Started and Running Tests

### Prerequisites
- Python $\ge 3.10$
- `numpy >= 1.24.0`
- `pytest >= 7.0.0`

### Running the Test Suite
```bash
pytest
```

### Running Stage 1.1 White-Noise Periodogram Validation
```bash
python scripts/validate_white_noise_psd.py --realizations 1000 --seed 20261102
```

### Running Stage 1.2 Welch PSD Numerical Validation
```bash
python scripts/validate_welch_psd.py --n-realizations 300 --seed 42
```

---

## 6. Scientific Review Status
- **Stage 0:** APPROVED by independent scientific reviewer (ChatGPT).
- **Stage 1.1:** APPROVED by independent scientific reviewer (ChatGPT).
- **Stage 1.2:** Revised, tested (147 tests passing), and submitted for independent evaluation.
