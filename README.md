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

## 2. Current Implementation Status: Stage 1.1

**Stage 1.1: Discrete Fourier Conventions, Scaling, and Synthetic Time-Series Generation** is currently implemented and under review.

### Scope of Stage 1.1
- [x] Discrete Fourier transform conventions and frequency-grid construction.
- [x] Continuous/discrete Fourier transform scaling ($\tilde{x}(f_k) \simeq \Delta t X_k$).
- [x] Real-to-complex FFT (`rfft`) folding and one-sided periodogram normalization.
- [x] Numerical validation of the discrete Parseval energy identity across deterministic and random signals (relative error $< 10^{-12}$).
- [x] Reproducible synthetic Gaussian white-noise generation (`numpy.random.default_rng`).
- [x] Controlled 1000-realization Monte Carlo ensemble validation of white-noise periodogram normalization against $S_0 = 2\sigma^2 / f_s$.
- [x] Automated unit test suite (44 tests, 100% passing).
- [x] Technical implementation report (`reports/Stage_1_1_Implementation_Report.md`).

### Strict Execution Boundary
The following belong to subsequent stages and are **strictly excluded** from Stage 1.1:
- Welch PSD estimation, segment windowing, or median-Welch bias correction (Stage 1.2).
- Overlapping-segment covariance and pseudo-covariance estimation (Stage 1.2).
- Butterworth bandpass conditioning and edge transient trimming (Stage 1.2).
- Noise-weighted inner products and matched filter complex SNR time-series (Stage 1.3–1.4).
- Waveform generation (`TaylorF2`, `IMRPhenomD`) (Stage 1.5).
- Likelihood evaluation, phase marginalization, and nested sampling (Stage 1.6+).
- GWOSC strain data download or detector processing (Stage 1.7+).

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
│       └── noise.py                                <- Reproducible Gaussian white-noise generator
│
├── tests/
│   ├── test_conventions.py                         <- 34 unit tests for Fourier conventions & Parseval
│   └── test_noise.py                               <- 10 unit tests for noise generator & statistics
│
├── scripts/
│   └── validate_white_noise_psd.py                 <- 1000-realization Monte Carlo validation script
│
├── reports/
│   └── Stage_1_1_Implementation_Report.md         <- Detailed technical report for scientific reviewer
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
python -m pytest tests -v
```

### Running the 1000-Realization Ensemble Validation
```bash
python scripts/validate_white_noise_psd.py --realizations 1000 --seed 20261102
```

---

## 6. Scientific Review Status
- **Stage 0:** APPROVED by independent scientific reviewer (ChatGPT).
- **Stage 1.1:** Complete, tested, and awaiting formal reviewer evaluation.
