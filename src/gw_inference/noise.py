"""Synthetic noise generation module for gravitational-wave data analysis.

This module provides reproducible generation of zero-mean discrete Gaussian
white noise using modern NumPy Generator instances (`numpy.random.default_rng`).

Statistical Assumptions:
-------------------------
1. Discrete real Gaussian white noise:
   n_j \\sim \\mathcal{N}(0, \\sigma^2),  \\quad j = 0, 1, \\dots, N-1

2. Independence across samples (delta-correlated):
   \\mathbb{E}[n_j] = 0,  \\quad \\mathbb{E}[n_j n_l] = \\sigma^2 \\delta_{j, l}

3. Flat power spectral density:
   - Two-sided discrete PSD: S_n^{(2)}(f) = \\sigma^2 \\Delta t = \\sigma^2 / f_s
   - One-sided discrete PSD: S_n(f) = 2 \\sigma^2 \\Delta t = 2 \\sigma^2 / f_s  (for f > 0)
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np


def generate_white_noise(
    n_samples: int,
    sigma: float,
    seed: Optional[Union[int, np.random.SeedSequence]] = None,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate reproducible zero-mean Gaussian white noise.

    Parameters
    ----------
    n_samples : int
        Number of discrete time-domain samples to generate. Must be a positive integer.
    sigma : float
        Standard deviation of the noise in physical signal units (e.g. strain).
        Must be a non-negative finite float.
    seed : int or np.random.SeedSequence, optional
        Seed for numpy.random.default_rng. Used only if `generator` is None.
    generator : np.random.Generator, optional
        Pre-existing NumPy Generator instance. If provided, `seed` is ignored.

    Returns
    -------
    np.ndarray
        1D float64 array of length `n_samples` containing the noise realization.

    Raises
    ------
    ValueError
        If `n_samples` is not a positive integer, or if `sigma` is negative or non-finite.
    TypeError
        If `generator` is not an instance of `np.random.Generator`.

    Notes
    -----
    This function avoids global random state by operating strictly through
    `numpy.random.Generator` instances.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

    if not isinstance(sigma, (int, float, np.floating)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be a non-negative finite float, got {sigma}")

    if generator is not None:
        if not isinstance(generator, np.random.Generator):
            raise TypeError(
                f"generator must be an instance of np.random.Generator, got {type(generator)}"
            )
        rng = generator
    else:
        rng = np.random.default_rng(seed)

    if sigma == 0.0:
        return np.zeros(n_samples, dtype=np.float64)

    return rng.normal(loc=0.0, scale=float(sigma), size=int(n_samples)).astype(np.float64)


def generate_colored_noise(
    n_samples: int,
    delta_t: float,
    psd: np.ndarray,
    seed: Optional[Union[int, np.random.SeedSequence]] = None,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate stationary discrete Gaussian noise with a specified one-sided PSD.

    Synthesizes a circularly stationary Gaussian time series on a finite grid of
    `n_samples` using continuous-scaled Fourier normalization.

    Parameters
    ----------
    n_samples : int
        Number of discrete time-domain samples to generate. Must be an integer >= 2.
    delta_t : float
        Sampling interval in seconds (1 / f_s). Must be a positive finite float.
    psd : np.ndarray
        1D float array of one-sided power spectral density values S_n(f_k) in [x]^2 / Hz.
        Length must equal `n_samples // 2 + 1`, matching frequencies
        f_k = k / (n_samples * delta_t) for k = 0, ..., n_samples // 2.
        All values must be finite and non-negative (>= 0). Cannot be identically zero
        across all interior bins.
    seed : int or np.random.SeedSequence, optional
        Seed for numpy.random.default_rng. Used only if `generator` is None.
    generator : np.random.Generator, optional
        Pre-existing NumPy Generator instance. If provided, `seed` is ignored.

    Returns
    -------
    np.ndarray
        1D float64 array of length `n_samples` containing the colored noise realization.

    Raises
    ------
    ValueError
        If `n_samples` < 2, `delta_t` <= 0 or non-finite, `psd` is not 1D, has wrong
        length, contains non-finite or negative values, or is identically zero across
        all interior bins.
    TypeError
        If `generator` is provided and is not an instance of `np.random.Generator`.

    Notes
    -----
    Generation Structure:
    1. Positive interior bins (1 <= k < n_samples / 2):
       X_k = sqrt(n_samples * S_n(f_k) / (4 * delta_t)) * (A_k + 1j * B_k)
       where A_k, B_k ~ i.i.d. N(0, 1). If S_n(f_k) == 0, X_k = 0.
    2. DC bin (k = 0):
       X_0 = sqrt(n_samples * S_n(0) / (2 * delta_t)) * A_0
       where A_0 ~ N(0, 1), Im(X_0) = 0. If S_n(0) == 0, X_0 = 0.
    3. Even-N Nyquist bin (k = n_samples / 2, even n_samples only):
       X_{N/2} = sqrt(n_samples * S_n(f_s/2) / (2 * delta_t)) * A_{N/2}
       where A_{N/2} ~ N(0, 1), Im(X_{N/2}) = 0. If S_n(f_s/2) == 0, X_{N/2} = 0.
    4. Odd-N grids:
       No Nyquist bin. All bins k = 1, ..., (n_samples - 1) / 2 are interior bins.
    5. Time-domain output:
       n = irfft(X, n=n_samples) as float64.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError(f"n_samples must be an integer >= 2, got {n_samples}")

    if not isinstance(delta_t, (int, float, np.floating)) or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError(f"delta_t must be a positive finite float, got {delta_t}")

    psd_arr = np.asarray(psd, dtype=np.float64)
    if psd_arr.ndim != 1:
        raise ValueError(f"psd must be a 1D array, got dimension {psd_arr.ndim}")

    expected_size = n_samples // 2 + 1
    if psd_arr.size != expected_size:
        raise ValueError(
            f"psd length ({psd_arr.size}) must match n_samples // 2 + 1 ({expected_size})"
        )

    if not np.all(np.isfinite(psd_arr)):
        raise ValueError("psd must contain only finite numbers (no NaN or Inf)")

    if np.any(psd_arr < 0.0):
        raise ValueError("psd values must be non-negative (>= 0)")

    N = int(n_samples)
    dt = float(delta_t)

    # Check interior bins:
    if N % 2 == 0:
        int_end = N // 2
    else:
        int_end = expected_size

    if int_end > 1:
        if np.all(psd_arr[1:int_end] == 0.0):
            raise ValueError("psd cannot be identically zero across all interior bins")
    else:
        if np.all(psd_arr == 0.0):
            raise ValueError("psd cannot be identically zero")

    if generator is not None:
        if not isinstance(generator, np.random.Generator):
            raise TypeError(
                f"generator must be an instance of np.random.Generator, got {type(generator)}"
            )
        rng = generator
    else:
        rng = np.random.default_rng(seed)

    X = np.zeros(expected_size, dtype=np.complex128)

    # DC component (k = 0)
    s_dc = psd_arr[0]
    if s_dc > 0.0:
        sigma_dc = math.sqrt(N * s_dc / (2.0 * dt))
        X[0] = complex(rng.normal(0.0, sigma_dc), 0.0)

    # Interior bins
    n_int = int_end - 1
    if n_int > 0:
        s_int = psd_arr[1:int_end]
        sigma_int = np.sqrt(np.maximum(0.0, N * s_int / (4.0 * dt)))
        re = rng.normal(0.0, 1.0, size=n_int) * sigma_int
        im = rng.normal(0.0, 1.0, size=n_int) * sigma_int
        X[1:int_end] = re + 1j * im

    # Nyquist component (k = N // 2, even N only)
    if N % 2 == 0:
        s_nyq = psd_arr[N // 2]
        if s_nyq > 0.0:
            sigma_nyq = math.sqrt(N * s_nyq / (2.0 * dt))
            X[N // 2] = complex(rng.normal(0.0, sigma_nyq), 0.0)

    time_series = np.fft.irfft(X, n=N).astype(np.float64)
    return time_series

