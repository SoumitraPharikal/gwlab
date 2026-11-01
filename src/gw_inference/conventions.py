"""Discrete Fourier transform conventions, scaling, and Parseval validation.

This module implements the authoritative mathematical and numerical conventions
approved in Stage 0 for discrete Fourier transforms, frequency grids,
continuous-transform scaling, Parseval energy conservation, and one-sided
periodogram normalization for real-valued discrete time series.

Mathematical Conventions:
-------------------------
1. Continuous Fourier Transform:
   \\tilde{x}(f) = \\int_{-\\infty}^{\\infty} x(t) e^{-2\\pi i f t} dt

2. Discrete Sampling:
   f_s = 1 / \\Delta t,  T = N \\Delta t,  \\Delta f = 1 / T = f_s / N

3. One-Sided Frequency Grid:
   f_k = k \\Delta f,  k = 0, 1, ..., N/2  (N/2 + 1 bins for even N)

4. NumPy FFT Scaling:
   X_k = rfft(x_j) (unnormalised)
   \\tilde{x}(f_k) \\simeq \\Delta t X_k

5. Discrete Parseval Identity:
   \\Delta t \\sum_{j=0}^{N-1} |x_j|^2 =
       \\Delta f \\Delta t^2 [|X_0|^2 + 2 \\sum_{k=1}^{N/2-1} |X_k|^2 + |X_{N/2}|^2]

6. One-Sided Periodogram Normalization:
   \\hat{S}_n(f_k) = (\\Delta t / N) |X_k|^2        for k = 0, N/2
   \\hat{S}_n(f_k) = (2 \\Delta t / N) |X_k|^2      for 1 <= k < N/2
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np


def build_frequency_grid(n_samples: int, delta_t: float) -> np.ndarray:
    """Construct the one-sided physical frequency grid for an even-length real time series.

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples N. Must be a positive even integer.
    delta_t : float
        Sampling interval in seconds (\\Delta t = 1 / f_s). Must be positive and finite.

    Returns
    -------
    np.ndarray
        Array of frequency bin centers f_k = k \\Delta f for k = 0, ..., N/2,
        with length N/2 + 1, starting at DC (0.0 Hz) and ending at Nyquist (f_s / 2).

    Raises
    ------
    ValueError
        If n_samples is not positive or not even, or if delta_t is not positive or finite.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
    if n_samples % 2 != 0:
        raise ValueError(f"n_samples must be an even integer, got {n_samples}")
    if not isinstance(delta_t, (int, float, np.floating)) or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError(f"delta_t must be a positive finite float, got {delta_t}")

    # Explicit construction f_k = k * delta_f
    delta_f = 1.0 / (n_samples * delta_t)
    n_freqs = n_samples // 2 + 1
    freqs = np.arange(n_freqs, dtype=np.float64) * delta_f
    return freqs


def continuous_rfft(x: np.ndarray, delta_t: float) -> np.ndarray:
    """Compute continuous Fourier transform approximation via unnormalised real FFT.

    Approximation:
        \\tilde{x}(f_k) \\simeq \\Delta t \\cdot \\operatorname{rfft}(x)_k

    Parameters
    ----------
    x : np.ndarray
        Real-valued 1D time-series array of even length N.
    delta_t : float
        Sampling interval in seconds.

    Returns
    -------
    np.ndarray
        Complex-valued array of length N/2 + 1 containing the continuous Fourier
        transform approximation at frequencies f_k = k / (N \\Delta t).

    Raises
    ------
    ValueError
        If x is not 1D, has odd length, contains non-finite values, or is complex.
        If delta_t is not a positive finite float.
    """
    x_arr = np.asarray(x)
    if x_arr.ndim != 1:
        raise ValueError(f"x must be a 1D array, got dimension {x_arr.ndim}")
    if x_arr.size == 0 or x_arr.size % 2 != 0:
        raise ValueError(f"x must have non-zero even length, got {x_arr.size}")
    if np.iscomplexobj(x_arr):
        raise ValueError("x must be real-valued")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must contain only finite numbers")
    if not isinstance(delta_t, (int, float, np.floating)) or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError(f"delta_t must be a positive finite float, got {delta_t}")

    unnormalised_rfft = np.fft.rfft(x_arr)
    return delta_t * unnormalised_rfft


def discrete_parseval_time_energy(x: np.ndarray, delta_t: float) -> float:
    """Compute total continuous-energy integral approximation in the time domain.

    E_t = \\Delta t \\sum_{j=0}^{N-1} |x_j|^2

    Parameters
    ----------
    x : np.ndarray
        Real-valued 1D time-series array of even length N.
    delta_t : float
        Sampling interval in seconds.

    Returns
    -------
    float
        Time-domain discrete energy sum with units of [x]^2 * seconds.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size % 2 != 0:
        raise ValueError("x must be a 1D array with even length")
    if not (math.isfinite(delta_t) and delta_t > 0):
        raise ValueError("delta_t must be positive and finite")

    return float(delta_t * np.sum(x_arr ** 2))


def discrete_parseval_freq_energy(
    X: np.ndarray, delta_t: float, delta_f: float
) -> float:
    """Compute total energy from unnormalised one-sided rfft coefficients.

    E_f = \\Delta f \\Delta t^2 \\left[ |X_0|^2 + 2 \\sum_{k=1}^{N/2-1} |X_k|^2 + |X_{N/2}|^2 \\right]

    Interior bins (1 <= k < N/2) receive a folding weight of 2.
    DC (k = 0) and Nyquist (k = N/2) receive a weight of 1.

    Parameters
    ----------
    X : np.ndarray
        Unnormalised complex rfft coefficients of length N/2 + 1.
    delta_t : float
        Sampling interval in seconds.
    delta_f : float
        Frequency resolution in Hz (\\Delta f = 1 / (N \\Delta t)).

    Returns
    -------
    float
        Frequency-domain folded energy sum with units of [x]^2 * seconds.
    """
    X_arr = np.asarray(X, dtype=np.complex128)
    if X_arr.ndim != 1 or X_arr.size < 2:
        raise ValueError("X must be a 1D array of at least 2 bins (DC and Nyquist)")
    if not (math.isfinite(delta_t) and delta_t > 0):
        raise ValueError("delta_t must be positive and finite")
    if not (math.isfinite(delta_f) and delta_f > 0):
        raise ValueError("delta_f must be positive and finite")

    # Squared magnitudes of rfft coefficients
    sq_mag = np.abs(X_arr) ** 2

    # Folded sum: DC (k=0) has weight 1, Nyquist (k=-1) has weight 1, interior bins have weight 2
    if sq_mag.size == 2:
        folded_sum = sq_mag[0] + sq_mag[1]
    else:
        folded_sum = sq_mag[0] + 2.0 * np.sum(sq_mag[1:-1]) + sq_mag[-1]

    return float(delta_f * (delta_t ** 2) * folded_sum)


def validate_parseval_identity(
    x: np.ndarray, delta_t: float
) -> Tuple[float, float, float]:
    """Validate discrete Parseval energy balance for a real-valued time series.

    Parameters
    ----------
    x : np.ndarray
        Real-valued 1D time series of even length N.
    delta_t : float
        Sampling interval in seconds.

    Returns
    -------
    tuple of (float, float, float)
        (time_energy, freq_energy, relative_error) where:
        relative_error = |time_energy - freq_energy| / time_energy
        (or absolute difference if time_energy == 0).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    n_samples = x_arr.size
    delta_f = 1.0 / (n_samples * delta_t)

    e_time = discrete_parseval_time_energy(x_arr, delta_t)
    X = np.fft.rfft(x_arr)
    e_freq = discrete_parseval_freq_energy(X, delta_t, delta_f)

    if e_time == 0.0:
        rel_err = abs(e_time - e_freq)
    else:
        rel_err = abs(e_time - e_freq) / e_time

    return e_time, e_freq, float(rel_err)


def one_sided_periodogram(x: np.ndarray, delta_t: float) -> np.ndarray:
    """Compute the standard one-sided periodogram for an unwindowed real time series.

    Normalization:
        \\hat{S}_n(f_k) = \\begin{cases}
            \\frac{\\Delta t}{N} |X_k|^2, & k = 0, N/2 \\\\[4pt]
            \\frac{2 \\Delta t}{N} |X_k|^2, & 1 \\le k < N/2
        \\end{cases}
    where X_k = \\operatorname{rfft}(x)_k.

    Units:
        [x]^2 / Hz (e.g. strain^2 / Hz).

    Integral property:
        \\sum_{k=0}^{N/2} \\hat{S}_n(f_k) \\Delta f = \\frac{1}{N} \\sum_{j=0}^{N-1} |x_j|^2

    Parameters
    ----------
    x : np.ndarray
        Real-valued 1D time series of even length N.
    delta_t : float
        Sampling interval in seconds.

    Returns
    -------
    np.ndarray
        1D real float array of length N/2 + 1 containing the one-sided PSD estimate.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size % 2 != 0:
        raise ValueError("x must be a 1D array with even length")
    if not (math.isfinite(delta_t) and delta_t > 0):
        raise ValueError("delta_t must be positive and finite")

    n_samples = x_arr.size
    X = np.fft.rfft(x_arr)
    sq_mag = np.abs(X) ** 2

    # Weighting: 2 * delta_t / N for interior bins, 1 * delta_t / N for DC and Nyquist
    scale = delta_t / n_samples
    psd = 2.0 * scale * sq_mag
    psd[0] = scale * sq_mag[0]
    psd[-1] = scale * sq_mag[-1]

    return psd


def expected_white_noise_psd(
    sigma: float,
    delta_t: float,
    bin_type: Literal["interior", "dc", "nyquist"] = "interior",
) -> float:
    """Return theoretical expected one-sided periodogram for discrete Gaussian white noise.

    For n_j ~ N(0, \\sigma^2):
        E[\\hat{S}_n(f_k)] = 2 \\sigma^2 \\Delta t = 2 \\sigma^2 / f_s   (interior bins: 1 <= k < N/2)
        E[\\hat{S}_n(f_k)] = \\sigma^2 \\Delta t = \\sigma^2 / f_s       (DC: k=0, and Nyquist: k=N/2)

    Parameters
    ----------
    sigma : float
        Standard deviation of white noise in signal units. Must be non-negative and finite.
    delta_t : float
        Sampling interval in seconds. Must be positive and finite.
    bin_type : {'interior', 'dc', 'nyquist'}, default='interior'
        Frequency bin category.

    Returns
    -------
    float
        Theoretical one-sided power spectral density in [signal]^2 / Hz.
    """
    if not isinstance(sigma, (int, float, np.floating)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be a non-negative finite float, got {sigma}")
    if not isinstance(delta_t, (int, float, np.floating)) or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError(f"delta_t must be a positive finite float, got {delta_t}")

    var = sigma ** 2
    if bin_type == "interior":
        return float(2.0 * var * delta_t)
    elif bin_type in ("dc", "nyquist"):
        return float(var * delta_t)
    else:
        raise ValueError(f"Invalid bin_type '{bin_type}', expected 'interior', 'dc', or 'nyquist'")
