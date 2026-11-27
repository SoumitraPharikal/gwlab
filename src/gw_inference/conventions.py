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
from typing import Literal, Optional, Tuple

import numpy as np


def _validate_real_time_series(x: object, delta_t: object) -> Tuple[np.ndarray, float]:
    """Validate 1D real-valued discrete time series and sampling interval.

    Ensures that input is 1D, non-empty, even-length, real-valued, and finite.
    Prevents silent discarding of imaginary parts by checking dtype before conversion.
    Supports both integer and floating-point real input.

    Parameters
    ----------
    x : array_like
        Input time-series data.
    delta_t : float
        Sampling interval in seconds.

    Returns
    -------
    tuple of (np.ndarray, float)
        Validated array as np.float64, and float delta_t.

    Raises
    ------
    ValueError
        If x is empty, has invalid dimension, odd length, non-real/complex dtype,
        or non-finite values (NaN, +Inf, -Inf).
        If delta_t is not a positive finite float.
    """
    if not isinstance(delta_t, (int, float, np.floating)) or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError(f"delta_t must be a positive finite float, got {delta_t}")

    # Inspect x without forced float64 cast to prevent silent discarding of imaginary part
    x_raw = np.asarray(x)
    if x_raw.ndim != 1:
        raise ValueError(f"x must be a 1D array, got dimension {x_raw.ndim}")
    if x_raw.size == 0:
        raise ValueError("x must be a non-empty array")
    if x_raw.size % 2 != 0:
        raise ValueError(f"x must have an even number of samples, got {x_raw.size}")
    if np.iscomplexobj(x_raw):
        raise ValueError(f"x must be real-valued, got complex dtype {x_raw.dtype}")
    if not np.all(np.isfinite(x_raw)):
        raise ValueError("x must contain only finite numbers (no NaN or Inf)")

    return x_raw.astype(np.float64, copy=False), float(delta_t)


def build_frequency_grid(
    n_samples: int, delta_t: float, allow_odd: bool = False
) -> np.ndarray:
    """Construct the one-sided physical frequency grid for a real time series.

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples N. Must be a positive integer (even by default unless allow_odd=True).
    delta_t : float
        Sampling interval in seconds (\\Delta t = 1 / f_s). Must be positive and finite.
    allow_odd : bool, default=False
        If True, allow odd n_samples where the grid ends at ((N - 1) / (2N)) * f_s < f_s / 2.

    Returns
    -------
    np.ndarray
        Array of frequency bin centers f_k = k \\Delta f for k = 0, ..., floor(N/2),
        with length floor(N/2) + 1, starting at DC (0.0 Hz). For even N, ends at
        Nyquist (f_s / 2); for odd N, ends at (N - 1) / (2 N \\Delta t).

    Raises
    ------
    ValueError
        If n_samples is not positive, or if n_samples is odd and allow_odd is False,
        or if delta_t is not positive or finite.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
    if not allow_odd and n_samples % 2 != 0:
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
    x_arr, dt = _validate_real_time_series(x, delta_t)
    unnormalised_rfft = np.fft.rfft(x_arr)
    return dt * unnormalised_rfft


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
    x_arr, dt = _validate_real_time_series(x, delta_t)
    return float(dt * np.sum(x_arr ** 2))


def discrete_parseval_freq_energy(
    X: np.ndarray, delta_t: float, delta_f: float, n_samples: Optional[int] = None
) -> float:
    """Compute total energy from unnormalised one-sided rfft coefficients.

    For even N:
        E_f = \\Delta f \\Delta t^2 \\left[ |X_0|^2 + 2 \\sum_{k=1}^{N/2-1} |X_k|^2 + |X_{N/2}|^2 \\right]
        Interior bins (1 <= k < N/2) receive a folding weight of 2.
        DC (k = 0) and Nyquist (k = N/2) receive a weight of 1.

    For odd N:
        E_f = \\Delta f \\Delta t^2 \\left[ |X_0|^2 + 2 \\sum_{k=1}^{(N-1)/2} |X_k|^2 \\right]
        DC (k = 0) receives weight 1. All positive frequency bins receive weight 2.
        There is no Nyquist bin.

    Parameters
    ----------
    X : np.ndarray
        Unnormalised complex rfft coefficients of length N//2 + 1.
    delta_t : float
        Sampling interval in seconds.
    delta_f : float
        Frequency resolution in Hz (\\Delta f = 1 / (N \\Delta t)).
    n_samples : int, optional
        Original time-domain sample count N. If None, assumes even N with N = 2 * (len(X) - 1).

    Returns
    -------
    float
        Frequency-domain folded energy sum with units of [x]^2 * seconds.
    """
    if not isinstance(delta_t, (int, float, np.floating)) or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError(f"delta_t must be a positive finite float, got {delta_t}")
    if not isinstance(delta_f, (int, float, np.floating)) or not math.isfinite(delta_f) or delta_f <= 0:
        raise ValueError(f"delta_f must be a positive finite float, got {delta_f}")

    X_raw = np.asarray(X)
    if X_raw.ndim != 1:
        raise ValueError(f"X must be a 1D array, got dimension {X_raw.ndim}")
    if X_raw.size < 2:
        raise ValueError(f"X must have at least 2 bins (DC and Nyquist/first harmonic), got {X_raw.size}")
    if not np.all(np.isfinite(X_raw)):
        raise ValueError("X must contain only finite numbers (no NaN or Inf)")

    if n_samples is not None:
        if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
        expected_len = (int(n_samples) // 2) + 1
        if X_raw.size != expected_len:
            raise ValueError(
                f"Length of frequency array X ({X_raw.size}) does not match expected length "
                f"({expected_len}) for n_samples={n_samples}"
            )

    X_arr = X_raw.astype(np.complex128, copy=False)

    # Squared magnitudes of rfft coefficients
    sq_mag = np.abs(X_arr) ** 2

    # Distinguish odd vs even N
    is_odd = (n_samples is not None) and (n_samples % 2 == 1)

    if is_odd:
        # Odd N: DC (k=0) has weight 1; all k >= 1 have weight 2. No Nyquist bin.
        folded_sum = sq_mag[0] + 2.0 * np.sum(sq_mag[1:])
    else:
        # Even N: DC (k=0) has weight 1, Nyquist (k=-1) has weight 1, interior bins have weight 2
        if sq_mag.size == 2:
            folded_sum = sq_mag[0] + sq_mag[1]
        else:
            folded_sum = sq_mag[0] + 2.0 * np.sum(sq_mag[1:-1]) + sq_mag[-1]

    return float(delta_f * (delta_t ** 2) * folded_sum)


def compute_parseval_error(
    time_energy: float,
    freq_energy: float,
    ref_scale: Optional[float] = None,
) -> float:
    """Compute Parseval energy discrepancy.

    When time_energy > 0, returns the relative error:
        \\epsilon_{\\mathrm{rel}} = \\frac{|E_{\\mathrm{time}} - E_{\\mathrm{freq}}|}{E_{\\mathrm{time}}}
    When time_energy == 0 (or near-zero), returns the discrepancy normalized by a fixed reference scale:
        \\epsilon_{\\mathrm{ref}} = \\frac{|E_{\\mathrm{time}} - E_{\\mathrm{freq}}|}{E_{\\mathrm{ref}}}

    Parameters
    ----------
    time_energy : float
        Time-domain discrete energy sum E_time.
    freq_energy : float
        Frequency-domain folded energy sum E_freq.
    ref_scale : float, optional
        Fixed reference energy scale E_ref used when time_energy == 0.0.
        Defaults to 1.0.

    Returns
    -------
    float
        epsilon_rel if time_energy > 0, else epsilon_ref.
    """
    if not isinstance(time_energy, (int, float, np.floating)) or not math.isfinite(time_energy) or time_energy < 0:
        raise ValueError(f"time_energy must be a non-negative finite float, got {time_energy}")
    if not isinstance(freq_energy, (int, float, np.floating)) or not math.isfinite(freq_energy) or freq_energy < 0:
        raise ValueError(f"freq_energy must be a non-negative finite float, got {freq_energy}")

    diff = abs(float(time_energy) - float(freq_energy))
    if time_energy > 0.0:
        return float(diff / time_energy)

    scale = 1.0 if ref_scale is None else float(ref_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"ref_scale must be a positive finite float, got {ref_scale}")
    return float(diff / scale)


def validate_parseval_identity(
    x: np.ndarray, delta_t: float, ref_scale: Optional[float] = None
) -> Tuple[float, float, float]:
    """Validate discrete Parseval energy balance for a real-valued time series.

    Validates inputs strictly before constructing the frequency step \\Delta f,
    preventing any potential division-by-zero on empty input.

    Parameters
    ----------
    x : np.ndarray
        Real-valued 1D time series of even length N.
    delta_t : float
        Sampling interval in seconds.
    ref_scale : float, optional
        Fixed reference scale E_ref for zero-energy signals. Defaults to 1.0.

    Returns
    -------
    tuple of (float, float, float)
        (time_energy, freq_energy, discrepancy) where:
        discrepancy = compute_parseval_error(time_energy, freq_energy, ref_scale).
    """
    x_arr, dt = _validate_real_time_series(x, delta_t)
    n_samples = x_arr.size
    delta_f = 1.0 / (n_samples * dt)

    e_time = discrete_parseval_time_energy(x_arr, dt)
    X = np.fft.rfft(x_arr)
    e_freq = discrete_parseval_freq_energy(X, dt, delta_f, n_samples=n_samples)
    err = compute_parseval_error(e_time, e_freq, ref_scale=ref_scale)

    return e_time, e_freq, err


def direct_dft(x: np.ndarray, delta_t: float) -> np.ndarray:
    """Compute continuous Fourier transform approximation via direct O(N^2) discrete sum.

    \\tilde{x}(f_k) = \\Delta t \\sum_{j=0}^{N-1} x_j e^{-i 2\\pi j k / N}

    Used as an independent reference calculation for small arrays (e.g. N <= 2048)
    to validate the Cooley-Tukey FFT implementation without relying on the FFT algorithm.

    Parameters
    ----------
    x : np.ndarray
        Real-valued 1D time series of even length N.
    delta_t : float
        Sampling interval in seconds.

    Returns
    -------
    np.ndarray
        Complex-valued array of length N/2 + 1 containing the continuous Fourier
        transform approximation at frequencies f_k = k / (N \\Delta t).
    """
    x_arr, dt = _validate_real_time_series(x, delta_t)
    N = x_arr.size
    n_freqs = N // 2 + 1

    # Matrix-vector implementation using outer product
    j = np.arange(N, dtype=np.float64)
    k = np.arange(n_freqs, dtype=np.float64)
    exponent = (-2.0j * np.pi / N) * np.outer(k, j)
    dft_matrix = np.exp(exponent)
    X = dft_matrix @ x_arr
    return dt * X


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
    x_arr, dt = _validate_real_time_series(x, delta_t)
    n_samples = x_arr.size
    X = np.fft.rfft(x_arr)
    sq_mag = np.abs(X) ** 2

    # Weighting: 2 * delta_t / N for interior bins, 1 * delta_t / N for DC and Nyquist
    scale = dt / n_samples
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
