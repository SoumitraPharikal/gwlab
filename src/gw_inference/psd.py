"""Welch power spectral density (PSD) estimation module.

This module implements mathematically rigorous, scientifically audited
power spectral density estimation for discrete real-valued time series,
following the conventions approved in Stage 0.6 and Stage 1.2 revisions.

Key Capabilities:
-----------------
1. Discrete Tukey window generation with exact symmetric midpoint cosine sampling.
2. Window power (S_2) and coherent gain (S_1) normalization factors.
3. Generalized multi-lag window overlap correlations (r_ell) and all-lags
   Welch variance inflation factor (kappa_overlap) under proper-complex Gaussian noise.
4. Exact discrete covariance and pseudo-covariance sums (P_11, P_12, C_11, C_12)
   and full Isserlis/Wick periodogram correlation evaluations.
5. Segment extraction with arbitrary valid overlap and strict non-truncated slicing.
6. Single-segment windowed one-sided periodogram calculation.
7. Standard Welch mean PSD estimator with configurable window and overlap.
8. Finite-sample bias-corrected Welch median PSD estimator with odd/even parity
   formulas and strict caller assertion of independent exponential variates.

Mathematical Conventions:
-------------------------
1. Sampling:
   f_s = 1 / \\Delta t,  \\Delta f = 1 / (M \\Delta t),  k = 0, 1, \\dots, M/2

2. Discrete Tukey Window (\\alpha = 0.25, M samples, M_taper = \\lfloor \\alpha M / 2 \\rfloor):
   w_j = \\begin{cases}
       \\frac{1}{2}\\left[1 - \\cos\\left(\\frac{\\pi(j + 0.5)}{M_{\\mathrm{taper}}}\\right)\\right],
           & 0 \\le j < M_{\\mathrm{taper}} \\\\[4pt]
       1.0, & M_{\\mathrm{taper}} \\le j < M - M_{\\mathrm{taper}} \\\\[4pt]
       \\frac{1}{2}\\left[1 - \\cos\\left(\\frac{\\pi(M - 0.5 - j)}{M_{\\mathrm{taper}}}\\right)\\right],
           & M - M_{\\mathrm{taper}} \\le j < M
   \\end{cases}

3. Window Power Normalization:
   S_2 = \\frac{1}{M} \\sum_{j=0}^{M-1} w_j^2
   Coherent Gain (Amplitude Normalization):
   S_1 = \\frac{1}{M} \\sum_{j=0}^{M-1} w_j

4. Windowed One-Sided Segment Periodogram:
   For unwindowed segment slice d_j^{(m)} = x_{j + m H}, the windowed segment is:
       x_j^{(m)} = d_j^{(m)} w_j = x_{j + m H} w_j
   and its discrete Fourier transform is:
       X_k^{(m)} = \\operatorname{rfft}(x^{(m)})_k = \\operatorname{rfft}(\\{x_{j + m H} w_j\\}_{j=0}^{M-1})_k
   The one-sided windowed periodogram is:
       \\widehat{S}_m(f_k) = \\begin{cases}
           \\frac{\\Delta t}{M S_2} |X_k^{(m)}|^2, & k = 0, M/2 \\\\[4pt]
           \\frac{2\\Delta t}{M S_2} |X_k^{(m)}|^2, & 1 \\le k < M/2
       \\end{cases}

5. Welch Mean Estimator:
   \\widehat{S}_{\\mathrm{Welch}}(f_k) = \\frac{1}{K} \\sum_{m=0}^{K-1} \\widehat{S}_m(f_k)

6. All-Lags Variance Inflation Factor (Proper-Complex Gaussian Approximation):
   For hop H = M - N_{\\mathrm{overlap}}, normalized window correlation at lag \\ell \\ge 1:
       r_\\ell = \\frac{\\sum_{j=0}^{M - \\ell H - 1} w_j w_{j + \\ell H}}{\\sum_{j=0}^{M-1} w_j^2}
   and:
       \\kappa_{\\mathrm{overlap}} = 1 + 2 \\sum_{\\ell=1}^{\\ell_{\\max}} \\left( 1 - \\frac{\\ell}{K} \\right) r_\\ell^2
   where \\ell_{\\max} = \\min(\\lfloor (M-1)/H \\rfloor, K - 1).

7. Welch Median Estimator:
   Raw median:
       \\widehat{S}_{\\mathrm{raw\\_median}}(f_k) = \\operatorname{median}_{m=0,\\dots,K-1} \\widehat{S}_m(f_k)
   Under the strict caller-asserted assumption that individual periodograms are
   independent exponential random variables (zero overlap, uncorrupted white noise),
   the finite-sample bias-corrected estimator is:
       \\widehat{S}_{\\mathrm{median}}(f_k) = \\frac{1}{C(K)} \\widehat{S}_{\\mathrm{raw\\_median}}(f_k)
   where C(K) satisfies:
   - For odd K = 2L + 1:   C(K) = \\sum_{j=L+1}^{2L+1} \\frac{1}{j}
   - For even K = 2L:      C(K) = \\sum_{j=L+1}^{2L} \\frac{1}{j} + \\frac{1}{2L}
   - Asymptotic limit:     \\lim_{K \\to \\infty} C(K) = \\ln 2 \\approx 0.693147
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple, Union

import numpy as np

from gw_inference.conventions import _validate_real_time_series, build_frequency_grid


def _validate_psd_time_series(x: object, delta_t: object) -> Tuple[np.ndarray, float]:
    """Validate 1D real-valued finite input time series and sampling interval.

    Unlike single-segment FFT operations which require x to be even length,
    a time series input for Welch estimation may have any length N >= segment_length,
    as trailing non-divisible samples are cleanly discarded.

    Parameters
    ----------
    x : array_like
        Input time-series data.
    delta_t : float
        Sampling interval in seconds.

    Returns
    -------
    tuple of (np.ndarray, float)
        Validated array as float64, and float delta_t.
    """
    if not isinstance(delta_t, (int, float, np.floating)) or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError(f"delta_t must be a positive finite float, got {delta_t}")

    x_raw = np.asarray(x)
    if x_raw.ndim != 1:
        raise ValueError(f"x must be a 1D array, got dimension {x_raw.ndim}")
    if x_raw.size == 0:
        raise ValueError("x must be a non-empty array")
    if np.iscomplexobj(x_raw):
        raise ValueError(f"x must be real-valued, got complex dtype {x_raw.dtype}")
    if not np.all(np.isfinite(x_raw)):
        raise ValueError("x must contain only finite numbers (no NaN or Inf)")

    return x_raw.astype(np.float64, copy=False), float(delta_t)


def tukey_window(length: int, alpha: float = 0.25) -> np.ndarray:
    """Generate a discrete symmetric Tukey (cosine-tapered) window.

    Evaluates the discrete Tukey window using exact half-sample midpoint cosine
    arguments as specified in Stage 0.6 Section 4:
        w_j = 0.5 * [1 - cos(pi * (j + 0.5) / M_taper)]  for 0 <= j < M_taper
        w_j = 1.0                                        for M_taper <= j < M - M_taper
        w_j = 0.5 * [1 - cos(pi * (M - 0.5 - j) / M_taper)] for M - M_taper <= j < M
    where M_taper = floor(alpha * length / 2).

    Parameters
    ----------
    length : int
        Total window length M in samples. Must be a positive even integer.
    alpha : float, default=0.25
        Fraction of the window inside the cosine taper (0 <= alpha <= 1).
        When alpha = 0.0, yields a rectangular window (all ones).
        When alpha = 1.0, yields a Hann window.

    Returns
    -------
    np.ndarray
        1D float64 array of shape (length,) containing the discrete window.

    Raises
    ------
    ValueError
        If length is not a positive even integer, or alpha is not in [0.0, 1.0].
    """
    if not isinstance(length, (int, np.integer)) or length <= 0:
        raise ValueError(f"length must be a positive integer, got {length}")
    if length % 2 != 0:
        raise ValueError(f"length must be an even integer, got {length}")
    if not isinstance(alpha, (int, float, np.floating)) or not math.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be a finite float between 0.0 and 1.0, got {alpha}")

    M = int(length)
    alpha_val = float(alpha)

    # Number of samples in each taper section
    M_taper = int(np.floor(alpha_val * M / 2.0))

    w = np.ones(M, dtype=np.float64)

    if M_taper <= 0:
        return w

    # Left taper: 0 <= j < M_taper
    j_left = np.arange(M_taper, dtype=np.float64)
    w[:M_taper] = 0.5 * (1.0 - np.cos(np.pi * (j_left + 0.5) / M_taper))

    # Right taper: M - M_taper <= j < M
    j_right = np.arange(M - M_taper, M, dtype=np.float64)
    w[M - M_taper:] = 0.5 * (1.0 - np.cos(np.pi * (M - 0.5 - j_right) / M_taper))

    return w


def _validate_window_array(window: object) -> np.ndarray:
    """Validate a 1D real-valued, finite, non-empty window array."""
    w_raw = np.asarray(window)
    if w_raw.ndim != 1:
        raise ValueError(f"window must be a 1D array, got dimension {w_raw.ndim}")
    if w_raw.size == 0:
        raise ValueError("window must be non-empty")
    if np.iscomplexobj(w_raw):
        raise ValueError(f"window must be real-valued, got complex dtype {w_raw.dtype}")
    if not np.all(np.isfinite(w_raw)):
        raise ValueError("window must contain only finite numbers (no NaN or Inf)")
    return w_raw.astype(np.float64, copy=False)


def window_power(window: np.ndarray) -> float:
    """Compute the discrete window power normalization factor S_2.

    S_2 = \\frac{1}{M} \\sum_{j=0}^{M-1} w_j^2

    Parameters
    ----------
    window : np.ndarray
        1D real-valued discrete window array.

    Returns
    -------
    float
        Mean squared value of the window samples S_2.
    """
    w = _validate_window_array(window)
    return float(np.mean(w ** 2))


def window_coherent_gain(window: np.ndarray) -> float:
    """Compute the discrete window coherent gain (amplitude scale) S_1.

    S_1 = \\frac{1}{M} \\sum_{j=0}^{M-1} w_j

    Parameters
    ----------
    window : np.ndarray
        1D real-valued discrete window array.

    Returns
    -------
    float
        Arithmetic mean of the window samples S_1.
    """
    w = _validate_window_array(window)
    return float(np.mean(w))


def window_overlap_correlation(window: np.ndarray, overlap_samples: int) -> float:
    """Compute the normalized window overlap correlation factor r_1 for adjacent segments.

    For hop length H = M - overlap_samples:
    r_1 = \\frac{\\sum_{j=0}^{\\text{overlap}-1} w_j w_{j+H}}{\\sum_{j=0}^{M-1} w_j^2}

    Parameters
    ----------
    window : np.ndarray
        1D real-valued discrete window array of length M.
    overlap_samples : int
        Number of overlapping samples between adjacent segments.
        Must satisfy 0 <= overlap_samples < M.

    Returns
    -------
    float
        Normalized adjacent-segment overlap correlation r_1 (or 0.0 if overlap_samples == 0).
    """
    w = _validate_window_array(window)
    M = w.size

    if not isinstance(overlap_samples, (int, np.integer)):
        raise ValueError(f"overlap_samples must be an integer, got {overlap_samples}")
    if overlap_samples < 0 or overlap_samples >= M:
        raise ValueError(f"overlap_samples must satisfy 0 <= overlap_samples < length ({M}), got {overlap_samples}")

    if overlap_samples == 0:
        return 0.0

    hop = M - overlap_samples
    denom = np.sum(w ** 2)
    if denom == 0.0:
        raise ValueError("Window has zero total energy sum")

    numer = np.sum(w[:overlap_samples] * w[hop:])
    return float(numer / denom)


def window_multi_lag_correlations(
    window: np.ndarray,
    hop_length: int,
    max_lag: Optional[int] = None,
) -> np.ndarray:
    """Compute normalized discrete window cross-correlations for all valid positive lags.

    For lag \\ell >= 1:
        r_\\ell = \\frac{\\sum_{j=0}^{M - \\ell H - 1} w_j w_{j + \\ell H}}{\\sum_{j=0}^{M-1} w_j^2}
    where \\ell H < M.

    Parameters
    ----------
    window : np.ndarray
        1D real-valued discrete window array of length M.
    hop_length : int
        Hop length H in samples (H = M - overlap_samples). Must be a positive integer.
    max_lag : int, optional
        Upper limit on lag \\ell (e.g. K - 1). If None, evaluates up to floor((M-1)/H).

    Returns
    -------
    np.ndarray
        1D float64 array of shape (ell_max,) containing [r_1, r_2, ..., r_ell_max].
        Returns an empty float64 array if ell_max < 1.

    Raises
    ------
    ValueError
        If hop_length is not a positive integer.
    """
    w = _validate_window_array(window)
    M = w.size

    if not isinstance(hop_length, (int, np.integer)) or hop_length <= 0:
        raise ValueError(f"hop_length must be a positive integer, got {hop_length}")

    H = int(hop_length)
    if H >= M:
        return np.empty(0, dtype=np.float64)

    max_possible_lag = (M - 1) // H
    if max_lag is not None:
        if not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
            raise ValueError(f"max_lag must be a non-negative integer, got {max_lag}")
        ell_max = min(max_possible_lag, int(max_lag))
    else:
        ell_max = max_possible_lag

    if ell_max < 1:
        return np.empty(0, dtype=np.float64)

    denom = np.sum(w ** 2)
    if denom == 0.0:
        raise ValueError("Window has zero total energy sum")

    r_lags = np.empty(ell_max, dtype=np.float64)
    for ell in range(1, ell_max + 1):
        shared_samples = M - ell * H
        r_lags[ell - 1] = np.sum(w[:shared_samples] * w[ell * H :]) / denom

    return r_lags


def welch_variance_inflation(
    window: np.ndarray,
    segment_length: int,
    overlap_samples: int,
    n_segments: int,
) -> float:
    """Compute the theoretical Welch PSD variance inflation factor kappa_overlap.

    Under the proper-complex Gaussian approximation, accounting for all shared lags:
        \\kappa_{\\mathrm{overlap}} = 1 + 2 \\sum_{\\ell=1}^{\\ell_{\\max}} \\left( 1 - \\frac{\\ell}{K} \\right) r_\\ell^2
    where \\ell_{\\max} = \\min(\\lfloor (M-1)/H \\rfloor, K - 1).

    Parameters
    ----------
    window : np.ndarray
        1D real-valued discrete window of length segment_length.
    segment_length : int
        Segment length M in samples.
    overlap_samples : int
        Number of overlapping samples between adjacent segments.
    n_segments : int
        Number of valid segments K.

    Returns
    -------
    float
        Variance inflation factor kappa_overlap.
        Returns 1.0 if overlap_samples == 0 or n_segments <= 1.
    """
    if n_segments <= 1 or overlap_samples <= 0:
        return 1.0

    w = _validate_window_array(window)
    if w.size != segment_length:
        raise ValueError(f"window length ({w.size}) must match segment_length ({segment_length})")

    H = segment_length - overlap_samples
    if H <= 0:
        raise ValueError(f"overlap_samples ({overlap_samples}) must be strictly less than segment_length ({segment_length})")

    r_lags = window_multi_lag_correlations(w, hop_length=H, max_lag=n_segments - 1)
    if r_lags.size == 0:
        return 1.0

    K = float(n_segments)
    lags = np.arange(1, r_lags.size + 1, dtype=np.float64)
    inflation = 1.0 + 2.0 * np.sum((1.0 - lags / K) * (r_lags ** 2))
    return float(inflation)


def discrete_pseudo_covariance_ratios(
    window: np.ndarray,
    overlap_samples: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    r"""Compute exact discrete covariance and pseudo-covariance sums for a windowed DFT.

    Under stationary discrete Gaussian white noise with variance sigma^2:
        C_{11} = \sigma^2 \sum_{j=0}^{M-1} w_j^2 = \sigma^2 M S_2
        C_{12}(k) = \sigma^2 e^{-2\pi i H k / M} \sum_{l=0}^{M - H - 1} w_l w_{l+H}
        P_{11}(k) = \sigma^2 \sum_{j=0}^{M-1} w_j^2 e^{-4\pi i j k / M}
        P_{12}(k) = \sigma^2 e^{-2\pi i H k / M} \sum_{l=0}^{M - H - 1} w_l w_{l+H} e^{-4\pi i l k / M}

    The complete Isserlis/Wick periodogram correlation across segments is:
        \rho_P(k) = \frac{|C_{12}(k)|^2 + |P_{12}(k)|^2}{C_{11}^2 + |P_{11}(k)|^2}
    whereas the proper-complex approximation assumes P_{11}(k) \approx 0 and P_{12}(k) \approx 0:
        \rho_P^{\mathrm{approx}} = \frac{|C_{12}|^2}{C_{11}^2} = r_w^2

    Parameters
    ----------
    window : np.ndarray
        1D real-valued discrete window of length M.
    overlap_samples : int, optional
        Number of overlapping samples between adjacent segments.
        Defaults to M // 2 (50% overlap).

    Returns
    -------
    dict
        Dictionary containing:
        - "C11": Total window energy sum \sum w_j^2 (float)
        - "C12": Window overlap product sum \sum w_l w_{l+H} (float)
        - "rw_squared": Proper-complex approximation (C12 / C11)^2 (float)
        - "P11": 1D complex128 array of shape (M // 2 + 1,)
        - "P12": 1D complex128 array of shape (M // 2 + 1,)
        - "ratio_P11": 1D float64 array of |P11(k)| / C11
        - "ratio_P12": 1D float64 array of |P12(k)| / |C12| (0.0 if C12 == 0)
        - "rho_full": 1D float64 array of exact periodogram correlation \rho_P(k)
        - "rel_diff": 1D float64 array of (\rho_full(k) - rw_squared) / rw_squared
    """
    w = _validate_window_array(window)
    M = w.size

    if overlap_samples is None:
        overlap = M // 2
    else:
        if not isinstance(overlap_samples, (int, np.integer)):
            raise ValueError(f"overlap_samples must be an integer, got {overlap_samples}")
        overlap = int(overlap_samples)
        if overlap < 0 or overlap >= M:
            raise ValueError(f"overlap_samples must satisfy 0 <= overlap < M ({M}), got {overlap}")

    H = M - overlap
    C11 = float(np.sum(w ** 2))
    if C11 == 0.0:
        raise ValueError("Window has zero total energy sum")

    n_freqs = M // 2 + 1
    k = np.arange(n_freqs, dtype=np.float64)

    # P11(k) = sum_{j=0}^{M-1} w_j^2 e^{-4pi i j k / M}
    # Length-M FFT of w^2 at index (2k) % M
    w2 = w ** 2
    w2_fft = np.fft.fft(w2)
    idx_2k = (2 * np.arange(n_freqs)) % M
    P11 = w2_fft[idx_2k]

    # C12 and P12
    if overlap > 0:
        w_overlap = w[:overlap] * w[H:]
        C12 = float(np.sum(w_overlap))
        phase_prefactor = np.exp(-2j * np.pi * H * k / float(M))
        # The FFT shortcut reduces \sum w_{overlap}[l] e^{-4\pi i l k / M} to a length-overlap DFT
        # \sum w_{overlap}[l] e^{-2\pi i l k / overlap} if and only if 2 * overlap == M
        # (which requires M to be even). For all other cases (odd M or overlap != M/2),
        # use the exact direct discrete phase sum.
        if 2 * overlap == M:
            ov_fft = np.fft.fft(w_overlap)
            idx_k_ov = np.arange(n_freqs) % overlap
            P12 = phase_prefactor * ov_fft[idx_k_ov]
        else:
            l_arr = np.arange(overlap, dtype=np.float64)
            phase_mat = np.exp(-4j * np.pi * np.outer(l_arr, k) / float(M))
            P12 = phase_prefactor * (w_overlap @ phase_mat)
    else:
        C12 = 0.0
        P12 = np.zeros(n_freqs, dtype=np.complex128)

    ratio_P11 = np.abs(P11) / C11
    ratio_P12 = np.abs(P12) / abs(C12) if abs(C12) > 0.0 else np.zeros(n_freqs, dtype=np.float64)

    rw_squared = (C12 / C11) ** 2 if C11 > 0 else 0.0
    num = (C12 ** 2) + (np.abs(P12) ** 2)
    denom = (C11 ** 2) + (np.abs(P11) ** 2)
    rho_full = num / denom

    rel_diff = (rho_full - rw_squared) / rw_squared if rw_squared > 0.0 else np.zeros(n_freqs, dtype=np.float64)

    return {
        "C11": C11,
        "C12": C12,
        "rw_squared": rw_squared,
        "P11": P11,
        "P12": P12,
        "ratio_P11": ratio_P11,
        "ratio_P12": ratio_P12,
        "rho_full": rho_full,
        "rel_diff": rel_diff,
    }


def extract_segments(
    x: np.ndarray,
    segment_length: int,
    overlap_samples: int = 0,
) -> np.ndarray:
    """Extract 2D array of overlapping segments from a 1D real time series.

    Extracts K non-truncated segments of length `segment_length` with fixed hop
    `segment_length - overlap_samples`. Any incomplete trailing samples that do
    not form a complete segment are discarded.

    Parameters
    ----------
    x : np.ndarray
        1D real-valued finite input time series.
    segment_length : int
        Length of each segment in samples. Must be a positive even integer.
    overlap_samples : int, default=0
        Number of samples shared between consecutive segments.
        Must satisfy 0 <= overlap_samples < segment_length.

    Returns
    -------
    np.ndarray
        2D float64 array of shape (K, segment_length) containing the K valid segments.

    Raises
    ------
    ValueError
        If inputs are invalid or if segment_length exceeds the length of x
        (yielding zero valid segments).
    """
    x_arr, _ = _validate_psd_time_series(x, delta_t=1.0)
    N = x_arr.size

    if not isinstance(segment_length, (int, np.integer)) or segment_length <= 0:
        raise ValueError(f"segment_length must be a positive integer, got {segment_length}")
    if segment_length % 2 != 0:
        raise ValueError(f"segment_length must be an even integer, got {segment_length}")

    if not isinstance(overlap_samples, (int, np.integer)):
        raise ValueError(f"overlap_samples must be an integer, got {overlap_samples}")
    if overlap_samples < 0 or overlap_samples >= segment_length:
        raise ValueError(
            f"overlap_samples must satisfy 0 <= overlap_samples < segment_length ({segment_length}), "
            f"got {overlap_samples}"
        )

    if segment_length > N:
        raise ValueError(
            f"segment_length ({segment_length}) cannot exceed input data length ({N})"
        )

    hop_length = segment_length - overlap_samples
    n_segments = (N - overlap_samples) // hop_length

    if n_segments <= 0:
        raise ValueError(
            f"Configuration yields zero valid segments (N={N}, segment_length={segment_length}, "
            f"overlap_samples={overlap_samples})"
        )

    segments = np.empty((n_segments, segment_length), dtype=np.float64)
    for m in range(n_segments):
        start = m * hop_length
        segments[m] = x_arr[start : start + segment_length]

    return segments


def single_segment_psd(
    segment: np.ndarray,
    delta_t: float,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the one-sided power spectral density for a single windowed data segment.

    Applies the window w_j to the unwindowed data segment d_j exactly once:
        x_j = d_j \\cdot w_j
    computes the unnormalised real FFT:
        X_k = \\operatorname{rfft}(x)_k
    and applies one-sided power normalization:
        \\widehat{S}_m(f_k) = \\begin{cases}
            \\frac{\\Delta t}{M S_2} |X_k|^2, & k = 0, M/2 \\\\[4pt]
            \\frac{2\\Delta t}{M S_2} |X_k|^2, & 1 \\le k < M/2
        \\end{cases}
    where S_2 = \\frac{1}{M}\\sum_{j=0}^{M-1} w_j^2.

    If window is None, a rectangular window (w_j = 1, S_2 = 1) is used, matching
    `one_sided_periodogram` from Stage 1.1 identically.

    Parameters
    ----------
    segment : np.ndarray
        1D real-valued time-series segment of even length M.
    delta_t : float
        Sampling interval in seconds.
    window : np.ndarray, optional
        1D real-valued discrete window of length M. If None, rectangular window is used.

    Returns
    -------
    np.ndarray
        1D float64 array of shape (M/2 + 1,) containing the one-sided PSD in [x]^2 / Hz.

    Raises
    ------
    ValueError
        If segment or window is invalid, lengths do not match, or window power is zero.
    """
    seg_arr, dt = _validate_real_time_series(segment, delta_t)
    M = seg_arr.size

    if window is None:
        w = np.ones(M, dtype=np.float64)
        s2 = 1.0
    else:
        w = _validate_window_array(window)
        if w.size != M:
            raise ValueError(f"window length ({w.size}) must match segment length ({M})")
        s2 = window_power(w)
        if s2 <= 0.0:
            raise ValueError(f"window power S_2 must be positive, got {s2}")

    # Windowed segment (window applied exactly once)
    x_win = seg_arr * w
    X = np.fft.rfft(x_win)
    sq_mag = np.abs(X) ** 2

    # Scale factor: dt / (M * S_2)
    scale = dt / (M * s2)

    psd = 2.0 * scale * sq_mag
    psd[0] = scale * sq_mag[0]
    psd[-1] = scale * sq_mag[-1]

    return psd


def median_bias_factor(n_segments: int) -> float:
    """Compute the finite-sample bias correction factor C(K) for the median of K periodograms.

    Derived under the assumption of independent, identically distributed exponential
    random variables.
    For odd K = 2L + 1:
        C(K) = \\sum_{j=L+1}^{2L+1} \\frac{1}{j}
    For even K = 2L:
        C(K) = \\sum_{j=L+1}^{2L} \\frac{1}{j} + \\frac{1}{2L}

    Parameters
    ----------
    n_segments : int
        Number of independent exponential variates K. Must be a positive integer.

    Returns
    -------
    float
        Finite-sample bias correction factor C(K).
        Converges to ln(2) approx 0.693147 as K -> infinity.

    Raises
    ------
    ValueError
        If n_segments is not a positive integer.
    """
    if not isinstance(n_segments, (int, np.integer)) or n_segments <= 0:
        raise ValueError(f"n_segments must be a positive integer, got {n_segments}")

    K = int(n_segments)
    if K == 1:
        return 1.0

    if K % 2 == 1:
        # Odd K = 2L + 1
        L = (K - 1) // 2
        return float(sum(1.0 / j for j in range(L + 1, 2 * L + 2)))
    else:
        # Even K = 2L
        L = K // 2
        return float(sum(1.0 / j for j in range(L + 1, 2 * L + 1)) + 1.0 / (2 * L))


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
    """Estimate Power Spectral Density using Welch's averaged or median periodogram method.

    Divides the input time series x into segments of length `segment_length` with
    `overlap_samples` overlap, applies the specified window exactly once to each segment,
    computes the one-sided periodogram, and aggregates across segments using either the
    arithmetic mean or the sample median.

    Parameters
    ----------
    x : np.ndarray
        1D real-valued discrete time series.
    delta_t : float
        Sampling interval in seconds (1 / f_s).
    segment_length : int
        Segment length in samples M. Must be a positive even integer.
    overlap_samples : int, optional
        Number of overlapping samples between adjacent segments.
        Defaults to segment_length // 2 (50% overlap).
    window : str or np.ndarray, optional, default="tukey"
        Window to apply to each segment. Options:
        - "tukey": Discrete Tukey window with alpha=0.25 (the project default).
        - "rect" or None: Rectangular window (w_j = 1.0).
        - np.ndarray: Pre-computed custom window of length `segment_length`.
    method : {"mean", "median"}, default="mean"
        Aggregation method across segments:
        - "mean": Standard Welch arithmetic average of segment periodograms.
        - "median": Sample median of segment periodograms.
    bias_correct : bool, default=False
        Applicable only when method="median". If True, applies finite-sample
        bias correction factor C(K). Requires zero overlap (overlap_samples=0)
        and an explicit assertion of independent exponential periodograms
        via assume_independent_exponential_periodograms=True.
        If False, returns the raw uncorrected sample median.
    assume_independent_exponential_periodograms : bool, default=False
        Applicable only when method="median" and bias_correct=True.
        Explicit caller assertion that individual periodograms follow independent
        exponential distributions. The software cannot independently verify this
        assumption, and zero overlap alone does NOT guarantee independence for
        temporally correlated or coloured noise.

    Returns
    -------
    tuple of (freqs, psd)
        freqs : np.ndarray
            One-sided frequency grid bin centers f_k = k / (M * delta_t) for k = 0, ..., M/2.
        psd : np.ndarray
            Estimated one-sided power spectral density in [x]^2 / Hz.

    Raises
    ------
    ValueError
        If inputs are invalid, segment length exceeds data length, method is invalid,
        or bias_correct is requested with nonzero overlap or without the explicit
        independence assertion.
    """
    x_arr, dt = _validate_psd_time_series(x, delta_t)

    if not isinstance(segment_length, (int, np.integer)) or segment_length <= 0:
        raise ValueError(f"segment_length must be a positive integer, got {segment_length}")
    if segment_length % 2 != 0:
        raise ValueError(f"segment_length must be an even integer, got {segment_length}")

    if overlap_samples is None:
        overlap = segment_length // 2
    else:
        if not isinstance(overlap_samples, (int, np.integer)):
            raise ValueError(f"overlap_samples must be an integer, got {overlap_samples}")
        overlap = int(overlap_samples)

    if method not in ("mean", "median"):
        raise ValueError(f"method must be either 'mean' or 'median', got '{method}'")

    if method == "mean":
        if bias_correct or assume_independent_exponential_periodograms:
            raise ValueError(
                "bias_correct and assume_independent_exponential_periodograms are only "
                "applicable when method='median'"
            )

    if method == "median":
        if bias_correct:
            if overlap > 0:
                raise ValueError(
                    f"Bias-corrected median estimation is rejected for overlapping segments "
                    f"(got overlap_samples={overlap}). Overlapping segments share time-domain "
                    f"samples and produce correlated periodograms, violating the independent-exponential "
                    f"periodogram assumption required by C(K). Use raw median (bias_correct=False) "
                    f"or set overlap_samples=0."
                )
            if not assume_independent_exponential_periodograms:
                raise ValueError(
                    "Bias-corrected median estimation requires the caller to explicitly assert that "
                    "individual periodograms are independent exponential variates by passing "
                    "assume_independent_exponential_periodograms=True. Note that zero overlap alone "
                    "does not guarantee statistical independence for coloured or temporally correlated "
                    "noise, and the software cannot independently verify this assumption."
                )
        else:
            if assume_independent_exponential_periodograms:
                raise ValueError(
                    "assume_independent_exponential_periodograms was set to True, but bias_correct is False. "
                    "Set bias_correct=True to enable finite-sample bias correction, or leave "
                    "assume_independent_exponential_periodograms=False for uncorrected raw median estimation."
                )

    # Window resolution
    if isinstance(window, str):
        if window.lower() == "tukey":
            w = tukey_window(segment_length, alpha=0.25)
        elif window.lower() in ("rect", "rectangular"):
            w = np.ones(segment_length, dtype=np.float64)
        else:
            raise ValueError(f"Unknown window string '{window}'. Supported: 'tukey', 'rect'")
    elif window is None:
        w = np.ones(segment_length, dtype=np.float64)
    elif isinstance(window, np.ndarray):
        w = _validate_window_array(window)
        if w.size != segment_length:
            raise ValueError(f"Custom window length ({w.size}) must match segment_length ({segment_length})")
    else:
        raise TypeError(f"window must be a str, np.ndarray, or None, got {type(window)}")

    # Extract non-truncated segments
    segments = extract_segments(x_arr, segment_length=segment_length, overlap_samples=overlap)
    n_segments = segments.shape[0]

    # Pre-allocate periodograms: (K, M // 2 + 1)
    n_freqs = segment_length // 2 + 1
    periodograms = np.empty((n_segments, n_freqs), dtype=np.float64)

    for m in range(n_segments):
        periodograms[m] = single_segment_psd(segments[m], delta_t=dt, window=w)

    # Aggregate
    if method == "mean":
        psd = np.mean(periodograms, axis=0)
    else:  # median
        raw_med = np.median(periodograms, axis=0)
        if bias_correct:
            c_k = median_bias_factor(n_segments)
            psd = raw_med / c_k
        else:
            psd = raw_med

    freqs = build_frequency_grid(segment_length, dt)
    return freqs, psd


def low_frequency_psd_continuation(
    frequencies: np.ndarray,
    psd: np.ndarray,
    f_start: float,
    f_0: float,
) -> np.ndarray:
    """Apply regularized operational low-frequency PSD continuation below f_start.

    Functional form (frozen Group A specification):
        S_n(f) = S_n(f_start) * [ (f_start^2 + f_0^2) / (f^2 + f_0^2) ]^2,  for 0 <= f < f_start

    Guarantees C^0 continuity at f_start. Note that C^1 derivative continuity is
    not guaranteed.

    Governance Note:
    ----------------
    The parameter f_0 regulates the low-frequency roll-off below f_start.
    Because Decision D8 remains open and unconfirmed by the Project Lead,
    f_0 MUST be provided explicitly; no default is provided.

    Parameters
    ----------
    frequencies : np.ndarray
        1D array of frequency values in Hz.
    psd : np.ndarray
        1D array of power spectral density values in [strain]^2 / Hz.
    f_start : float
        Lower boundary of the active analysis band in Hz. Must be positive and finite.
    f_0 : float
        Regularization frequency scale in Hz. Must be positive and finite (explicit, Decision D8 open).

    Returns
    -------
    np.ndarray
        Modified PSD array with continuation applied for all f < f_start.

    Raises
    ------
    ValueError
        If inputs are invalid or non-finite, if f_start <= 0 or f_0 <= 0,
        or if psd contains non-positive values.
    """
    if (
        not isinstance(f_start, (int, float, np.floating))
        or not math.isfinite(f_start)
        or f_start <= 0
    ):
        raise ValueError(f"f_start must be a positive finite float, got {f_start}")
    if (
        not isinstance(f_0, (int, float, np.floating))
        or not math.isfinite(f_0)
        or f_0 <= 0
    ):
        raise ValueError(f"f_0 must be a positive finite float, got {f_0}")

    freqs_arr = np.asarray(frequencies, dtype=np.float64)
    psd_arr = np.asarray(psd, dtype=np.float64)

    if freqs_arr.ndim != 1 or psd_arr.ndim != 1:
        raise ValueError("frequencies and psd must be 1D arrays")
    if freqs_arr.size != psd_arr.size:
        raise ValueError(
            f"frequencies length ({freqs_arr.size}) must match psd length ({psd_arr.size})"
        )
    if not np.all(np.isfinite(freqs_arr)) or not np.all(np.isfinite(psd_arr)):
        raise ValueError("frequencies and psd must contain only finite numbers")
    if np.any(psd_arr <= 0):
        raise ValueError("psd values must be strictly positive")

    # Find or interpolate S_n(f_start)
    if f_start in freqs_arr:
        idx = np.where(freqs_arr == f_start)[0][0]
        s_at_start = psd_arr[idx]
    else:
        # Linear interpolation in log-log space at f_start
        pos_mask = freqs_arr > 0
        if not np.any(pos_mask):
            raise ValueError("frequencies must contain positive values")
        pos_f = freqs_arr[pos_mask]
        pos_p = psd_arr[pos_mask]
        if f_start < pos_f[0] or f_start > pos_f[-1]:
            raise ValueError(
                f"f_start ({f_start} Hz) is outside the measured frequency range "
                f"[{pos_f[0]}, {pos_f[-1]}] Hz"
            )
        s_at_start = float(
            np.exp(np.interp(np.log(f_start), np.log(pos_f), np.log(pos_p)))
        )

    out_psd = psd_arr.copy()
    low_mask = freqs_arr < float(f_start)

    if np.any(low_mask):
        f_low = freqs_arr[low_mask]
        ratio = (f_start**2 + f_0**2) / (f_low**2 + f_0**2)
        out_psd[low_mask] = s_at_start * (ratio**2)

    return out_psd


class PSDEngine:
    """Authoritative power spectral density (PSD) interpolation and management engine.

    Provides continuous-spectrum interpolation, low-frequency regularization below f_start,
    strict positivity enforcement, and active-bin PSD extraction.
    """

    def __init__(self, frequencies: np.ndarray, psd: np.ndarray):
        """Initialize PSDEngine with discrete frequency and PSD arrays.

        Parameters
        ----------
        frequencies : np.ndarray
            1D array of frequency values in Hz. Must be strictly increasing.
        psd : np.ndarray
            1D array of one-sided PSD values in [strain]^2 / Hz. Must be strictly positive.
        """
        f_arr = np.asarray(frequencies, dtype=np.float64)
        p_arr = np.asarray(psd, dtype=np.float64)

        if f_arr.ndim != 1 or p_arr.ndim != 1:
            raise ValueError("frequencies and psd must be 1D arrays")
        if f_arr.size != p_arr.size:
            raise ValueError(
                f"frequencies length ({f_arr.size}) must match psd length ({p_arr.size})"
            )
        if f_arr.size < 2:
            raise ValueError("frequencies and psd must have at least 2 points")
        if not np.all(np.isfinite(f_arr)) or not np.all(np.isfinite(p_arr)):
            raise ValueError("frequencies and psd must contain only finite numbers")
        if np.any(p_arr <= 0):
            idx_bad = np.where(p_arr <= 0)[0][0]
            raise ValueError(
                f"PSD values must be strictly positive (> 0); found non-positive value "
                f"{p_arr[idx_bad]} at frequency {f_arr[idx_bad]} Hz (index {idx_bad})"
            )
        if not np.all(np.diff(f_arr) > 0):
            raise ValueError("frequencies array must be strictly monotonically increasing")

        self._frequencies = f_arr
        self._psd = p_arr

    @property
    def frequencies(self) -> np.ndarray:
        """Return measured frequency array."""
        return self._frequencies

    @property
    def psd(self) -> np.ndarray:
        """Return measured PSD array."""
        return self._psd

    @property
    def f_min(self) -> float:
        """Return minimum frequency in measured PSD."""
        return float(self._frequencies[0])

    @property
    def f_max(self) -> float:
        """Return maximum frequency in measured PSD."""
        return float(self._frequencies[-1])

    def interpolate(
        self,
        target_frequencies: np.ndarray,
        f_start: float,
        f_0: float,
    ) -> np.ndarray:
        """Interpolate PSD onto target frequencies with low-frequency continuation below f_start.

        Interpolation Method:
        ---------------------
        - For f < f_start: Evaluates the frozen Group A analytical continuation:
            S_n(f) = S_n(f_start) * [ (f_start^2 + f_0^2) / (f^2 + f_0^2) ]^2
        - For f_start <= f <= f_max: Piecewise linear interpolation in (ln f, ln S_n) space.
          This guarantees strict positivity (exp(interpolant) > 0) and preserves power laws.
        - For f > f_max: Strictly rejected with a ValueError. Extrapolation beyond the
          measured band is unsupported.

        Parameters
        ----------
        target_frequencies : np.ndarray
            1D array of physical frequencies in Hz.
        f_start : float
            Lower analysis boundary in Hz. Must be positive and <= f_max.
        f_0 : float
            Continuation frequency scale in Hz. Must be positive (explicit, Decision D8 open).

        Returns
        -------
        np.ndarray
            1D array of interpolated PSD values matching target_frequencies shape.
        """
        targets = np.asarray(target_frequencies, dtype=np.float64)
        if targets.ndim != 1:
            raise ValueError("target_frequencies must be a 1D array")
        if not np.all(np.isfinite(targets)):
            raise ValueError("target_frequencies must contain only finite numbers")
        if np.any(targets < 0):
            raise ValueError("target_frequencies cannot contain negative frequencies")

        if (
            not isinstance(f_start, (int, float, np.floating))
            or not math.isfinite(f_start)
            or f_start <= 0
        ):
            raise ValueError(f"f_start must be a positive finite float, got {f_start}")
        if (
            not isinstance(f_0, (int, float, np.floating))
            or not math.isfinite(f_0)
            or f_0 <= 0
        ):
            raise ValueError(f"f_0 must be a positive finite float, got {f_0}")

        if np.any(targets > self.f_max):
            max_exceed = float(np.max(targets))
            raise ValueError(
                f"Requested frequency {max_exceed} Hz exceeds maximum measured PSD frequency "
                f"({self.f_max} Hz); high-frequency extrapolation is unsupported"
            )

        # Baseline value at f_start
        pos_mask = self._frequencies > 0
        pos_f = self._frequencies[pos_mask]
        pos_p = self._psd[pos_mask]

        if f_start < pos_f[0] or f_start > pos_f[-1]:
            raise ValueError(
                f"f_start ({f_start} Hz) is outside the positive frequency range "
                f"[{pos_f[0]}, {pos_f[-1]}] Hz"
            )

        ln_f_pos = np.log(pos_f)
        ln_p_pos = np.log(pos_p)
        s_at_start = float(np.exp(np.interp(np.log(f_start), ln_f_pos, ln_p_pos)))

        out = np.empty_like(targets)

        # Sub-band continuation: f < f_start
        low_mask = targets < float(f_start)
        if np.any(low_mask):
            f_low = targets[low_mask]
            ratio = (f_start**2 + f_0**2) / (f_low**2 + f_0**2)
            out[low_mask] = s_at_start * (ratio**2)

        # In-band interpolation: f >= f_start
        in_mask = ~low_mask
        if np.any(in_mask):
            f_in = targets[in_mask]
            ln_targets_in = np.log(f_in)
            out[in_mask] = np.exp(np.interp(ln_targets_in, ln_f_pos, ln_p_pos))

        return out

    def get_inverse_psd_on_active_bins(
        self,
        frequencies: np.ndarray,
        mask: np.ndarray,
        f_start: float,
        f_0: float,
    ) -> np.ndarray:
        """Extract inverse PSD weights 1 / S_n(f_k) strictly on the active bins.

        Parameters
        ----------
        frequencies : np.ndarray
            Full frequency grid array.
        mask : np.ndarray
            Boolean active observation mask.
        f_start : float
            Lower analysis frequency in Hz.
        f_0 : float
            Continuation scale in Hz (explicit, Decision D8 open).

        Returns
        -------
        np.ndarray
            1D array of inverse PSD weights 1 / S_n(f_k) for all bins where mask is True.
        """
        active_f = frequencies[mask]
        psd_active = self.interpolate(active_f, f_start=f_start, f_0=f_0)
        if np.any(psd_active <= 0):
            raise ValueError("Active-bin PSD evaluated to non-positive value")
        return 1.0 / psd_active


def analytical_detector_psd(
    frequencies: np.ndarray,
    model: str = "aligo_representative",
) -> np.ndarray:
    """Evaluate standard analytical reference power spectral density (PSD) curves.

    Parameters
    ----------
    frequencies : np.ndarray
        1D array of non-negative physical frequencies in Hz.
    model : str, default="aligo_representative"
        Analytical model name. Supported options:
        - "aligo_representative": Custom, regularized, hand-constructed representative
          detector-noise proxy designed for controlled numerical benchmarking:
              S_n(f) = S_0 * [ (f_knee^2 / (f^2 + f_reg^2))^2 + 1.0 + (f / f_high)^2 ]
          with S_0 = 1.0e-46 Hz^-1, f_knee = 35.0 Hz, f_reg = 15.0 Hz, f_high = 300.0 Hz.
          Features finite DC S_n(0) = S_0 * [(35/15)^4 + 1] approx 30.642e-46 Hz^-1.
          Note: This model is a custom, regularized, hand-constructed representative
          detector-noise proxy designed for controlled numerical benchmarking. It is
          NOT an official Advanced LIGO design-sensitivity curve, measured detector
          PSD, or standard reference formula.
        - "flat": Constant white-noise reference spectrum:
              S_n(f) = S_0
          with S_0 = 1.0e-46 Hz^-1.

    Returns
    -------
    np.ndarray
        1D float64 array of one-sided PSD values in [strain]^2 / Hz matching
        `frequencies.shape`.

    Raises
    ------
    ValueError
        If `frequencies` is not 1D, contains non-finite values (NaN/Inf), contains
        negative frequencies, or if `model` is unrecognized.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.ndim != 1:
        raise ValueError(f"frequencies must be a 1D array, got dimension {freqs.ndim}")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("frequencies must contain only finite numbers (no NaN or Inf)")
    if np.any(freqs < 0.0):
        raise ValueError("frequencies cannot contain negative values")

    if not isinstance(model, str):
        raise ValueError(f"model must be a string, got {type(model)}")

    if model == "aligo_representative":
        s0 = 1.0e-46
        f_knee = 35.0
        f_reg = 15.0
        f_high = 300.0
        seismic = (f_knee**2 / (freqs**2 + f_reg**2)) ** 2
        shot = 1.0
        mirror = (freqs / f_high) ** 2
        return (s0 * (seismic + shot + mirror)).astype(np.float64)
    elif model == "flat":
        s0 = 1.0e-46
        return np.full_like(freqs, fill_value=s0, dtype=np.float64)
    else:
        raise ValueError(
            f"Unknown PSD model '{model}'. Supported models: 'aligo_representative', 'flat'"
        )

