"""Work Package 3: Matched filtering, SNR, phase maximization, and detection statistics.

This module implements the scientifically approved matched-filtering core for
compact binary gravitational-wave signals:
1. Noise-weighted active-band inner product (a|b)_{K_+}.
2. Template normalization sigma_h = sqrt((h|h)) and optimal SNR.
3. Direct finite frequency-sum reference matched filter at arbitrary lag.
4. Fast circular complex matched-filter time series z(t) via N-point analytic IFFT.
5. Signed and circular time-lag coordinate arrays for even and odd N.
6. Coalescence-phase maximization via two orthogonal quadratures.
7. Binned Allen chi-square time-frequency consistency diagnostic with
   degenerate sub-band handling.
8. Consistent Allen (2005) reweighted SNR statistic.
9. High-level orchestrator matched_filter producing an immutable MatchedFilterResult.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import numpy as np

from gw_inference.conventions import continuous_rfft
from gw_inference.grid import FrequencyGrid
from gw_inference.psd import PSDEngine


@dataclass(frozen=True)
class MatchedFilterResult:
    """Immutable container for matched-filter outputs and search diagnostics.

    Attributes
    ----------
    snr_time_series : np.ndarray
        1D complex128 array rho(t_j) = z(t_j) / sigma_h of length N.
        Re[rho] is 0-phase SNR; Im[rho] is pi/2-phase SNR.
    snr_max : np.ndarray
        1D float64 array of phase-maximized SNR |rho(t_j)| of length N.
    phase_max : np.ndarray
        1D float64 array of best-fit coalescence phase in radians in (-pi, pi].
    signed_lags : np.ndarray
        1D float64 array of signed physical time lags in seconds [-T/2, T/2).
    circular_lags : np.ndarray
        1D float64 array of circular time lags in seconds [0, T).
    valid_mask : np.ndarray
        1D bool array indicating valid lags under the operational boundary guard.
    sigma_h : float
        Scalar template norm (standard deviation) in active band.
    optimal_snr : float
        Optimal SNR rho_opt = sigma_h.
    peak_snr : float
        Maximum phase-maximized SNR over valid uncorrupted lags.
    peak_time : float
        Physical time lag in seconds corresponding to peak_snr.
    peak_phase : float
        Best-fit coalescence phase in radians corresponding to peak_snr.
    chisq : Optional[np.ndarray]
        1D float64 array of binned Allen chi^2 time series, if requested and q >= 2.
    chisq_dof : Optional[int]
        Effective degrees of freedom nu = 2q - 2, if chisq available.
    reweighted_snr : Optional[np.ndarray]
        1D float64 array of reweighted SNR time series, if chisq requested.
    """

    snr_time_series: np.ndarray
    snr_max: np.ndarray
    phase_max: np.ndarray
    signed_lags: np.ndarray
    circular_lags: np.ndarray
    valid_mask: np.ndarray
    sigma_h: float
    optimal_snr: float
    peak_snr: float
    peak_time: float
    peak_phase: float
    chisq: Optional[np.ndarray] = None
    chisq_dof: Optional[int] = None
    reweighted_snr: Optional[np.ndarray] = None


def compute_signed_lags(n_samples: int, delta_t: float) -> np.ndarray:
    """Generate signed physical time lag array for even or odd N.

    Follows standard discrete Fourier transform circular lag unwrapping:
    - For even N: lags span [-T/2, T/2) where T = N * delta_t.
      Indices 0 <= m < N/2 map to non-negative lags m * delta_t.
      Indices N/2 <= m < N map to negative lags (m - N) * delta_t.
    - For odd N: lags span [-(N - 1)*delta_t/2, (N - 1)*delta_t/2].
      Indices 0 <= m <= (N - 1)/2 map to non-negative lags m * delta_t.
      Indices (N + 1)/2 <= m < N map to negative lags (m - N) * delta_t.

    Parameters
    ----------
    n_samples : int
        Number of discrete time-domain samples N. Must be a positive integer.
    delta_t : float
        Sampling interval in seconds. Must be positive and finite.

    Returns
    -------
    np.ndarray
        1D float64 array of signed time lags of length n_samples in seconds.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
    if not isinstance(delta_t, (int, float, np.floating)) or not math.isfinite(delta_t) or delta_t <= 0:
        raise ValueError(f"delta_t must be a positive finite float, got {delta_t}")

    N = int(n_samples)
    dt = float(delta_t)
    m = np.arange(N, dtype=np.float64)
    cutoff = (N + 1) // 2 if (N % 2 != 0) else (N // 2)
    signed_lags = np.where(m < cutoff, m * dt, (m - N) * dt)
    return signed_lags


def noise_weighted_inner_product(
    a_fd: np.ndarray,
    b_fd: np.ndarray,
    psd: np.ndarray,
    delta_f: float,
    active_indices: Optional[np.ndarray] = None,
) -> float:
    """Compute the noise-weighted active-band inner product (a|b)_{K_+}.

    Mathematical Definition:
        (a|b)_{K_+} = 4 * delta_f * Re[ sum_{k in K_+} (a_k * b_k^*) / S_n(f_k) ]

    Conventions and Constraints:
    - Endpoints DC (k=0) and Nyquist (k=N/2 for even N) must not be included
      in the active summation, as defined by WP1 mask K_+.
    - The active mask/indices define the domain of integration.
    - If active_indices is None, a_fd, b_fd, and psd are assumed to be
      pre-sliced arrays over active bins.
    - If active_indices is provided, a_fd and b_fd are indexed by active_indices,
      while psd may be full-grid or active-bin size.

    Parameters
    ----------
    a_fd : np.ndarray
        First frequency-domain signal or template. 1D complex or float array.
    b_fd : np.ndarray
        Second frequency-domain signal or template. 1D complex or float array.
    psd : np.ndarray
        One-sided power spectral density S_n(f_k). Must be strictly positive.
    delta_f : float
        Frequency resolution Delta f in Hz. Must be positive and finite.
    active_indices : np.ndarray, optional
        1D integer array of active frequency-bin indices K_+.

    Returns
    -------
    float
        Real-valued inner product (a|b)_{K_+}.
    """
    if not isinstance(delta_f, (int, float, np.floating)) or not math.isfinite(delta_f) or delta_f <= 0:
        raise ValueError(f"delta_f must be a positive finite float, got {delta_f}")

    a = np.asarray(a_fd, dtype=np.complex128)
    b = np.asarray(b_fd, dtype=np.complex128)
    p = np.asarray(psd, dtype=np.float64)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a_fd and b_fd must be 1D arrays")
    if a.shape != b.shape:
        raise ValueError(f"a_fd shape {a.shape} must match b_fd shape {b.shape}")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("a_fd and b_fd must contain only finite numbers")

    if active_indices is not None:
        idx = np.asarray(active_indices, dtype=np.intp)
        if idx.ndim != 1:
            raise ValueError("active_indices must be a 1D array")
        if idx.size == 0:
            raise ValueError("active_indices cannot be empty")
        if np.any(idx < 0) or np.any(idx >= a.size):
            raise ValueError("active_indices contains out-of-bounds indices for input arrays")
        a_act = a[idx]
        b_act = b[idx]
        if p.size == a.size:
            p_act = p[idx]
        elif p.size == idx.size:
            p_act = p
        else:
            raise ValueError(
                f"psd shape {p.shape} does not match full grid ({a.size}) or active bins ({idx.size})"
            )
    else:
        if a.size == 0:
            raise ValueError("Input frequency-domain arrays cannot be empty")
        if p.shape != a.shape:
            raise ValueError(f"psd shape {p.shape} must match a_fd shape {a.shape}")
        a_act = a
        b_act = b
        p_act = p

    if not np.all(np.isfinite(p_act)):
        raise ValueError("psd must contain only finite numbers")
    if np.any(p_act <= 0):
        raise ValueError("psd values in active band must be strictly positive (> 0)")

    # 4 * delta_f * Re[ sum (a_k * b_k^*) / S_n(f_k) ]
    integrand = (a_act * np.conj(b_act)) / p_act
    inner_prod = 4.0 * float(delta_f) * float(np.real(np.sum(integrand)))
    return inner_prod


def compute_template_norm(
    template_fd: np.ndarray,
    inv_psd_active: np.ndarray,
    active_indices: np.ndarray,
    delta_f: float,
) -> float:
    """Compute active-band template norm sigma_h = sqrt((h|h)_{K_+}).

    Mathematical Definition:
        sigma_h^2 = 4 * delta_f * sum_{k in K_+} |h~(f_k)|^2 / S_n(f_k)
        sigma_h = sqrt(sigma_h^2)

    Parameters
    ----------
    template_fd : np.ndarray
        1D complex frequency-domain template array h~(f_k).
    inv_psd_active : np.ndarray
        1D float64 array of inverse PSD values 1 / S_n(f_k) over active bins.
    active_indices : np.ndarray
        1D integer array of active frequency-bin indices K_+.
    delta_f : float
        Frequency resolution Delta f in Hz.

    Returns
    -------
    float
        Strictly positive template norm sigma_h.

    Raises
    ------
    ValueError
        If template has zero power in the active band or inputs are invalid.
    """
    if not isinstance(delta_f, (int, float, np.floating)) or not math.isfinite(delta_f) or delta_f <= 0:
        raise ValueError(f"delta_f must be a positive finite float, got {delta_f}")

    h = np.asarray(template_fd, dtype=np.complex128)
    inv_p = np.asarray(inv_psd_active, dtype=np.float64)
    idx = np.asarray(active_indices, dtype=np.intp)

    if h.ndim != 1 or inv_p.ndim != 1 or idx.ndim != 1:
        raise ValueError("template_fd, inv_psd_active, and active_indices must be 1D arrays")
    if inv_p.size != idx.size:
        raise ValueError(
            f"inv_psd_active length ({inv_p.size}) must match active_indices length ({idx.size})"
        )
    if not np.all(np.isfinite(h)):
        raise ValueError("template_fd must contain only finite numbers")
    if not np.all(np.isfinite(inv_p)) or np.any(inv_p <= 0):
        raise ValueError("inv_psd_active must contain strictly positive finite values")

    h_act = h[idx]
    power_terms = (np.abs(h_act) ** 2) * inv_p
    sigma_h_sq = 4.0 * float(delta_f) * float(np.sum(power_terms))

    if sigma_h_sq <= 0:
        raise ValueError(
            f"Template has non-positive power in active band: sigma_h^2 = {sigma_h_sq}. "
            "Cannot normalize matched filter with a zero template."
        )

    return float(math.sqrt(sigma_h_sq))


def direct_matched_filter_at_lag(
    data_fd: np.ndarray,
    template_fd: np.ndarray,
    inv_psd_active: np.ndarray,
    active_indices: np.ndarray,
    freqs: np.ndarray,
    delta_f: float,
    time_lag: float,
) -> complex:
    """Evaluate complex matched filter at a specified lag via the authoritative direct sum.

    Mathematical Definition:
        z(tau) = 4 * delta_f * sum_{k in K_+} [ d~(f_k) * h~^*(f_k) / S_n(f_k) ] * exp(i * 2 * pi * f_k * tau)

    This is the authoritative, ground-truth reference implementation defined on
    the frozen WP1 grid without FFT approximations or circular wraparound.

    Parameters
    ----------
    data_fd : np.ndarray
        1D complex continuous-scaled frequency-domain data d~(f_k).
    template_fd : np.ndarray
        1D complex continuous-scaled frequency-domain template h~(f_k).
    inv_psd_active : np.ndarray
        1D float64 inverse PSD array over active bins 1 / S_n(f_k).
    active_indices : np.ndarray
        1D integer array of active frequency-bin indices K_+.
    freqs : np.ndarray
        1D float64 array of physical frequency bin centers in Hz.
    delta_f : float
        Frequency resolution Delta f in Hz.
    time_lag : float
        Physical time lag tau in seconds.

    Returns
    -------
    complex
        Complex matched filter output z(tau) = x(tau) + i y(tau).
    """
    if not isinstance(delta_f, (int, float, np.floating)) or not math.isfinite(delta_f) or delta_f <= 0:
        raise ValueError(f"delta_f must be a positive finite float, got {delta_f}")
    if not isinstance(time_lag, (int, float, np.floating)) or not math.isfinite(time_lag):
        raise ValueError(f"time_lag must be a finite float, got {time_lag}")

    d = np.asarray(data_fd, dtype=np.complex128)
    h = np.asarray(template_fd, dtype=np.complex128)
    inv_p = np.asarray(inv_psd_active, dtype=np.float64)
    idx = np.asarray(active_indices, dtype=np.intp)
    f = np.asarray(freqs, dtype=np.float64)

    if d.ndim != 1 or h.ndim != 1 or inv_p.ndim != 1 or idx.ndim != 1 or f.ndim != 1:
        raise ValueError("All input arrays must be 1D")
    if d.shape != h.shape or d.shape != f.shape:
        raise ValueError("data_fd, template_fd, and freqs must have identical shapes")
    if inv_p.size != idx.size:
        raise ValueError("inv_psd_active length must match active_indices length")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(h)):
        raise ValueError("data_fd and template_fd must contain only finite numbers")
    if not np.all(np.isfinite(inv_p)) or np.any(inv_p <= 0):
        raise ValueError("inv_psd_active must contain strictly positive finite values")

    d_act = d[idx]
    h_act = h[idx]
    f_act = f[idx]

    # Kernel: 4 * delta_f * (d * h^* / S_n) * exp(i * 2pi * f * tau)
    shift = np.exp(2.0j * np.pi * f_act * float(time_lag))
    terms = (d_act * np.conj(h_act) * inv_p) * shift
    z = 4.0 * float(delta_f) * complex(np.sum(terms))
    return z


def matched_filter_core(
    data_fd: np.ndarray,
    template_fd: np.ndarray,
    inv_psd_active: np.ndarray,
    active_indices: np.ndarray,
    n_samples: int,
    delta_f: float,
) -> Tuple[np.ndarray, float]:
    """Execute complex matched-filter correlation via N-point analytic IFFT.

    Mathematical Construction:
        1. Construct full N-point complex array Z in C^N initialized to 0.
        2. Populate active positive frequencies k in K_+:
           Z_k = 4 * delta_f * [ d~(f_k) * h~^*(f_k) / S_n(f_k) ]
        3. All other bins (DC k=0, Nyquist k=N/2 for even N, and negative frequencies
           k >= ceil(N/2)) remain identically 0.
        4. Invert via NumPy's N-point ifft:
           z(t_m) = N * ifft(Z, n=N)_m
        5. Compute template norm sigma_h = sqrt((h|h)_{K_+}).

    Normalization and Statistics:
        Under stationary Gaussian noise colored by S_n(f):
        - Var(Re z) = Var(Im z) = sigma_h^2
        - Cov(Re z, Im z) = 0
        - E[|z|^2] = 2 * sigma_h^2
        - Complex SNR rho(t) = z(t) / sigma_h satisfies Re rho ~ N(0, 1), Im rho ~ N(0, 1)
        - Phase-maximized SNR |rho(t)| ~ Rayleigh(scale 1).

    Parameters
    ----------
    data_fd : np.ndarray
        1D complex continuous-scaled frequency-domain data d~(f_k).
    template_fd : np.ndarray
        1D complex continuous-scaled frequency-domain template h~(f_k).
    inv_psd_active : np.ndarray
        1D float64 inverse PSD values over active bins 1 / S_n(f_k).
    active_indices : np.ndarray
        1D integer array of active frequency-bin indices K_+.
    n_samples : int
        Number of time-domain samples N.
    delta_f : float
        Frequency resolution Delta f in Hz.

    Returns
    -------
    tuple of (z_time_series, sigma_h)
        z_time_series : 1D complex128 array of length n_samples.
        sigma_h : float active-band template norm.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
    if not isinstance(delta_f, (int, float, np.floating)) or not math.isfinite(delta_f) or delta_f <= 0:
        raise ValueError(f"delta_f must be a positive finite float, got {delta_f}")

    N = int(n_samples)
    df = float(delta_f)

    d = np.asarray(data_fd, dtype=np.complex128)
    h = np.asarray(template_fd, dtype=np.complex128)
    inv_p = np.asarray(inv_psd_active, dtype=np.float64)
    idx = np.asarray(active_indices, dtype=np.intp)

    if d.ndim != 1 or h.ndim != 1 or inv_p.ndim != 1 or idx.ndim != 1:
        raise ValueError("All input arrays must be 1D")
    if d.shape != h.shape:
        raise ValueError(f"data_fd shape {d.shape} must match template_fd shape {h.shape}")
    if inv_p.size != idx.size:
        raise ValueError(
            f"inv_psd_active size ({inv_p.size}) must match active_indices size ({idx.size})"
        )
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(h)):
        raise ValueError("data_fd and template_fd must contain only finite numbers")
    if not np.all(np.isfinite(inv_p)) or np.any(inv_p <= 0):
        raise ValueError("inv_psd_active must contain strictly positive finite values")

    # Compute template norm sigma_h
    sigma_h = compute_template_norm(
        template_fd=h,
        inv_psd_active=inv_p,
        active_indices=idx,
        delta_f=df,
    )

    # Embed active bins into N-point complex array Z
    Z = np.zeros(N, dtype=np.complex128)
    d_act = d[idx]
    h_act = h[idx]
    Z[idx] = 4.0 * df * (d_act * np.conj(h_act) * inv_p)

    # N * ifft(Z) evaluates z(t_m) = sum_{k in K_+} Z_k exp(i * 2pi * k * m / N)
    z_time_series = N * np.fft.ifft(Z, n=N)

    return z_time_series, sigma_h


def compute_chisq_binned(
    data_fd: np.ndarray,
    template_fd: np.ndarray,
    inv_psd_active: np.ndarray,
    active_indices: np.ndarray,
    n_samples: int,
    delta_f: float,
    n_bins: int = 16,
    power_tol: float = 1e-12,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Compute Allen (2005) binned chi^2 time series with adaptive sub-band merging.

    Algorithm:
    ----------
    1. Divide active frequency band K_+ into p nominal contiguous sub-bands
       such that each sub-band has roughly equal template power sigma_{h, l}^2.
    2. Check power in each sub-band:
       P = { l : sigma_{h, l}^2 > power_tol * sigma_h^2 }, with q = |P|.
    3. If q < 2:
       Degenerate case. All power is concentrated in <= 1 sub-band.
       Chi^2 is mathematically undefined and unavailable.
       Returns (None, None).
    4. If 2 <= q < p:
       Merge sub-bands with insufficient power into adjacent active sub-bands.
       Recompute weights w~_l = sigma~_{h, l}^2 / sigma_h^2 (sum w~_l = 1).
       Degrees of freedom are strictly nu = 2q - 2.
    5. For each retained sub-band l in {1, ..., q}:
       Compute sub-band matched filter time series z_l(t_m) via N-point IFFT.
    6. Residual vector:
       Delta z_l(t_m) = z_l(t_m) - w~_l * z(t_m)
    7. Chi^2 time series:
       chi^2(t_m) = sum_{l=1}^q |Delta z_l(t_m)|^2 / sigma~_{h, l}^2.

    Parameters
    ----------
    data_fd : np.ndarray
        1D complex continuous-scaled frequency-domain data d~(f_k).
    template_fd : np.ndarray
        1D complex continuous-scaled frequency-domain template h~(f_k).
    inv_psd_active : np.ndarray
        1D float64 inverse PSD values over active bins.
    active_indices : np.ndarray
        1D integer array of active frequency-bin indices K_+.
    n_samples : int
        Number of time-domain samples N.
    delta_f : float
        Frequency resolution Delta f in Hz.
    n_bins : int, default=16
        Nominal number of frequency sub-bands p. Must be >= 2.
    power_tol : float, default=1e-12
        Relative template power threshold below which sub-bands are considered empty.

    Returns
    -------
    tuple of (chisq_time_series, degrees_of_freedom)
        chisq_time_series : Optional 1D float64 array of length n_samples.
        degrees_of_freedom : Optional int nu = 2q - 2.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError(f"n_bins must be an integer >= 2, got {n_bins}")
    if not isinstance(power_tol, (int, float, np.floating)) or power_tol <= 0:
        raise ValueError(f"power_tol must be a positive float, got {power_tol}")

    p_nominal = int(n_bins)
    idx = np.asarray(active_indices, dtype=np.intp)
    h = np.asarray(template_fd, dtype=np.complex128)
    inv_p = np.asarray(inv_psd_active, dtype=np.float64)
    N = int(n_samples)
    df = float(delta_f)

    # Bin-by-bin template power elements
    h_act = h[idx]
    bin_powers = 4.0 * df * ((np.abs(h_act) ** 2) * inv_p)
    total_power = float(np.sum(bin_powers))

    if total_power <= 0:
        return None, None

    # Construct p nominal sub-bands partitioned by cumulative template power
    cum_power = np.cumsum(bin_powers)
    target_power_per_bin = total_power / p_nominal

    sub_band_indices = []
    start_idx = 0
    for l in range(1, p_nominal):
        target = l * target_power_per_bin
        split_idx = int(np.searchsorted(cum_power, target))
        split_idx = max(start_idx, min(split_idx, len(idx)))
        sub_band_indices.append((start_idx, split_idx))
        start_idx = split_idx
    sub_band_indices.append((start_idx, len(idx)))

    # Compute power in each nominal sub-band
    nominal_sub_band_powers = [
        float(np.sum(bin_powers[s:e])) if e > s else 0.0 for s, e in sub_band_indices
    ]

    # Identify active sub-bands exceeding relative tolerance
    tol = float(power_tol) * total_power
    active_bands = [
        (s, e, pwr)
        for (s, e), pwr in zip(sub_band_indices, nominal_sub_band_powers)
        if (e > s and pwr > tol)
    ]
    q = len(active_bands)

    if q < 2:
        # Degenerate: power in <= 1 sub-band. Chi^2 is mathematically undefined.
        return None, None

    # Re-normalize retained sub-band powers
    retained_powers = np.array([pwr for _, _, pwr in active_bands], dtype=np.float64)
    retained_slices = [(s, e) for s, e, _ in active_bands]
    sigma_sq_l = retained_powers
    weights = sigma_sq_l / np.sum(sigma_sq_l)

    # Evaluate sub-band IFFTs
    d = np.asarray(data_fd, dtype=np.complex128)
    d_act = d[idx]
    z_sub_list = []
    for s, e in retained_slices:
        Z_l = np.zeros(N, dtype=np.complex128)
        sub_idx = idx[s:e]
        sub_integrand = 4.0 * df * (d_act[s:e] * np.conj(h_act[s:e]) * inv_p[s:e])
        Z_l[sub_idx] = sub_integrand
        z_l = N * np.fft.ifft(Z_l, n=N)
        z_sub_list.append(z_l)

    # Total filter output is sum of sub-band outputs
    z_total = np.zeros(N, dtype=np.complex128)
    for z_l in z_sub_list:
        z_total += z_l

    # Compute chi^2 = sum_l |z_l - w_l * z|^2 / sigma_{h, l}^2
    chisq = np.zeros(N, dtype=np.float64)
    for l in range(q):
        residual = z_sub_list[l] - weights[l] * z_total
        chisq += (np.abs(residual) ** 2) / sigma_sq_l[l]

    chisq_dof = 2 * q - 2
    return chisq, chisq_dof


def compute_reweighted_snr(
    snr_max: np.ndarray,
    chisq: Optional[np.ndarray],
    chisq_dof: Optional[int],
) -> np.ndarray:
    """Compute Allen (2005) reweighted SNR time series.

    Mathematical Definition:
        chi_r^2(t) = chi^2(t) / (2q - 2)
        rho_tilde(t) = rho_max(t)                                    if chi_r^2(t) <= 1
        rho_tilde(t) = rho_max(t) / [ (1 + (chi_r^2(t))^3) / 2 ]^(1/6)  if chi_r^2(t) > 1

    When chi^2 is unavailable (q < 2), rho_tilde(t) = rho_max(t) unconditionally.

    Parameters
    ----------
    snr_max : np.ndarray
        1D float64 phase-maximized SNR array |rho(t)|.
    chisq : np.ndarray, optional
        1D float64 Allen chi^2 time series.
    chisq_dof : int, optional
        Degrees of freedom nu = 2q - 2.

    Returns
    -------
    np.ndarray
        1D float64 reweighted SNR time series rho_tilde(t).
    """
    rho = np.asarray(snr_max, dtype=np.float64)
    if chisq is None or chisq_dof is None or chisq_dof <= 0:
        return rho.copy()

    chi2 = np.asarray(chisq, dtype=np.float64)
    if chi2.shape != rho.shape:
        raise ValueError(f"chisq shape {chi2.shape} must match snr_max shape {rho.shape}")

    chi_r_sq = np.maximum(0.0, chi2 / float(chisq_dof))
    penalty_mask = chi_r_sq > 1.0

    reweighted = rho.copy()
    if np.any(penalty_mask):
        denom = ((1.0 + (chi_r_sq[penalty_mask] ** 3)) / 2.0) ** (1.0 / 6.0)
        reweighted[penalty_mask] = rho[penalty_mask] / denom

    return reweighted


def matched_filter(
    data: np.ndarray,
    template_fd: np.ndarray,
    grid: FrequencyGrid,
    psd_engine: PSDEngine,
    mask: np.ndarray,
    f_start: float,
    f_0: float,
    tau_chirp: Optional[float] = None,
    tau_filter_diag: float = 0.2,
    compute_chisq: bool = False,
    chisq_bins: int = 16,
) -> MatchedFilterResult:
    """High-level authoritative matched-filtering entry point.

    Synthesizes detector time series or frequency series, one-sided noise PSD,
    and frequency-domain template into optimal SNR time series, phase-maximized
    statistics, and time-frequency consistency diagnostics.

    Parameters
    ----------
    data : np.ndarray
        Input detector data. Either 1D real time series of length grid.n_samples,
        or 1D complex continuous-scaled frequency-domain data of length grid.n_frequencies.
    template_fd : np.ndarray
        1D complex continuous-scaled frequency-domain template h~(f_k) of length
        grid.n_frequencies.
    grid : FrequencyGrid
        Authoritative WP1 frequency grid.
    psd_engine : PSDEngine
        Authoritative WP1 PSD engine.
    mask : np.ndarray
        1D boolean active observation mask K_+ on grid.
    f_start : float
        Lower analysis frequency in Hz. Must be positive and <= grid.nyquist.
    f_0 : float
        Mandatory continuation frequency scale in Hz (Decision D8 open; no default).
    tau_chirp : float, optional
        In-band chirp duration in seconds. If provided, used to construct the
        operational boundary diagnostic guard.
    tau_filter_diag : float, default=0.2
        Operational boundary diagnostic width in seconds for inverse PSD filter
        transient (uncertified diagnostic heuristic).
    compute_chisq : bool, default=False
        Whether to compute the binned Allen chi^2 diagnostic time series.
    chisq_bins : int, default=16
        Nominal number of sub-bands for Allen chi^2.

    Returns
    -------
    MatchedFilterResult
        Immutable result container with SNR series, phase series, lags, and diagnostics.
    """
    if not isinstance(grid, FrequencyGrid):
        raise TypeError(f"grid must be a FrequencyGrid instance, got {type(grid)}")
    if not isinstance(psd_engine, PSDEngine):
        raise TypeError(f"psd_engine must be a PSDEngine instance, got {type(psd_engine)}")
    if not isinstance(f_start, (int, float, np.floating)) or not math.isfinite(f_start) or f_start <= 0:
        raise ValueError(f"f_start must be a positive finite float, got {f_start}")
    if not isinstance(f_0, (int, float, np.floating)) or not math.isfinite(f_0) or f_0 <= 0:
        raise ValueError(f"f_0 must be a positive finite float (Decision D8 mandatory), got {f_0}")

    grid._validate_mask_shape(mask)
    active_indices = grid.get_active_indices(mask)
    if active_indices.size == 0:
        raise ValueError("Active mask K_+ contains zero active bins")

    # Validate template_fd
    h = np.asarray(template_fd, dtype=np.complex128)
    if h.ndim != 1 or h.size != grid.n_frequencies:
        raise ValueError(
            f"template_fd shape {h.shape} must be 1D with length grid.n_frequencies ({grid.n_frequencies})"
        )
    if not np.all(np.isfinite(h)):
        raise ValueError("template_fd must contain only finite numbers")

    # Interpolate PSD on target frequencies with mandatory f_0
    psd_full = psd_engine.interpolate(grid.frequencies, f_start=float(f_start), f_0=float(f_0))
    psd_active = psd_full[active_indices]
    if np.any(psd_active <= 0):
        raise ValueError("Interpolated PSD values in active band must be strictly positive")
    inv_psd_active = 1.0 / psd_active

    # Prepare frequency-domain data d~(f_k)
    d_raw = np.asarray(data)
    if d_raw.ndim != 1:
        raise ValueError(f"data must be a 1D array, got dimension {d_raw.ndim}")
    if not np.all(np.isfinite(d_raw)):
        raise ValueError("data must contain only finite numbers (no NaN or Inf)")

    if d_raw.size == grid.n_samples and not np.iscomplexobj(d_raw):
        # Time-domain data: compute continuous-scaled rfft
        data_fd = continuous_rfft(d_raw.astype(np.float64), grid.delta_t)
    elif d_raw.size == grid.n_frequencies:
        # Already frequency-domain data
        data_fd = d_raw.astype(np.complex128)
    else:
        raise ValueError(
            f"data length ({d_raw.size}) does not match grid.n_samples ({grid.n_samples}) "
            f"or grid.n_frequencies ({grid.n_frequencies})"
        )

    # Core matched filter execution
    z_time_series, sigma_h = matched_filter_core(
        data_fd=data_fd,
        template_fd=h,
        inv_psd_active=inv_psd_active,
        active_indices=active_indices,
        n_samples=grid.n_samples,
        delta_f=grid.delta_f,
    )

    # SNR time series
    rho_complex = z_time_series / sigma_h
    snr_max = np.abs(rho_complex)
    phase_max = np.angle(z_time_series)

    # Lag coordinate arrays
    signed_lags = compute_signed_lags(grid.n_samples, grid.delta_t)
    circular_lags = np.arange(grid.n_samples, dtype=np.float64) * grid.delta_t

    # Operational boundary guard mask (uncertified diagnostic)
    T = grid.duration
    guard_tau = float(tau_filter_diag) if tau_filter_diag is not None else 0.2
    if tau_chirp is not None and tau_chirp > 0:
        guard_start = float(tau_chirp) + guard_tau
    else:
        guard_start = guard_tau
    guard_end = T - guard_tau

    if guard_start < guard_end:
        valid_mask = (circular_lags >= guard_start) & (circular_lags <= guard_end)
    else:
        # Observation time shorter than guard window: retain all lags with warning flag
        valid_mask = np.ones(grid.n_samples, dtype=bool)

    # Peak search over valid lags
    if np.any(valid_mask):
        valid_indices = np.flatnonzero(valid_mask)
        rel_peak_idx = int(np.argmax(snr_max[valid_mask]))
        peak_idx = valid_indices[rel_peak_idx]
    else:
        peak_idx = int(np.argmax(snr_max))

    peak_snr = float(snr_max[peak_idx])
    peak_time = float(signed_lags[peak_idx])
    peak_phase = float(phase_max[peak_idx])

    # Optional Allen chi-square diagnostic
    chisq_arr = None
    chisq_dof = None
    reweighted_snr = None
    if compute_chisq:
        chisq_arr, chisq_dof = compute_chisq_binned(
            data_fd=data_fd,
            template_fd=h,
            inv_psd_active=inv_psd_active,
            active_indices=active_indices,
            n_samples=grid.n_samples,
            delta_f=grid.delta_f,
            n_bins=chisq_bins,
        )
        reweighted_snr = compute_reweighted_snr(
            snr_max=snr_max,
            chisq=chisq_arr,
            chisq_dof=chisq_dof,
        )

    return MatchedFilterResult(
        snr_time_series=rho_complex,
        snr_max=snr_max,
        phase_max=phase_max,
        signed_lags=signed_lags,
        circular_lags=circular_lags,
        valid_mask=valid_mask,
        sigma_h=sigma_h,
        optimal_snr=sigma_h,
        peak_snr=peak_snr,
        peak_time=peak_time,
        peak_phase=peak_phase,
        chisq=chisq_arr,
        chisq_dof=chisq_dof,
        reweighted_snr=reweighted_snr,
    )
