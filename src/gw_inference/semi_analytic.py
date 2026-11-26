"""Semi-analytic extrinsic parameter marginalization and complex matched filtering.

This module implements the canonical semi-analytic extrinsic marginalization
(phase and time) for compact binary gravitational-wave signals in Work Package 6 (WP6) Stage 3:
    theta = (M_c, eta, t_c, phi_c)

Mathematical Formulation:
-------------------------
1. Waveform factorization:
   The frequency-domain template factors exactly into intrinsic and extrinsic contributions:
       h~(f; M_c, eta, t_c, phi_c) = h~_0(f; M_c, eta) * exp(-2*pi*i*f*t_c) * exp(i*phi_c)
   where h~_0 is the zero-extrinsic template (t_c = 0, phi_c = 0).

2. Noise-weighted inner product expansion:
   (r|r) = (d - h | d - h) = (d|d) - 2 Re(d|h) + (h|h)
   where:
       (h|h) = (h_0|h_0)  [independent of t_c and phi_c]
       (d|h) = z(t_c) * exp(-i*phi_c)
   with complex matched-filter statistic:
       z(t_c) = 4 * Delta_f * sum_{k in K_+} (d_k * h~_{0,k}^* / S_n(f_k)) * exp(+2*pi*i*f_k*t_c)

3. Exact analytic phase marginalization:
   For uniform prior phi_c in [0, 2*pi):
       int_0^{2*pi} (d phi_c / (2*pi)) exp( Re(z(t_c) * exp(-i*phi_c)) ) = I_0(|z(t_c)|)
   giving the phase-marginalized log-likelihood:
       ln L_{phase_marg}(M_c, eta, t_c) = ln L_norm - 0.5 * (d|d) - 0.5 * (h_0|h_0) + ln I_0(|z(t_c)|)
   where ln I_0(|z|) is evaluated stably in log-domain as |z| + ln I_{0e}(|z|).

4. Numerical time marginalization:
   For uniform prior t_c in [t_{c,min}, t_{c,max}]:
       L_{marg}(M_c, eta) = (1 / Delta_t_c) * int_{t_{c,min}}^{t_{c,max}} dt_c exp( ln L_{phase_marg}(M_c, eta, t_c) )
   evaluated via positive-weight quadrature on a fine time grid using complex IFFT.

5. Complex IFFT convention:
   For N_fft = M * N with frequency resolution Delta_f:
       Y_k = 4 * Delta_f * (d_k * h~_{0,k}^* / S_n(f_k))  for k in K_+
       Y_m = 0                                           otherwise
       z(t_j) = N_fft * ifft(Y)_j  at times t_j = j / (M * f_s)
   Zero-padding factor M >= 1 provides exact bandlimited sinc interpolation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import i0e, logsumexp


def compute_complex_matched_filter_statistic_direct(
    data_fd: np.ndarray,
    template_zero_fd: np.ndarray,
    psd: np.ndarray,
    delta_f: float,
    active_indices: np.ndarray,
    t_c: Union[float, Sequence[float], np.ndarray],
) -> Union[complex, np.ndarray]:
    """Compute the complex matched-filter statistic z(t_c) by direct positive-frequency summation.

    z(t_c) = 4 * Delta_f * sum_{k in K_+} (d_k * h~_{0,k}^* / S_n(f_k)) * exp(+2*pi*i*f_k*t_c)

    Parameters
    ----------
    data_fd : np.ndarray
        1D complex128 array of frequency-domain detector data d~(f).
    template_zero_fd : np.ndarray
        1D complex128 array of zero-extrinsic template h~_0(f) (t_c = 0, phi_c = 0).
    psd : np.ndarray
        1D float64 array of one-sided power spectral density S_n(f).
    delta_f : float
        Frequency resolution Delta_f in Hz. Must be positive and finite.
    active_indices : np.ndarray
        1D integer array of active positive-frequency bin indices K_+.
    t_c : float or array-like
        Coalescence time(s) in seconds at which to evaluate z(t_c).

    Returns
    -------
    complex or np.ndarray
        Complex matched-filter statistic z(t_c).
        Returns a complex scalar for scalar t_c input, or a 1D complex128 array
        matching the input shape for array-like t_c.

    Raises
    ------
    ValueError
        If inputs have incompatible dimensions, non-positive PSD/delta_f, or non-finite values.
    """
    d = np.asarray(data_fd, dtype=np.complex128)
    h0 = np.asarray(template_zero_fd, dtype=np.complex128)
    sn = np.asarray(psd, dtype=np.float64)
    k_act = np.asarray(active_indices, dtype=np.int64)

    if d.ndim != 1 or h0.ndim != 1 or sn.ndim != 1:
        raise ValueError("data_fd, template_zero_fd, and psd must be 1D arrays")
    if not (d.shape == h0.shape == sn.shape):
        raise ValueError(f"Array shape mismatch: data={d.shape}, template={h0.shape}, psd={sn.shape}")
    if not math.isfinite(delta_f) or delta_f <= 0.0:
        raise ValueError(f"delta_f must be positive and finite, got {delta_f}")
    if k_act.ndim != 1 or k_act.size == 0:
        raise ValueError("active_indices must be a non-empty 1D array")
    if np.any(sn[k_act] <= 0.0) or not np.all(np.isfinite(sn[k_act])):
        raise ValueError("psd values on active indices must be strictly positive and finite")

    # Base integrand: 4 * delta_f * (d_k * h0_k^* / S_n(f_k))
    integrand = (4.0 * delta_f * d[k_act] * np.conj(h0[k_act])) / sn[k_act]
    f_act = k_act * delta_f

    # Handle scalar vs array-like t_c
    tc_arr = np.asarray(t_c, dtype=np.float64)
    is_scalar = (tc_arr.ndim == 0)

    if is_scalar:
        tc_val = float(tc_arr)
        # exp(+2*pi*i*f_k*t_c)
        phase_factors = np.exp(2.0j * np.pi * f_act * tc_val)
        return complex(np.sum(integrand * phase_factors))
    else:
        tc_flat = tc_arr.ravel()
        # Matrix multiplication: f_act (N_act,) x tc_flat (N_tc,) -> (N_act, N_tc)
        # z(t_c) = sum_k integrand_k * exp(2*pi*i * f_k * t_c)
        phases = 2.0 * np.pi * np.outer(f_act, tc_flat)
        phase_factors = np.exp(1.0j * phases)
        z_vals = np.dot(integrand, phase_factors)
        return z_vals.reshape(tc_arr.shape)


def compute_complex_matched_filter_statistic_ifft(
    data_fd: np.ndarray,
    template_zero_fd: np.ndarray,
    psd: np.ndarray,
    delta_f: float,
    active_indices: np.ndarray,
    n_base: int,
    zero_pad_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the complex matched-filter statistic z(t_j) via zero-padded complex IFFT.

    Evaluates:
        z(t_j) = 4 * Delta_f * sum_{k in K_+} (d_k * h~_{0,k}^* / S_n(f_k)) * exp(+2*pi*i*f_k*t_j)
    on the uniform fine time grid:
        t_j = j / (zero_pad_factor * f_s) = j / (N_fft * Delta_f)
    for j = 0, 1, ..., N_fft - 1, where N_fft = zero_pad_factor * n_base.

    Parameters
    ----------
    data_fd : np.ndarray
        1D complex128 array of frequency-domain detector data d~(f).
    template_zero_fd : np.ndarray
        1D complex128 array of zero-extrinsic template h~_0(f) (t_c = 0, phi_c = 0).
    psd : np.ndarray
        1D float64 array of one-sided power spectral density S_n(f).
    delta_f : float
        Frequency resolution Delta_f in Hz.
    active_indices : np.ndarray
        1D integer array of active positive-frequency bin indices K_+.
    n_base : int
        Full discrete time-domain length N = 2 * (len(psd) - 1) corresponding to Nyquist.
    zero_pad_factor : int, default=1
        Integer oversampling factor M >= 1. Yields N_fft = M * n_base bins.

    Returns
    -------
    t_grid : np.ndarray
        1D float64 array of sample times in seconds [0, T_padded).
    z_grid : np.ndarray
        1D complex128 array of matched-filter values z(t_j).

    Raises
    ------
    ValueError
        If inputs have invalid shapes, non-positive parameters, or invalid pad factor.
    """
    if not isinstance(n_base, (int, np.integer)) or n_base < 2:
        raise ValueError(f"n_base must be an integer >= 2, got {n_base}")
    if not isinstance(zero_pad_factor, (int, np.integer)) or zero_pad_factor < 1:
        raise ValueError(f"zero_pad_factor must be an integer >= 1, got {zero_pad_factor}")

    d = np.asarray(data_fd, dtype=np.complex128)
    h0 = np.asarray(template_zero_fd, dtype=np.complex128)
    sn = np.asarray(psd, dtype=np.float64)
    k_act = np.asarray(active_indices, dtype=np.int64)

    if d.ndim != 1 or h0.ndim != 1 or sn.ndim != 1:
        raise ValueError("data_fd, template_zero_fd, and psd must be 1D arrays")
    if not (d.shape == h0.shape == sn.shape):
        raise ValueError(f"Array shape mismatch: data={d.shape}, template={h0.shape}, psd={sn.shape}")
    if not math.isfinite(delta_f) or delta_f <= 0.0:
        raise ValueError(f"delta_f must be positive and finite, got {delta_f}")
    if np.any(sn[k_act] <= 0.0) or not np.all(np.isfinite(sn[k_act])):
        raise ValueError("psd values on active indices must be strictly positive and finite")

    n_fft = int(zero_pad_factor * n_base)

    # Validate that active indices fit within positive half of N_fft
    max_k = int(np.max(k_act))
    if max_k >= n_fft // 2:
        raise ValueError(f"Maximum active index {max_k} exceeds N_fft // 2 ({n_fft // 2})")

    # Allocate full complex array of length N_fft
    y_fft = np.zeros(n_fft, dtype=np.complex128)

    # Place positive-frequency coefficients Y_k = 4 * Delta_f * d_k * h0_k^* / S_n(f_k)
    y_fft[k_act] = (4.0 * delta_f * d[k_act] * np.conj(h0[k_act])) / sn[k_act]

    # Standard complex inverse FFT: ifft(Y)_j = (1 / N_fft) * sum_m Y_m * exp(+2*pi*i * j * m / N_fft)
    # Multiply by N_fft to cancel the (1 / N_fft) normalization factor:
    z_grid = n_fft * np.fft.ifft(y_fft)

    # Time grid: t_j = j / (N_fft * Delta_f)
    dt_fine = 1.0 / (n_fft * delta_f)
    t_grid = np.arange(n_fft, dtype=np.float64) * dt_fine

    return t_grid, z_grid


def log_bessel_i0(x: Union[float, Sequence[float], np.ndarray]) -> Union[float, np.ndarray]:
    """Compute ln I_0(x) stably in the log domain using exponentially scaled Bessel function.

    ln I_0(x) = |x| + ln I_{0e}(|x|)
    where I_{0e}(x) = exp(-x) * I_0(x) is numerically stable across all finite inputs.

    Parameters
    ----------
    x : float or array-like
        Argument(s) at which to evaluate ln I_0(x).

    Returns
    -------
    float or np.ndarray
        Value(s) of ln I_0(x).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    is_scalar = (x_arr.ndim == 0)

    abs_x = np.abs(x_arr)
    # i0e(0) = 1.0, ln(1.0) = 0.0 -> log_i0(0) = 0.0
    val_i0e = i0e(abs_x)
    # Clip tiny float underflow if any
    safe_i0e = np.maximum(val_i0e, np.finfo(np.float64).tiny)
    res = abs_x + np.log(safe_i0e)

    if is_scalar:
        return float(res)
    return res


def log_phase_marginalized_likelihood(
    z_abs: Union[float, Sequence[float], np.ndarray],
    d_inner_d: float,
    h0_inner_h0: float,
    log_l_norm: float,
) -> Union[float, np.ndarray]:
    """Compute the exact phase-marginalized log-likelihood ln L_{phase_marg}(t_c).

    ln L_{phase_marg}(t_c) = ln L_norm - 0.5 * (d|d) - 0.5 * (h_0|h_0) + ln I_0(|z(t_c)|)

    Parameters
    ----------
    z_abs : float or array-like
        Magnitude |z(t_c)| of the complex matched-filter statistic.
    d_inner_d : float
        Active-band noise-weighted data inner product (d|d).
    h0_inner_h0 : float
        Active-band noise-weighted template inner product (h_0|h_0).
    log_l_norm : float
        Log-likelihood normalization constant ln L_norm.

    Returns
    -------
    float or np.ndarray
        Phase-marginalized log-likelihood.
    """
    ln_i0 = log_bessel_i0(z_abs)
    base_const = float(log_l_norm) - 0.5 * float(d_inner_d) - 0.5 * float(h0_inner_h0)
    return base_const + ln_i0


def marginalize_extrinsic_parameters_4d(
    t_grid: np.ndarray,
    log_l_phase_marg: np.ndarray,
    tc_min: float = 1.90,
    tc_max: float = 2.10,
) -> float:
    """Numerically marginalize over coalescence time t_c using trapezoidal quadrature in log-space.

    Computes:
        ln L_{marg}(M_c, eta) = -ln(Delta_t_c) + ln int_{tc_min}^{tc_max} dt_c exp(ln L_{phase_marg}(t_c))
    where Delta_t_c = tc_max - tc_min.

    Parameters
    ----------
    t_grid : np.ndarray
        1D monotonically increasing array of time grid points in seconds.
    log_l_phase_marg : np.ndarray
        1D array of phase-marginalized log-likelihoods on t_grid.
    tc_min : float, default=1.90
        Lower integration boundary for t_c in seconds.
    tc_max : float, default=2.10
        Upper integration boundary for t_c in seconds.

    Returns
    -------
    float
        Extrinsic-marginalized log-likelihood ln L_{marg}(M_c, eta).

    Raises
    ------
    ValueError
        If integration domain is invalid or contains fewer than 2 points.
    """
    if not math.isfinite(tc_min):
        raise ValueError(f"tc_min must be finite, got {tc_min}")
    if not math.isfinite(tc_max) or tc_max <= tc_min:
        raise ValueError(f"tc_max must be greater than tc_min ({tc_min}), got {tc_max}")

    delta_tc = float(tc_max - tc_min)
    t = np.asarray(t_grid, dtype=np.float64)
    ln_l = np.asarray(log_l_phase_marg, dtype=np.float64)

    if t.ndim != 1 or ln_l.ndim != 1 or t.shape != ln_l.shape:
        raise ValueError(f"t_grid and log_l_phase_marg must be 1D arrays of matching shape, got {t.shape} and {ln_l.shape}")

    # Select points within [tc_min, tc_max]
    mask = (t >= tc_min) & (t <= tc_max)
    t_sub = t[mask]
    ln_l_sub = ln_l[mask]

    if t_sub.size < 2:
        raise ValueError(
            f"Fewer than 2 grid points found in [{tc_min}, {tc_max}]; grid range is [{t[0]}, {t[-1]}]"
        )

    # Compute positive trapezoidal quadrature weights
    # w_0 = (t_1 - t_0) / 2
    # w_j = (t_{j+1} - t_{j-1}) / 2
    # w_{end} = (t_{end} - t_{end-1}) / 2
    dt = np.diff(t_sub)
    if np.any(dt <= 0.0):
        raise ValueError("t_grid must be strictly monotonically increasing")

    weights = np.empty_like(t_sub)
    weights[0] = 0.5 * dt[0]
    weights[-1] = 0.5 * dt[-1]
    weights[1:-1] = 0.5 * (dt[:-1] + dt[1:])

    # Verify weight positivity
    if np.any(weights <= 0.0):
        raise ValueError("Trapezoidal weights must be strictly positive")

    # Log-sum-exp: ln int L dt = logsumexp(ln_l + ln(w))
    log_weights = np.log(weights)
    log_integral = float(logsumexp(ln_l_sub + log_weights))

    # Normalized marginal likelihood: ln L_marg = -ln(Delta_t_c) + log_integral
    return log_integral - math.log(delta_tc)


def compare_direct_and_ifft_matched_filter(
    data_fd: np.ndarray,
    template_zero_fd: np.ndarray,
    psd: np.ndarray,
    delta_f: float,
    active_indices: np.ndarray,
    n_base: int,
    test_times: Sequence[float],
    zero_pad_factor: int = 16,
) -> Dict[str, float]:
    """Compare direct-summation and IFFT matched-filter statistics across test times.

    Computes maximum absolute differences in:
    - Real part: max |Re(z_direct) - Re(z_ifft)|
    - Imaginary part: max |Im(z_direct) - Im(z_ifft)|
    - Magnitude: max ||z_direct| - |z_ifft||
    - Wrapped phase: max |angle(z_direct / z_ifft)|

    Parameters
    ----------
    data_fd : np.ndarray
        Frequency-domain detector data d~(f).
    template_zero_fd : np.ndarray
        Zero-extrinsic template h~_0(f).
    psd : np.ndarray
        Power spectral density S_n(f).
    delta_f : float
        Frequency resolution in Hz.
    active_indices : np.ndarray
        Active positive-frequency bin indices K_+.
    n_base : int
        Base discrete Fourier transform length.
    test_times : sequence of float
        Test coalescence times t_c in seconds.
    zero_pad_factor : int, default=16
        IFFT zero-padding factor.

    Returns
    -------
    dict
        Dictionary containing max differences:
        "max_diff_real", "max_diff_imag", "max_diff_magnitude", "max_diff_phase".
    """
    times = np.asarray(test_times, dtype=np.float64)
    z_direct = compute_complex_matched_filter_statistic_direct(
        data_fd=data_fd,
        template_zero_fd=template_zero_fd,
        psd=psd,
        delta_f=delta_f,
        active_indices=active_indices,
        t_c=times,
    )

    t_grid, z_grid = compute_complex_matched_filter_statistic_ifft(
        data_fd=data_fd,
        template_zero_fd=template_zero_fd,
        psd=psd,
        delta_f=delta_f,
        active_indices=active_indices,
        n_base=n_base,
        zero_pad_factor=zero_pad_factor,
    )

    # For each test time, interpolate or find nearest grid point
    # Since IFFT grid has spacing dt_fine, we interpolate linearly or find exact match
    dt_fine = t_grid[1] - t_grid[0]
    z_ifft_interp = np.empty_like(z_direct)

    for i, t_val in enumerate(times):
        # Index on periodic grid [0, T_padded)
        t_wrapped = t_val % (t_grid[-1] + dt_fine)
        idx_float = t_wrapped / dt_fine
        idx0 = int(math.floor(idx_float)) % len(t_grid)
        idx1 = (idx0 + 1) % len(t_grid)
        frac = idx_float - math.floor(idx_float)
        # Linear interpolation between adjacent fine bins
        z_ifft_interp[i] = (1.0 - frac) * z_grid[idx0] + frac * z_grid[idx1]

    diff_real = float(np.max(np.abs(np.real(z_direct) - np.real(z_ifft_interp))))
    diff_imag = float(np.max(np.abs(np.imag(z_direct) - np.imag(z_ifft_interp))))
    diff_mag = float(np.max(np.abs(np.abs(z_direct) - np.abs(z_ifft_interp))))

    # Angular difference wrapped to [-pi, pi]
    phase_ratio = z_direct * np.conj(z_ifft_interp)
    diff_phase = float(np.max(np.abs(np.angle(phase_ratio))))

    return {
        "max_diff_real": diff_real,
        "max_diff_imag": diff_imag,
        "max_diff_magnitude": diff_mag,
        "max_diff_phase": diff_phase,
    }
