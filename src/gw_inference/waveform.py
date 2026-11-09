"""TaylorF2 frequency-domain gravitational waveform engine.

This module implements the frozen Group C specification from Stage 1.3 Phase B Revision 9
and Stage 1.4:
- Nonspinning, quasi-circular compact binary inspiral.
- Restricted-amplitude stationary-phase approximation (SPA).
- 3.5PN post-Newtonian Fourier phase.
- SI-consistent physical dimensions and constants.
- Exact Schwarzschild ISCO cutoff with inclusive bin-boundary convention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from gw_inference.constants import C_ISCO_SI, C_SI, GAMMA_E, G_SI, M_SUN_SI


def component_masses_to_chirp_mass(
    m1_kg: float, m2_kg: float
) -> Tuple[float, float, float]:
    """Convert component masses to total mass, symmetric mass ratio, and chirp mass.

    Parameters
    ----------
    m1_kg : float
        Primary mass in kg. Must be positive and finite.
    m2_kg : float
        Secondary mass in kg. Must be positive and finite.

    Returns
    -------
    tuple of (M, eta, M_chirp)
        M : Total mass in kg (m1 + m2).
        eta : Symmetric mass ratio (m1 * m2 / M^2), in (0, 0.25].
        M_chirp : Chirp mass in kg (M * eta^(3/5)).

    Raises
    ------
    ValueError
        If masses are non-positive or non-finite.
    """
    if not isinstance(m1_kg, (int, float, np.floating)) or not math.isfinite(m1_kg) or m1_kg <= 0:
        raise ValueError(f"m1_kg must be a positive finite float, got {m1_kg}")
    if not isinstance(m2_kg, (int, float, np.floating)) or not math.isfinite(m2_kg) or m2_kg <= 0:
        raise ValueError(f"m2_kg must be a positive finite float, got {m2_kg}")

    m1 = float(m1_kg)
    m2 = float(m2_kg)
    M = m1 + m2
    eta = (m1 * m2) / (M**2)
    # Numerical clip to 0.25 to prevent tiny float roundoff exceeding 0.25 on equal masses
    eta = min(0.25, float(eta))
    M_chirp = M * (eta ** (3.0 / 5.0))
    return M, eta, M_chirp


def schwarzschild_isco_frequency(m_total_msun: float) -> float:
    """Calculate Schwarzschild ISCO frequency in Hz.

    f_ISCO = C_ISCO / (M_total / M_sun) = 4397.17 / (M_total / M_sun) Hz

    Parameters
    ----------
    m_total_msun : float
        Total detector-frame mass in solar masses (M / M_sun). Must be positive.

    Returns
    -------
    float
        Schwarzschild ISCO frequency in Hz.
    """
    if (
        not isinstance(m_total_msun, (int, float, np.floating))
        or not math.isfinite(m_total_msun)
        or m_total_msun <= 0
    ):
        raise ValueError(f"m_total_msun must be a positive finite float, got {m_total_msun}")
    return float(C_ISCO_SI / m_total_msun)


def effective_termination_frequency(
    f_end: Optional[float], f_isco: float
) -> float:
    """Compute effective waveform termination frequency.

    f_term = min(f_end, f_ISCO) if f_end is provided, else f_ISCO.

    Parameters
    ----------
    f_end : float, optional
        Upper analysis frequency band limit in Hz.
    f_isco : float
        Schwarzschild ISCO frequency in Hz.

    Returns
    -------
    float
        Effective termination frequency in Hz.
    """
    if f_end is None:
        return float(f_isco)
    if (
        not isinstance(f_end, (int, float, np.floating))
        or not math.isfinite(f_end)
        or f_end <= 0
    ):
        raise ValueError(f"f_end must be a positive finite float, got {f_end}")
    return float(min(float(f_end), float(f_isco)))


def taylorf2_amplitude_prefactor(chirp_mass_kg: float, distance_m: float) -> float:
    """Compute SI constant amplitude prefactor A for restricted TaylorF2.

    A = sqrt(5 / 24) * pi^(-2/3) * (c / D_L) * (G * M_chirp / c^3)^(5/6)

    Dimensional Verification:
    -------------------------
    [c / D_L] = s^-1
    [G * M_chirp / c^3] = s
    [A] = s^-1 * s^(5/6) = s^(-1/6)

    Parameters
    ----------
    chirp_mass_kg : float
        Detector-frame chirp mass in kg. Must be positive.
    distance_m : float
        Luminosity distance in meters. Must be positive.

    Returns
    -------
    float
        Constant amplitude prefactor A in s^(-1/6).
    """
    if (
        not isinstance(chirp_mass_kg, (int, float, np.floating))
        or not math.isfinite(chirp_mass_kg)
        or chirp_mass_kg <= 0
    ):
        raise ValueError(f"chirp_mass_kg must be a positive finite float, got {chirp_mass_kg}")
    if (
        not isinstance(distance_m, (int, float, np.floating))
        or not math.isfinite(distance_m)
        or distance_m <= 0
    ):
        raise ValueError(f"distance_m must be a positive finite float, got {distance_m}")

    c = C_SI
    G = G_SI
    M_c = float(chirp_mass_kg)
    D_L = float(distance_m)

    factor_const = math.sqrt(5.0 / 24.0) * (math.pi ** (-2.0 / 3.0))
    inv_time_dist = c / D_L
    char_time = (G * M_c) / (c**3)

    return float(factor_const * inv_time_dist * (char_time ** (5.0 / 6.0)))


def taylorf2_35pn_phase(
    freqs: np.ndarray,
    m1_kg: float,
    m2_kg: float,
    phi_c: float = 0.0,
) -> np.ndarray:
    """Evaluate 3.5PN TaylorF2 Fourier phase Psi_{TF2}^+(f) at t_c = 0.

    Phase definition (Group C frozen convention):
        Psi_{TF2}^+(f) = - phi_c - pi / 4 + Psi_{int}(f)
    where:
        Psi_{int}(f) = (3 / (128 * eta * v^5)) * sum_{i=0}^7 alpha_i(v) * v^i
        v = (pi * G * M * f / c^3)^(1/3) = (pi * G * M_chirp * f / c^3)^(1/3) * eta^(-1/5)

    Coefficients alpha_0 through alpha_7:
        alpha_0 = 1.0
        alpha_1 = 0.0
        alpha_2 = 3715/756 + (55/9) * eta
        alpha_3 = -16 * pi
        alpha_4 = 15293365/508032 + (27145/504) * eta + (3085/72) * eta^2
        alpha_5 = pi * (38645/756 - (65/9) * eta) * (1 + 3 * ln(v))
        alpha_6 = 11583231236531/46942156800 - (6848/21) * gamma_E - (640/3) * pi^2
                  + (-15737765635/3048192 + (2255/12) * pi^2) * eta
                  + (76055/1728) * eta^2 - (127825/1296) * eta^3
                  - (6848/21) * ln(4 * v)
        alpha_7 = pi * (77096675/254016 + (378515/1512) * eta - (74045/756) * eta^2)

    Parameters
    ----------
    freqs : np.ndarray
        1D array of positive physical frequencies in Hz.
    m1_kg : float
        Primary mass in kg.
    m2_kg : float
        Secondary mass in kg.
    phi_c : float, default=0.0
        Coalescence phase in radians.

    Returns
    -------
    np.ndarray
        1D array of phase values Psi_{TF2}^+(f) in radians.
    """
    M, eta, _ = component_masses_to_chirp_mass(m1_kg, m2_kg)
    f_arr = np.asarray(freqs, dtype=np.float64)

    if np.any(f_arr <= 0):
        raise ValueError("Frequencies must be strictly positive (> 0) for TaylorF2 phase evaluation")

    # Dimensionless velocity v = (pi * G * M * f / c^3)^(1/3)
    c = C_SI
    G = G_SI
    geom_factor = math.pi * G * M / (c**3)
    v = (geom_factor * f_arr) ** (1.0 / 3.0)

    # Individual PN coefficients
    a0 = 1.0
    a1 = 0.0
    a2 = 3715.0 / 756.0 + (55.0 / 9.0) * eta
    a3 = -16.0 * math.pi
    a4 = 15293365.0 / 508032.0 + (27145.0 / 504.0) * eta + (3085.0 / 72.0) * (eta**2)

    ln_v = np.log(v)
    a5 = math.pi * (38645.0 / 756.0 - (65.0 / 9.0) * eta) * (1.0 + 3.0 * ln_v)

    gamma_E = GAMMA_E
    a6_const = (
        11583231236531.0 / 46942156800.0
        - (6848.0 / 21.0) * gamma_E
        - (640.0 / 3.0) * (math.pi**2)
        + (-15737765635.0 / 3048192.0 + (2255.0 / 12.0) * (math.pi**2)) * eta
        + (76055.0 / 1728.0) * (eta**2)
        - (127825.0 / 1296.0) * (eta**3)
    )
    a6 = a6_const - (6848.0 / 21.0) * np.log(4.0 * v)

    a7 = math.pi * (77096675.0 / 254016.0 + (378515.0 / 1512.0) * eta - (74045.0 / 756.0) * (eta**2))

    # Horner-like or direct summation for intrinsic phase
    poly = (
        a0
        + a1 * v
        + a2 * (v**2)
        + a3 * (v**3)
        + a4 * (v**4)
        + a5 * (v**5)
        + a6 * (v**6)
        + a7 * (v**7)
    )

    psi_int = (3.0 / (128.0 * eta * (v**5))) * poly
    psi_tf2 = -float(phi_c) - (math.pi / 4.0) + psi_int
    return psi_tf2


def taylorf2_fd_waveform(
    freqs: np.ndarray,
    m1_kg: float,
    m2_kg: float,
    distance_m: float,
    phi_c: float = 0.0,
    f_end: Optional[float] = None,
) -> np.ndarray:
    """Generate frequency-domain TaylorF2 strain template h~(f) with inclusive ISCO cutoff.

    Mathematical Formulation:
    -------------------------
    h~(f) = A * f^(-7/6) * exp(-i * Psi_{TF2}^+(f))    for f <= f_term
    h~(f) = 0                                          for f > f_term
    where f_term = min(f_end, f_ISCO(M)).

    Inclusive Bin-Boundary Convention:
    ----------------------------------
    - All discrete bins satisfying f_k <= f_term are RETAINED (nonzero).
    - All discrete bins satisfying f_k > f_term are set to IDENTICALLY ZERO.

    Parameters
    ----------
    freqs : np.ndarray
        1D array of discrete frequency bin centers in Hz. Can include DC (0 Hz).
    m1_kg : float
        Primary mass in kg.
    m2_kg : float
        Secondary mass in kg.
    distance_m : float
        Luminosity distance in meters.
    phi_c : float, default=0.0
        Coalescence phase in radians.
    f_end : float, optional
        Upper analysis band frequency in Hz. If None, cutoff is determined solely by f_ISCO.

    Returns
    -------
    np.ndarray
        1D complex128 array containing the frequency-domain template h~(f_k).
        Bins with f_k == 0 or f_k > f_term are identically 0.0 + 0.0j.
    """
    f_arr = np.asarray(freqs, dtype=np.float64)
    if f_arr.ndim != 1:
        raise ValueError("freqs must be a 1D array")
    if not np.all(np.isfinite(f_arr)):
        raise ValueError("freqs must contain only finite numbers")
    if np.any(f_arr < 0):
        raise ValueError("freqs cannot contain negative frequencies")

    M, eta, M_c = component_masses_to_chirp_mass(m1_kg, m2_kg)
    m_total_msun = M / M_SUN_SI
    f_isco = schwarzschild_isco_frequency(m_total_msun)
    f_term = effective_termination_frequency(f_end=f_end, f_isco=f_isco)

    # Pre-allocate output complex array with zeros
    h_tilde = np.zeros(f_arr.size, dtype=np.complex128)

    # Eligible bins: strictly positive and <= f_term (INCLUSIVE)
    eligible_mask = (f_arr > 0.0) & (f_arr <= float(f_term))

    if np.any(eligible_mask):
        f_eval = f_arr[eligible_mask]
        A = taylorf2_amplitude_prefactor(M_c, distance_m)
        amp = A * (f_eval ** (-7.0 / 6.0))
        psi = taylorf2_35pn_phase(f_eval, m1_kg, m2_kg, phi_c=phi_c)

        # Frozen Group C Fourier convention: exp(-i * Psi_{TF2}^+)
        h_tilde[eligible_mask] = amp * np.exp(-1.0j * psi)

    return h_tilde


def apply_coalescence_time_shift(
    waveform: np.ndarray, freqs: np.ndarray, t_c: float
) -> np.ndarray:
    """Apply time-translation translation to frequency-domain waveform.

    Fourier time-shift property:
        F{h(t - t_c)} = h~(f) * exp(-i * 2 * pi * f * t_c)

    Parameters
    ----------
    waveform : np.ndarray
        1D complex frequency-domain waveform h~(f).
    freqs : np.ndarray
        1D physical frequency array in Hz matching waveform shape.
    t_c : float
        Coalescence time shift in seconds.

    Returns
    -------
    np.ndarray
        Time-shifted complex waveform h~(f) * exp(-i * 2pi * f * t_c).
    """
    w = np.asarray(waveform, dtype=np.complex128)
    f = np.asarray(freqs, dtype=np.float64)
    if w.shape != f.shape:
        raise ValueError("waveform and freqs must have identical shapes")
    if t_c == 0.0:
        return w.copy()
    shift_kernel = np.exp(-2.0j * np.pi * f * float(t_c))
    return w * shift_kernel


def newtonian_chirp_time(f: float, chirp_mass_kg: float) -> float:
    """Compute Newtonian chirp time tau_N(f) in seconds.

    tau_N(f) = (5 / 256) * (G * M_c / c^3)^(-5/3) * (pi * f)^(-8/3)

    Strictly positive for all f > 0, M_c > 0.

    Parameters
    ----------
    f : float
        Frequency in Hz.
    chirp_mass_kg : float
        Chirp mass in kg.

    Returns
    -------
    float
        Newtonian time to coalescence in seconds.
    """
    if f <= 0 or chirp_mass_kg <= 0:
        raise ValueError("f and chirp_mass_kg must be positive")
    c = C_SI
    G = G_SI
    tau = (5.0 / 256.0) * (((G * chirp_mass_kg) / (c**3)) ** (-5.0 / 3.0)) * ((math.pi * f) ** (-8.0 / 3.0))
    return float(tau)


def taylorf2_35pn_chirp_time(
    f: float, m1_kg: float, m2_kg: float, delta_f: float = 1e-4
) -> float:
    """Evaluate full 3.5PN chirp time tau_{3.5PN}(f) = - (1 / (2 * pi)) * (dPsi_int / df).

    Evaluated via high-precision central difference derivative of the intrinsic phase.

    Governance Note:
    ----------------
    As established in Stage 1.3 Phase B Revision 9, positivity tau_{3.5PN}(f) > 0 is
    not an unconditional mathematical theorem for arbitrary mass parameters and
    frequencies; it must be empirically validated on the specific analysis domain.

    Parameters
    ----------
    f : float
        Frequency in Hz.
    m1_kg : float
        Primary mass in kg.
    m2_kg : float
        Secondary mass in kg.
    delta_f : float, default=1e-4
        Numerical step for central finite difference in Hz.

    Returns
    -------
    float
        3.5PN time to coalescence in seconds.
    """
    if f <= delta_f:
        raise ValueError("Frequency must be greater than numerical step delta_f")
    f_plus = f + delta_f
    f_minus = f - delta_f
    psi_plus = taylorf2_35pn_phase(np.array([f_plus]), m1_kg, m2_kg, phi_c=0.0)[0]
    psi_minus = taylorf2_35pn_phase(np.array([f_minus]), m1_kg, m2_kg, phi_c=0.0)[0]
    dpsi_df = (psi_plus - psi_minus) / (2.0 * delta_f)
    tau = - (1.0 / (2.0 * math.pi)) * dpsi_df
    return float(tau)
