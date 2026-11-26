"""Physics utilities and parameter transforms for GWLab.

This module provides pure, deterministic parameter transformations and waveform
generation helpers for the GWLab scientific workstation.

Governance:
- Pure functions only.
- Zero likelihood, noise, PSD, or sampler imports.
- Explicit SI units and physical parameter validation.
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple
import numpy as np

from gw_inference.constants import (
    C_ISCO_SI,
    M_SUN_SI,
    MPC_SI,
)
from gw_inference.grid import FrequencyGrid
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    effective_termination_frequency,
    newtonian_chirp_time,
    schwarzschild_isco_frequency,
    taylorf2_35pn_chirp_time,
    taylorf2_amplitude_prefactor,
    taylorf2_fd_waveform,
)

# Authoritative Stage 6C Run A Injection Benchmark Parameters
STAGE6C_INJECTION = {
    "mc_msun": 28.095555795460438,
    "eta": 0.24710059171597634,
    "tc_s": 2.0,
    "phic_rad": 0.0,
    "distance_mpc": 410.0,
    "f_lower": 20.0,
    "f_upper": 512.0,
    "sampling_rate": 2048.0,
    "duration": 4.0,
}


def mchirp_eta_to_component_masses(
    mc_msun: float, eta: float
) -> Tuple[float, float, float]:
    """Convert chirp mass and symmetric mass ratio to component masses (m1, m2, M).

    Mathematical Formulation:
        M = M_c * eta^(-3/5)
        m_{1,2} = (M / 2) * (1 +/- sqrt(max(0, 1 - 4 * eta)))
        with m1 >= m2.

    Parameters
    ----------
    mc_msun : float
        Chirp mass in solar masses (M_sun). Must be positive and finite.
    eta : float
        Symmetric mass ratio. Must be in (0, 0.25].

    Returns
    -------
    tuple of (m1_msun, m2_msun, m_total_msun)
        m1_msun : Primary component mass in M_sun (m1 >= m2).
        m2_msun : Secondary component mass in M_sun.
        m_total_msun : Total binary mass in M_sun (m1 + m2).

    Raises
    ------
    ValueError
        If mc_msun <= 0, non-finite, or eta not in (0, 0.25].
    """
    if not isinstance(mc_msun, (int, float, np.floating)) or not math.isfinite(mc_msun) or mc_msun <= 0:
        raise ValueError(f"Chirp mass must be a positive finite float, got {mc_msun}")
    if not isinstance(eta, (int, float, np.floating)) or not math.isfinite(eta) or eta <= 0 or eta > 0.25:
        raise ValueError(f"Symmetric mass ratio eta must be in (0, 0.25], got {eta}")

    mc = float(mc_msun)
    sym_eta = float(eta)

    # Total binary mass: M = M_c * eta^(-3/5)
    total_m = mc * (sym_eta ** (-0.6))

    # Discriminant radicand with strict numerical guard at equal-mass boundary
    radicand = max(0.0, 1.0 - 4.0 * sym_eta)
    disc = math.sqrt(radicand)

    m1 = 0.5 * total_m * (1.0 + disc)
    m2 = 0.5 * total_m * (1.0 - disc)

    return float(m1), float(m2), float(total_m)


def component_masses_to_mchirp_eta(
    m1_msun: float, m2_msun: float
) -> Tuple[float, float, float]:
    """Convert component masses to (M_c, eta, M).

    Parameters
    ----------
    m1_msun : float
        Primary mass in M_sun. Must be positive and finite.
    m2_msun : float
        Secondary mass in M_sun. Must be positive and finite.

    Returns
    -------
    tuple of (mc_msun, eta, m_total_msun)
    """
    if not isinstance(m1_msun, (int, float, np.floating)) or not math.isfinite(m1_msun) or m1_msun <= 0:
        raise ValueError(f"m1_msun must be a positive finite float, got {m1_msun}")
    if not isinstance(m2_msun, (int, float, np.floating)) or not math.isfinite(m2_msun) or m2_msun <= 0:
        raise ValueError(f"m2_msun must be a positive finite float, got {m2_msun}")

    m1 = float(m1_msun)
    m2 = float(m2_msun)
    total_m = m1 + m2
    eta = min(0.25, (m1 * m2) / (total_m**2))
    mc = total_m * (eta**0.6)
    return float(mc), float(eta), float(total_m)


def generate_taylorf2_strain(
    mc_msun: float,
    eta: float,
    distance_mpc: float,
    phic_rad: float = 0.0,
    tc_s: float = 0.0,
    duration_s: float = 4.0,
    sampling_rate_hz: float = 2048.0,
    f_end_hz: float | None = None,
    pn_order: float = 3.5,
) -> Dict[str, Any]:
    """Generate pure, deterministic TaylorF2 frequency-domain strain and diagnostics.

    Parameters
    ----------
    mc_msun : float
        Chirp mass in solar masses.
    eta : float
        Symmetric mass ratio in [0.10, 0.25].
    distance_mpc : float
        Luminosity distance in Mpc.
    phic_rad : float, default=0.0
        Coalescence reference phase in radians.
    tc_s : float, default=0.0
        Coalescence time shift in seconds.
    duration_s : float, default=4.0
        Fourier observation segment duration in seconds.
    sampling_rate_hz : float, default=2048.0
        Discrete sampling frequency in Hz.
    f_end_hz : float, optional
        Upper analysis band frequency. If None, cutoff is solely f_ISCO.
    pn_order : float, default=3.5
        Post-Newtonian order.

    Returns
    -------
    dict
        Dictionary containing:
        - 'frequencies': 1D array of frequency bins in Hz.
        - 'h_tilde': 1D complex128 strain array.
        - 'amplitude': 1D float64 strain amplitude |h~(f)|.
        - 'unwrapped_phase': 1D float64 unwrapped phase in radians.
        - 'wrapped_phase': 1D float64 wrapped phase in [-pi, pi] radians.
        - 'm1_msun', 'm2_msun', 'm_total_msun': Component and total masses in M_sun.
        - 'q': Mass ratio m1 / m2 >= 1.
        - 'f_isco': Schwarzschild ISCO frequency in Hz.
        - 'f_term': Effective termination frequency in Hz.
        - 'amp_prefactor': SI amplitude prefactor A in s^(-1/6).
        - 'nonzero_mask': Boolean mask of active bins (f > 0 and f <= f_term).
    """
    m1_msun, m2_msun, m_total_msun = mchirp_eta_to_component_masses(mc_msun, eta)
    m1_kg = m1_msun * M_SUN_SI
    m2_kg = m2_msun * M_SUN_SI
    dist_m = float(distance_mpc) * MPC_SI

    grid = FrequencyGrid.from_duration_and_sampling_rate(
        duration=float(duration_s), sampling_rate=float(sampling_rate_hz)
    )
    freqs = grid.frequencies

    # Base TaylorF2 waveform at t_c = 0
    h_base = taylorf2_fd_waveform(
        freqs=freqs,
        m1_kg=m1_kg,
        m2_kg=m2_kg,
        distance_m=dist_m,
        phi_c=float(phic_rad),
        f_end=f_end_hz,
        pn_order=pn_order,
    )

    # Apply coalescence time translation if tc != 0
    if float(tc_s) != 0.0:
        h_tilde = apply_coalescence_time_shift(h_base, freqs, float(tc_s))
    else:
        h_tilde = h_base

    amplitude = np.abs(h_tilde)
    f_isco = schwarzschild_isco_frequency(m_total_msun)
    f_term = effective_termination_frequency(f_end=f_end_hz, f_isco=f_isco)

    nonzero_mask = (freqs > 0.0) & (freqs <= float(f_term)) & (amplitude > 0.0)

    # Compute phase diagnostics for active bins
    wrapped_phase = np.zeros_like(freqs)
    unwrapped_phase = np.zeros_like(freqs)

    if np.any(nonzero_mask):
        active_h = h_tilde[nonzero_mask]
        # Wrapped phase: angle in [-pi, pi]
        wrapped = np.angle(active_h)
        wrapped_phase[nonzero_mask] = wrapped
        # Unwrapped phase: continuous phase trajectory
        unwrapped = np.unwrap(wrapped)
        unwrapped_phase[nonzero_mask] = unwrapped

    q = m1_msun / m2_msun if m2_msun > 0 else 1.0
    amp_prefactor = taylorf2_amplitude_prefactor(mc_msun * M_SUN_SI, dist_m)

    return {
        "frequencies": freqs,
        "h_tilde": h_tilde,
        "amplitude": amplitude,
        "unwrapped_phase": unwrapped_phase,
        "wrapped_phase": wrapped_phase,
        "m1_msun": m1_msun,
        "m2_msun": m2_msun,
        "m_total_msun": m_total_msun,
        "q": q,
        "f_isco": f_isco,
        "f_term": f_term,
        "amp_prefactor": amp_prefactor,
        "nonzero_mask": nonzero_mask,
        "delta_f": grid.delta_f,
        "nyquist": grid.nyquist,
    }


def compute_chirp_time_series(
    freqs: np.ndarray,
    m1_msun: float,
    m2_msun: float,
    mc_msun: float,
    f_isco: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Newtonian and 3.5PN time to coalescence tau(f) = t_c - t(f).

    Mathematical Formulation:
        tau_N(f) = (5 / 256) * (G * M_c / c^3)^(-5/3) * (pi * f)^(-8/3)
        tau_{3.5PN}(f) = - (1 / (2 * pi)) * (dPsi_int / df)

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequencies in Hz.
    m1_msun, m2_msun, mc_msun : float
        Masses in solar masses.
    f_isco : float
        Cutoff frequency in Hz.

    Returns
    -------
    tuple of (valid_freqs, tau_newtonian, tau_35pn)
    """
    valid = (freqs >= 15.0) & (freqs <= float(f_isco))
    eval_f = freqs[valid]

    if eval_f.size == 0:
        return np.array([]), np.array([]), np.array([])

    mc_kg = float(mc_msun) * M_SUN_SI
    m1_kg = float(m1_msun) * M_SUN_SI
    m2_kg = float(m2_msun) * M_SUN_SI

    tau_n = np.array([newtonian_chirp_time(f, mc_kg) for f in eval_f])
    tau_pn = np.array([taylorf2_35pn_chirp_time(f, m1_kg, m2_kg) for f in eval_f])

    return eval_f, tau_n, tau_pn
