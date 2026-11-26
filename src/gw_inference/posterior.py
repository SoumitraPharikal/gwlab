"""Likelihood and posterior wrappers connecting physical GW parameters to sampler interfaces.

This module integrates the existing FrequencyDomainGaussianLikelihood (WP4)
with the TaylorF2 frequency-domain waveform engine (WP2) to provide a standardized,
sampler-compatible likelihood callable:
    log_likelihood_fn(theta) -> float

Mathematical Formulation:
-------------------------
For proposed parameter vector theta = (M_c, eta) where:
- M_c: Chirp mass in solar masses (M_sun)
- eta: Symmetric mass ratio in (0, 0.25]

1. Total detector-frame mass in solar masses:
   M_total_msun = M_c * eta^(-3/5)

2. Physical cutoff check:
   f_ISCO = C_ISCO / M_total_msun
   If f_ISCO <= f_start, the binary merges below the active analysis band.
   The wrapper returns -inf immediately.

3. Component masses in kilograms:
   M_total_kg = M_total_msun * M_SUN_SI
   Delta_m_kg = M_total_kg * sqrt(1.0 - 4.0 * eta)
   m1_kg = (M_total_kg + Delta_m_kg) / 2.0
   m2_kg = (M_total_kg - Delta_m_kg) / 2.0

4. Frequency-domain waveform template:
   h~(f_k) = taylorf2_fd_waveform(freqs, m1_kg, m2_kg, distance_m, phi_c, f_end)
   If t_c != 0.0:
       h~(f_k) = apply_coalescence_time_shift(h~, freqs, t_c)

5. Active-band Whittle Gaussian log-likelihood:
   ln L_active(theta) = ln L_norm - 0.5 * (r|r)_{K_+}
   where ln L_norm is the full active-band normalization precomputed by
   FrequencyDomainGaussianLikelihood, ensuring full consistency for Bayesian evidence.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np

from gw_inference.constants import C_ISCO_SI, M_SUN_SI, MPC_SI
from gw_inference.likelihood import FrequencyDomainGaussianLikelihood
from gw_inference.prior import (
    Uniform4DParameterPrior,
    UniformChirpMassEtaPrior,
)
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    taylorf2_fd_waveform,
)


class GravitationalWaveLikelihoodWrapper:
    """Thin wrapper binding FrequencyDomainGaussianLikelihood to physical parameters (M_c, eta).

    Evaluates the full active-band Whittle Gaussian log-likelihood for compact binary
    inspiral signals, retaining the complete normalization constant ln L_norm.

    Parameters
    ----------
    likelihood : FrequencyDomainGaussianLikelihood
        Precomputed active-band Gaussian Whittle likelihood instance from WP4.
    distance_m : float, default=410.0 * MPC_SI
        Luminosity distance in meters (default corresponds to 410 Mpc).
    phi_c : float, default=0.0
        Coalescence orbital phase in radians.
    t_c : float, default=0.0
        Coalescence arrival time shift in seconds.
    f_start : float, default=20.0
        Lower frequency analysis boundary in Hz. Used to verify f_ISCO > f_start.
    f_end : float, optional
        Upper analysis band frequency in Hz. If None, cutoff is determined by f_ISCO.
    prior : UniformChirpMassEtaPrior, optional
        Prior instance for evaluating joint log-posterior.
    """

    def __init__(
        self,
        likelihood: FrequencyDomainGaussianLikelihood,
        distance_m: float = 410.0 * MPC_SI,
        phi_c: float = 0.0,
        t_c: float = 0.0,
        f_start: float = 20.0,
        f_end: Optional[float] = None,
        prior: Optional[UniformChirpMassEtaPrior] = None,
    ) -> None:
        if not isinstance(likelihood, FrequencyDomainGaussianLikelihood):
            raise TypeError(
                f"likelihood must be a FrequencyDomainGaussianLikelihood instance, got {type(likelihood)}"
            )
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            raise ValueError(f"distance_m must be positive and finite, got {distance_m}")
        if not math.isfinite(phi_c):
            raise ValueError(f"phi_c must be finite, got {phi_c}")
        if not math.isfinite(t_c):
            raise ValueError(f"t_c must be finite, got {t_c}")
        if not math.isfinite(f_start) or f_start <= 0.0:
            raise ValueError(f"f_start must be positive and finite, got {f_start}")
        if f_end is not None and (not math.isfinite(f_end) or f_end <= f_start):
            raise ValueError(f"f_end must be > f_start ({f_start}), got {f_end}")

        self._likelihood = likelihood
        self._distance_m = float(distance_m)
        self._phi_c = float(phi_c)
        self._t_c = float(t_c)
        self._f_start = float(f_start)
        self._f_end = float(f_end) if f_end is not None else None
        self._prior = prior

        # Cache frequencies array from grid
        self._frequencies = likelihood.grid.frequencies

    @property
    def likelihood(self) -> FrequencyDomainGaussianLikelihood:
        """Return the underlying FrequencyDomainGaussianLikelihood instance."""
        return self._likelihood

    @property
    def distance_m(self) -> float:
        """Return fixed luminosity distance in meters."""
        return self._distance_m

    @property
    def phi_c(self) -> float:
        """Return fixed coalescence phase in radians."""
        return self._phi_c

    @property
    def t_c(self) -> float:
        """Return fixed coalescence time shift in seconds."""
        return self._t_c

    @property
    def f_start(self) -> float:
        """Return lower analysis frequency boundary in Hz."""
        return self._f_start

    @property
    def f_end(self) -> Optional[float]:
        """Return upper analysis frequency boundary in Hz."""
        return self._f_end

    @property
    def prior(self) -> Optional[UniformChirpMassEtaPrior]:
        """Return attached prior instance, if any."""
        return self._prior

    def evaluate_log_likelihood(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate active-band Gaussian Whittle log-likelihood for parameter vector (M_c, eta).

        Parameters
        ----------
        theta : array-like of shape (2,)
            Parameter vector [M_c, eta] with M_c in solar masses (M_sun) and
            eta in (0, 0.25].

        Returns
        -------
        float
            Log-likelihood ln L_active(theta) = ln L_norm - 0.5 * (r|r)_{K_+}.
            Returns -inf if parameters are non-physical or if f_ISCO <= f_start.
        """
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (2,):
            raise ValueError(f"theta must have shape (2,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            return -math.inf

        mc_msun = float(arr[0])
        eta = float(arr[1])

        # Physical boundary checks
        if mc_msun <= 0.0 or eta <= 0.0 or eta > 0.25:
            return -math.inf

        # Total mass and ISCO cutoff
        m_total_msun = mc_msun * (eta ** (-3.0 / 5.0))
        f_isco = C_ISCO_SI / m_total_msun

        if f_isco <= self._f_start:
            # Merges below analysis band; template has zero active support
            return -math.inf

        # Component masses in kg
        m_total_kg = m_total_msun * M_SUN_SI
        delta_m_kg = m_total_kg * math.sqrt(max(0.0, 1.0 - 4.0 * eta))
        m1_kg = (m_total_kg + delta_m_kg) / 2.0
        m2_kg = (m_total_kg - delta_m_kg) / 2.0

        # Synthesize TaylorF2 frequency-domain template
        try:
            h_tilde = taylorf2_fd_waveform(
                freqs=self._frequencies,
                m1_kg=m1_kg,
                m2_kg=m2_kg,
                distance_m=self._distance_m,
                phi_c=self._phi_c,
                f_end=self._f_end,
            )
            if self._t_c != 0.0:
                h_tilde = apply_coalescence_time_shift(h_tilde, self._frequencies, self._t_c)

            # Evaluate likelihood (pre-sliced active inner product)
            return self._likelihood.evaluate(h_tilde)
        except Exception:
            return -math.inf

    def evaluate_log_posterior(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate unnormalized log-posterior ln pi(theta) = ln L(theta) + ln p(theta).

        Requires an attached prior instance.

        Parameters
        ----------
        theta : array-like of shape (2,)
            Parameter vector [M_c, eta].

        Returns
        -------
        float
            Log-posterior density.
        """
        if self._prior is None:
            raise ValueError("No prior instance attached; cannot evaluate log_posterior")

        ln_p = self._prior.log_prior(theta)
        if not math.isfinite(ln_p):
            return -math.inf

        ln_l = self.evaluate_log_likelihood(theta)
        if not math.isfinite(ln_l):
            return -math.inf

        return ln_l + ln_p

    def __call__(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        """Callable interface forwarding directly to evaluate_log_likelihood."""
        return self.evaluate_log_likelihood(theta)


class GravitationalWaveLikelihood4DWrapper:
    """Wrapper binding FrequencyDomainGaussianLikelihood to 4D physical parameters (M_c, eta, t_c, phi_c).

    Evaluates the full active-band Whittle Gaussian log-likelihood for compact binary
    inspiral signals, retaining the complete normalization constant ln L_norm.

    Parameters
    ----------
    likelihood : FrequencyDomainGaussianLikelihood
        Precomputed active-band Gaussian Whittle likelihood instance from WP4.
    distance_m : float, default=410.0 * MPC_SI
        Luminosity distance in meters (default corresponds to 410 Mpc).
    f_start : float, default=20.0
        Lower frequency analysis boundary in Hz. Used to verify f_ISCO > f_start.
    f_end : float, optional
        Upper analysis band frequency in Hz. If None, cutoff is determined by f_ISCO.
    prior : Uniform4DParameterPrior, optional
        Prior instance for evaluating joint 4D log-posterior.
    """

    def __init__(
        self,
        likelihood: FrequencyDomainGaussianLikelihood,
        distance_m: float = 410.0 * MPC_SI,
        f_start: float = 20.0,
        f_end: Optional[float] = None,
        prior: Optional[Uniform4DParameterPrior] = None,
        pn_order: float = 3.5,
    ) -> None:
        if not isinstance(likelihood, FrequencyDomainGaussianLikelihood):
            raise TypeError(
                f"likelihood must be a FrequencyDomainGaussianLikelihood instance, got {type(likelihood)}"
            )
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            raise ValueError(f"distance_m must be positive and finite, got {distance_m}")
        if not math.isfinite(f_start) or f_start <= 0.0:
            raise ValueError(f"f_start must be positive and finite, got {f_start}")
        if f_end is not None and (not math.isfinite(f_end) or f_end <= f_start):
            raise ValueError(f"f_end must be > f_start ({f_start}), got {f_end}")
        if prior is not None and not isinstance(prior, Uniform4DParameterPrior):
            raise TypeError(
                f"prior must be a Uniform4DParameterPrior instance, got {type(prior)}"
            )

        self._likelihood = likelihood
        self._distance_m = float(distance_m)
        self._f_start = float(f_start)
        self._f_end = float(f_end) if f_end is not None else None
        self._prior = prior
        self._pn_order = float(pn_order)

        # Cache frequencies array from grid
        self._frequencies = likelihood.grid.frequencies

    @property
    def likelihood(self) -> FrequencyDomainGaussianLikelihood:
        """Return the underlying FrequencyDomainGaussianLikelihood instance."""
        return self._likelihood

    @property
    def distance_m(self) -> float:
        """Return fixed luminosity distance in meters."""
        return self._distance_m

    @property
    def f_start(self) -> float:
        """Return lower analysis frequency boundary in Hz."""
        return self._f_start

    @property
    def f_end(self) -> Optional[float]:
        """Return upper analysis frequency boundary in Hz."""
        return self._f_end

    @property
    def prior(self) -> Optional[Uniform4DParameterPrior]:
        """Return attached 4D prior instance, if any."""
        return self._prior

    def evaluate_log_likelihood(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate active-band Gaussian Whittle log-likelihood for parameter vector (M_c, eta, t_c, phi_c).

        Parameters
        ----------
        theta : array-like of shape (4,)
            Parameter vector [M_c, eta, t_c, phi_c] with M_c in solar masses (M_sun),
            eta in (0, 0.25], t_c in seconds, and phi_c in radians.

        Returns
        -------
        float
            Log-likelihood ln L_active(theta) = ln L_norm - 0.5 * (r|r)_{K_+}.
            Returns -inf if parameters are non-physical or if f_ISCO <= f_start.
        """
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (4,):
            raise ValueError(f"theta must have shape (4,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            return -math.inf

        mc_msun = float(arr[0])
        eta = float(arr[1])
        tc = float(arr[2])
        phic = float(arr[3])

        # Physical boundary checks
        if mc_msun <= 0.0 or eta <= 0.0 or eta > 0.25:
            return -math.inf

        # Total mass and ISCO cutoff
        m_total_msun = mc_msun * (eta ** (-3.0 / 5.0))
        f_isco = C_ISCO_SI / m_total_msun

        if f_isco <= self._f_start:
            # Merges below analysis band; template has zero active support
            return -math.inf

        # Component masses in kg
        m_total_kg = m_total_msun * M_SUN_SI
        delta_m_kg = m_total_kg * math.sqrt(max(0.0, 1.0 - 4.0 * eta))
        m1_kg = (m_total_kg + delta_m_kg) / 2.0
        m2_kg = (m_total_kg - delta_m_kg) / 2.0

        # Synthesize TaylorF2 frequency-domain template with exact shifts
        try:
            h_tilde = taylorf2_fd_waveform(
                freqs=self._frequencies,
                m1_kg=m1_kg,
                m2_kg=m2_kg,
                distance_m=self._distance_m,
                phi_c=phic,
                f_end=self._f_end,
                pn_order=self._pn_order,
            )
            if tc != 0.0:
                h_tilde = apply_coalescence_time_shift(h_tilde, self._frequencies, tc)

            # Evaluate likelihood (pre-sliced active inner product)
            return self._likelihood.evaluate(h_tilde)
        except Exception:
            return -math.inf

    def evaluate_log_posterior(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate unnormalized 4D log-posterior ln pi(theta) = ln L(theta) + ln p(theta).

        Requires an attached prior instance.

        Parameters
        ----------
        theta : array-like of shape (4,)
            Parameter vector [M_c, eta, t_c, phi_c].

        Returns
        -------
        float
            Log-posterior density.
        """
        if self._prior is None:
            raise ValueError("No prior instance attached; cannot evaluate log_posterior")

        ln_p = self._prior.log_prior(theta)
        if not math.isfinite(ln_p):
            return -math.inf

        ln_l = self.evaluate_log_likelihood(theta)
        if not math.isfinite(ln_l):
            return -math.inf

        return ln_l + ln_p

    def __call__(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        """Callable interface forwarding directly to evaluate_log_likelihood."""
        return self.evaluate_log_likelihood(theta)
