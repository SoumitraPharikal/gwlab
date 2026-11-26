"""Prior distributions and coordinate transformations for Bayesian parameter estimation.

This module implements the approved baseline prior distributions and analytic unit-cube
transformations for Work Package 6 (WP6) Stage 1:
- Primary 2D parameter vector: theta = (chirp_mass, symmetric_mass_ratio) = (M_c, eta)
- Uniform prior measure directly over the rectangular support:
    M_c in [20.0, 40.0] M_sun (solar masses)
    eta in [0.10, 0.25] (dimensionless symmetric mass ratio)
- Normalized prior probability density:
    pi(M_c, eta) = 1 / ((40.0 - 20.0) * (0.25 - 0.10)) = 1 / 3.0 (M_sun)^(-1)
- Analytic unit-hypercube mapping u in [0, 1]^2 -> theta in [20, 40] x [0.10, 0.25]:
    M_c = 20.0 + 20.0 * u_0
    eta = 0.10 + 0.15 * u_1

Important Scientific and Mathematical Notes:
---------------------------------------------
1. Coordinate Measure: This prior is strictly uniform with respect to the (M_c, eta)
   coordinate volume element dM_c deta. It does NOT introduce any component-mass
   Jacobian factor |d(m1, m2) / d(M_c, eta)|.
2. Non-Informativeness Qualification: In Bayesian inference, a flat prior in one
   coordinate chart represents a non-flat, structured prior in any non-linearly
   transformed chart (e.g. component masses). Therefore, this prior is NOT universally
   or objectively uninformative; it represents a deliberate baseline choice for a
   controlled, reproducible scientific injection/recovery experiment.
3. Dimensionality and Units:
   - M_c: Solar masses (M_sun). Note: when passed to waveform generators (e.g. TaylorF2),
     masses must be converted to kilograms via M_SUN_SI.
   - eta: Dimensionless, with physical boundary eta <= 0.25 (equal masses).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np


def prior_transform_2d(u: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Analytic unit-square transformation to the approved 2D baseline coordinates (M_c, eta).

    Maps points u = (u_0, u_1) from the unit hypercube [0, 1]^2 to the parameter space
    theta = (M_c, eta) in [20.0, 40.0] M_sun x [0.10, 0.25]:
        M_c = 20.0 + 20.0 * u_0
        eta = 0.10 + 0.15 * u_1

    Parameters
    ----------
    u : array-like
        1D array of shape (2,) or 2D array of shape (N, 2) containing unit-cube
        coordinates u_i in [0, 1].

    Returns
    -------
    np.ndarray
        Transformed physical parameters theta of matching leading shape.
        For 1D input of shape (2,), returns 1D float64 array [M_c, eta].
        For 2D input of shape (N, 2), returns 2D float64 array of shape (N, 2).

    Raises
    ------
    TypeError
        If input cannot be converted to a NumPy float array.
    ValueError
        If u has invalid dimensionality (not 1D length 2 or 2D shape (N, 2)),
        contains non-finite values (NaN, +inf, -inf), or contains values outside [0, 1].
    """
    try:
        arr = np.asarray(u, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"u must be convertible to a float numpy array, got {type(u)}") from e

    # Dimension validation
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"1D input u must have exactly 2 elements, got shape {arr.shape}")
        is_1d = True
    elif arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"2D input u must have shape (N, 2), got shape {arr.shape}")
        is_1d = False
    else:
        raise ValueError(f"u must be 1D or 2D with last dimension 2, got ndim={arr.ndim} and shape={arr.shape}")

    # Finite value validation
    if not np.all(np.isfinite(arr)):
        raise ValueError("u contains non-finite values (NaN or Inf)")

    # Unit-cube bounds validation [0, 1]
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        raise ValueError(
            f"All elements of u must lie in the unit interval [0, 1]; found min={min_val}, max={max_val}"
        )

    # Analytic transformation:
    # M_c in [20.0, 40.0] M_sun
    # eta in [0.10, 0.25]
    if is_1d:
        mc = 20.0 + 20.0 * float(arr[0])
        eta = 0.10 + 0.15 * float(arr[1])
        return np.array([mc, eta], dtype=np.float64)
    else:
        out = np.empty_like(arr)
        out[:, 0] = 20.0 + 20.0 * arr[:, 0]
        out[:, 1] = 0.10 + 0.15 * arr[:, 1]
        return out


class UniformChirpMassEtaPrior:
    """Uniform prior density and unit-cube transform for (M_c, eta) parameter space.

    Encapsulates the approved baseline prior:
        M_c in [mc_min, mc_max] M_sun (default [20.0, 40.0])
        eta in [eta_min, eta_max] (default [0.10, 0.25])

    Parameters
    ----------
    mc_min : float, default=20.0
        Minimum chirp mass in solar masses (M_sun). Must be > 0.
    mc_max : float, default=40.0
        Maximum chirp mass in solar masses (M_sun). Must be > mc_min.
    eta_min : float, default=0.10
        Minimum symmetric mass ratio. Must be > 0.
    eta_max : float, default=0.25
        Maximum symmetric mass ratio. Must be > eta_min and <= 0.25.
    """

    PARAM_NAMES: Tuple[str, str] = ("chirp_mass_msun", "symmetric_mass_ratio")
    PARAM_UNITS: Tuple[str, str] = ("M_sun", "dimensionless")

    def __init__(
        self,
        mc_min: float = 20.0,
        mc_max: float = 40.0,
        eta_min: float = 0.10,
        eta_max: float = 0.25,
    ) -> None:
        if not math.isfinite(mc_min) or mc_min <= 0.0:
            raise ValueError(f"mc_min must be positive and finite, got {mc_min}")
        if not math.isfinite(mc_max) or mc_max <= mc_min:
            raise ValueError(f"mc_max must be greater than mc_min ({mc_min}), got {mc_max}")
        if not math.isfinite(eta_min) or eta_min <= 0.0:
            raise ValueError(f"eta_min must be positive and finite, got {eta_min}")
        if not math.isfinite(eta_max) or eta_max <= eta_min or eta_max > 0.25:
            raise ValueError(f"eta_max must satisfy eta_min < eta_max <= 0.25, got {eta_max}")

        self._mc_min = float(mc_min)
        self._mc_max = float(mc_max)
        self._eta_min = float(eta_min)
        self._eta_max = float(eta_max)

        # Coordinate volume: V = Delta M_c * Delta eta
        self._volume = (self._mc_max - self._mc_min) * (self._eta_max - self._eta_min)
        self._density = 1.0 / self._volume
        self._log_density = -math.log(self._volume)

    @property
    def mc_min(self) -> float:
        """Return minimum chirp mass in solar masses."""
        return self._mc_min

    @property
    def mc_max(self) -> float:
        """Return maximum chirp mass in solar masses."""
        return self._mc_max

    @property
    def eta_min(self) -> float:
        """Return minimum symmetric mass ratio."""
        return self._eta_min

    @property
    def eta_max(self) -> float:
        """Return maximum symmetric mass ratio."""
        return self._eta_max

    @property
    def volume(self) -> float:
        """Return coordinate hypervolume V = Delta M_c * Delta eta in M_sun."""
        return self._volume

    @property
    def normalized_density(self) -> float:
        """Return constant normalized prior probability density pi(theta) inside support."""
        return self._density

    @property
    def log_density(self) -> float:
        """Return log prior probability density ln pi(theta) inside support."""
        return self._log_density

    @property
    def param_names(self) -> Tuple[str, str]:
        """Return ordered parameter names."""
        return self.PARAM_NAMES

    @property
    def param_units(self) -> Tuple[str, str]:
        """Return ordered parameter physical units."""
        return self.PARAM_UNITS

    def is_within_bounds(self, theta: Union[Sequence[float], np.ndarray]) -> bool:
        """Check whether proposed parameter vector lies inside the prior support.

        Parameters
        ----------
        theta : array-like of shape (2,)
            Parameter vector [M_c, eta].

        Returns
        -------
        bool
            True if within [mc_min, mc_max] x [eta_min, eta_max], False otherwise.
        """
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (2,) or not np.all(np.isfinite(arr)):
            return False
        mc, eta = float(arr[0]), float(arr[1])
        return (self._mc_min <= mc <= self._mc_max) and (self._eta_min <= eta <= self._eta_max)

    def log_prior(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate log-prior probability density ln pi(theta).

        Returns -ln(volume) inside support, and -inf outside.

        Parameters
        ----------
        theta : array-like of shape (2,)
            Parameter vector [M_c, eta].

        Returns
        -------
        float
            Log-prior density.
        """
        if self.is_within_bounds(theta):
            return self._log_density
        return -math.inf

    def prior_transform(self, u: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Transform unit hypercube coordinates u in [0, 1]^2 to (M_c, eta).

        Parameters
        ----------
        u : array-like of shape (2,) or (N, 2)
            Unit-cube coordinates.

        Returns
        -------
        np.ndarray
            Transformed parameter coordinates [M_c, eta].
        """
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != 2:
                raise ValueError(f"u must have 2 elements, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("u contains non-finite values (NaN or Inf)")
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError(f"u must lie in [0, 1]^2, got {arr}")
            mc = self._mc_min + float(arr[0]) * (self._mc_max - self._mc_min)
            eta = self._eta_min + float(arr[1]) * (self._eta_max - self._eta_min)
            return np.array([mc, eta], dtype=np.float64)
        elif arr.ndim == 2:
            if arr.shape[1] != 2:
                raise ValueError(f"2D u must have shape (N, 2), got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("u contains non-finite values (NaN or Inf)")
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError("All elements of u must lie in [0, 1]")
            out = np.empty_like(arr)
            out[:, 0] = self._mc_min + arr[:, 0] * (self._mc_max - self._mc_min)
            out[:, 1] = self._eta_min + arr[:, 1] * (self._eta_max - self._eta_min)
            return out
        else:
            raise ValueError(f"u must be 1D or 2D, got ndim={arr.ndim}")


def prior_transform_4d(u: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Analytic unit-hypercube transformation to the approved 4D baseline coordinates (M_c, eta, t_c, phi_c).

    Maps points u = (u_0, u_1, u_2, u_3) from the unit hypercube [0, 1]^4 to the parameter space
    theta = (M_c, eta, t_c, phi_c) in [20.0, 40.0] M_sun x [0.10, 0.25] x [1.90, 2.10] s x [0, 2*pi) rad:
        M_c = 20.0 + 20.0 * u_0
        eta = 0.10 + 0.15 * u_1
        t_c = 1.90 + 0.20 * u_2
        phi_c = 2 * pi * u_3

    Parameters
    ----------
    u : array-like
        1D array of shape (4,) or 2D array of shape (N, 4) containing unit-cube
        coordinates u_i in [0, 1].

    Returns
    -------
    np.ndarray
        Transformed physical parameters theta of matching leading shape.
        For 1D input of shape (4,), returns 1D float64 array [M_c, eta, t_c, phi_c].
        For 2D input of shape (N, 4), returns 2D float64 array of shape (N, 4).

    Raises
    ------
    TypeError
        If input cannot be converted to a NumPy float array.
    ValueError
        If u has invalid dimensionality (not 1D length 4 or 2D shape (N, 4)),
        contains non-finite values (NaN, +inf, -inf), or contains values outside [0, 1].
    """
    try:
        arr = np.asarray(u, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"u must be convertible to a float numpy array, got {type(u)}") from e

    # Dimension validation
    if arr.ndim == 1:
        if arr.size != 4:
            raise ValueError(f"1D input u must have exactly 4 elements, got shape {arr.shape}")
        is_1d = True
    elif arr.ndim == 2:
        if arr.shape[1] != 4:
            raise ValueError(f"2D input u must have shape (N, 4), got shape {arr.shape}")
        is_1d = False
    else:
        raise ValueError(f"u must be 1D or 2D with last dimension 4, got ndim={arr.ndim} and shape={arr.shape}")

    # Finite value validation
    if not np.all(np.isfinite(arr)):
        raise ValueError("u contains non-finite values (NaN or Inf)")

    # Unit-cube bounds validation: non-periodic dimensions [0, 1, 2] must lie in [0, 1].
    # Periodic dimension 3 (phase) is allowed to exceed [0, 1] during sampler boundary wrapping
    # (e.g. dynesty periodic proposals) and is mapped smoothly modulo 1.
    if is_1d:
        if np.any(arr[:3] < 0.0) or np.any(arr[:3] > 1.0):
            min_val = float(np.min(arr[:3]))
            max_val = float(np.max(arr[:3]))
            raise ValueError(
                f"Non-periodic elements u_0, u_1, u_2 must lie in the unit interval [0, 1]; found min={min_val}, max={max_val}"
            )
        mc = 20.0 + 20.0 * float(arr[0])
        eta = 0.10 + 0.15 * float(arr[1])
        tc = 1.90 + 0.20 * float(arr[2])
        u3 = float(arr[3])
        phic = 2.0 * math.pi if u3 == 1.0 else (u3 % 1.0) * (2.0 * math.pi)
        return np.array([mc, eta, tc, phic], dtype=np.float64)
    else:
        if np.any(arr[:, :3] < 0.0) or np.any(arr[:, :3] > 1.0):
            min_val = float(np.min(arr[:, :3]))
            max_val = float(np.max(arr[:, :3]))
            raise ValueError(
                f"Non-periodic elements u_0, u_1, u_2 must lie in the unit interval [0, 1]; found min={min_val}, max={max_val}"
            )
        out = np.empty_like(arr)
        out[:, 0] = 20.0 + 20.0 * arr[:, 0]
        out[:, 1] = 0.10 + 0.15 * arr[:, 1]
        out[:, 2] = 1.90 + 0.20 * arr[:, 2]
        u3 = arr[:, 3]
        out[:, 3] = np.where(u3 == 1.0, 2.0 * math.pi, (u3 % 1.0) * (2.0 * math.pi))
        return out


class Uniform4DParameterPrior:
    """Uniform prior density and unit-cube transform for 4D parameter space (M_c, eta, t_c, phi_c).

    Encapsulates the approved 4D baseline prior:
        M_c in [mc_min, mc_max] M_sun (default [20.0, 40.0])
        eta in [eta_min, eta_max] (default [0.10, 0.25])
        t_c in [tc_min, tc_max] s (default [1.90, 2.10])
        phi_c in [phic_min, phic_max] rad (default [0.0, 2*pi])

    Prior volume:
        V_Pi = Delta M_c * Delta eta * Delta t_c * Delta phi_c
        Default: 20.0 * 0.15 * 0.20 * (2 * pi) = 1.2 * pi ~= 3.769911 M_sun * s * rad
        p(theta) = 1 / (1.2 * pi) ~= 0.265258
        ln p(theta) = -ln(1.2 * pi) ~= -1.327053
    """

    PARAM_NAMES: Tuple[str, str, str, str] = (
        "chirp_mass_msun",
        "symmetric_mass_ratio",
        "coalescence_time_s",
        "coalescence_phase_rad",
    )
    PARAM_UNITS: Tuple[str, str, str, str] = ("M_sun", "dimensionless", "s", "rad")

    def __init__(
        self,
        mc_min: float = 20.0,
        mc_max: float = 40.0,
        eta_min: float = 0.10,
        eta_max: float = 0.25,
        tc_min: float = 1.90,
        tc_max: float = 2.10,
        phic_min: float = 0.0,
        phic_max: float = 2.0 * math.pi,
    ) -> None:
        if not math.isfinite(mc_min) or mc_min <= 0.0:
            raise ValueError(f"mc_min must be positive and finite, got {mc_min}")
        if not math.isfinite(mc_max) or mc_max <= mc_min:
            raise ValueError(f"mc_max must be greater than mc_min ({mc_min}), got {mc_max}")
        if not math.isfinite(eta_min) or eta_min <= 0.0:
            raise ValueError(f"eta_min must be positive and finite, got {eta_min}")
        if not math.isfinite(eta_max) or eta_max <= eta_min or eta_max > 0.25:
            raise ValueError(f"eta_max must satisfy eta_min < eta_max <= 0.25, got {eta_max}")
        if not math.isfinite(tc_min):
            raise ValueError(f"tc_min must be finite, got {tc_min}")
        if not math.isfinite(tc_max) or tc_max <= tc_min:
            raise ValueError(f"tc_max must be greater than tc_min ({tc_min}), got {tc_max}")
        if not math.isfinite(phic_min):
            raise ValueError(f"phic_min must be finite, got {phic_min}")
        if not math.isfinite(phic_max) or phic_max <= phic_min:
            raise ValueError(f"phic_max must be greater than phic_min ({phic_min}), got {phic_max}")

        self._mc_min = float(mc_min)
        self._mc_max = float(mc_max)
        self._eta_min = float(eta_min)
        self._eta_max = float(eta_max)
        self._tc_min = float(tc_min)
        self._tc_max = float(tc_max)
        self._phic_min = float(phic_min)
        self._phic_max = float(phic_max)

        # Coordinate hypervolume V = Delta M_c * Delta eta * Delta t_c * Delta phi_c
        self._volume = (
            (self._mc_max - self._mc_min)
            * (self._eta_max - self._eta_min)
            * (self._tc_max - self._tc_min)
            * (self._phic_max - self._phic_min)
        )
        self._density = 1.0 / self._volume
        self._log_density = -math.log(self._volume)

    @property
    def mc_min(self) -> float:
        """Return minimum chirp mass in solar masses."""
        return self._mc_min

    @property
    def mc_max(self) -> float:
        """Return maximum chirp mass in solar masses."""
        return self._mc_max

    @property
    def eta_min(self) -> float:
        """Return minimum symmetric mass ratio."""
        return self._eta_min

    @property
    def eta_max(self) -> float:
        """Return maximum symmetric mass ratio."""
        return self._eta_max

    @property
    def tc_min(self) -> float:
        """Return minimum coalescence time in seconds."""
        return self._tc_min

    @property
    def tc_max(self) -> float:
        """Return maximum coalescence time in seconds."""
        return self._tc_max

    @property
    def phic_min(self) -> float:
        """Return minimum coalescence phase in radians."""
        return self._phic_min

    @property
    def phic_max(self) -> float:
        """Return maximum coalescence phase in radians."""
        return self._phic_max

    @property
    def volume(self) -> float:
        """Return coordinate hypervolume V in M_sun * s * rad."""
        return self._volume

    @property
    def normalized_density(self) -> float:
        """Return constant normalized prior probability density pi(theta) inside support."""
        return self._density

    @property
    def log_density(self) -> float:
        """Return log prior probability density ln pi(theta) inside support."""
        return self._log_density

    @property
    def param_names(self) -> Tuple[str, str, str, str]:
        """Return ordered parameter names."""
        return self.PARAM_NAMES

    @property
    def param_units(self) -> Tuple[str, str, str, str]:
        """Return ordered parameter physical units."""
        return self.PARAM_UNITS

    def is_within_bounds(self, theta: Union[Sequence[float], np.ndarray]) -> bool:
        """Check whether proposed parameter vector lies inside the 4D prior support.

        Parameters
        ----------
        theta : array-like of shape (4,)
            Parameter vector [M_c, eta, t_c, phi_c].

        Returns
        -------
        bool
            True if within bounds, False otherwise.
        """
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (4,) or not np.all(np.isfinite(arr)):
            return False
        mc, eta, tc, phic = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        return (
            (self._mc_min <= mc <= self._mc_max)
            and (self._eta_min <= eta <= self._eta_max)
            and (self._tc_min <= tc <= self._tc_max)
            and (self._phic_min <= phic <= self._phic_max)
        )

    def log_prior(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        """Evaluate log-prior probability density ln pi(theta).

        Returns -ln(volume) inside support, and -inf outside.

        Parameters
        ----------
        theta : array-like of shape (4,)
            Parameter vector [M_c, eta, t_c, phi_c].

        Returns
        -------
        float
            Log-prior density.
        """
        if self.is_within_bounds(theta):
            return self._log_density
        return -math.inf

    def prior_transform(self, u: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Transform unit hypercube coordinates u in [0, 1]^4 to (M_c, eta, t_c, phi_c).

        Parameters
        ----------
        u : array-like of shape (4,) or (N, 4)
            Unit-cube coordinates.

        Returns
        -------
        np.ndarray
            Transformed parameter coordinates [M_c, eta, t_c, phi_c].
        """
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != 4:
                raise ValueError(f"u must have 4 elements, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("u contains non-finite values (NaN or Inf)")
            if np.any(arr[:3] < 0.0) or np.any(arr[:3] > 1.0):
                raise ValueError(f"Non-periodic elements u_0, u_1, u_2 must lie in [0, 1], got {arr[:3]}")
            mc = self._mc_min + float(arr[0]) * (self._mc_max - self._mc_min)
            eta = self._eta_min + float(arr[1]) * (self._eta_max - self._eta_min)
            tc = self._tc_min + float(arr[2]) * (self._tc_max - self._tc_min)
            u3 = float(arr[3])
            phic = self._phic_max if u3 == 1.0 else self._phic_min + (u3 % 1.0) * (self._phic_max - self._phic_min)
            return np.array([mc, eta, tc, phic], dtype=np.float64)
        elif arr.ndim == 2:
            if arr.shape[1] != 4:
                raise ValueError(f"2D u must have shape (N, 4), got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("u contains non-finite values (NaN or Inf)")
            if np.any(arr[:, :3] < 0.0) or np.any(arr[:, :3] > 1.0):
                raise ValueError("Non-periodic elements u_0, u_1, u_2 must lie in [0, 1]")
            out = np.empty_like(arr)
            out[:, 0] = self._mc_min + arr[:, 0] * (self._mc_max - self._mc_min)
            out[:, 1] = self._eta_min + arr[:, 1] * (self._eta_max - self._eta_min)
            out[:, 2] = self._tc_min + arr[:, 2] * (self._tc_max - self._tc_min)
            u3 = arr[:, 3]
            out[:, 3] = np.where(
                u3 == 1.0,
                self._phic_max,
                self._phic_min + (u3 % 1.0) * (self._phic_max - self._phic_min),
            )
            return out
        else:
            raise ValueError(f"u must be 1D or 2D, got ndim={arr.ndim}")


# ==============================================================================
# Work Package 6 (WP6) Stage 6B: Alternative Mass Parameterization (M_total, eta)
# ==============================================================================


def total_mass_eta_jacobian(
    eta: Union[float, Sequence[float], np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Evaluate the absolute Jacobian determinant \|d(M_c, eta) / d(M_total, eta)\| = eta^(3/5).

    Mathematical derivation:
    ------------------------
    With forward map M_c = M_total * eta^(3/5), eta = eta:
        d(M_c) / d(M_total) = eta^(3/5)
        d(M_c) / d(eta)     = (3/5) * M_total * eta^(-2/5)
        d(eta) / d(M_total) = 0
        d(eta) / d(eta)     = 1

    The Jacobian determinant is:
        det J = eta^(3/5) * 1 - 0 = eta^(3/5).
    For physical symmetric mass ratios eta > 0, |det J| = eta^(3/5).

    Parameters
    ----------
    eta : float or array-like
        Symmetric mass ratio value(s). Must be positive and finite.

    Returns
    -------
    float or np.ndarray
        Absolute Jacobian determinant eta^(3/5).
    """
    arr = np.asarray(eta, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("eta contains non-finite values (NaN or Inf)")
    if np.any(arr <= 0.0):
        raise ValueError("eta must be strictly positive to evaluate Jacobian")

    j = arr ** 0.6
    return float(j) if arr.ndim == 0 else j


def mchirp_eta_to_total_mass_eta(
    theta: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    r"""Convert parameter vector from (M_c, eta, ...) to (M_total, eta, ...).

    Coordinate mapping:
        M_total = M_c * eta^(-3/5)
        eta = eta
        Extrinsic parameters (t_c, phi_c) are preserved identically if present.

    Parameters
    ----------
    theta : array-like
        1D array of shape (2,) or (4,), or 2D array of shape (N, 2) or (N, 4)
        containing [M_c, eta, ...] with M_c in solar masses (M_sun).

    Returns
    -------
    np.ndarray
        Transformed parameter array [M_total, eta, ...] of matching shape and float64 dtype.

    Raises
    ------
    TypeError
        If input cannot be converted to a NumPy float array.
    ValueError
        If theta has invalid dimensionality, non-finite values, or non-positive eta.
    """
    try:
        arr = np.asarray(theta, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"theta must be convertible to a float numpy array, got {type(theta)}") from e

    if not np.all(np.isfinite(arr)):
        raise ValueError("theta contains non-finite values (NaN or Inf)")

    if arr.ndim == 1:
        if arr.size not in (2, 4):
            raise ValueError(f"1D theta must have 2 or 4 elements, got shape {arr.shape}")
        mc = float(arr[0])
        eta = float(arr[1])
        if eta <= 0.0:
            raise ValueError(f"eta must be strictly positive, got {eta}")
        mtotal = mc * (eta ** -0.6)
        if arr.size == 2:
            return np.array([mtotal, eta], dtype=np.float64)
        return np.array([mtotal, eta, float(arr[2]), float(arr[3])], dtype=np.float64)

    elif arr.ndim == 2:
        if arr.shape[1] not in (2, 4):
            raise ValueError(f"2D theta must have shape (N, 2) or (N, 4), got shape {arr.shape}")
        eta = arr[:, 1]
        if np.any(eta <= 0.0):
            raise ValueError("All eta elements must be strictly positive")
        out = arr.copy()
        out[:, 0] = arr[:, 0] * (eta ** -0.6)
        return out
    else:
        raise ValueError(f"theta must be 1D or 2D, got ndim={arr.ndim}")


def total_mass_eta_to_mchirp_eta(
    theta: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    r"""Convert parameter vector from (M_total, eta, ...) to (M_c, eta, ...).

    Inverse coordinate mapping:
        M_c = M_total * eta^(3/5)
        eta = eta
        Extrinsic parameters (t_c, phi_c) are preserved identically if present.

    Parameters
    ----------
    theta : array-like
        1D array of shape (2,) or (4,), or 2D array of shape (N, 2) or (N, 4)
        containing [M_total, eta, ...] with M_total in solar masses (M_sun).

    Returns
    -------
    np.ndarray
        Transformed parameter array [M_c, eta, ...] of matching shape and float64 dtype.

    Raises
    ------
    TypeError
        If input cannot be converted to a NumPy float array.
    ValueError
        If theta has invalid dimensionality, non-finite values, or non-positive eta.
    """
    try:
        arr = np.asarray(theta, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"theta must be convertible to a float numpy array, got {type(theta)}") from e

    if not np.all(np.isfinite(arr)):
        raise ValueError("theta contains non-finite values (NaN or Inf)")

    if arr.ndim == 1:
        if arr.size not in (2, 4):
            raise ValueError(f"1D theta must have 2 or 4 elements, got shape {arr.shape}")
        mtotal = float(arr[0])
        eta = float(arr[1])
        if eta <= 0.0:
            raise ValueError(f"eta must be strictly positive, got {eta}")
        mc = mtotal * (eta ** 0.6)
        if arr.size == 2:
            return np.array([mc, eta], dtype=np.float64)
        return np.array([mc, eta, float(arr[2]), float(arr[3])], dtype=np.float64)

    elif arr.ndim == 2:
        if arr.shape[1] not in (2, 4):
            raise ValueError(f"2D theta must have shape (N, 2) or (N, 4), got shape {arr.shape}")
        eta = arr[:, 1]
        if np.any(eta <= 0.0):
            raise ValueError("All eta elements must be strictly positive")
        out = arr.copy()
        out[:, 0] = arr[:, 0] * (eta ** 0.6)
        return out
    else:
        raise ValueError(f"theta must be 1D or 2D, got ndim={arr.ndim}")


def prior_transform_total_mass_eta_2d(
    u: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    r"""Analytic unit-square transformation to alternative 2D coordinates (M_total, eta).

    For independent u_1, u_2 in [0, 1] (0-indexed as u[0], u[1]):
        eta = 0.10 + 0.15 * u_1
        M_total = eta^(-3/5) * (20.0 + 20.0 * u_2)

    This exact closed-form transformation preserves the original uniform prior
    p(M_c, eta) = 1/3 (M_sun)^(-1) over [20.0, 40.0] M_sun x [0.10, 0.25].

    Parameters
    ----------
    u : array-like
        1D array of shape (2,) or 2D array of shape (N, 2) with elements in [0, 1].

    Returns
    -------
    np.ndarray
        Transformed physical parameters [M_total, eta] in M_sun x dimensionless.
    """
    try:
        arr = np.asarray(u, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"u must be convertible to a float numpy array, got {type(u)}") from e

    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"1D input u must have exactly 2 elements, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("u contains non-finite values (NaN or Inf)")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            raise ValueError(f"All elements of u must lie in [0, 1]; found min={min_val}, max={max_val}")

        u1 = float(arr[0])
        u2 = float(arr[1])
        eta = 0.10 + 0.15 * u1
        mtotal = (eta ** -0.6) * (20.0 + 20.0 * u2)
        return np.array([mtotal, eta], dtype=np.float64)

    elif arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"2D input u must have shape (N, 2), got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("u contains non-finite values (NaN or Inf)")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError("All elements of u must lie in [0, 1]")

        u1 = arr[:, 0]
        u2 = arr[:, 1]
        eta = 0.10 + 0.15 * u1
        mtotal = (eta ** -0.6) * (20.0 + 20.0 * u2)
        out = np.empty_like(arr)
        out[:, 0] = mtotal
        out[:, 1] = eta
        return out
    else:
        raise ValueError(f"u must be 1D or 2D with last dimension 2, got ndim={arr.ndim} and shape={arr.shape}")


def prior_transform_total_mass_eta_4d(
    u: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    r"""Analytic unit-hypercube transformation to alternative 4D coordinates (M_total, eta, t_c, phi_c).

    For independent u_1, u_2, u_3, u_4 in [0, 1] (0-indexed as u[0], u[1], u[2], u[3]):
        eta = 0.10 + 0.15 * u_1
        M_total = eta^(-3/5) * (20.0 + 20.0 * u_2)
        t_c = 1.90 + 0.20 * u_3
        phi_c = 2 * pi * u_4  (with periodic boundary wrapping)

    Preserves the exact original physical prior in (M_c, eta, t_c, phi_c).

    Parameters
    ----------
    u : array-like
        1D array of shape (4,) or 2D array of shape (N, 4). Non-periodic dimensions
        u_1, u_2, u_3 must lie in [0, 1]. Periodic dimension u_4 supports wrapping.

    Returns
    -------
    np.ndarray
        Transformed physical parameters [M_total, eta, t_c, phi_c] in M_sun x dimless x s x rad.
    """
    try:
        arr = np.asarray(u, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"u must be convertible to a float numpy array, got {type(u)}") from e

    if arr.ndim == 1:
        if arr.size != 4:
            raise ValueError(f"1D input u must have exactly 4 elements, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("u contains non-finite values (NaN or Inf)")
        if np.any(arr[:3] < 0.0) or np.any(arr[:3] > 1.0):
            min_val = float(np.min(arr[:3]))
            max_val = float(np.max(arr[:3]))
            raise ValueError(f"Non-periodic elements u_0, u_1, u_2 must lie in [0, 1]; found min={min_val}, max={max_val}")

        u1 = float(arr[0])
        u2 = float(arr[1])
        u3 = float(arr[2])
        u4 = float(arr[3])

        eta = 0.10 + 0.15 * u1
        mtotal = (eta ** -0.6) * (20.0 + 20.0 * u2)
        tc = 1.90 + 0.20 * u3
        phic = 2.0 * math.pi if u4 == 1.0 else (u4 % 1.0) * (2.0 * math.pi)
        return np.array([mtotal, eta, tc, phic], dtype=np.float64)

    elif arr.ndim == 2:
        if arr.shape[1] != 4:
            raise ValueError(f"2D input u must have shape (N, 4), got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("u contains non-finite values (NaN or Inf)")
        if np.any(arr[:, :3] < 0.0) or np.any(arr[:, :3] > 1.0):
            raise ValueError("Non-periodic elements u_0, u_1, u_2 must lie in [0, 1]")

        u1 = arr[:, 0]
        u2 = arr[:, 1]
        u3 = arr[:, 2]
        u4 = arr[:, 3]

        eta = 0.10 + 0.15 * u1
        mtotal = (eta ** -0.6) * (20.0 + 20.0 * u2)
        tc = 1.90 + 0.20 * u3
        phic = np.where(u4 == 1.0, 2.0 * math.pi, (u4 % 1.0) * (2.0 * math.pi))

        out = np.empty_like(arr)
        out[:, 0] = mtotal
        out[:, 1] = eta
        out[:, 2] = tc
        out[:, 3] = phic
        return out
    else:
        raise ValueError(f"u must be 1D or 2D with last dimension 4, got ndim={arr.ndim} and shape={arr.shape}")


class CorrelatedTotalMassEtaPrior:
    r"""Prior density and unit-cube transform for alternative (M_total, eta) mass coordinates.

    Represents the exact pushforward measure of the approved uniform (M_c, eta) prior under:
        M_total = M_c * eta^(-3/5),  eta = eta.

    Mathematical properties:
    ------------------------
    1. Transformed probability density:
           p(M_total, eta) = p(M_c, eta) * |d(M_c, eta) / d(M_total, eta)|
                           = (1/3) * eta^(3/5)  (M_sun)^(-1)
       on the correlated support:
           eta in [0.10, 0.25],
           M_total * eta^(3/5) in [20.0, 40.0] M_sun.
       Outside this correlated support, p(M_total, eta) = 0.

    2. Normalization:
           \iint p(M_total, eta) dM_total deta = 1.0 (exact).

    3. Exact unit-cube mapping (u_1, u_2 in [0, 1]):
           eta = eta_min + u_1 * (eta_max - eta_min)
           M_total = eta^(-3/5) * (mc_min + u_2 * (mc_max - mc_min))

    Parameters
    ----------
    mc_min : float, default=20.0
        Minimum physical chirp mass in solar masses (M_sun). Must be > 0.
    mc_max : float, default=40.0
        Maximum physical chirp mass in solar masses (M_sun). Must be > mc_min.
    eta_min : float, default=0.10
        Minimum symmetric mass ratio. Must be > 0.
    eta_max : float, default=0.25
        Maximum symmetric mass ratio. Must satisfy eta_min < eta_max <= 0.25.
    """

    PARAM_NAMES: Tuple[str, str] = ("total_mass_msun", "symmetric_mass_ratio")
    PARAM_UNITS: Tuple[str, str] = ("M_sun", "dimensionless")

    def __init__(
        self,
        mc_min: float = 20.0,
        mc_max: float = 40.0,
        eta_min: float = 0.10,
        eta_max: float = 0.25,
    ) -> None:
        if not math.isfinite(mc_min) or mc_min <= 0.0:
            raise ValueError(f"mc_min must be positive and finite, got {mc_min}")
        if not math.isfinite(mc_max) or mc_max <= mc_min:
            raise ValueError(f"mc_max must be greater than mc_min ({mc_min}), got {mc_max}")
        if not math.isfinite(eta_min) or eta_min <= 0.0:
            raise ValueError(f"eta_min must be positive and finite, got {eta_min}")
        if not math.isfinite(eta_max) or eta_max <= eta_min or eta_max > 0.25:
            raise ValueError(f"eta_max must satisfy eta_min < eta_max <= 0.25, got {eta_max}")

        self._mc_min = float(mc_min)
        self._mc_max = float(mc_max)
        self._eta_min = float(eta_min)
        self._eta_max = float(eta_max)

        # Original hypervolume V_orig = Delta M_c * Delta eta
        self._volume_orig = (self._mc_max - self._mc_min) * (self._eta_max - self._eta_min)
        self._base_density = 1.0 / self._volume_orig
        self._base_log_density = -math.log(self._volume_orig)

    @property
    def mc_min(self) -> float:
        """Return minimum physical chirp mass in solar masses."""
        return self._mc_min

    @property
    def mc_max(self) -> float:
        """Return maximum physical chirp mass in solar masses."""
        return self._mc_max

    @property
    def eta_min(self) -> float:
        """Return minimum symmetric mass ratio."""
        return self._eta_min

    @property
    def eta_max(self) -> float:
        """Return maximum symmetric mass ratio."""
        return self._eta_max

    @property
    def param_names(self) -> Tuple[str, str]:
        """Return ordered parameter names."""
        return self.PARAM_NAMES

    @property
    def param_units(self) -> Tuple[str, str]:
        """Return ordered parameter physical units."""
        return self.PARAM_UNITS

    def is_within_bounds(self, theta: Union[Sequence[float], np.ndarray]) -> bool:
        """Check whether proposed parameter vector lies inside the correlated support.

        Parameters
        ----------
        theta : array-like of shape (2,)
            Parameter vector [M_total, eta].

        Returns
        -------
        bool
            True if inside correlated support, False otherwise.
        """
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (2,) or not np.all(np.isfinite(arr)):
            return False
        mtotal, eta = float(arr[0]), float(arr[1])
        if mtotal <= 0.0 or not (self._eta_min <= eta <= self._eta_max):
            return False
        mc = mtotal * (eta ** 0.6)
        return self._mc_min <= mc <= self._mc_max

    def jacobian(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        r"""Evaluate absolute Jacobian determinant \|d(M_c, eta) / d(M_total, eta)\| = eta^(3/5)."""
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError(f"theta must have at least 2 elements, got shape {arr.shape}")
        eta = float(arr[1])
        if eta <= 0.0:
            raise ValueError(f"eta must be strictly positive, got {eta}")
        return eta ** 0.6

    def density(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        r"""Evaluate transformed prior probability density p(M_total, eta) = (1/3) * eta^(3/5)."""
        if self.is_within_bounds(theta):
            eta = float(theta[1])
            return self._base_density * (eta ** 0.6)
        return 0.0

    def log_prior(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        r"""Evaluate transformed log-prior density ln p(M_total, eta) = -ln(3) + (3/5)*ln(eta)."""
        if self.is_within_bounds(theta):
            eta = float(theta[1])
            return self._base_log_density + 0.6 * math.log(eta)
        return -math.inf

    def prior_transform(self, u: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        r"""Transform unit-square coordinates u in [0, 1]^2 to (M_total, eta)."""
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != 2:
                raise ValueError(f"u must have 2 elements, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("u contains non-finite values (NaN or Inf)")
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError(f"u must lie in [0, 1]^2, got {arr}")
            u1 = float(arr[0])
            u2 = float(arr[1])
            eta = self._eta_min + u1 * (self._eta_max - self._eta_min)
            mc = self._mc_min + u2 * (self._mc_max - self._mc_min)
            mtotal = mc * (eta ** -0.6)
            return np.array([mtotal, eta], dtype=np.float64)
        elif arr.ndim == 2:
            if arr.shape[1] != 2:
                raise ValueError(f"2D u must have shape (N, 2), got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("u contains non-finite values (NaN or Inf)")
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError("All elements of u must lie in [0, 1]")
            u1 = arr[:, 0]
            u2 = arr[:, 1]
            eta = self._eta_min + u1 * (self._eta_max - self._eta_min)
            mc = self._mc_min + u2 * (self._mc_max - self._mc_min)
            mtotal = mc * (eta ** -0.6)
            out = np.empty_like(arr)
            out[:, 0] = mtotal
            out[:, 1] = eta
            return out
        else:
            raise ValueError(f"u must be 1D or 2D, got ndim={arr.ndim}")

    def to_physical_mchirp_eta(
        self, theta: Union[Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """Map parameter vector [M_total, eta] to physical [M_c, eta]."""
        return total_mass_eta_to_mchirp_eta(theta)

    def from_physical_mchirp_eta(
        self, theta: Union[Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """Map physical parameter vector [M_c, eta] to [M_total, eta]."""
        return mchirp_eta_to_total_mass_eta(theta)


class Correlated4DTotalMassEtaPrior:
    r"""Prior density and unit-cube transform for alternative 4D (M_total, eta, t_c, phi_c) coordinates.

    Represents the exact pushforward measure of the approved uniform 4D prior under:
        M_total = M_c * eta^(-3/5),  eta = eta,  t_c = t_c,  phi_c = phi_c.

    The transformed prior density is:
        p(M_total, eta, t_c, phi_c) = (1 / (1.2 * pi)) * eta^(3/5)
    on the correlated support:
        eta in [0.10, 0.25],
        M_total * eta^(3/5) in [20.0, 40.0] M_sun,
        t_c in [1.90, 2.10] s,
        phi_c in [0.0, 2*pi] rad.

    Parameters
    ----------
    mc_min : float, default=20.0
        Minimum physical chirp mass in solar masses (M_sun).
    mc_max : float, default=40.0
        Maximum physical chirp mass in solar masses (M_sun).
    eta_min : float, default=0.10
        Minimum symmetric mass ratio.
    eta_max : float, default=0.25
        Maximum symmetric mass ratio.
    tc_min : float, default=1.90
        Minimum coalescence time in seconds.
    tc_max : float, default=2.10
        Maximum coalescence time in seconds.
    phic_min : float, default=0.0
        Minimum coalescence phase in radians.
    phic_max : float, default=2*pi
        Maximum coalescence phase in radians.
    """

    PARAM_NAMES: Tuple[str, str, str, str] = (
        "total_mass_msun",
        "symmetric_mass_ratio",
        "coalescence_time_s",
        "coalescence_phase_rad",
    )
    PARAM_UNITS: Tuple[str, str, str, str] = ("M_sun", "dimensionless", "s", "rad")

    def __init__(
        self,
        mc_min: float = 20.0,
        mc_max: float = 40.0,
        eta_min: float = 0.10,
        eta_max: float = 0.25,
        tc_min: float = 1.90,
        tc_max: float = 2.10,
        phic_min: float = 0.0,
        phic_max: float = 2.0 * math.pi,
    ) -> None:
        if not math.isfinite(mc_min) or mc_min <= 0.0:
            raise ValueError(f"mc_min must be positive and finite, got {mc_min}")
        if not math.isfinite(mc_max) or mc_max <= mc_min:
            raise ValueError(f"mc_max must be greater than mc_min ({mc_min}), got {mc_max}")
        if not math.isfinite(eta_min) or eta_min <= 0.0:
            raise ValueError(f"eta_min must be positive and finite, got {eta_min}")
        if not math.isfinite(eta_max) or eta_max <= eta_min or eta_max > 0.25:
            raise ValueError(f"eta_max must satisfy eta_min < eta_max <= 0.25, got {eta_max}")
        if not math.isfinite(tc_min):
            raise ValueError(f"tc_min must be finite, got {tc_min}")
        if not math.isfinite(tc_max) or tc_max <= tc_min:
            raise ValueError(f"tc_max must be greater than tc_min ({tc_min}), got {tc_max}")
        if not math.isfinite(phic_min):
            raise ValueError(f"phic_min must be finite, got {phic_min}")
        if not math.isfinite(phic_max) or phic_max <= phic_min:
            raise ValueError(f"phic_max must be greater than phic_min ({phic_min}), got {phic_max}")

        self._mc_min = float(mc_min)
        self._mc_max = float(mc_max)
        self._eta_min = float(eta_min)
        self._eta_max = float(eta_max)
        self._tc_min = float(tc_min)
        self._tc_max = float(tc_max)
        self._phic_min = float(phic_min)
        self._phic_max = float(phic_max)

        self._volume_orig = (
            (self._mc_max - self._mc_min)
            * (self._eta_max - self._eta_min)
            * (self._tc_max - self._tc_min)
            * (self._phic_max - self._phic_min)
        )
        self._base_density = 1.0 / self._volume_orig
        self._base_log_density = -math.log(self._volume_orig)

    @property
    def mc_min(self) -> float:
        """Return minimum physical chirp mass in solar masses."""
        return self._mc_min

    @property
    def mc_max(self) -> float:
        """Return maximum physical chirp mass in solar masses."""
        return self._mc_max

    @property
    def eta_min(self) -> float:
        """Return minimum symmetric mass ratio."""
        return self._eta_min

    @property
    def eta_max(self) -> float:
        """Return maximum symmetric mass ratio."""
        return self._eta_max

    @property
    def tc_min(self) -> float:
        """Return minimum coalescence time in seconds."""
        return self._tc_min

    @property
    def tc_max(self) -> float:
        """Return maximum coalescence time in seconds."""
        return self._tc_max

    @property
    def phic_min(self) -> float:
        """Return minimum coalescence phase in radians."""
        return self._phic_min

    @property
    def phic_max(self) -> float:
        """Return maximum coalescence phase in radians."""
        return self._phic_max

    @property
    def param_names(self) -> Tuple[str, str, str, str]:
        """Return ordered parameter names."""
        return self.PARAM_NAMES

    @property
    def param_units(self) -> Tuple[str, str, str, str]:
        """Return ordered parameter physical units."""
        return self.PARAM_UNITS

    def is_within_bounds(self, theta: Union[Sequence[float], np.ndarray]) -> bool:
        """Check whether proposed parameter vector lies inside the correlated 4D support.

        Parameters
        ----------
        theta : array-like of shape (4,)
            Parameter vector [M_total, eta, t_c, phi_c].

        Returns
        -------
        bool
            True if inside correlated 4D support, False otherwise.
        """
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (4,) or not np.all(np.isfinite(arr)):
            return False
        mtotal, eta, tc, phic = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        if mtotal <= 0.0 or not (self._eta_min <= eta <= self._eta_max):
            return False
        mc = mtotal * (eta ** 0.6)
        if not (self._mc_min <= mc <= self._mc_max):
            return False
        return (self._tc_min <= tc <= self._tc_max) and (self._phic_min <= phic <= self._phic_max)

    def jacobian(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        r"""Evaluate absolute Jacobian determinant \|d(M_c, eta, t_c, phi_c) / d(M_total, eta, t_c, phi_c)\| = eta^(3/5)."""
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError(f"theta must have at least 2 elements, got shape {arr.shape}")
        eta = float(arr[1])
        if eta <= 0.0:
            raise ValueError(f"eta must be strictly positive, got {eta}")
        return eta ** 0.6

    def density(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        r"""Evaluate transformed 4D prior probability density p(theta) = (1 / V_orig) * eta^(3/5)."""
        if self.is_within_bounds(theta):
            eta = float(theta[1])
            return self._base_density * (eta ** 0.6)
        return 0.0

    def log_prior(self, theta: Union[Sequence[float], np.ndarray]) -> float:
        r"""Evaluate transformed 4D log-prior density ln p(theta) = -ln(V_orig) + (3/5)*ln(eta)."""
        if self.is_within_bounds(theta):
            eta = float(theta[1])
            return self._base_log_density + 0.6 * math.log(eta)
        return -math.inf

    def prior_transform(self, u: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        r"""Transform unit-hypercube coordinates u in [0, 1]^4 to (M_total, eta, t_c, phi_c)."""
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != 4:
                raise ValueError(f"u must have 4 elements, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("u contains non-finite values (NaN or Inf)")
            if np.any(arr[:3] < 0.0) or np.any(arr[:3] > 1.0):
                raise ValueError(f"Non-periodic elements u_0, u_1, u_2 must lie in [0, 1], got {arr[:3]}")

            u1 = float(arr[0])
            u2 = float(arr[1])
            u3 = float(arr[2])
            u4 = float(arr[3])

            eta = self._eta_min + u1 * (self._eta_max - self._eta_min)
            mc = self._mc_min + u2 * (self._mc_max - self._mc_min)
            mtotal = mc * (eta ** -0.6)
            tc = self._tc_min + u3 * (self._tc_max - self._tc_min)
            phic = self._phic_max if u4 == 1.0 else self._phic_min + (u4 % 1.0) * (self._phic_max - self._phic_min)
            return np.array([mtotal, eta, tc, phic], dtype=np.float64)

        elif arr.ndim == 2:
            if arr.shape[1] != 4:
                raise ValueError(f"2D u must have shape (N, 4), got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("u contains non-finite values (NaN or Inf)")
            if np.any(arr[:, :3] < 0.0) or np.any(arr[:, :3] > 1.0):
                raise ValueError("Non-periodic elements u_0, u_1, u_2 must lie in [0, 1]")

            u1 = arr[:, 0]
            u2 = arr[:, 1]
            u3 = arr[:, 2]
            u4 = arr[:, 3]

            eta = self._eta_min + u1 * (self._eta_max - self._eta_min)
            mc = self._mc_min + u2 * (self._mc_max - self._mc_min)
            mtotal = mc * (eta ** -0.6)
            tc = self._tc_min + u3 * (self._tc_max - self._tc_min)
            phic = np.where(
                u4 == 1.0,
                self._phic_max,
                self._phic_min + (u4 % 1.0) * (self._phic_max - self._phic_min),
            )

            out = np.empty_like(arr)
            out[:, 0] = mtotal
            out[:, 1] = eta
            out[:, 2] = tc
            out[:, 3] = phic
            return out
        else:
            raise ValueError(f"u must be 1D or 2D, got ndim={arr.ndim}")

    def to_physical_mchirp_eta(
        self, theta: Union[Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """Map parameter vector [M_total, eta, t_c, phi_c] to physical [M_c, eta, t_c, phi_c]."""
        return total_mass_eta_to_mchirp_eta(theta)

    def from_physical_mchirp_eta(
        self, theta: Union[Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """Map physical parameter vector [M_c, eta, t_c, phi_c] to [M_total, eta, t_c, phi_c]."""
        return mchirp_eta_to_total_mass_eta(theta)

