"""Frequency grid and observation mask abstraction for gravitational-wave inference.

This module implements:
1. Full discrete Fourier frequency grid derived from N and f_s (Delta t).
2. Static positive-frequency active observation mask K_+ on [f_start, f_end].
3. Strict separation between the data analysis mask and parameter-dependent waveform cutoffs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from gw_inference.conventions import build_frequency_grid


@dataclass(frozen=True)
class FrequencyGrid:
    """Immutable representation of a discrete one-sided Fourier frequency grid.

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples N. Must be a positive integer (supports both even and odd N).
    delta_t : float
        Sampling interval in seconds (\\Delta t = 1 / f_s). Must be positive and finite.

    Attributes
    ----------
    n_samples : int
        Number of samples N.
    delta_t : float
        Sampling interval \\Delta t in seconds.
    sampling_rate : float
        Sampling frequency f_s = 1 / \\Delta t in Hz.
    duration : float
        Total observation duration T = N \\Delta t in seconds.
    delta_f : float
        Frequency resolution \\Delta f = 1 / T = f_s / N in Hz.
    nyquist : float
        Nyquist frequency f_{\\mathrm{Nyq}} = f_s / 2 in Hz.
    n_frequencies : int
        Number of one-sided frequency bins floor(N/2) + 1.
    frequencies : np.ndarray
        Array of frequency bin centers f_k = k \\Delta f for k = 0, ..., floor(N/2).
    """

    n_samples: int
    delta_t: float

    def __post_init__(self):
        if not isinstance(self.n_samples, (int, np.integer)) or self.n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {self.n_samples}")
        if (
            not isinstance(self.delta_t, (int, float, np.floating))
            or not math.isfinite(self.delta_t)
            or self.delta_t <= 0
        ):
            raise ValueError(f"delta_t must be a positive finite float, got {self.delta_t}")

        object.__setattr__(self, "n_samples", int(self.n_samples))
        object.__setattr__(self, "delta_t", float(self.delta_t))
        object.__setattr__(self, "sampling_rate", float(1.0 / self.delta_t))
        object.__setattr__(self, "duration", float(self.n_samples * self.delta_t))
        object.__setattr__(self, "delta_f", float(1.0 / (self.n_samples * self.delta_t)))
        object.__setattr__(self, "nyquist", float(0.5 / self.delta_t))
        object.__setattr__(self, "n_frequencies", int(self.n_samples // 2 + 1))
        object.__setattr__(
            self,
            "frequencies",
            build_frequency_grid(self.n_samples, self.delta_t, allow_odd=True),
        )

    @classmethod
    def from_duration_and_sampling_rate(
        cls, duration: float, sampling_rate: float
    ) -> FrequencyGrid:
        """Construct FrequencyGrid from total observation duration T and sampling rate f_s.

        Parameters
        ----------
        duration : float
            Observation time in seconds. Must be positive and finite.
        sampling_rate : float
            Sampling frequency in Hz. Must be positive and finite.

        Returns
        -------
        FrequencyGrid
            Constructed frequency grid.
        """
        if (
            not isinstance(duration, (int, float, np.floating))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(f"duration must be a positive finite float, got {duration}")
        if (
            not isinstance(sampling_rate, (int, float, np.floating))
            or not math.isfinite(sampling_rate)
            or sampling_rate <= 0
        ):
            raise ValueError(f"sampling_rate must be a positive finite float, got {sampling_rate}")

        delta_t = 1.0 / float(sampling_rate)
        n_samples = int(round(float(duration) * float(sampling_rate)))
        return cls(n_samples=n_samples, delta_t=delta_t)

    def create_active_mask(self, f_start: float, f_end: float) -> np.ndarray:
        """Construct a static positive-frequency active observation mask K_+.

        K_+ = { k : f_start <= f_k <= f_end, 1 <= k <= (N - 1) // 2 } (for odd N)
        K_+ = { k : f_start <= f_k <= f_end, 1 <= k <= N/2 - 1 }       (for even N)

        Exclusion of Endpoints:
        -----------------------
        - DC (k = 0, f_0 = 0.0 Hz) is ALWAYS False.
        - Nyquist (k = N/2, f_{N/2} = f_s / 2) is ALWAYS False when N is even.
        - For odd N, there is no Nyquist bin in the DFT; the highest bin k = (N - 1) / 2
          has f_k = ((N - 1)/(2N)) * f_s < f_s / 2 and is treated as an ordinary interior
          bin (retained if f_start <= f_k <= f_end).
        - Interior bins are True if and only if f_start <= f_k <= f_end.

        Invariance Property:
        --------------------
        The returned mask is defined strictly by the detector observation band
        and sampling parameters. It is STATIC and MUST NEVER be dynamically altered
        during Bayesian parameter estimation or template generation.

        Parameters
        ----------
        f_start : float
            Lower frequency boundary in Hz. Must satisfy 0 < f_start < f_end.
        f_end : float
            Upper frequency boundary in Hz. Must satisfy f_start < f_end <= f_Nyquist.

        Returns
        -------
        np.ndarray
            1D boolean array of shape (n_frequencies,) where True indicates an active bin.

        Raises
        ------
        ValueError
            If frequency bounds are invalid or result in zero active bins.
        """
        if (
            not isinstance(f_start, (int, float, np.floating))
            or not math.isfinite(f_start)
            or f_start <= 0
        ):
            raise ValueError(f"f_start must be a positive finite float, got {f_start}")
        if (
            not isinstance(f_end, (int, float, np.floating))
            or not math.isfinite(f_end)
            or f_end <= f_start
        ):
            raise ValueError(f"f_end must be finite and strictly greater than f_start ({f_start}), got {f_end}")
        if f_end > self.nyquist:
            raise ValueError(
                f"f_end ({f_end} Hz) cannot exceed Nyquist frequency ({self.nyquist} Hz)"
            )

        freqs = self.frequencies
        # Standard inclusive interval f_start <= f_k <= f_end
        mask = (freqs >= float(f_start)) & (freqs <= float(f_end))

        # Strictly enforce exclusion of DC (index 0)
        mask[0] = False

        # Strictly enforce exclusion of Nyquist (index -1) ONLY when N is even.
        # For odd N, there is no Nyquist bin in the DFT; bin -1 is an interior bin k = (N - 1) / 2
        # with f_k < f_s / 2, which must be retained if it falls within [f_start, f_end].
        if self.n_samples % 2 == 0:
            mask[-1] = False

        if not np.any(mask):
            raise ValueError(
                f"Frequency band [{f_start}, {f_end}] Hz contains zero interior frequency bins "
                f"on grid with delta_f = {self.delta_f} Hz"
            )

        return mask

    def get_active_indices(self, mask: np.ndarray) -> np.ndarray:
        """Extract array of 0-based DFT indices where mask is True.

        Parameters
        ----------
        mask : np.ndarray
            1D boolean mask of shape (N/2 + 1,).

        Returns
        -------
        np.ndarray
            1D integer array of active DFT indices.
        """
        self._validate_mask_shape(mask)
        return np.flatnonzero(mask)

    def get_active_frequencies(self, mask: np.ndarray) -> np.ndarray:
        """Extract array of active physical frequencies f_k where mask is True.

        Parameters
        ----------
        mask : np.ndarray
            1D boolean mask of shape (N/2 + 1,).

        Returns
        -------
        np.ndarray
            1D float64 array of active frequency values in Hz.
        """
        self._validate_mask_shape(mask)
        return self.frequencies[mask]

    def _validate_mask_shape(self, mask: np.ndarray) -> None:
        """Ensure mask matches frequency grid dimensions."""
        m = np.asarray(mask)
        if m.ndim != 1 or m.size != self.n_frequencies:
            raise ValueError(
                f"mask shape {m.shape} must be 1D with length {self.n_frequencies}"
            )
        if m.dtype != bool:
            raise ValueError(f"mask must have boolean dtype, got {m.dtype}")
