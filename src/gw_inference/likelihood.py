"""Frequency-domain Gaussian likelihood for gravitational-wave Bayesian inference.

This module implements the active-band Whittle likelihood approximation for
stationary, colored Gaussian detector noise according to the frozen project
conventions approved in Work Packages 0 through 4.

Mathematical Formulation:
-------------------------
For sampling rate f_s = 1 / \\Delta t, observation duration T = N \\Delta t, and
frequency resolution \\Delta f = f_s / N = 1 / T:

1. Time-scaled discrete Fourier transform:
   \\tilde{h}(f_k) = \\Delta t \\sum_{j=0}^{N-1} h(t_j) \\exp(-i 2\\pi j k / N)

2. Frequency-domain residual:
   \\tilde{r}(f_k; \\theta) = \\tilde{d}(f_k) - \\tilde{h}(f_k; \\theta)

3. Active positive-frequency set:
   \\mathcal{K}_+ = \\{ k : 1 \\le k \\le \\lfloor (N-1)/2 \\rfloor, M_k = 1 \\}
   DC (k = 0) and Nyquist (k = N/2 for even N) are strictly excluded.
   The active mask is static, determined solely by the analysis band [f_start, f_end],
   and strictly independent of parameter-dependent waveform cutoffs.

4. Active-band noise-weighted quadratic form:
   (r|r)_{\\mathcal{K}_+} = 4 \\Delta f \\sum_{k \\in \\mathcal{K}_+} \\frac{|\\tilde{r}(f_k)|^2}{S_n(f_k)}

5. Active-band Whittle log-likelihood:
   \\ln L_{\\mathrm{active}}(\\theta) = \\ln L_{\\mathrm{norm}} - \\frac{1}{2} (r|r)_{\\mathcal{K}_+}
   where:
   \\ln L_{\\mathrm{norm}} = -\\sum_{k \\in \\mathcal{K}_+} \\ln\\left[ \\frac{\\pi N S_n(f_k)}{2 f_s} \\right]

Note on Whittle Approximation:
------------------------------
This is the standard positive-frequency active-band Whittle likelihood approximation.
It diagonalizes the noise covariance in the discrete Fourier basis, which is asymptotically
exact in the long-duration limit (T \\gg \\tau_{\\mathrm{autocorr}}) but neglects off-diagonal
Toeplitz covariance contributions from finite-duration windowing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from gw_inference.grid import FrequencyGrid
from gw_inference.psd import PSDEngine


@dataclass(frozen=True)
class GaussianLikelihoodResult:
    """Immutable evaluation result of a frequency-domain Gaussian likelihood.

    Attributes
    ----------
    log_likelihood : float
        Total active-band log-likelihood: ln L_active = ln L_norm - 0.5 * (r|r)_{K_+}.
    log_likelihood_norm : float
        Log-likelihood normalization term: -sum_{k in K_+} ln[pi * N * S_n(f_k) / (2 * f_s)].
    quadratic_form : float
        Noise-weighted active-band residual quadratic form: (r|r)_{K_+} = 4 * df * sum |r_k|^2 / S_n(f_k).
    n_active_bins : int
        Number of active frequency bins in K_+.
    """

    log_likelihood: float
    log_likelihood_norm: float
    quadratic_form: float
    n_active_bins: int


def compute_residual(
    data_fd: np.ndarray,
    template_fd: np.ndarray,
) -> np.ndarray:
    """Compute the complex frequency-domain residual r~(f) = d~(f) - h~(f).

    Parameters
    ----------
    data_fd : np.ndarray
        1D array of frequency-domain detector data d~(f_k).
    template_fd : np.ndarray
        1D array of frequency-domain template h~(f_k).

    Returns
    -------
    np.ndarray
        1D complex128 array containing r~(f_k) = d~(f_k) - h~(f_k).

    Raises
    ------
    ValueError
        If inputs have invalid dimensionality, mismatched shapes, or non-finite values.
    """
    d = np.asarray(data_fd, dtype=np.complex128)
    h = np.asarray(template_fd, dtype=np.complex128)

    if d.ndim != 1 or h.ndim != 1:
        raise ValueError(f"data_fd (ndim={d.ndim}) and template_fd (ndim={h.ndim}) must be 1D arrays")
    if d.shape != h.shape:
        raise ValueError(f"data_fd shape {d.shape} does not match template_fd shape {h.shape}")
    if d.size == 0:
        raise ValueError("data_fd and template_fd cannot be empty")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(h)):
        raise ValueError("data_fd and template_fd must contain only finite numbers (no NaN or Inf)")

    return d - h


def compute_quadratic_form(
    residual_fd: np.ndarray,
    psd: np.ndarray,
    delta_f: float,
    active_indices: Optional[np.ndarray] = None,
) -> float:
    """Compute the active-band noise-weighted quadratic form (r|r)_{K_+}.

    (r|r)_{K_+} = 4 * \\Delta f * \\sum_{k \\in \\mathcal{K}_+} \\frac{|\\tilde{r}(f_k)|^2}{S_n(f_k)}

    Parameters
    ----------
    residual_fd : np.ndarray
        1D complex or float frequency-domain residual array r~(f_k).
    psd : np.ndarray
        1D float one-sided power spectral density S_n(f_k). Must be strictly positive.
        Can be either full-grid size matching residual_fd, or pre-sliced to active bins.
    delta_f : float
        Frequency resolution \\Delta f in Hz. Must be positive and finite.
    active_indices : np.ndarray, optional
        1D integer array of active frequency bin indices K_+.
        If None, residual_fd and psd are assumed to be already pre-sliced over active bins.

    Returns
    -------
    float
        Real-valued quadratic form (r|r)_{K_+} >= 0.

    Raises
    ------
    ValueError
        If inputs have invalid dimensions, mismatched shapes, non-finite values,
        non-positive PSD values, or invalid delta_f.
    """
    if not isinstance(delta_f, (int, float, np.floating)) or not math.isfinite(delta_f) or delta_f <= 0:
        raise ValueError(f"delta_f must be a positive finite float, got {delta_f}")

    r = np.asarray(residual_fd, dtype=np.complex128)
    p = np.asarray(psd, dtype=np.float64)

    if r.ndim != 1 or p.ndim != 1:
        raise ValueError(f"residual_fd (ndim={r.ndim}) and psd (ndim={p.ndim}) must be 1D arrays")

    if active_indices is not None:
        idx = np.asarray(active_indices, dtype=np.intp)
        if idx.ndim != 1:
            raise ValueError(f"active_indices must be a 1D array, got ndim={idx.ndim}")
        if idx.size == 0:
            raise ValueError("active_indices cannot be empty")
        if np.any(idx < 0) or np.any(idx >= r.size):
            raise ValueError(f"active_indices contains out-of-bounds indices for residual array of size {r.size}")

        r_act = r[idx]
        if p.size == r.size:
            p_act = p[idx]
        elif p.size == idx.size:
            p_act = p
        else:
            raise ValueError(
                f"psd shape {p.shape} matches neither full grid ({r.size}) nor active bins ({idx.size})"
            )
    else:
        if r.size == 0:
            raise ValueError("residual_fd cannot be empty")
        if p.shape != r.shape:
            raise ValueError(f"psd shape {p.shape} does not match residual_fd shape {r.shape}")
        r_act = r
        p_act = p

    if not np.all(np.isfinite(r_act)):
        raise ValueError("residual_fd on active bins contains non-finite numbers (NaN or Inf)")
    if not np.all(np.isfinite(p_act)):
        raise ValueError("psd on active bins contains non-finite numbers (NaN or Inf)")
    if np.any(p_act <= 0):
        idx_bad = int(np.where(p_act <= 0)[0][0])
        raise ValueError(f"psd values in active band must be strictly positive (> 0); found {p_act[idx_bad]} at index {idx_bad}")

    # 4 * delta_f * sum |r_k|^2 / S_n(f_k)
    # Using real-squared: |r_k|^2 = Re(r_k)^2 + Im(r_k)^2
    r_power = (r_act.real ** 2) + (r_act.imag ** 2)
    quad_form = 4.0 * float(delta_f) * float(np.sum(r_power / p_act))
    return float(max(0.0, quad_form))


def compute_log_likelihood_normalization(
    psd: np.ndarray,
    n_samples: int,
    sampling_rate: float,
    active_indices: Optional[np.ndarray] = None,
) -> float:
    """Compute the active-band Whittle log-likelihood normalization term.

    \\ln L_{\\mathrm{norm}} = -\\sum_{k \\in \\mathcal{K}_+} \\ln\\left[ \\frac{\\pi N S_n(f_k)}{2 f_s} \\right]

    Parameters
    ----------
    psd : np.ndarray
        1D float array of one-sided PSD values S_n(f_k). Must be strictly positive.
        Can be full grid or pre-sliced to active bins.
    n_samples : int
        Number of time-domain samples N. Must be a positive integer.
    sampling_rate : float
        Sampling frequency f_s in Hz. Must be positive and finite.
    active_indices : np.ndarray, optional
        1D integer array of active frequency bin indices K_+.
        If None, psd is assumed to be pre-sliced to the active bins.

    Returns
    -------
    float
        Real-valued log-likelihood normalization constant ln L_norm.

    Raises
    ------
    ValueError
        If inputs are non-positive, non-finite, or have invalid dimensions.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
    if (
        not isinstance(sampling_rate, (int, float, np.floating))
        or not math.isfinite(sampling_rate)
        or sampling_rate <= 0
    ):
        raise ValueError(f"sampling_rate must be a positive finite float, got {sampling_rate}")

    p = np.asarray(psd, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"psd must be a 1D array, got ndim={p.ndim}")

    if active_indices is not None:
        idx = np.asarray(active_indices, dtype=np.intp)
        if idx.ndim != 1:
            raise ValueError(f"active_indices must be a 1D array, got ndim={idx.ndim}")
        if idx.size == 0:
            raise ValueError("active_indices cannot be empty")
        if p.size == idx.size:
            p_act = p
        elif np.any(idx < 0) or np.any(idx >= p.size):
            raise ValueError(f"active_indices contains out-of-bounds indices for psd array of size {p.size}")
        else:
            p_act = p[idx]
    else:
        if p.size == 0:
            raise ValueError("psd array cannot be empty")
        p_act = p

    if not np.all(np.isfinite(p_act)):
        raise ValueError("psd on active bins contains non-finite numbers (NaN or Inf)")
    if np.any(p_act <= 0):
        idx_bad = int(np.where(p_act <= 0)[0][0])
        raise ValueError(f"psd values in active band must be strictly positive (> 0); found {p_act[idx_bad]} at index {idx_bad}")

    # factor = pi * N / (2 * f_s)
    scale_factor = (math.pi * float(n_samples)) / (2.0 * float(sampling_rate))
    variances = scale_factor * p_act
    log_norm = -float(np.sum(np.log(variances)))
    return log_norm


def validate_psd_frequency_alignment(
    psd: np.ndarray,
    psd_frequencies: Optional[np.ndarray],
    grid: FrequencyGrid,
    active_indices: Optional[np.ndarray] = None,
    require_frequencies: bool = False,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate PSD array shape, active-bin finiteness, positivity, and physical frequency alignment.

    Precision Semantics:
    -------------------
    Frequency coordinate alignment is verified using `np.allclose` (numerical
    agreement within stated tolerances atol and rtol), NOT exact array equality
    `np.array_equal`. This accounts for benign floating-point representation
    differences arising from distinct arithmetic generation paths (e.g. cumulative
    sums vs discrete multiplications), while strictly rejecting physical frequency
    shifts or differing bin spacings Delta f.

    Active Index Mapping:
    --------------------
    - Full-grid inputs (len(psd) == grid.n_frequencies) are validated against
      grid.frequencies, and sliced to active_indices.
    - Pre-sliced active inputs (len(psd) == len(active_indices)) are validated
      against grid.frequencies[active_indices].

    Parameters
    ----------
    psd : np.ndarray
        1D float array of one-sided PSD values. Can be full-grid size matching
        grid.n_frequencies or pre-sliced to active_indices.size.
    psd_frequencies : np.ndarray, optional
        1D float array of physical frequency coordinates (in Hz) corresponding to psd.
        If provided, coordinates are validated for numerical agreement with grid.frequencies
        within tolerance (np.allclose).
    grid : FrequencyGrid
        FrequencyGrid instance defining the discrete frequency grid.
    active_indices : np.ndarray
        1D integer array of active frequency bin indices K_+.
    require_frequencies : bool, default=False
        If True, psd_frequencies cannot be None (enforced on empirical PSD paths).
    atol : float, default=1e-12
        Absolute tolerance for frequency coordinate numerical agreement in Hz.
    rtol : float, default=1e-12
        Relative tolerance for frequency coordinate numerical agreement.

    Returns
    -------
    tuple of (p_act, validated_freqs)
        p_act : np.ndarray
            1D float64 array of PSD values on the active bins.
        validated_freqs : np.ndarray or None
            1D float64 array of validated frequency coordinates matching psd shape if
            psd_frequencies was provided, else None.

    Raises
    ------
    TypeError
        If grid is not a FrequencyGrid instance.
    ValueError
        If psd or psd_frequencies has invalid dimensions, mismatched shapes, non-finite
        values, if active PSD has non-positive values, if psd_frequencies is omitted when
        required, or if psd_frequencies deviates from the expected likelihood frequency coordinates.
    """
    if not isinstance(grid, FrequencyGrid):
        raise TypeError(f"grid must be a FrequencyGrid instance, got {type(grid)}")

    idx = np.asarray(active_indices, dtype=np.intp)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("active_indices must be a non-empty 1D array")

    p = np.asarray(psd, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"psd must be a 1D array, got ndim={p.ndim}")

    if p.size == grid.n_frequencies:
        p_act = p[idx]
        expected_freqs = grid.frequencies
    elif p.size == idx.size:
        p_act = p
        expected_freqs = grid.frequencies[idx]
    else:
        raise ValueError(
            f"psd shape {p.shape} does not match full grid ({grid.n_frequencies}) "
            f"or active bins ({idx.size})"
        )

    if require_frequencies and psd_frequencies is None:
        raise ValueError(
            "psd_frequencies is strictly required for empirical PSD ingestion. "
            "Omitting frequency coordinates is not permitted on the empirical-PSD path."
        )

    validated_freqs: Optional[np.ndarray] = None
    if psd_frequencies is not None:
        f_p = np.asarray(psd_frequencies, dtype=np.float64)
        if f_p.shape != p.shape:
            raise ValueError(
                f"psd_frequencies shape {f_p.shape} does not match psd shape {p.shape}"
            )
        if not np.all(np.isfinite(f_p)):
            raise ValueError("psd_frequencies contains non-finite numbers (NaN or Inf)")
        # Numerical agreement within tolerance via np.allclose (not np.array_equal)
        if not np.allclose(f_p, expected_freqs, atol=atol, rtol=rtol):
            max_dev = float(np.max(np.abs(f_p - expected_freqs)))
            raise ValueError(
                f"psd_frequencies do not match expected likelihood grid frequencies within tolerance "
                f"(maximum absolute discrepancy: {max_dev:.6e} Hz). "
                f"Array length alone does not guarantee frequency-grid identity."
            )
        validated_freqs = np.array(f_p, copy=True)

    if not np.all(np.isfinite(p_act)):
        raise ValueError("psd on active bins contains non-finite values (NaN or Inf)")
    if np.any(p_act <= 0):
        idx_bad = int(np.where(p_act <= 0)[0][0])
        raise ValueError(
            f"psd on active bins must be strictly positive (> 0); found {p_act[idx_bad]} at index {idx_bad}"
        )

    return p_act, validated_freqs


def compute_log_likelihood(
    data_fd: np.ndarray,
    template_fd: np.ndarray,
    psd: np.ndarray,
    grid: FrequencyGrid,
    mask: Optional[np.ndarray] = None,
    active_indices: Optional[np.ndarray] = None,
    psd_frequencies: Optional[np.ndarray] = None,
    require_psd_frequencies: bool = False,
) -> float:
    """Compute the active-band Whittle Gaussian log-likelihood.

    \\ln L_{\\mathrm{active}}(\\theta) = -\\sum_{k \\in \\mathcal{K}_+} \\ln\\left[ \\frac{\\pi N S_n(f_k)}{2 f_s} \\right]
                                      - \\frac{1}{2} (r|r)_{\\mathcal{K}_+}

    Parameters
    ----------
    data_fd : np.ndarray
        1D complex or float frequency-domain detector data d~(f_k).
    template_fd : np.ndarray
        1D complex or float frequency-domain template h~(f_k).
    psd : np.ndarray
        1D float one-sided power spectral density S_n(f_k).
    grid : FrequencyGrid
        FrequencyGrid instance defining sampling rate, sample count, and frequency resolution.
    mask : np.ndarray, optional
        1D boolean active mask K_+ on the grid.
    active_indices : np.ndarray, optional
        1D integer active indices K_+ on the grid. Mutually exclusive with mask.
    psd_frequencies : np.ndarray, optional
        1D float array of physical frequency coordinates corresponding to psd.
        If provided, coordinates are validated against grid.frequencies within tolerance (np.allclose).
    require_psd_frequencies : bool, default=False
        If True, psd_frequencies cannot be omitted / None (enforced on empirical PSD paths).

    Returns
    -------
    float
        Real-valued active-band log-likelihood ln L_active.

    Raises
    ------
    ValueError
        If inputs are inconsistent, contain invalid values, or include DC/Nyquist in active set.
    """
    if not isinstance(grid, FrequencyGrid):
        raise TypeError(f"grid must be a FrequencyGrid instance, got {type(grid)}")

    # Resolve active indices
    if mask is not None and active_indices is not None:
        raise ValueError("Specify either mask or active_indices, not both")
    elif mask is not None:
        grid._validate_mask_shape(mask)
        idx = grid.get_active_indices(mask)
    elif active_indices is not None:
        idx = np.asarray(active_indices, dtype=np.intp)
    else:
        raise ValueError("Either mask or active_indices must be provided")

    _validate_active_indices_against_grid(idx, grid)

    p_act, _ = validate_psd_frequency_alignment(
        psd=psd,
        psd_frequencies=psd_frequencies,
        grid=grid,
        active_indices=idx,
        require_frequencies=require_psd_frequencies,
    )

    residual = compute_residual(data_fd, template_fd)
    quad_form = compute_quadratic_form(
        residual_fd=residual,
        psd=p_act,
        delta_f=grid.delta_f,
        active_indices=idx,
    )
    log_norm = compute_log_likelihood_normalization(
        psd=p_act,
        n_samples=grid.n_samples,
        sampling_rate=grid.sampling_rate,
        active_indices=idx,
    )

    return float(log_norm - 0.5 * quad_form)


class FrequencyDomainGaussianLikelihood:
    """Precomputed active-band Gaussian Whittle likelihood evaluator.

    Designed for high-throughput Bayesian inference (MCMC / nested sampling).
    Precomputes the static active mask, active PSD values, active inverse PSD,
    and log-likelihood normalization constant ln L_norm at initialization,
    ensuring minimal overhead per likelihood evaluation.

    Parameters
    ----------
    data_fd : np.ndarray
        1D complex or float frequency-domain detector data d~(f_k).
    psd : np.ndarray
        1D float one-sided PSD array S_n(f_k). Can be full grid or pre-sliced to active bins.
    grid : FrequencyGrid
        FrequencyGrid instance defining discrete analysis parameters.
    mask : np.ndarray
        1D boolean active observation mask K_+.
    psd_frequencies : np.ndarray, optional
        1D float array of physical frequency coordinates corresponding to psd.
        If provided, coordinates are strictly validated against grid.frequencies.
    require_psd_frequencies : bool, default=False
        If True, psd_frequencies cannot be omitted / None (enforced on empirical PSD paths).
    """

    def __init__(
        self,
        data_fd: np.ndarray,
        psd: np.ndarray,
        grid: FrequencyGrid,
        mask: np.ndarray,
        psd_frequencies: Optional[np.ndarray] = None,
        require_psd_frequencies: bool = False,
    ):
        if not isinstance(grid, FrequencyGrid):
            raise TypeError(f"grid must be a FrequencyGrid instance, got {type(grid)}")

        grid._validate_mask_shape(mask)
        self._grid = grid
        self._mask = np.array(mask, dtype=bool, copy=True)
        self._active_indices = grid.get_active_indices(self._mask)

        _validate_active_indices_against_grid(self._active_indices, grid)

        d = np.asarray(data_fd, dtype=np.complex128)
        if d.ndim != 1 or d.size != grid.n_frequencies:
            raise ValueError(f"data_fd shape {d.shape} must be 1D matching grid frequencies ({grid.n_frequencies})")
        if not np.all(np.isfinite(d)):
            raise ValueError("data_fd contains non-finite values (NaN or Inf)")

        # Cache pre-sliced active data
        self._data_active = np.array(d[self._active_indices], copy=True)

        p_act, validated_freqs = validate_psd_frequency_alignment(
            psd=psd,
            psd_frequencies=psd_frequencies,
            grid=grid,
            active_indices=self._active_indices,
            require_frequencies=require_psd_frequencies,
        )

        self._psd_active = np.array(p_act, copy=True)
        self._psd_frequencies = validated_freqs
        self._inv_psd_active = 1.0 / self._psd_active
        self._delta_f = grid.delta_f

        # Precompute normalization term
        self._log_norm = compute_log_likelihood_normalization(
            psd=self._psd_active,
            n_samples=grid.n_samples,
            sampling_rate=grid.sampling_rate,
            active_indices=None,
        )

    @classmethod
    def from_psd_engine(
        cls,
        data_fd: np.ndarray,
        psd_engine: PSDEngine,
        grid: FrequencyGrid,
        mask: np.ndarray,
        f_start: float,
        f_0: float,
    ) -> FrequencyDomainGaussianLikelihood:
        """Construct likelihood from a PSDEngine with explicit continuation parameters.

        Note
        ----
        This route evaluates continuous-spectrum interpolation via PSDEngine.interpolate
        onto grid.frequencies and applies analytical low-frequency continuation below f_start.
        It is intended exclusively for analytical/tabulated sensitivity curves.
        For empirical PSDs (e.g. Welch periodograms), use from_empirical_psd instead,
        which forbids interpolation and enforces explicit frequency-coordinate validation.

        Parameters
        ----------
        data_fd : np.ndarray
            Frequency-domain data array.
        psd_engine : PSDEngine
            Interpolation engine instance.
        grid : FrequencyGrid
            Analysis grid.
        mask : np.ndarray
            Active mask K_+.
        f_start : float
            Lower analysis frequency boundary in Hz.
        f_0 : float
            Explicit PSD low-frequency continuation scale in Hz (Decision D8 open).
            Must be strictly provided; no default value is permitted.

        Returns
        -------
        FrequencyDomainGaussianLikelihood
            Constructed likelihood evaluator.
        """
        if not isinstance(psd_engine, PSDEngine):
            raise TypeError(f"psd_engine must be a PSDEngine instance, got {type(psd_engine)}")

        # Mandatory explicit f_0 and f_start enforcement
        interpolated_psd = psd_engine.interpolate(grid.frequencies, f_start=f_start, f_0=f_0)
        return cls(
            data_fd=data_fd,
            psd=interpolated_psd,
            grid=grid,
            mask=mask,
            psd_frequencies=grid.frequencies,
        )

    @classmethod
    def from_empirical_psd(
        cls,
        data_fd: np.ndarray,
        psd: np.ndarray,
        grid: Union[FrequencyGrid, np.ndarray],
        mask: Union[np.ndarray, FrequencyGrid],
        psd_frequencies: Optional[np.ndarray] = None,
    ) -> FrequencyDomainGaussianLikelihood:
        """Construct likelihood from an empirical PSD (e.g. Welch periodogram).

        This is the authoritative Stage 5 empirical-PSD ingestion route.
        Unlike analytical PSD ingestion via PSDEngine, empirical PSD ingestion
        strictly requires explicit physical frequency coordinates that agree with
        the likelihood's FrequencyGrid within numerical tolerance (np.allclose).
        No interpolation, resampling, or analytical low-frequency continuation
        is performed.

        Parameters
        ----------
        data_fd : np.ndarray
            1D complex or float frequency-domain detector data d~(f_k).
        psd : np.ndarray
            1D float one-sided empirical PSD array. Can be full grid or pre-sliced to active bins.
        grid : FrequencyGrid
            FrequencyGrid instance defining discrete analysis parameters.
        mask : np.ndarray
            1D boolean active observation mask K_+.
        psd_frequencies : np.ndarray
            1D float array of physical frequency coordinates corresponding to psd.
            Must be explicitly provided (cannot be omitted / None) and must numerically
            agree with grid.frequencies (or active subset) within tolerance (np.allclose).

        Returns
        -------
        FrequencyDomainGaussianLikelihood
            Constructed likelihood evaluator with validated empirical PSD.

        Raises
        ------
        ValueError
            If psd_frequencies is None (omitted on the empirical path), or if
            frequency coordinates do not match the expected likelihood grid frequencies.
        TypeError
            If grid is not a FrequencyGrid instance.
        """
        if isinstance(grid, FrequencyGrid):
            target_grid = grid
            target_mask = mask
            target_freqs = psd_frequencies
        elif isinstance(mask, FrequencyGrid):
            # Positional calling pattern: (data_fd, psd, psd_frequencies, grid, mask)
            target_freqs = grid
            target_grid = mask
            target_mask = psd_frequencies
        else:
            raise TypeError("A FrequencyGrid instance must be provided to from_empirical_psd")

        if target_freqs is None:
            raise ValueError(
                "psd_frequencies is strictly required for empirical PSD ingestion. "
                "Omitting frequency coordinates is not permitted on the Stage 5 empirical-PSD path."
            )

        return cls(
            data_fd=data_fd,
            psd=psd,
            grid=target_grid,
            mask=target_mask,
            psd_frequencies=target_freqs,
            require_psd_frequencies=True,
        )

    @property
    def grid(self) -> FrequencyGrid:
        """Return the underlying FrequencyGrid."""
        return self._grid

    @property
    def mask(self) -> np.ndarray:
        """Return a copy of the active boolean mask."""
        return np.array(self._mask, copy=True)

    @property
    def active_indices(self) -> np.ndarray:
        """Return a copy of active bin indices K_+."""
        return np.array(self._active_indices, copy=True)

    @property
    def n_active_bins(self) -> int:
        """Return the count of active bins in K_+."""
        return int(self._active_indices.size)

    @property
    def log_likelihood_norm(self) -> float:
        """Return the precomputed log-likelihood normalization constant ln L_norm."""
        return self._log_norm

    @property
    def psd_frequencies(self) -> Optional[np.ndarray]:
        """Return a copy of the validated PSD physical frequency coordinates, or None if unvalidated."""
        if self._psd_frequencies is None:
            return None
        return np.array(self._psd_frequencies, copy=True)

    def evaluate(self, template_fd: np.ndarray) -> float:
        """Evaluate active-band Gaussian Whittle log-likelihood for a given template.

        Parameters
        ----------
        template_fd : np.ndarray
            1D frequency-domain template array h~(f_k). Can be full grid or pre-sliced to active bins.

        Returns
        -------
        float
            Log-likelihood ln L_active.
        """
        t = np.asarray(template_fd, dtype=np.complex128)
        if t.ndim != 1:
            raise ValueError(f"template_fd must be 1D, got ndim={t.ndim}")

        if t.size == self._grid.n_frequencies:
            t_act = t[self._active_indices]
        elif t.size == self._active_indices.size:
            t_act = t
        else:
            raise ValueError(
                f"template_fd size ({t.size}) matches neither full grid ({self._grid.n_frequencies}) "
                f"nor active bins ({self._active_indices.size})"
            )

        if not np.all(np.isfinite(t_act)):
            raise ValueError("template_fd on active bins contains non-finite values (NaN or Inf)")

        # Fast active residual evaluation
        r_act = self._data_active - t_act
        r_power = (r_act.real ** 2) + (r_act.imag ** 2)
        quad_form = 4.0 * self._delta_f * float(np.sum(r_power * self._inv_psd_active))
        return float(self._log_norm - 0.5 * quad_form)

    def evaluate_detailed(self, template_fd: np.ndarray) -> GaussianLikelihoodResult:
        """Evaluate likelihood and return detailed constituent diagnostics.

        Parameters
        ----------
        template_fd : np.ndarray
            1D frequency-domain template array h~(f_k).

        Returns
        -------
        GaussianLikelihoodResult
            Dataclass holding log_likelihood, log_likelihood_norm, quadratic_form, and n_active_bins.
        """
        t = np.asarray(template_fd, dtype=np.complex128)
        if t.ndim != 1:
            raise ValueError(f"template_fd must be 1D, got ndim={t.ndim}")

        if t.size == self._grid.n_frequencies:
            t_act = t[self._active_indices]
        elif t.size == self._active_indices.size:
            t_act = t
        else:
            raise ValueError(
                f"template_fd size ({t.size}) matches neither full grid ({self._grid.n_frequencies}) "
                f"nor active bins ({self._active_indices.size})"
            )

        if not np.all(np.isfinite(t_act)):
            raise ValueError("template_fd on active bins contains non-finite values (NaN or Inf)")

        r_act = self._data_active - t_act
        r_power = (r_act.real ** 2) + (r_act.imag ** 2)
        quad_form = 4.0 * self._delta_f * float(np.sum(r_power * self._inv_psd_active))
        log_like = float(self._log_norm - 0.5 * quad_form)

        return GaussianLikelihoodResult(
            log_likelihood=log_like,
            log_likelihood_norm=self._log_norm,
            quadratic_form=float(max(0.0, quad_form)),
            n_active_bins=int(self._active_indices.size),
        )

    def __call__(self, template_fd: np.ndarray) -> float:
        """Callable interface for seamless sampler / MCMC integration."""
        return self.evaluate(template_fd)


def _validate_active_indices_against_grid(
    indices: np.ndarray,
    grid: FrequencyGrid,
) -> None:
    """Validate that active indices strictly exclude DC and Nyquist."""
    if indices.ndim != 1:
        raise ValueError(f"active_indices must be 1D, got ndim={indices.ndim}")
    if indices.size == 0:
        raise ValueError("active_indices cannot be empty")
    if np.any(indices < 0) or np.any(indices >= grid.n_frequencies):
        raise ValueError(f"active_indices out of bounds for grid with {grid.n_frequencies} frequency bins")

    # Strict DC exclusion: index 0 is 0.0 Hz
    if 0 in indices:
        raise ValueError("active_indices must strictly exclude DC (bin index 0)")

    # Strict Nyquist exclusion for even N: index n_frequencies - 1 = N / 2
    if grid.n_samples % 2 == 0:
        nyquist_idx = grid.n_frequencies - 1
        if nyquist_idx in indices:
            raise ValueError(f"active_indices must strictly exclude Nyquist bin ({nyquist_idx}) for even N = {grid.n_samples}")
