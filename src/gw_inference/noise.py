"""Synthetic noise generation module for gravitational-wave data analysis.

This module provides reproducible generation of zero-mean discrete Gaussian
white noise using modern NumPy Generator instances (`numpy.random.default_rng`).

Statistical Assumptions:
-------------------------
1. Discrete real Gaussian white noise:
   n_j \\sim \\mathcal{N}(0, \\sigma^2),  \\quad j = 0, 1, \\dots, N-1

2. Independence across samples (delta-correlated):
   \\mathbb{E}[n_j] = 0,  \\quad \\mathbb{E}[n_j n_l] = \\sigma^2 \\delta_{j, l}

3. Flat power spectral density:
   - Two-sided discrete PSD: S_n^{(2)}(f) = \\sigma^2 \\Delta t = \\sigma^2 / f_s
   - One-sided discrete PSD: S_n(f) = 2 \\sigma^2 \\Delta t = 2 \\sigma^2 / f_s  (for f > 0)
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np


def generate_white_noise(
    n_samples: int,
    sigma: float,
    seed: Optional[Union[int, np.random.SeedSequence]] = None,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate reproducible zero-mean Gaussian white noise.

    Parameters
    ----------
    n_samples : int
        Number of discrete time-domain samples to generate. Must be a positive integer.
    sigma : float
        Standard deviation of the noise in physical signal units (e.g. strain).
        Must be a non-negative finite float.
    seed : int or np.random.SeedSequence, optional
        Seed for numpy.random.default_rng. Used only if `generator` is None.
    generator : np.random.Generator, optional
        Pre-existing NumPy Generator instance. If provided, `seed` is ignored.

    Returns
    -------
    np.ndarray
        1D float64 array of length `n_samples` containing the noise realization.

    Raises
    ------
    ValueError
        If `n_samples` is not a positive integer, or if `sigma` is negative or non-finite.
    TypeError
        If `generator` is not an instance of `np.random.Generator`.

    Notes
    -----
    This function avoids global random state by operating strictly through
    `numpy.random.Generator` instances.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

    if not isinstance(sigma, (int, float, np.floating)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be a non-negative finite float, got {sigma}")

    if generator is not None:
        if not isinstance(generator, np.random.Generator):
            raise TypeError(
                f"generator must be an instance of np.random.Generator, got {type(generator)}"
            )
        rng = generator
    else:
        rng = np.random.default_rng(seed)

    if sigma == 0.0:
        return np.zeros(n_samples, dtype=np.float64)

    return rng.normal(loc=0.0, scale=float(sigma), size=int(n_samples)).astype(np.float64)
