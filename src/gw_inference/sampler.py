"""Nested sampling adapter architecture for Bayesian parameter estimation.

This module implements a thin, modular nested-sampling adapter wrapping the
community-standard `dynesty` library (Speagle 2020), conforming to the approved
WP6 specification:
- Accepts arbitrary likelihood callables (e.g. GravitationalWaveLikelihoodWrapper)
- Accepts analytic prior transformations from the unit hypercube
- Preserves the full normalized likelihood ln L_norm for exact Bayesian evidence ln(Z)
- Enforces strict input validation and configuration bounds
- Handles the optional `dynesty` dependency gracefully with clear diagnostic errors
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np


def is_dynesty_available() -> bool:
    """Check whether the optional `dynesty` package is installed and importable.

    Returns
    -------
    bool
        True if `dynesty` can be imported, False otherwise.
    """
    try:
        import dynesty  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class NestedSamplingResult:
    """Immutable container for nested-sampling evaluation results.

    Attributes
    ----------
    samples : np.ndarray
        Array of dead-point parameter samples of shape (n_samples, n_dim).
    log_weights : np.ndarray
        Array of log statistical weights for each sample of shape (n_samples,).
    weights : np.ndarray
        Array of normalized posterior weights summing to 1.0 of shape (n_samples,).
    log_evidence : float
        Estimated log-evidence ln(Z), preserving full likelihood normalization.
    log_evidence_err : float
        Estimated statistical uncertainty Delta ln(Z) on the log-evidence.
    n_evaluations : int
        Total number of likelihood function evaluations performed.
    param_names : Tuple[str, ...]
        Ordered names of the inferred parameters.
    posterior_samples : np.ndarray
        Equally-weighted posterior samples of shape (n_posterior_samples, n_dim).
    raw_results : Any
        Raw results object or dictionary returned by dynesty.
    """

    samples: np.ndarray
    log_weights: np.ndarray
    weights: np.ndarray
    log_evidence: float
    log_evidence_err: float
    n_evaluations: int
    param_names: Tuple[str, ...]
    posterior_samples: np.ndarray
    raw_results: Any

    def summary(self) -> dict[str, Any]:
        """Return a human-readable dictionary summary of key inference results."""
        param_summaries = {}
        for i, name in enumerate(self.param_names):
            vals = self.posterior_samples[:, i]
            median = float(np.median(vals))
            p5 = float(np.percentile(vals, 5.0))
            p95 = float(np.percentile(vals, 95.0))
            param_summaries[name] = {
                "median": median,
                "p5": p5,
                "p95": p95,
                "ci90_width": p95 - p5,
            }

        return {
            "log_evidence": self.log_evidence,
            "log_evidence_err": self.log_evidence_err,
            "n_evaluations": self.n_evaluations,
            "n_dead_points": len(self.samples),
            "n_posterior_samples": len(self.posterior_samples),
            "parameters": param_summaries,
        }


def extract_total_evaluations(res: Any, n_samples: Optional[int] = None) -> int:
    """Extract total likelihood evaluation count from dynesty results.

    In the dynesty Results object, `res.ncall` can be:
    - A 1D NumPy array or sequence of per-iteration call counts, where the
      total number of likelihood evaluations is the sum: N_call,total = sum_i N_call,i.
    - A non-negative integer or float scalar representing the total call count directly.

    Parameters
    ----------
    res : Any
        Results object or dictionary returned by dynesty or a test mock.
    n_samples : int, optional
        Number of dead-point samples, used for physical consistency validation.

    Returns
    -------
    int
        Total number of likelihood function evaluations N_call,total >= 0.

    Raises
    ------
    ValueError
        If `ncall` is missing, empty, non-finite, negative, fractional,
        has invalid shape, or is physically inconsistent with the number of samples.
    TypeError
        If `ncall` is complex-valued, non-numeric, or has an unsupported dtype.
    """
    ncall_raw = getattr(res, "ncall", None)
    if ncall_raw is None and isinstance(res, dict):
        ncall_raw = res.get("ncall", None)

    if ncall_raw is None:
        raise ValueError(
            "dynesty results object is missing 'ncall'; cannot determine total likelihood evaluations. "
            "A valid dynesty run must record likelihood call counts."
        )

    # Reject complex scalars immediately
    if isinstance(ncall_raw, complex) or np.iscomplexobj(ncall_raw):
        raise TypeError(f"ncall cannot be complex-valued, got {type(ncall_raw)}")

    # Handle numeric scalars (int, float, numpy scalar)
    if isinstance(ncall_raw, (int, np.integer)):
        val = int(ncall_raw)
        if val < 0:
            raise ValueError(f"Scalar ncall must be non-negative, got {val}")
        if n_samples is not None and n_samples > 0 and val < n_samples:
            raise ValueError(
                f"Scalar ncall ({val}) cannot be less than number of dead points ({n_samples})"
            )
        return val

    if isinstance(ncall_raw, (float, np.floating)):
        if not math.isfinite(ncall_raw):
            raise ValueError(f"Scalar ncall must be finite, got {ncall_raw}")
        val = float(ncall_raw)
        if val < 0:
            raise ValueError(f"Scalar ncall must be non-negative, got {val}")
        if not val.is_integer():
            raise ValueError(f"Scalar ncall must represent an integer count, got {val}")
        int_val = int(val)
        if n_samples is not None and n_samples > 0 and int_val < n_samples:
            raise ValueError(
                f"Scalar ncall ({int_val}) cannot be less than number of dead points ({n_samples})"
            )
        return int_val

    # Handle array-like or sequence-like objects
    try:
        arr = np.asarray(ncall_raw)
    except Exception as e:
        raise TypeError(f"Could not convert ncall to array: {e}") from e

    # Check for complex dtypes
    if np.iscomplexobj(arr) or np.issubdtype(arr.dtype, np.complexfloating):
        raise TypeError(f"ncall array cannot be complex-valued, got dtype {arr.dtype}")

    # Check for 0D array (e.g. np.array(420))
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, complex):
            raise TypeError(f"0D ncall array cannot be complex-valued: {item}")
        if isinstance(item, (int, float, np.integer, np.floating)):
            class _ScalarMock:
                ncall = item
            return extract_total_evaluations(_ScalarMock(), n_samples=n_samples)
        raise TypeError(f"0D ncall array has unsupported dtype or value: {item}")

    if arr.ndim != 1:
        raise ValueError(f"ncall array must be 1D, got ndim={arr.ndim} with shape {arr.shape}")

    if arr.size == 0:
        raise ValueError("ncall array is empty; cannot determine total likelihood evaluations")

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"ncall array must have numeric real dtype, got {arr.dtype}")

    if not np.all(np.isfinite(arr)):
        raise ValueError("ncall array contains non-finite values (NaN or Inf)")

    if np.any(arr < 0):
        raise ValueError("ncall array contains negative evaluation counts")

    # Reject fractional / non-integer values
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.equal(arr, np.floor(arr))):
            raise ValueError("ncall array contains fractional (non-integer) evaluation counts")

    # Accumulate safely using arbitrary-precision Python integers to avoid fixed-width overflow
    total_calls = sum(int(x) for x in arr)

    if n_samples is not None and n_samples > 0 and total_calls < n_samples:
        raise ValueError(
            f"Total likelihood calls ({total_calls}) cannot be less than number of dead points ({n_samples})"
        )

    return total_calls


class NestedSamplerAdapter:
    """Thin adapter interfacing likelihoods and prior transforms to dynesty.

    Parameters
    ----------
    log_likelihood_fn : Callable[[np.ndarray], float]
        Callable evaluating the full log-likelihood for parameter vector theta.
    prior_transform_fn : Callable[[np.ndarray], np.ndarray]
        Callable mapping unit hypercube u in [0, 1]^ndim to physical parameters theta.
    ndim : int, default=2
        Dimensionality of the parameter space. Must be >= 1.
    nlive : int, default=500
        Number of active live points. Must be >= 1.
    dlogz : float, default=0.1
        Stopping tolerance on remaining live-point evidence Delta ln(Z). Must be > 0.
    rstate : Any, optional
        NumPy Generator or RandomState instance for reproducible sampling.
    seed : int, optional
        Integer seed to initialize a NumPy default_rng if rstate is not provided.
    param_names : Sequence[str], optional
        Labels for inferred parameters matching coordinate ordering.
    """

    def __init__(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        prior_transform_fn: Callable[[np.ndarray], np.ndarray],
        ndim: int = 2,
        nlive: int = 500,
        dlogz: float = 0.1,
        rstate: Optional[Any] = None,
        seed: Optional[int] = None,
        param_names: Optional[Sequence[str]] = None,
    ) -> None:
        if not callable(log_likelihood_fn):
            raise TypeError(f"log_likelihood_fn must be callable, got {type(log_likelihood_fn)}")
        if not callable(prior_transform_fn):
            raise TypeError(f"prior_transform_fn must be callable, got {type(prior_transform_fn)}")

        if not isinstance(ndim, (int, np.integer)) or ndim < 1:
            raise ValueError(f"ndim must be an integer >= 1, got {ndim}")
        if not isinstance(nlive, (int, np.integer)) or nlive < 1:
            raise ValueError(f"nlive must be an integer >= 1, got {nlive}")
        if not isinstance(dlogz, (int, float, np.floating)) or not math.isfinite(dlogz) or dlogz <= 0.0:
            raise ValueError(f"dlogz must be a positive finite float, got {dlogz}")

        self._log_likelihood_fn = log_likelihood_fn
        self._prior_transform_fn = prior_transform_fn
        self._ndim = int(ndim)
        self._nlive = int(nlive)
        self._dlogz = float(dlogz)
        self._rstate = rstate
        self._seed = int(seed) if seed is not None else None

        if param_names is not None:
            if len(param_names) != self._ndim:
                raise ValueError(
                    f"param_names length ({len(param_names)}) must match ndim ({self._ndim})"
                )
            self._param_names = tuple(str(p) for p in param_names)
        else:
            self._param_names = tuple(f"theta_{i}" for i in range(self._ndim))

    @property
    def ndim(self) -> int:
        """Return dimensionality of the parameter space."""
        return self._ndim

    @property
    def nlive(self) -> int:
        """Return number of live points."""
        return self._nlive

    @property
    def dlogz(self) -> float:
        """Return evidence termination tolerance."""
        return self._dlogz

    @property
    def param_names(self) -> Tuple[str, ...]:
        """Return parameter names in coordinate ordering."""
        return self._param_names

    def run(self, **dynesty_kwargs: Any) -> NestedSamplingResult:
        """Execute nested sampling and return structured inference results.

        Parameters
        ----------
        **dynesty_kwargs : Any
            Additional keyword arguments forwarded directly to dynesty.NestedSampler
            or run_nested (e.g. bound, sample, walks, maxiter, print_progress).

        Returns
        -------
        NestedSamplingResult
            Immutable container holding dead points, weights, evidence, and posterior samples.

        Raises
        ------
        ImportError
            If `dynesty` is not installed in the environment.
        """
        if not is_dynesty_available():
            raise ImportError(
                "dynesty is required to run nested sampling. "
                "Install it as an optional dependency via `pip install '.[sampling]'` "
                "or `pip install dynesty>=2.1.0`."
            )

        import dynesty
        from dynesty import NestedSampler

        # Resolve RNG state for reproducibility
        if self._rstate is not None:
            rstate = self._rstate
        elif self._seed is not None:
            rstate = np.random.default_rng(self._seed)
        else:
            rstate = None

        # Filter kwargs between constructor and run_nested
        constructor_keys = {
            "bound", "sample", "periodic", "reflective", "update_interval",
            "first_update", "npdim", "live_points", "logl_args", "logl_kwargs",
            "ptform_args", "ptform_kwargs", "enlarge", "bootstrap", "walks",
            "facc", "slices", "fmove", "max_move",
        }
        init_kwargs = {k: v for k, v in dynesty_kwargs.items() if k in constructor_keys}
        run_kwargs = {k: v for k, v in dynesty_kwargs.items() if k not in constructor_keys}

        # Initialize dynesty NestedSampler
        sampler = NestedSampler(
            loglikelihood=self._log_likelihood_fn,
            prior_transform=self._prior_transform_fn,
            ndim=self._ndim,
            nlive=self._nlive,
            rstate=rstate,
            **init_kwargs,
        )

        # Execute nested sampling
        sampler.run_nested(dlogz=self._dlogz, **run_kwargs)
        res = sampler.results

        # Extract and unpack results
        if not hasattr(res, "samples") or not hasattr(res, "logwt") or not hasattr(res, "logz") or not hasattr(res, "logzerr"):
            raise ValueError("dynesty results object must contain 'samples', 'logwt', 'logz', and 'logzerr'.")

        try:
            samples = np.asarray(res.samples, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Could not convert res.samples to float64 array: {e}") from e

        try:
            logwt = np.asarray(res.logwt, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Could not convert res.logwt to float64 array: {e}") from e

        # Validate non-empty dimensions and matching sample counts
        if samples.ndim != 2:
            raise ValueError(f"res.samples must be a 2D array, got ndim={samples.ndim} with shape {samples.shape}")
        if samples.shape[1] != self._ndim:
            raise ValueError(
                f"res.samples second dimension ({samples.shape[1]}) must match adapter ndim ({self._ndim})"
            )
        if len(samples) == 0:
            raise ValueError("res.samples is empty; nested sampling must produce at least one sample")

        if logwt.ndim != 1:
            raise ValueError(f"res.logwt must be a 1D array, got ndim={logwt.ndim} with shape {logwt.shape}")
        if len(logwt) == 0:
            raise ValueError("res.logwt is empty; nested sampling must produce at least one weight")

        if len(samples) != len(logwt):
            raise ValueError(
                f"Sample count mismatch: len(samples)={len(samples)} != len(logwt)={len(logwt)}"
            )

        n_samples = len(samples)

        # Validate logz and logzerr
        try:
            logz_arr = np.asarray(res.logz)
            if logz_arr.size == 0:
                raise ValueError("res.logz is empty")
            logz = float(logz_arr[-1])
        except Exception as e:
            raise ValueError(f"Could not extract logz from dynesty results: {e}") from e

        if not math.isfinite(logz):
            raise ValueError(f"Evidence logz must be finite, got {logz}")

        try:
            logzerr_arr = np.asarray(res.logzerr)
            if logzerr_arr.size == 0:
                raise ValueError("res.logzerr is empty")
            logzerr = float(logzerr_arr[-1])
        except Exception as e:
            raise ValueError(f"Could not extract logzerr from dynesty results: {e}") from e

        if not math.isfinite(logzerr) or logzerr < 0.0:
            raise ValueError(f"Evidence error logzerr must be finite and non-negative, got {logzerr}")

        # Extract likelihood call count
        ncall = extract_total_evaluations(res, n_samples=n_samples)

        # Validate logwt entries:
        # Reject NaN and positive-infinite logwt entries
        if np.any(np.isnan(logwt)):
            raise ValueError("res.logwt contains NaN values")
        if np.any(np.isposinf(logwt)):
            raise ValueError("res.logwt contains positive-infinite (+inf) values")

        # Permit negative-infinite individual logwt entries (-inf corresponds to 0 weight),
        # but require at least one finite log weight
        finite_mask = np.isfinite(logwt)
        if not np.any(finite_mask):
            raise ValueError(
                "All logwt entries are negative-infinite (-inf); cannot compute posterior weights."
            )

        # Stable max-shift normalization across finite weights:
        # Mathematically, unnormalized dead-point weights are w_i = exp(logwt_i - logz).
        # We subtract max(logwt - logz) over finite entries so the maximum exponent is 0.0.
        rel_log_weights = logwt - logz
        max_rel = float(np.max(rel_log_weights[finite_mask]))

        # Initialize weights with zeros (so -inf logwt entries naturally remain 0.0)
        shifted_weights = np.zeros(n_samples, dtype=np.float64)
        shifted_weights[finite_mask] = np.exp(rel_log_weights[finite_mask] - max_rel)

        sum_w = float(np.sum(shifted_weights))
        if not (math.isfinite(sum_w) and sum_w > 0.0):
            raise ValueError(
                f"Weight sum is non-finite or non-positive ({sum_w}); cannot normalize posterior weights."
            )

        weights = shifted_weights / sum_w

        # Verify final normalization
        sum_norm = float(np.sum(weights))
        if not math.isclose(sum_norm, 1.0, rel_tol=1e-12):
            raise ValueError(f"Normalized weights do not sum to 1.0 (sum={sum_norm})")

        # Resample equal-weight posterior samples
        posterior_samples = None
        try:
            from dynesty import utils as dyn_utils
            resampled = dyn_utils.resample_equal(samples, weights)
            if isinstance(resampled, np.ndarray) and resampled.ndim == samples.ndim:
                posterior_samples = np.array(resampled, dtype=np.float64, copy=True)
        except Exception:
            pass

        if posterior_samples is None:
            # Direct weighted selection with replacement
            rng = rstate if isinstance(rstate, np.random.Generator) else np.random.default_rng(self._seed)
            n_resample = len(samples)
            idx = rng.choice(len(samples), size=n_resample, p=weights)
            posterior_samples = np.array(samples[idx], dtype=np.float64, copy=True)

        return NestedSamplingResult(
            samples=samples,
            log_weights=logwt,
            weights=weights,
            log_evidence=logz,
            log_evidence_err=logzerr,
            n_evaluations=ncall,
            param_names=self._param_names,
            posterior_samples=posterior_samples,
            raw_results=res,
        )
