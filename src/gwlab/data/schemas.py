"""Data schemas and immutable validated data models for GWLab."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from gwlab.data.provenance import ProvenanceRecord


@dataclass(frozen=True)
class ParameterSummary:
    """Summary statistics for a single physical inference parameter."""
    name: str
    unit: str
    median: float
    p5: Optional[float] = None
    p95: Optional[float] = None
    ci90_width: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None


@dataclass(frozen=True)
class RunResult:
    """Validated data model for a Bayesian parameter estimation result artifact."""
    run_id: str
    target: str
    status: str
    log_evidence: float
    log_evidence_err: float
    n_evaluations: int
    guard_evaluations_completed: Optional[int]
    guard_evaluations_started: Optional[int]
    wall_time_s: float
    cpu_time_s: float
    timestamp: float
    parameters: Dict[str, ParameterSummary]
    samples_npz: str
    provenance: ProvenanceRecord

    @property
    def authoritative_call_count(self) -> int:
        """Return the guard-reconciled likelihood evaluations if available."""
        if self.guard_evaluations_completed is not None:
            return self.guard_evaluations_completed
        return self.n_evaluations


@dataclass(frozen=True)
class AggregateLedger:
    """Validated data model for campaign-level computational resource tracking."""
    cumulative_calls: int
    cumulative_cpu_s: float
    cumulative_wall_s: float
    remaining_calls: int
    remaining_cpu_s: float
    remaining_wall_s: float
    runs: List[Dict[str, Any]]
    run_b_gate: Dict[str, Any]
    provenance: ProvenanceRecord

    @property
    def total_budget_calls(self) -> int:
        """Global project cap on likelihood evaluations."""
        return self.cumulative_calls + self.remaining_calls

    @property
    def total_budget_cpu_s(self) -> float:
        """Global project cap on CPU seconds."""
        return self.cumulative_cpu_s + self.remaining_cpu_s

    @property
    def total_budget_wall_s(self) -> float:
        """Global project cap on wall seconds."""
        return self.cumulative_wall_s + self.remaining_wall_s


@dataclass(frozen=True)
class PosteriorSamples:
    """Validated array container for posterior parameter samples and weights."""
    run_id: str
    samples: np.ndarray  # Shape: (N, 4), dtype: float64
    weights: np.ndarray  # Shape: (N,), dtype: float64, sum == 1.0
    posterior_samples: Optional[np.ndarray]  # Resampled points if present
    parameter_names: List[str]  # ['mc', 'eta', 'tc', 'phic']
    parameter_units: Dict[str, str]
    provenance: ProvenanceRecord

    @property
    def n_samples(self) -> int:
        """Number of saved nested sampling points."""
        return int(self.samples.shape[0])

    @property
    def n_params(self) -> int:
        """Dimensionality of the physical inference space."""
        return int(self.samples.shape[1])


@dataclass(frozen=True)
class SeedComparisonResult:
    """Validated comparison model between two independent-PRNG-seed runs."""
    run_1_id: str
    seed_1: int
    logz_1: float
    logzerr_1: float
    n_eval_1: int
    run_2_id: str
    seed_2: int
    logz_2: float
    logzerr_2: float
    n_eval_2: int
    delta_logz: float
    sigma_delta: float
    z_discrepancy: float
    qualification: str
    provenance: ProvenanceRecord
