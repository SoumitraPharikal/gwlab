"""GW-Inference: Robust Matched Filtering and Bayesian Parameter Estimation.

Stage 1.1: Discrete Fourier Conventions, Scaling, and Synthetic Time-Series Generation.
"""

from gw_inference.conventions import (
    build_frequency_grid,
    continuous_rfft,
    discrete_parseval_freq_energy,
    discrete_parseval_time_energy,
    expected_white_noise_psd,
    one_sided_periodogram,
    validate_parseval_identity,
)
from gw_inference.noise import generate_white_noise

__all__ = [
    "build_frequency_grid",
    "continuous_rfft",
    "discrete_parseval_freq_energy",
    "discrete_parseval_time_energy",
    "expected_white_noise_psd",
    "one_sided_periodogram",
    "validate_parseval_identity",
    "generate_white_noise",
]
