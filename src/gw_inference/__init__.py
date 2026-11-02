"""GW-Inference: Robust Matched Filtering and Bayesian Parameter Estimation.

Stage 1.1: Discrete Fourier Conventions, Scaling, and Synthetic Time-Series Generation.
Stage 1.2: Welch Power Spectral Density Estimation.
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
from gw_inference.psd import (
    extract_segments,
    median_bias_factor,
    single_segment_psd,
    tukey_window,
    welch_psd,
    window_coherent_gain,
    window_overlap_correlation,
    window_power,
)

__all__ = [
    "build_frequency_grid",
    "continuous_rfft",
    "discrete_parseval_freq_energy",
    "discrete_parseval_time_energy",
    "expected_white_noise_psd",
    "one_sided_periodogram",
    "validate_parseval_identity",
    "generate_white_noise",
    "tukey_window",
    "window_power",
    "window_coherent_gain",
    "window_overlap_correlation",
    "extract_segments",
    "single_segment_psd",
    "median_bias_factor",
    "welch_psd",
]
