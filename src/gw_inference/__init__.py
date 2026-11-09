"""GW-Inference: Robust Matched Filtering and Bayesian Parameter Estimation.

Stage 1.1: Discrete Fourier Conventions, Scaling, and Synthetic Time-Series Generation.
Stage 1.2: Welch Power Spectral Density Estimation.
Stage 1.4A: Foundation, Frequency Grid, PSD Engine, and TaylorF2 Waveforms.
"""

from gw_inference.constants import (
    C_ISCO_SI,
    C_SI,
    GAMMA_E,
    G_SI,
    M_SUN_SI,
    MPC_SI,
)
from gw_inference.conventions import (
    build_frequency_grid,
    compute_parseval_error,
    continuous_rfft,
    direct_dft,
    discrete_parseval_freq_energy,
    discrete_parseval_time_energy,
    expected_white_noise_psd,
    one_sided_periodogram,
    validate_parseval_identity,
)
from gw_inference.grid import FrequencyGrid
from gw_inference.noise import generate_white_noise
from gw_inference.psd import (
    PSDEngine,
    discrete_pseudo_covariance_ratios,
    extract_segments,
    low_frequency_psd_continuation,
    median_bias_factor,
    single_segment_psd,
    tukey_window,
    welch_psd,
    welch_variance_inflation,
    window_coherent_gain,
    window_multi_lag_correlations,
    window_overlap_correlation,
    window_power,
)
from gw_inference.filter import (
    MatchedFilterResult,
    compute_chisq_binned,
    compute_reweighted_snr,
    compute_signed_lags,
    compute_template_norm,
    direct_matched_filter_at_lag,
    matched_filter,
    matched_filter_core,
    noise_weighted_inner_product,
)
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    component_masses_to_chirp_mass,
    effective_termination_frequency,
    newtonian_chirp_time,
    schwarzschild_isco_frequency,
    taylorf2_35pn_chirp_time,
    taylorf2_35pn_phase,
    taylorf2_amplitude_prefactor,
    taylorf2_fd_waveform,
)

__all__ = [
    # Constants
    "C_SI",
    "G_SI",
    "M_SUN_SI",
    "MPC_SI",
    "GAMMA_E",
    "C_ISCO_SI",
    # Conventions (WP0)
    "build_frequency_grid",
    "continuous_rfft",
    "direct_dft",
    "discrete_parseval_freq_energy",
    "discrete_parseval_time_energy",
    "compute_parseval_error",
    "expected_white_noise_psd",
    "one_sided_periodogram",
    "validate_parseval_identity",
    # Noise
    "generate_white_noise",
    # Grid & Mask (WP1)
    "FrequencyGrid",
    # PSD & Engine (WP1)
    "PSDEngine",
    "low_frequency_psd_continuation",
    "tukey_window",
    "window_power",
    "window_coherent_gain",
    "window_overlap_correlation",
    "window_multi_lag_correlations",
    "welch_variance_inflation",
    "discrete_pseudo_covariance_ratios",
    "extract_segments",
    "single_segment_psd",
    "median_bias_factor",
    "welch_psd",
    # Waveform (WP2)
    "component_masses_to_chirp_mass",
    "schwarzschild_isco_frequency",
    "effective_termination_frequency",
    "taylorf2_amplitude_prefactor",
    "taylorf2_35pn_phase",
    "taylorf2_fd_waveform",
    "apply_coalescence_time_shift",
    "newtonian_chirp_time",
    "taylorf2_35pn_chirp_time",
    # Matched Filtering & Detection (WP3)
    "MatchedFilterResult",
    "compute_chisq_binned",
    "compute_reweighted_snr",
    "compute_signed_lags",
    "compute_template_norm",
    "direct_matched_filter_at_lag",
    "matched_filter",
    "matched_filter_core",
    "noise_weighted_inner_product",
]
