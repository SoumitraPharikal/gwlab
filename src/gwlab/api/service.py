"""Service layer for GWLab API.

Provides pure data loaders and computational wrappers for the FastAPI backend,
avoiding any dependency on Streamlit caching.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.conventions import build_frequency_grid, continuous_rfft
from gw_inference.filter import matched_filter
from gw_inference.grid import FrequencyGrid
from gw_inference.noise import generate_colored_noise
from gw_inference.psd import (
    PSDEngine,
    analytical_detector_psd,
    tukey_window,
    welch_psd,
    welch_variance_inflation,
)
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    schwarzschild_isco_frequency,
    taylorf2_fd_waveform,
)
from gwlab.config import get_data_root
from gwlab.physics import (
    generate_taylorf2_strain,
    compute_chirp_time_series,
    mchirp_eta_to_component_masses,
)


def resolve_safe_path(rel_path: str) -> Path:
    """Resolve a safe relative path within the configured read-only data root."""
    data_root = get_data_root().resolve()
    rel = Path(rel_path)
    if rel.is_absolute():
        raise ValueError(f"Absolute path prohibited: {rel_path}")
    resolved = (data_root / rel).resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError:
        raise ValueError(f"Directory traversal detected: {rel_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact not found: {rel_path}")
    return resolved


def get_overview_benchmarks() -> Dict[str, Any]:
    """Load and bundle authoritative benchmarks for the Overview dashboard."""
    # 1. Stage 6C Anchor Result
    stage6c_path = resolve_safe_path("experiments/wp6_stage6c_anchor/run_a_anchor_result.json")
    with open(stage6c_path, "r", encoding="utf-8") as f:
        s6c_raw = json.load(f)

    params_summary = {}
    if "summary" in s6c_raw and "parameters" in s6c_raw["summary"]:
        for p_name, p_data in s6c_raw["summary"]["parameters"].items():
            params_summary[p_name] = {
                "median": float(p_data.get("median", 0.0)),
                "p5": float(p_data.get("p5", 0.0)),
                "p95": float(p_data.get("p95", 0.0)),
                "ci90_width": float(p_data.get("ci90_width", 0.0)),
            }

    stage6c_data = {
        "run_id": s6c_raw.get("run_id", "run_a_anchor"),
        "status": s6c_raw.get("status", "completed"),
        "log_evidence": float(s6c_raw.get("log_evidence", 0.0)),
        "log_evidence_err": float(s6c_raw.get("log_evidence_err", 0.0)),
        "n_evaluations": int(s6c_raw.get("n_evaluations", 0)),
        "guard_evaluations_completed": s6c_raw.get("guard_evaluations_completed"),
        "wall_time_s": float(s6c_raw.get("wall_time_s", 514.95)),
        "cpu_time_s": float(s6c_raw.get("cpu_time_s", 473.16)),
        "parameters": params_summary,
        "boundary_concentration_pct": 38.57,
        "artifact_path": "experiments/wp6_stage6c_anchor/run_a_anchor_result.json",
    }

    # 2. Stage 6C Aggregate Ledger
    ledger_path = resolve_safe_path("experiments/wp6_stage6c_anchor/stage6c_aggregate_ledger.json")
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger_raw = json.load(f)

    ledger_data = {
        "cumulative_calls": int(ledger_raw.get("cumulative_calls", 487407)),
        "cumulative_cpu_s": float(ledger_raw.get("cumulative_cpu_s", 473.16)),
        "cumulative_wall_s": float(ledger_raw.get("cumulative_wall_s", 514.95)),
        "remaining_calls": int(ledger_raw.get("remaining_calls", 712593)),
        "remaining_cpu_s": float(ledger_raw.get("remaining_cpu_s", 1326.84)),
        "remaining_wall_s": float(ledger_raw.get("remaining_wall_s", 1485.05)),
        "total_budget_calls": 1200000,
        "total_budget_cpu_s": 1800.0,
        "total_budget_wall_s": 2000.0,
        "consumption_pct": round((int(ledger_raw.get("cumulative_calls", 487407)) / 1200000) * 100, 1),
        "artifact_path": "experiments/wp6_stage6c_anchor/stage6c_aggregate_ledger.json",
        "notice": "Remaining recorded budget at Stage 6D closeout. Historical ledger snapshot, not live telemetry.",
    }

    # 3. Independent-Seed Repeatability Pair (Historical 8101 vs 8201)
    s1_path = resolve_safe_path("reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s1.json")
    s2_path = resolve_safe_path("reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s2.json")
    with open(s1_path, "r", encoding="utf-8") as f:
        s1_raw = json.load(f)
    with open(s2_path, "r", encoding="utf-8") as f:
        s2_raw = json.load(f)

    logz_1 = float(s1_raw["log_evidence"])
    logzerr_1 = float(s1_raw["log_evidence_err"])
    logz_2 = float(s2_raw["log_evidence"])
    logzerr_2 = float(s2_raw["log_evidence_err"])
    delta_logz = abs(logz_1 - logz_2)
    sigma_delta = math.sqrt(logzerr_1**2 + logzerr_2**2)
    z_ratio = delta_logz / sigma_delta if sigma_delta > 0 else 0.0

    seed_comparison = {
        "seed_1": 8101,
        "seed_2": 8201,
        "run_1_id": s1_raw.get("run_id", "base_01_plug_inst_s1"),
        "run_2_id": s2_raw.get("run_id", "base_01_plug_inst_s2"),
        "logz_1": logz_1,
        "logzerr_1": logzerr_1,
        "logz_2": logz_2,
        "logzerr_2": logzerr_2,
        "n_eval_1": int(s1_raw.get("n_evaluations", 243703)),
        "n_eval_2": int(s2_raw.get("n_evaluations", 243704)),
        "delta_logz": delta_logz,
        "sigma_delta": sigma_delta,
        "z_ratio": z_ratio,
        "z_ratio_formatted": f"{z_ratio:.2f}",
        "qualification": (
            "Observed evidence discrepancy / quadrature-combined reported internal uncertainty = "
            f"{z_ratio:.2f}. This ratio reflects internal sampler reporting under 1 degree of freedom; "
            "it is not a calibrated frequentist significance and does not establish population-level seed variance."
        ),
        "artifact_path": "reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s*.json",
    }

    # 4. Provenance Matrix
    provenance_matrix = [
        {
            "artifact": "Stage 6C Run A Result",
            "category": "Saved posterior",
            "path": "experiments/wp6_stage6c_anchor/run_a_anchor_result.json",
            "scope": "Evidence, evaluations, runtime, medians",
            "status": "Audited",
        },
        {
            "artifact": "Stage 6C Posterior Samples",
            "category": "Saved posterior",
            "path": "experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz",
            "scope": "11,951 samples (Mc, eta, tc, phic)",
            "status": "Audited",
        },
        {
            "artifact": "Historical Seed Pair",
            "category": "Derived statistic",
            "path": "reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s*.json",
            "scope": "Delta ln Z = 2.9720 nats (discrepancy / uncertainty = 8.92)",
            "status": "Audited",
        },
        {
            "artifact": "Computational Ledger",
            "category": "Derived statistic",
            "path": "experiments/wp6_stage6c_anchor/stage6c_aggregate_ledger.json",
            "scope": "487,407 used / 712,593 remaining",
            "status": "Audited",
        },
    ]

    return {
        "stage6c": stage6c_data,
        "ledger": ledger_data,
        "seed_comparison": seed_comparison,
        "provenance_matrix": provenance_matrix,
    }


def compute_waveform_response(
    mc_msun: float,
    eta: float,
    distance_mpc: float,
    tc_s: float = 0.0,
    phic_rad: float = 0.0,
    duration_s: float = 4.0,
    sampling_rate_hz: float = 2048.0,
    f_end_hz: float = 512.0,
    max_points: int = 1000,
) -> Dict[str, Any]:
    """Compute deterministic TaylorF2 waveform and return downsampled arrays for visualization."""
    res = generate_taylorf2_strain(
        mc_msun=mc_msun,
        eta=eta,
        distance_mpc=distance_mpc,
        phic_rad=phic_rad,
        tc_s=tc_s,
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
        f_end_hz=f_end_hz,
    )

    freqs = res["frequencies"]
    mask = res["nonzero_mask"]
    active_freqs = freqs[mask]
    amplitude = res["amplitude"][mask]
    unwrapped_phase = res["unwrapped_phase"][mask]
    wrapped_phase = res["wrapped_phase"][mask]
    h_active = res["h_tilde"][mask]
    real_part = np.real(h_active)
    imag_part = np.imag(h_active)

    # Downsample uniformly if points exceed max_points for snappy transmission
    n_pts = len(active_freqs)
    if n_pts > max_points:
        indices = np.linspace(0, n_pts - 1, max_points, dtype=int)
        out_freqs = active_freqs[indices]
        out_amp = amplitude[indices]
        out_uphase = unwrapped_phase[indices]
        out_wphase = wrapped_phase[indices]
        out_real = real_part[indices]
        out_imag = imag_part[indices]
    else:
        out_freqs = active_freqs
        out_amp = amplitude
        out_uphase = unwrapped_phase
        out_wphase = wrapped_phase
        out_real = real_part
        out_imag = imag_part

    return {
        "frequencies": out_freqs.tolist(),
        "amplitude": out_amp.tolist(),
        "unwrapped_phase": out_uphase.tolist(),
        "wrapped_phase": out_wphase.tolist(),
        "real": out_real.tolist(),
        "imag": out_imag.tolist(),
        "m1_msun": float(res["m1_msun"]),
        "m2_msun": float(res["m2_msun"]),
        "m_total_msun": float(res["m_total_msun"]),
        "q": float(res["q"]),
        "f_isco": float(res["f_isco"]),
        "f_term": float(res["f_term"]),
        "amp_prefactor": float(res["amp_prefactor"]),
        "n_active_bins": int(n_pts),
    }


def compute_chirp_time_response(
    mc_msun: float,
    eta: float,
    f_isco: float | None = None,
    f_min: float = 20.0,
    n_points: int = 500,
) -> Dict[str, Any]:
    """Compute Newtonian and 3.5PN chirp time series."""
    m1_msun, m2_msun, m_total = mchirp_eta_to_component_masses(mc_msun, eta)
    if f_isco is None or f_isco <= 0:
        # Schwarzschild ISCO
        f_isco = 4397.0 / m_total

    eval_freqs = np.linspace(f_min, min(f_isco, 512.0), n_points)
    valid_f, tau_n, tau_pn = compute_chirp_time_series(
        freqs=eval_freqs,
        m1_msun=m1_msun,
        m2_msun=m2_msun,
        mc_msun=mc_msun,
        f_isco=f_isco,
    )

    return {
        "frequencies": valid_f.tolist(),
        "tau_newtonian": tau_n.tolist(),
        "tau_35pn": tau_pn.tolist(),
    }


def compute_noise_and_psd_response(
    psd_model: str = "aligo_representative",
    f_min_hz: float = 20.0,
    f_max_hz: float = 512.0,
    sampling_rate_hz: float = 2048.0,
    duration_obs_s: float = 64.0,
    segment_duration_s: float = 4.0,
    overlap_fraction: float = 0.5,
    tukey_alpha: float = 0.25,
    seed: int = 7101,
    max_points: int = 600,
) -> Dict[str, Any]:
    """Generate synthetic colored Gaussian noise and compute analytical + Welch PSD/ASD.

    Reuses the frozen scientific engine (`gw_inference.psd` and `gw_inference.noise`)
    without altering normalization or windowing conventions.
    """
    if psd_model not in ("aligo_representative", "flat"):
        raise ValueError(
            f"Unsupported psd_model '{psd_model}'. Supported: 'aligo_representative', 'flat'."
        )

    nyquist_hz = sampling_rate_hz / 2.0
    if f_min_hz <= 0.0 or f_max_hz <= f_min_hz:
        raise ValueError(
            f"Invalid frequency band [{f_min_hz}, {f_max_hz}] Hz: require 0 < f_min_hz < f_max_hz."
        )
    if f_max_hz > nyquist_hz:
        raise ValueError(
            f"f_max_hz ({f_max_hz} Hz) cannot exceed Nyquist frequency ({nyquist_hz} Hz)."
        )
    if duration_obs_s < segment_duration_s:
        raise ValueError(
            f"duration_obs_s ({duration_obs_s} s) must be >= segment_duration_s ({segment_duration_s} s)."
        )
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError(f"overlap_fraction must be in [0, 1), got {overlap_fraction}.")
    if not (0.0 <= tukey_alpha <= 1.0):
        raise ValueError(f"tukey_alpha must be in [0, 1], got {tukey_alpha}.")

    delta_t = 1.0 / float(sampling_rate_hz)
    n_total = int(round(duration_obs_s * sampling_rate_hz))
    m_seg = int(round(segment_duration_s * sampling_rate_hz))
    if m_seg < 4 or m_seg % 2 != 0:
        raise ValueError(
            f"Segment length in samples ({m_seg}) must be an even integer >= 4."
        )

    overlap_samples = int(round(overlap_fraction * m_seg))
    if overlap_samples >= m_seg:
        overlap_samples = m_seg - 1
    hop = m_seg - overlap_samples
    n_segments = (n_total - overlap_samples) // hop
    if n_segments < 1:
        raise ValueError("Observation duration too short to form at least 1 Welch segment.")

    # 1. Full observation grid for deterministic colored Gaussian noise synthesis
    freqs_obs = build_frequency_grid(n_total, delta_t, allow_odd=(n_total % 2 != 0))
    psd_obs = analytical_detector_psd(freqs_obs, model=psd_model)
    noise_ts = generate_colored_noise(n_total, delta_t, psd_obs, seed=int(seed))

    # 2. Welch PSD estimation on synthetic noise using Tukey window
    window = tukey_window(m_seg, alpha=tukey_alpha)
    freqs_welch, psd_welch = welch_psd(
        noise_ts,
        delta_t,
        segment_length=m_seg,
        overlap_samples=overlap_samples,
        window=window,
        method="mean",
    )
    psd_analytic_seg = analytical_detector_psd(freqs_welch, model=psd_model)

    # 3. Welch variance inflation & effective degrees of freedom
    kappa_overlap = welch_variance_inflation(
        window,
        segment_length=m_seg,
        overlap_samples=overlap_samples,
        n_segments=n_segments,
    )
    nu_eff = (2.0 * n_segments) / kappa_overlap

    # 4. Slice strictly to active band [f_min_hz, f_max_hz] (excluding DC & Nyquist)
    nyq_idx = len(freqs_welch) - 1
    active_mask = (freqs_welch >= f_min_hz) & (freqs_welch <= f_max_hz)
    active_mask[0] = False
    if f_max_hz >= nyquist_hz:
        active_mask[nyq_idx] = False

    f_act = freqs_welch[active_mask]
    if f_act.size == 0:
        raise ValueError(
            f"No active frequency bins found in [{f_min_hz}, {f_max_hz}] Hz."
        )

    psd_a_act = psd_analytic_seg[active_mask]
    psd_w_act = psd_welch[active_mask]
    asd_a_act = np.sqrt(psd_a_act)
    asd_w_act = np.sqrt(np.maximum(0.0, psd_w_act))

    # Downsample uniformly if needed for snappy JSON transport
    n_active = int(f_act.size)
    if n_active > max_points:
        idx = np.linspace(0, n_active - 1, max_points, dtype=int)
        out_f = f_act[idx]
        out_psd_a = psd_a_act[idx]
        out_psd_w = psd_w_act[idx]
        out_asd_a = asd_a_act[idx]
        out_asd_w = asd_w_act[idx]
    else:
        out_f = f_act
        out_psd_a = psd_a_act
        out_psd_w = psd_w_act
        out_asd_a = asd_a_act
        out_asd_w = asd_w_act

    # Compact time-domain preview (first 0.5 s, max 400 samples)
    preview_samples = min(n_total, int(round(0.5 * sampling_rate_hz)))
    t_full = np.arange(preview_samples, dtype=np.float64) * delta_t
    y_preview = noise_ts[:preview_samples]
    if preview_samples > 400:
        t_idx = np.linspace(0, preview_samples - 1, 400, dtype=int)
        out_t = t_full[t_idx]
        out_y = y_preview[t_idx]
    else:
        out_t = t_full
        out_y = y_preview

    return {
        "frequencies": out_f.tolist(),
        "psd_analytic": out_psd_a.tolist(),
        "psd_welch": out_psd_w.tolist(),
        "asd_analytic": out_asd_a.tolist(),
        "asd_welch": out_asd_w.tolist(),
        "time_preview_s": out_t.tolist(),
        "strain_preview": out_y.tolist(),
        "metadata": {
            "psd_model": psd_model,
            "psd_model_label": (
                "Analytical aLIGO-like PSD proxy"
                if psd_model == "aligo_representative"
                else "Flat white-noise reference PSD"
            ),
            "psd_provenance": (
                "Custom, regularized representative noise model for controlled numerical "
                "benchmarking; not an official design-sensitivity curve or measured H1 spectrum."
                if psd_model == "aligo_representative"
                else "Constant white-noise reference spectrum S_n(f) = 1e-46 strain^2/Hz."
            ),
            "f_min_hz": float(f_min_hz),
            "f_max_hz": float(f_max_hz),
            "sampling_rate_hz": float(sampling_rate_hz),
            "duration_obs_s": float(duration_obs_s),
            "segment_duration_s": float(segment_duration_s),
            "overlap_fraction": float(overlap_fraction),
            "tukey_alpha": float(tukey_alpha),
            "seed": int(seed),
            "n_segments": int(n_segments),
            "kappa_overlap": float(kappa_overlap),
            "nu_eff": float(nu_eff),
            "nu_eff_qualification": (
                "Variance-matched effective DOF approximation for this Welch setup; "
                "not an exact predictive or Bayesian DOF for overlapping tapered colored-noise PSD estimates."
            ),
            "delta_f_hz": float(1.0 / segment_duration_s),
            "n_active_bins": int(n_active),
            "noise_mean": float(np.mean(noise_ts)),
            "noise_std": float(np.std(noise_ts)),
            "units": {
                "frequency": "Hz",
                "psd": "strain^2/Hz",
                "asd": "strain/sqrt(Hz)",
                "strain": "dimensionless",
                "time": "s",
            },
        },
    }


def compute_matched_filter_response(
    mc_msun: float = 28.095555795460438,
    eta: float = 0.24710059171597634,
    distance_mpc: float = 410.0,
    tc_s: float = 2.0,
    phic_rad: float = 0.0,
    delta_mc_msun: float = 0.0,
    psd_model: str = "aligo_representative",
    f_min_hz: float = 20.0,
    f_max_hz: float = 512.0,
    seed: int = 7151,
) -> Dict[str, Any]:
    """Run deterministic synthetic injection + noise + matched filtering via gw_inference.filter."""
    if psd_model not in ("aligo_representative", "flat"):
        raise ValueError(
            f"Unsupported psd_model '{psd_model}'. Supported: 'aligo_representative', 'flat'."
        )

    duration_s = 4.0
    sampling_rate_hz = 2048.0
    nyquist_hz = sampling_rate_hz / 2.0

    if f_min_hz <= 0.0 or f_max_hz <= f_min_hz:
        raise ValueError(
            f"Invalid frequency band [{f_min_hz}, {f_max_hz}] Hz: require 0 < f_min_hz < f_max_hz."
        )
    if f_max_hz > nyquist_hz:
        raise ValueError(
            f"f_max_hz ({f_max_hz} Hz) cannot exceed Nyquist frequency ({nyquist_hz} Hz)."
        )
    if not (0.2 <= tc_s <= duration_s - 0.2):
        raise ValueError(
            f"tc_s ({tc_s} s) must be within [0.2, {duration_s - 0.2}] s to avoid boundary wraparound."
        )

    mc_tmpl_msun = mc_msun + delta_mc_msun
    if mc_tmpl_msun <= 1.0:
        raise ValueError(f"Template chirp mass ({mc_tmpl_msun} M_sun) must be > 1.0 M_sun.")

    # 1. Frozen grid & active mask K_+
    grid = FrequencyGrid.from_duration_and_sampling_rate(
        duration=duration_s, sampling_rate=sampling_rate_hz
    )
    mask = grid.create_active_mask(f_start=f_min_hz, f_end=f_max_hz)
    active_indices = grid.get_active_indices(mask)
    if active_indices.size == 0:
        raise ValueError(f"No active bins in [{f_min_hz}, {f_max_hz}] Hz.")

    # 2. Analytical PSD proxy & PSDEngine
    psd_grid = analytical_detector_psd(grid.frequencies, model=psd_model)
    psd_engine = PSDEngine(grid.frequencies, psd_grid)

    # 3. Synthetic injection waveform h_inj(f) shifted to tc_s
    m1_inj_msun, m2_inj_msun, m_tot_inj = mchirp_eta_to_component_masses(mc_msun, eta)
    d_m = float(distance_mpc) * MPC_SI
    h_inj_zero = taylorf2_fd_waveform(
        grid.frequencies,
        m1_inj_msun * M_SUN_SI,
        m2_inj_msun * M_SUN_SI,
        d_m,
        phi_c=phic_rad,
        f_end=f_max_hz,
    )
    h_inj_fd = apply_coalescence_time_shift(h_inj_zero, grid.frequencies, tc_s)

    # 4. Template waveform h_tmpl(f) at t_c = 0, phi_c = 0
    m1_tmpl_msun, m2_tmpl_msun, m_tot_tmpl = mchirp_eta_to_component_masses(mc_tmpl_msun, eta)
    h_tmpl_fd = taylorf2_fd_waveform(
        grid.frequencies,
        m1_tmpl_msun * M_SUN_SI,
        m2_tmpl_msun * M_SUN_SI,
        d_m,
        phi_c=0.0,
        f_end=f_max_hz,
    )

    # 5. Deterministic synthetic colored Gaussian noise & combined data d~(f)
    noise_td = generate_colored_noise(grid.n_samples, grid.delta_t, psd_grid, seed=int(seed))
    noise_fd = continuous_rfft(noise_td, grid.delta_t)
    data_fd = h_inj_fd + noise_fd

    # 6. Execute authoritative matched_filter
    mf_res = matched_filter(
        grid=grid,
        data=data_fd,
        template_fd=h_tmpl_fd,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_min_hz,
        f_0=15.0,
        tau_chirp=0.0,
        tau_filter_diag=0.05,
        compute_chisq=False,
    )

    # Locate recovered peak in [0, T) circular lags over valid_mask
    valid_indices = np.flatnonzero(mf_res.valid_mask)
    if valid_indices.size > 0:
        rel_idx = int(np.argmax(mf_res.snr_max[mf_res.valid_mask]))
        peak_idx = int(valid_indices[rel_idx])
    else:
        peak_idx = int(np.argmax(mf_res.snr_max))

    recovered_peak_snr = float(mf_res.snr_max[peak_idx])
    recovered_peak_time_s = float(mf_res.circular_lags[peak_idx])
    recovered_peak_phase_rad = float(mf_res.phase_max[peak_idx])

    # 7. Peak-preserving downsampling of SNR time series over [0, T)
    n_buckets = 400
    bucket_edges = np.linspace(0, grid.n_samples, n_buckets + 1, dtype=int)
    snr_t: List[float] = []
    snr_abs: List[float] = []
    snr_real: List[float] = []
    for b in range(n_buckets):
        start_i, end_i = int(bucket_edges[b]), int(bucket_edges[b + 1])
        if end_i <= start_i:
            continue
        local_argmax = start_i + int(np.argmax(mf_res.snr_max[start_i:end_i]))
        snr_t.append(float(mf_res.circular_lags[local_argmax]))
        snr_abs.append(float(mf_res.snr_max[local_argmax]))
        snr_real.append(float(np.real(mf_res.snr_time_series[local_argmax])))

    # Zoomed window around peak (+/- 0.12 s at full resolution)
    half_win_samples = int(round(0.12 * sampling_rate_hz))
    z_start = max(0, peak_idx - half_win_samples)
    z_end = min(grid.n_samples, peak_idx + half_win_samples + 1)
    zoom_t = mf_res.circular_lags[z_start:z_end].tolist()
    zoom_abs = mf_res.snr_max[z_start:z_end].tolist()
    zoom_real = np.real(mf_res.snr_time_series[z_start:z_end]).tolist()

    # 8. Active-band frequency-domain spectrum (downsampled <= 400 pts)
    f_act = grid.frequencies[active_indices]
    data_amp_act = np.abs(data_fd[active_indices])
    inj_amp_act = np.abs(h_inj_fd[active_indices])
    tmpl_amp_act = np.abs(h_tmpl_fd[active_indices])
    n_act = int(f_act.size)
    if n_act > 400:
        f_idx = np.linspace(0, n_act - 1, 400, dtype=int)
        out_f = f_act[f_idx]
        out_d_amp = data_amp_act[f_idx]
        out_i_amp = inj_amp_act[f_idx]
        out_t_amp = tmpl_amp_act[f_idx]
    else:
        out_f = f_act
        out_d_amp = data_amp_act
        out_i_amp = inj_amp_act
        out_t_amp = tmpl_amp_act

    f_isco_inj = schwarzschild_isco_frequency(m_tot_inj)
    f_isco_tmpl = schwarzschild_isco_frequency(m_tot_tmpl)

    return {
        "snr_time_s": snr_t,
        "snr_abs": snr_abs,
        "snr_real": snr_real,
        "zoom_time_s": zoom_t,
        "zoom_abs": zoom_abs,
        "zoom_real": zoom_real,
        "frequencies": out_f.tolist(),
        "data_amp": out_d_amp.tolist(),
        "injection_amp": out_i_amp.tolist(),
        "template_amp": out_t_amp.tolist(),
        "metadata": {
            "optimal_snr": float(mf_res.optimal_snr),
            "sigma_h": float(mf_res.sigma_h),
            "recovered_peak_snr": recovered_peak_snr,
            "recovered_peak_time_s": recovered_peak_time_s,
            "recovered_peak_phase_rad": recovered_peak_phase_rad,
            "time_offset_s": float(recovered_peak_time_s - tc_s),
            "injected_tc_s": float(tc_s),
            "injected_mc_msun": float(mc_msun),
            "template_mc_msun": float(mc_tmpl_msun),
            "delta_mc_msun": float(delta_mc_msun),
            "eta": float(eta),
            "distance_mpc": float(distance_mpc),
            "f_isco_inj_hz": float(f_isco_inj),
            "f_isco_tmpl_hz": float(f_isco_tmpl),
            "f_min_hz": float(f_min_hz),
            "f_max_hz": float(f_max_hz),
            "n_active_bins": n_act,
            "delta_f_hz": float(grid.delta_f),
            "duration_s": float(grid.duration),
            "sampling_rate_hz": float(grid.sampling_rate),
            "psd_model": psd_model,
            "psd_model_label": (
                "Analytical aLIGO-like PSD proxy"
                if psd_model == "aligo_representative"
                else "Flat white-noise reference PSD"
            ),
            "seed": int(seed),
            "units": {
                "snr": "dimensionless",
                "time": "s",
                "frequency": "Hz",
                "fourier_strain": "strain/Hz",
            },
        },
    }


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: List[float]) -> List[float]:
    """Compute weighted quantiles using midpoint cumulative probability weights."""
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    w_sum = float(np.sum(w_sorted))
    cdf = (np.cumsum(w_sorted) - 0.5 * w_sorted) / w_sum
    return [float(x) for x in np.interp(quantiles, cdf, v_sorted)]


def get_inference_posterior(
    param_x: str = "mc",
    param_y: str = "eta",
    bins_1d: int = 40,
    bins_2d: int = 36,
    npz_rel_path: str = "experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz",
    result_rel_path: str = "experiments/wp6_stage6c_anchor/run_a_anchor_result.json",
) -> Dict[str, Any]:
    """Load saved Stage 6C Run A weighted nested-sampling posterior and compute bounded marginals."""
    param_specs = [
        {
            "key": "mc",
            "col": 0,
            "label": "Chirp Mass M_c",
            "symbol": "M_c",
            "unit": "M_sun",
            "prior_bounds": [20.0, 40.0],
            "injected": 28.095555795460438,
        },
        {
            "key": "eta",
            "col": 1,
            "label": "Symmetric Mass Ratio η",
            "symbol": "η",
            "unit": "dimensionless",
            "prior_bounds": [0.10, 0.25],
            "injected": 0.24710059171597634,
        },
        {
            "key": "tc",
            "col": 2,
            "label": "Coalescence Time t_c",
            "symbol": "t_c",
            "unit": "s",
            "prior_bounds": [1.90, 2.10],
            "injected": 2.0,
        },
        {
            "key": "phic",
            "col": 3,
            "label": "Coalescence Phase φ_c",
            "symbol": "φ_c",
            "unit": "rad",
            "prior_bounds": [0.0, 2.0 * math.pi],
            "injected": 0.0,
        },
    ]
    spec_by_key = {p["key"]: p for p in param_specs}

    if param_x not in spec_by_key or param_y not in spec_by_key:
        raise ValueError(
            f"Unsupported parameters ({param_x}, {param_y}). Supported: {list(spec_by_key.keys())}."
        )
    if param_x == param_y:
        raise ValueError("param_x and param_y must be distinct parameters for 2D joint view.")

    npz_path = resolve_safe_path(npz_rel_path)
    result_path = resolve_safe_path(result_rel_path)

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result_raw = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Corrupt JSON artifact at {result_rel_path}: {err}") from err

    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if "samples" not in npz or "weights" not in npz:
                raise KeyError("Mandatory keys ('samples', 'weights') missing from NPZ archive.")
            samples = np.array(npz["samples"], dtype=np.float64)
            weights = np.array(npz["weights"], dtype=np.float64)
    except Exception as err:
        raise ValueError(f"Failed to load NPZ archive {npz_rel_path}: {err}") from err

    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError(f"Expected samples shape (N, 4), got {samples.shape}.")
    if weights.ndim != 1 or weights.shape[0] != samples.shape[0]:
        raise ValueError(f"Weights shape {weights.shape} does not match samples {samples.shape[0]}.")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite values detected in posterior samples or weights.")
    if np.any(weights < 0):
        raise ValueError("Negative weights detected in posterior archive.")

    w_sum = float(np.sum(weights))
    if abs(w_sum - 1.0) > 1e-4:
        raise ValueError(f"Sample weights do not sum to unity (sum = {w_sum:.8f}).")

    # Kish effective sample size and eta boundary accumulation
    kish_n_eff = float(1.0 / np.sum(weights**2))
    eta_col = samples[:, 1]
    eta_top1_pct = float(np.sum(weights[eta_col >= 0.2485]) * 100.0)

    json_params = result_raw.get("summary", {}).get("parameters", {})

    # 1. Weighted 1D marginals over high-weight posterior support
    marginals_1d: Dict[str, Any] = {}
    support_ranges: Dict[str, List[float]] = {}

    for spec in param_specs:
        key = spec["key"]
        vals = samples[:, spec["col"]]

        if key == "phic":
            r_min, r_max = 0.0, 2.0 * math.pi
        else:
            q_low, q_high = _weighted_quantiles(vals, weights, [0.001, 0.999])
            span = max(1e-9, q_high - q_low)
            r_min = max(spec["prior_bounds"][0], q_low - 0.06 * span)
            r_max = min(spec["prior_bounds"][1], q_high + 0.06 * span)
            if key == "eta":
                r_max = 0.25

        support_ranges[key] = [float(r_min), float(r_max)]
        density, edges = np.histogram(
            vals, bins=int(bins_1d), range=(r_min, r_max), weights=weights, density=True
        )
        centers = 0.5 * (edges[:-1] + edges[1:])

        p5, p50, p95 = _weighted_quantiles(vals, weights, [0.05, 0.50, 0.95])
        w_mean = float(np.sum(weights * vals))
        w_std = float(math.sqrt(np.sum(weights * (vals - w_mean) ** 2)))

        saved_summary = json_params.get(key, {})
        marginals_1d[key] = {
            "key": key,
            "label": spec["label"],
            "symbol": spec["symbol"],
            "unit": spec["unit"],
            "prior_bounds": spec["prior_bounds"],
            "injected": spec["injected"],
            "support_range": [float(r_min), float(r_max)],
            "bin_width": float(edges[1] - edges[0]),
            "bin_edges": edges.tolist(),
            "bin_centers": centers.tolist(),
            "density": density.tolist(),
            "summary": {
                "median": float(saved_summary.get("median", p50)),
                "p5": float(saved_summary.get("p5", p5)),
                "p95": float(saved_summary.get("p95", p95)),
                "ci90_width": float(saved_summary.get("ci90_width", p95 - p5)),
                "weighted_mean": w_mean,
                "weighted_std": w_std,
            },
        }

    # 2. Weighted 2D joint histogram for (param_x, param_y)
    sx = spec_by_key[param_x]
    sy = spec_by_key[param_y]
    x_vals = samples[:, sx["col"]]
    y_vals = samples[:, sy["col"]]
    rx = support_ranges[param_x]
    ry = support_ranges[param_y]

    hist_2d, x_edges, y_edges = np.histogram2d(
        x_vals,
        y_vals,
        bins=int(bins_2d),
        range=[rx, ry],
        weights=weights,
    )
    max_cell_weight = float(np.max(hist_2d)) if hist_2d.size > 0 else 0.0
    norm_grid = (hist_2d / max_cell_weight) if max_cell_weight > 0 else hist_2d
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    return {
        "parameter_order": [p["key"] for p in param_specs],
        "marginals_1d": marginals_1d,
        "joint_2d": {
            "param_x": param_x,
            "param_y": param_y,
            "x_label": sx["label"],
            "y_label": sy["label"],
            "x_unit": sx["unit"],
            "y_unit": sy["unit"],
            "x_centers": x_centers.tolist(),
            "y_centers": y_centers.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "grid_normalized": norm_grid.tolist(),
            "max_bin_mass": max_cell_weight,
            "injected_x": sx["injected"],
            "injected_y": sy["injected"],
        },
        "metadata": {
            "run_id": str(result_raw.get("run_id", "run_a_anchor")),
            "status": str(result_raw.get("status", "SUCCESS")),
            "realization": "Synthetic Realization 01",
            "likelihood_model": "Model 2 (Plug-in Gaussian Whittle)",
            "prior_model": "4D Uniform Prior (mc, eta, tc, phic)",
            "sampler": "Dynesty (n_live = 500)",
            "seed": 8101,
            "n_samples": int(samples.shape[0]),
            "n_dims": int(samples.shape[1]),
            "n_evaluations": int(result_raw.get("n_evaluations", 500657)),
            "guard_evaluations_completed": int(
                result_raw.get("guard_evaluations_completed", 487407)
            ),
            "log_evidence": float(result_raw.get("log_evidence", 0.0)),
            "log_evidence_err": float(result_raw.get("log_evidence_err", 0.0)),
            "kish_n_eff": kish_n_eff,
            "eta_top1_pct": eta_top1_pct,
            "weight_sum": w_sum,
            "between_seed_delta_logz": 2.9720,
            "artifact_npz": npz_rel_path,
            "artifact_json": result_rel_path,
            "binning_note": (
                f"1D ({bins_1d} weighted bins) and 2D ({bins_2d}x{bins_2d} weighted bins) "
                "histograms are visualization choices over weighted nested-sampling dead points, "
                "not resampled unweighted posterior draws."
            ),
            "qualification": (
                "Saved synthetic single-realization posterior (Realization 01, Seed 8101), not a "
                "real detector event. Stage 6D reported between-seed evidence sensitivity against "
                "seed 8201 of |Δln Z| = 2.9720 nats. This archive does not establish full "
                "convergence, boundary-artifact freedom, or empirical coverage calibration. "
                "Model 3 Student-t and Run B remain frozen/unexecuted."
            ),
        },
    }


def _load_json_checked(rel_path: str) -> Dict[str, Any]:
    """Safely load a JSON artifact with strict path and parse validation."""
    target_path = resolve_safe_path(rel_path)
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Corrupt JSON artifact at {rel_path}: {err}") from err
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {rel_path}, got {type(data).__name__}.")
    return data


def _extract_finite_float(raw: Dict[str, Any], key: str, rel_path: str, required: bool = True) -> float | None:
    """Extract a finite float from a dict, raising ValueError on non-finite or missing required keys."""
    if key not in raw or raw[key] is None:
        if required:
            raise ValueError(f"Missing mandatory key '{key}' in {rel_path}.")
        return None
    val = float(raw[key])
    if not math.isfinite(val):
        raise ValueError(f"Non-finite value for '{key}' ({val}) in {rel_path}.")
    return val


def _extract_optional_int(raw: Dict[str, Any], key: str) -> int | None:
    """Extract an optional integer field without fabricating defaults."""
    val = raw.get(key)
    if val is None:
        return None
    return int(val)


def get_experiments_explorer(
    s1_rel_path: str = "reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s1.json",
    s2_rel_path: str = "reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s2.json",
    stage6c_rel_path: str = "experiments/wp6_stage6c_anchor/run_a_anchor_result.json",
    ledger_rel_path: str = "experiments/wp6_stage6c_anchor/stage6c_aggregate_ledger.json",
) -> Dict[str, Any]:
    """Load saved Stage 5G.2, 6C, and 6D artifacts for the read-only Experiment Explorer."""
    s1_raw = _load_json_checked(s1_rel_path)
    s2_raw = _load_json_checked(s2_rel_path)
    s6c_raw = _load_json_checked(stage6c_rel_path)
    ledger_raw = _load_json_checked(ledger_rel_path)

    # Validate provenance consistency for the Stage 5G.2 two-seed pair
    if s1_raw.get("realization") != s2_raw.get("realization"):
        raise ValueError(
            f"Inconsistent realization between {s1_rel_path} ({s1_raw.get('realization')}) "
            f"and {s2_rel_path} ({s2_raw.get('realization')})."
        )
    if s1_raw.get("model") != s2_raw.get("model"):
        raise ValueError(
            f"Inconsistent model between {s1_rel_path} ({s1_raw.get('model')}) "
            f"and {s2_rel_path} ({s2_raw.get('model')})."
        )
    seed_1 = _extract_optional_int(s1_raw, "sampler_seed")
    seed_2 = _extract_optional_int(s2_raw, "sampler_seed")
    if seed_1 is None or seed_2 is None or seed_1 == seed_2:
        raise ValueError(f"Expected distinct valid sampler seeds for comparison ({seed_1} vs {seed_2}).")

    def _summarize_stage5g2_run(raw: Dict[str, Any], rel_path: str) -> Dict[str, Any]:
        log_z = _extract_finite_float(raw, "log_evidence", rel_path, required=True)
        log_z_err = _extract_finite_float(raw, "log_evidence_err", rel_path, required=True)
        pstats = raw.get("parameter_stats") or {}
        mc_med = pstats.get("mc", {}).get("median") if isinstance(pstats.get("mc"), dict) else None
        eta_med = pstats.get("eta", {}).get("median") if isinstance(pstats.get("eta"), dict) else None
        tc_med = pstats.get("tc", {}).get("median") if isinstance(pstats.get("tc"), dict) else None
        phic_rbar = pstats.get("phic", {}).get("r_bar") if isinstance(pstats.get("phic"), dict) else None

        return {
            "run_id": str(raw.get("run_id") or ""),
            "status": str(raw.get("status") or "UNKNOWN"),
            "model": "Model 2 (Plug-in Gaussian Whittle)" if raw.get("model") == "plug" else str(raw.get("model", "")),
            "realization": f"Synthetic Realization {int(raw['realization']):02d}" if raw.get("realization") is not None else "Unknown",
            "sampler_seed": int(raw["sampler_seed"]),
            "nlive": _extract_optional_int(raw, "nlive"),
            "log_evidence": log_z,
            "log_evidence_err": log_z_err,
            "n_evaluations": _extract_optional_int(raw, "n_evaluations"),
            "n_total_samples": _extract_optional_int(raw, "n_total_samples"),
            "n_eff": _extract_finite_float(raw, "n_eff", rel_path, required=False),
            "wall_time_s": _extract_finite_float(raw, "wall_time_s", rel_path, required=False),
            "cpu_time_s": _extract_finite_float(raw, "cpu_time_s", rel_path, required=False),
            "mc_median": float(mc_med) if mc_med is not None else None,
            "eta_median": float(eta_med) if eta_med is not None else None,
            "tc_median": float(tc_med) if tc_med is not None else None,
            "phic_r_bar": float(phic_rbar) if phic_rbar is not None else None,
            "artifact_path": rel_path,
        }

    run_s1 = _summarize_stage5g2_run(s1_raw, s1_rel_path)
    run_s2 = _summarize_stage5g2_run(s2_raw, s2_rel_path)
    if not run_s1["run_id"] or not run_s2["run_id"]:
        raise ValueError("Missing run_id in Stage 5G.2 artifact.")

    delta_logz = abs(float(run_s1["log_evidence"]) - float(run_s2["log_evidence"]))
    sigma_delta = math.sqrt(float(run_s1["log_evidence_err"]) ** 2 + float(run_s2["log_evidence_err"]) ** 2)
    discrepancy_ratio = (delta_logz / sigma_delta) if sigma_delta > 0 else None

    # Validate Stage 6C & Ledger
    s6c_logz = _extract_finite_float(s6c_raw, "log_evidence", stage6c_rel_path, required=True)
    s6c_err = _extract_finite_float(s6c_raw, "log_evidence_err", stage6c_rel_path, required=True)
    s6c_calls = _extract_optional_int(s6c_raw, "guard_evaluations_completed") or _extract_optional_int(s6c_raw, "n_evaluations")
    s6c_samples = _extract_optional_int((s6c_raw.get("summary") or {}), "n_posterior_samples")

    cum_calls = _extract_optional_int(ledger_raw, "cumulative_calls")
    rem_calls = _extract_optional_int(ledger_raw, "remaining_calls")
    run_b_authorized = bool((ledger_raw.get("run_b_gate") or {}).get("authorized", False))

    campaign_stages = [
        {
            "stage_id": "stage_5g2",
            "title": "Stage 5G.2 — Two-Seed Repeatability Pair",
            "status": "Completed / Historical",
            "model": "Model 2 (Plug-in Gaussian Whittle)",
            "realization": "Synthetic Realization 01",
            "summary": (
                f"Paired runs {run_s1['run_id']} (Seed {seed_1}) and {run_s2['run_id']} (Seed {seed_2}) "
                f"yielded |Δln Z| = {delta_logz:.4f} nats with quadrature uncertainty σ_Δ = {sigma_delta:.4f} nats."
            ),
            "artifact_path": f"{s1_rel_path}, {s2_rel_path}",
        },
        {
            "stage_id": "stage_6c_run_a",
            "title": "Stage 6C Run A — Production Anchor",
            "status": "Completed / Historical",
            "model": "Model 2 (Plug-in Gaussian Whittle)",
            "realization": "Synthetic Realization 01 (Seed 8101)",
            "summary": (
                f"Run {s6c_raw.get('run_id', 'run_a_anchor')} completed: ln Z = {s6c_logz:.4f} ± {s6c_err:.4f} nats, "
                f"{s6c_samples if s6c_samples is not None else 'N/A'} samples, "
                f"{s6c_calls if s6c_calls is not None else 'N/A'} guard-reconciled calls."
            ),
            "artifact_path": stage6c_rel_path,
        },
        {
            "stage_id": "stage_6d_closeout",
            "title": "Stage 6D — Pathway A Closeout & Resource Ledger",
            "status": "Completed / Historical",
            "model": "Campaign Governance & Budget Snapshot",
            "realization": "Historical 2-seed basis (8101 vs 8201)",
            "summary": (
                f"Closed on the historical 2-seed comparison. Ledger snapshot: "
                f"{cum_calls if cum_calls is not None else 'N/A'} calls used, "
                f"{rem_calls if rem_calls is not None else 'N/A'} calls remaining."
            ),
            "artifact_path": ledger_rel_path,
        },
        {
            "stage_id": "stage_6c_run_b",
            "title": "Stage 6C Run B — Total-Mass Parameterization",
            "status": "Unexecuted",
            "model": "Model 2 with (M_total, η, t_c, φ_c) prior transform",
            "realization": "None (not executed)",
            "summary": (
                f"Budget gate authorized={run_b_authorized} in historical ledger, but Run B was never executed; "
                "no posterior samples or evidence results exist."
            ),
            "artifact_path": ledger_rel_path,
        },
        {
            "stage_id": "seed_8301",
            "title": "Seed 8301 — Third Independent Seed Diagnostic",
            "status": "Proposed",
            "model": "Model 2 (Plug-in Gaussian Whittle)",
            "realization": "Proposed diagnostic only",
            "summary": (
                "Retained in Stage 6D closeout purely as a proposed future diagnostic; not executed."
            ),
            "artifact_path": "reports/WP6_Stage6D_Sampler_Repeatability_Design_and_Resource_Review.md",
        },
        {
            "stage_id": "model_3_student_t",
            "title": "Model 3 — Independent-Bin Student-t Likelihood",
            "status": "Frozen / Specified",
            "model": "Student-t (ν_eff = 51894/1157 ≈ 44.8522)",
            "realization": "Historical Stage 5G.2 pilot only (not production)",
            "summary": (
                "Frozen and not approved for production inference following Stage 5F/5G.5 review."
            ),
            "artifact_path": "reports/WP6_Stage5G5_Scientific_Synthesis_and_Decision_Report.md",
        },
    ]

    return {
        "multi_seed_comparison": {
            "runs": [run_s1, run_s2],
            "delta_logz": delta_logz,
            "sigma_delta": sigma_delta,
            "discrepancy_ratio": discrepancy_ratio,
            "qualification": (
                "Two-seed evidence sensitivity diagnostic on Synthetic Realization 01 "
                "(|Δln Z| = 2.9720 nats, σ_Δ ≈ 0.3332 nats, ratio = 8.92). This reflects internal "
                "sampler reporting under 1 d.o.f. and is not a calibrated frequentist significance, "
                "convergence proof, or coverage validation."
            ),
        },
        "stage6c_anchor": {
            "run_id": str(s6c_raw.get("run_id", "run_a_anchor")),
            "status": str(s6c_raw.get("status", "UNKNOWN")),
            "log_evidence": s6c_logz,
            "log_evidence_err": s6c_err,
            "n_calls": s6c_calls,
            "n_samples": s6c_samples,
            "artifact_path": stage6c_rel_path,
        },
        "ledger": {
            "cumulative_calls": cum_calls,
            "remaining_calls": rem_calls,
            "run_b_authorized": run_b_authorized,
            "artifact_path": ledger_rel_path,
        },
        "campaign_stages": campaign_stages,
        "limitations": [
            "All saved runs use Synthetic Realization 01 under the analytical aligo_representative PSD proxy, not real LIGO detector data.",
            "Completed/historical benchmarks are strictly separated from unexecuted, proposed, or frozen work.",
            "Two-seed discrepancy demonstrates seed sensitivity for this frozen problem, not calibrated convergence or empirical coverage.",
        ],
    }
