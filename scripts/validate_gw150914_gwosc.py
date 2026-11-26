"""External validation of GWLab conditioning, Welch PSD, and matched filtering on public GWOSC GW150914 data.

Downloads (if not cached) the official 32s 4096Hz H1 and L1 HDF5 strain files from GWOSC,
verifies finite values and DQmask quality flags, conditions and downsamples (4096 -> 2048 Hz),
estimates off-source PSDs excluding the event window, runs `gw_inference.filter.matched_filter`
with a fixed nonspinning TaylorF2 inspiral template, and saves figures + JSON summary.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import ssl
import urllib.request
from typing import Any, Dict, Tuple

import certifi
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.conventions import continuous_rfft
from gw_inference.filter import matched_filter
from gw_inference.grid import FrequencyGrid
from gw_inference.psd import PSDEngine, tukey_window, welch_psd
from gw_inference.waveform import schwarzschild_isco_frequency, taylorf2_fd_waveform

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "gwosc_cache"
FIG_DIR = REPO_ROOT / "reports" / "figures" / "gw150914_external"

GW150914_GPS = 1126259462.4
GWOSC_URLS = {
    "H1": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
    "L1": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def fetch_gwosc_hdf5(detector: str) -> pathlib.Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = GWOSC_URLS[detector]
    dest = CACHE_DIR / url.split("/")[-1]
    if not dest.exists():
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(url, headers={"User-Agent": "GWLab-External-Validation/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            dest.write_bytes(resp.read())
    return dest


def load_and_condition_strain(path: pathlib.Path, fs_target: float = 2048.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    with h5py.File(path, "r") as f:
        strain_raw = np.asarray(f["strain"]["Strain"][()], dtype=np.float64)
        dt_raw = float(f["strain"]["Strain"].attrs["Xspacing"])
        fs_raw = 1.0 / dt_raw
        gps_start = float(f["meta"]["GPSstart"][()])
        duration = float(f["meta"]["Duration"][()])
        utc_start = f["meta"]["UTCstart"][()].decode() if isinstance(f["meta"]["UTCstart"][()], bytes) else str(f["meta"]["UTCstart"][()])
        dq_mask = np.asarray(f["quality"]["simple"]["DQmask"][()], dtype=np.int64)

    if not np.all(np.isfinite(strain_raw)):
        raise ValueError(f"Non-finite samples in {path.name}")
    if np.any(dq_mask <= 0):
        raise ValueError(f"Data-quality mask failure in {path.name}")

    # 1. Remove DC offset
    demeaned = strain_raw - np.mean(strain_raw)

    # 2. Zero-phase 4th-order Butterworth high-pass at 15 Hz to suppress sub-band seismic drift
    sos = butter(4, 15.0, btype="highpass", fs=fs_raw, output="sos")
    highpassed = sosfiltfilt(sos, demeaned)

    # 3. Anti-alias polyphase downsampling 4096 Hz -> 2048 Hz
    down = int(round(fs_raw / fs_target))
    conditioned = resample_poly(highpassed, up=1, down=down).astype(np.float64)

    meta = {
        "file": path.name,
        "size_bytes": len(raw_bytes),
        "sha256": sha256,
        "gps_start": gps_start,
        "duration_s": duration,
        "utc_start": utc_start,
        "fs_raw_hz": fs_raw,
        "fs_target_hz": fs_target,
        "n_samples_raw": int(strain_raw.size),
        "n_samples_conditioned": int(conditioned.size),
        "dq_mask_min": int(np.min(dq_mask)),
        "dq_mask_max": int(np.max(dq_mask)),
    }
    return conditioned, meta


def estimate_offsource_psd(
    conditioned: np.ndarray,
    fs: float,
    t_event_rel: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate Welch PSD from clean pre-event [1, 13] s and post-event [18, 30] s intervals."""
    delta_t = 1.0 / fs
    m_seg = int(round(4.0 * fs))
    overlap = m_seg // 2
    win = tukey_window(m_seg, alpha=0.25)

    # Strictly exclude filter edges [0, 1] s, [31, 32] s and event guard [t_event-2.4, t_event+2.6] s
    pre_slice = conditioned[int(1.0 * fs) : int(13.0 * fs)]
    post_slice = conditioned[int(18.0 * fs) : int(30.0 * fs)]

    freqs_w, psd_pre = welch_psd(pre_slice, delta_t, segment_length=m_seg, overlap_samples=overlap, window=win)
    _, psd_post = welch_psd(post_slice, delta_t, segment_length=m_seg, overlap_samples=overlap, window=win)
    psd_avg = 0.5 * (psd_pre + psd_post)
    return freqs_w, psd_avg


def run_validation() -> Dict[str, Any]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fs = 2048.0
    delta_t = 1.0 / fs
    duration_on = 4.0
    n_on = int(round(duration_on * fs))

    # Fixed, non-tuned inspiral template: equal-mass nonspinning 30 + 30 M_sun (M_c ~ 26.12 M_sun)
    # Using a generic stellar-mass BBH template without fitting to GW150914's posterior
    m1_msun = 35.0
    m2_msun = 30.0
    mc_msun = ((m1_msun * m2_msun) ** 0.6) / ((m1_msun + m2_msun) ** 0.2)
    eta = (m1_msun * m2_msun) / ((m1_msun + m2_msun) ** 2)
    f_isco = schwarzschild_isco_frequency(m1_msun + m2_msun)

    grid = FrequencyGrid.from_duration_and_sampling_rate(duration=duration_on, sampling_rate=fs)
    mask = grid.create_active_mask(f_start=20.0, f_end=512.0)
    active_idx = grid.get_active_indices(mask)

    template_fd = taylorf2_fd_waveform(
        grid.frequencies,
        m1_kg=m1_msun * M_SUN_SI,
        m2_kg=m2_msun * M_SUN_SI,
        distance_m=410.0 * MPC_SI,
        phi_c=0.0,
        f_end=512.0,
    )

    results: Dict[str, Any] = {
        "event": "GW150914",
        "catalog": "GWTC-1-confident",
        "event_gps": GW150914_GPS,
        "template": {
            "approximant": "TaylorF2 (3.5PN phase, Newtonian amplitude)",
            "m1_msun": m1_msun,
            "m2_msun": m2_msun,
            "mc_msun": mc_msun,
            "eta": eta,
            "f_isco_hz": f_isco,
            "active_band_hz": [20.0, 512.0],
        },
        "detectors": {},
    }

    per_det_arrays: Dict[str, Dict[str, np.ndarray]] = {}

    for det in ("H1", "L1"):
        hdf5_path = fetch_gwosc_hdf5(det)
        conditioned, meta = load_and_condition_strain(hdf5_path, fs_target=fs)
        t_event_rel = GW150914_GPS - meta["gps_start"]  # 15.4 s

        freqs_w, psd_off = estimate_offsource_psd(conditioned, fs=fs, t_event_rel=t_event_rel)
        psd_engine = PSDEngine(freqs_w, psd_off)

        # On-source 4.0s window centered around t_event_rel: [t_event_rel - 2.0, t_event_rel + 2.0]
        on_start_s = t_event_rel - 2.0
        idx_start = int(round(on_start_s * fs))
        on_source_td = conditioned[idx_start : idx_start + n_on]
        on_win = tukey_window(n_on, alpha=0.1)
        data_fd = continuous_rfft(on_source_td * on_win, delta_t)

        mf_res = matched_filter(
            grid=grid,
            data=data_fd,
            template_fd=template_fd,
            psd_engine=psd_engine,
            mask=mask,
            f_start=20.0,
            f_0=15.0,
            tau_chirp=0.0,
            tau_filter_diag=0.05,
            compute_chisq=False,
        )

        snr_abs = np.asarray(mf_res.snr_max, dtype=np.float64)
        t_on_rel_event = np.arange(n_on) * delta_t - 2.0
        # Restrict peak search to [-1.5, +1.5] s around event center to avoid window edge
        inner = (t_on_rel_event >= -1.5) & (t_on_rel_event <= 1.5)
        peak_idx = int(np.argmax(snr_abs * inner))
        peak_snr = float(snr_abs[peak_idx])
        peak_dt = float(t_on_rel_event[peak_idx])
        peak_gps = on_start_s + meta["gps_start"] + peak_idx * delta_t

        # Compute whitened time series (bandpass [25, 350] Hz for clear visual SNR)
        white_fd = np.zeros_like(data_fd)
        vis_mask = (grid.frequencies >= 25.0) & (grid.frequencies <= 350.0)
        white_fd[vis_mask] = data_fd[vis_mask] / np.sqrt(psd_engine.psd[vis_mask] / (2.0 * grid.delta_f))
        whitened_td = np.fft.irfft(white_fd, n=n_on) / delta_t

        per_det_arrays[det] = {
            "t_on": t_on_rel_event,
            "on_source_td": on_source_td,
            "whitened_td": whitened_td,
            "freqs": freqs_w,
            "psd": psd_off,
            "snr_abs": snr_abs,
        }

        results["detectors"][det] = {
            "metadata": meta,
            "peak_snr": peak_snr,
            "peak_offset_from_event_s": peak_dt,
            "peak_gps": peak_gps,
            "template_sigma": float(mf_res.sigma_h),
        }
        print(f"{det}: peak_snr={peak_snr:.2f} at dt={peak_dt:+.4f} s (GPS={peak_gps:.4f}), sigma_h={mf_res.sigma_h:.2f}")

    # --- Generate 5 external-data figures ---
    # 1. H1/L1 conditioned strain around event
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.2), sharex=True)
    for ax, det, col in zip(axes, ("H1", "L1"), ("#1d4ed8", "#0f766e")):
        arr = per_det_arrays[det]
        zoom = np.abs(arr["t_on"]) <= 0.5
        ax.plot(arr["t_on"][zoom], arr["on_source_td"][zoom] * 1e21, color=col, lw=0.9, label=f"{det} conditioned strain")
        ax.axvline(0.0, color="#dc2626", ls="--", lw=1.0, label="GW150914 GPS (1126259462.4)")
        ax.set_ylabel("Strain [$10^{-21}$]")
        ax.legend(loc="upper right", fontsize=7.8)
    axes[-1].set_xlabel("Time relative to GW150914 GPS [s]")
    axes[0].set_title("GWOSC GW150914 Conditioned Strain (15 Hz High-Pass, 2048 Hz)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gw150914_strain.pdf")
    plt.close(fig)

    # 2. Off-source PSD / ASD
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for det, col in (("H1", "#1d4ed8"), ("L1", "#0f766e")):
        arr = per_det_arrays[det]
        m = (arr["freqs"] >= 15.0) & (arr["freqs"] <= 512.0)
        ax.loglog(arr["freqs"][m], np.sqrt(arr["psd"][m]), color=col, lw=1.4, label=f"{det} off-source Welch ASD")
    ax.axvspan(20.0, 512.0, color="#fef3c7", alpha=0.25, label="Active band $[20, 512]$ Hz")
    ax.set_xlabel("Frequency $f$ [Hz]")
    ax.set_ylabel("ASD $\\sqrt{S_n(f)}$ [$\\mathrm{Hz}^{-1/2}$]")
    ax.set_title("GW150914 Off-Source Welch ASD (Pre/Post Event, Tukey $\\alpha=0.25$, 4 s Segments)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gw150914_psd_asd.pdf")
    plt.close(fig)

    # 3. Whitened strain
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.2), sharex=True)
    for ax, det, col in zip(axes, ("H1", "L1"), ("#1d4ed8", "#0f766e")):
        arr = per_det_arrays[det]
        zoom = np.abs(arr["t_on"]) <= 0.25
        ax.plot(arr["t_on"][zoom], arr["whitened_td"][zoom], color=col, lw=1.0, label=f"{det} whitened ([25, 350] Hz)")
        ax.axvline(0.0, color="#dc2626", ls="--", lw=1.0)
        ax.set_ylabel("Whitened [$\\sigma$]")
        ax.legend(loc="upper right", fontsize=7.8)
    axes[-1].set_xlabel("Time relative to GW150914 GPS [s]")
    axes[0].set_title("GW150914 Whitened Strain Around Coalescence")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gw150914_whitened.pdf")
    plt.close(fig)

    # 4. Matched-filter SNR time series
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.2), sharex=True)
    for ax, det, col in zip(axes, ("H1", "L1"), ("#1d4ed8", "#0f766e")):
        arr = per_det_arrays[det]
        dres = results["detectors"][det]
        zoom = np.abs(arr["t_on"]) <= 0.5
        ax.plot(arr["t_on"][zoom], arr["snr_abs"][zoom], color=col, lw=1.2, label=f"{det} $|\\rho(t)|$ (peak={dres['peak_snr']:.2f} at {dres['peak_offset_from_event_s']:+.4f} s)")
        ax.axvline(0.0, color="#dc2626", ls="--", lw=1.0, label="Catalog GPS (1126259462.4)")
        ax.set_ylabel("SNR $|\\rho(t)|$")
        ax.legend(loc="upper right", fontsize=7.8)
    axes[-1].set_xlabel("Time relative to GW150914 GPS [s]")
    axes[0].set_title("TaylorF2 Inspiral Matched-Filter SNR on Public GWOSC GW150914 Data")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gw150914_snr.pdf")
    plt.close(fig)

    # 5. Concise detector comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.1))
    dets = ["H1", "L1"]
    peaks = [results["detectors"][d]["peak_snr"] for d in dets]
    offsets_ms = [results["detectors"][d]["peak_offset_from_event_s"] * 1000.0 for d in dets]
    ax1.bar(dets, peaks, color=["#1d4ed8", "#0f766e"], width=0.48)
    ax1.set_ylim(0, max(peaks) * 1.18)
    ax1.set_ylabel("Peak TaylorF2 SNR $|\\rho|$")
    ax1.set_title("Detector Peak SNR")
    for i, v in enumerate(peaks):
        ax1.text(i, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax2.bar(dets, offsets_ms, color=["#1d4ed8", "#0f766e"], width=0.48)
    ax2.set_ylim(0, max(offsets_ms) * 1.18)
    ax2.axhline(0.0, color="#dc2626", ls="--", lw=1.0)
    ax2.set_ylabel("Peak Offset from Catalog GPS [ms]")
    ax2.set_title("Peak Timing Offset")
    for i, v in enumerate(offsets_ms):
        ax2.text(i, v + (1.5 if v >= 0 else -4.5), f"{v:+.1f} ms", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gw150914_detector_comparison.pdf")
    plt.close(fig)

    summary_path = FIG_DIR / "gw150914_validation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    run_validation()
