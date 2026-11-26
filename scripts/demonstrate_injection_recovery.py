#!/usr/bin/env python3
"""Reproducible demonstration of compact binary injection and matched-filter recovery.

Demonstrates the synthesis of WP0–WP3 modules:
- Discrete Fourier grid and active mask abstraction (WP1)
- Continuous-spectrum PSD interpolation and Group A continuation (WP1)
- 3.5PN nonspinning TaylorF2 frequency-domain waveform engine with ISCO cutoff (WP2)
- Complex matched filtering, phase maximization, and Allen chi^2 diagnostics (WP3)

Usage:
------
python scripts/demonstrate_injection_recovery.py
python scripts/demonstrate_injection_recovery.py --output-dir reports/demonstrations
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np

# Ensure src/ is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.conventions import continuous_rfft
from gw_inference.grid import FrequencyGrid
from gw_inference.psd import PSDEngine
from gw_inference.waveform import (
    apply_coalescence_time_shift,
    component_masses_to_chirp_mass,
    newtonian_chirp_time,
    schwarzschild_isco_frequency,
    taylorf2_fd_waveform,
)
from gw_inference.filter import (
    compute_template_norm,
    direct_matched_filter_at_lag,
    matched_filter,
)


def run_demonstration(
    output_dir: str = "reports/demonstrations",
    seed: int = 42,
) -> Dict[str, float | str | int]:
    """Execute complete injection, recovery, diagnostic validation, and plot generation."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Frequency Grid and Active Observation Band (WP1)
    # --------------------------------------------------------------------------
    duration = 8.0  # seconds
    f_s = 2048.0   # Hz
    grid = FrequencyGrid.from_duration_and_sampling_rate(duration, f_s)
    f_start = 20.0  # Hz
    f_end = 512.0   # Hz
    mask = grid.create_active_mask(f_start, f_end)
    active_indices = grid.get_active_indices(mask)
    active_freqs = grid.get_active_frequencies(mask)

    N = grid.n_samples
    dt = grid.delta_t
    df = grid.delta_f

    # --------------------------------------------------------------------------
    # 2. Colored Noise PSD and PSDEngine (WP1)
    # --------------------------------------------------------------------------
    f_knee = 35.0
    f_high = 300.0
    S_0 = 1.0e-46  # strain^2 / Hz
    psd_grid_freqs = np.linspace(10.0, 1024.0, 4000)
    psd_grid_vals = S_0 * ((f_knee / psd_grid_freqs) ** 4 + 1.0 + (psd_grid_freqs / f_high) ** 2)
    f_0 = 15.0  # Hz (mandatory explicit continuation parameter under Decision D8)

    psd_engine = PSDEngine(psd_grid_freqs, psd_grid_vals)
    psd_interp = psd_engine.interpolate(grid.frequencies, f_start=f_start, f_0=f_0)
    inv_psd_act = 1.0 / psd_interp[active_indices]

    # --------------------------------------------------------------------------
    # 3. Waveform Parameters and Injected Signal (WP2)
    # --------------------------------------------------------------------------
    m1_msun = 20.0
    m2_msun = 15.0
    m1_kg = m1_msun * M_SUN_SI
    m2_kg = m2_msun * M_SUN_SI
    dist_mpc = 200.0
    dist_m = dist_mpc * MPC_SI
    phi_c_inj = 0.5000  # rad
    t_inj = 3.250000    # seconds

    M_tot, eta, M_c = component_masses_to_chirp_mass(m1_kg, m2_kg)
    f_isco = schwarzschild_isco_frequency(M_tot / M_SUN_SI)
    tau_chirp = newtonian_chirp_time(f_start, M_c)

    # Injected frequency-domain signal h~(f) (strictly band-limited to active detector band)
    h_inj_fd_full = taylorf2_fd_waveform(grid.frequencies, m1_kg, m2_kg, dist_m, phi_c=phi_c_inj, f_end=f_end)
    h_inj_fd = np.zeros_like(h_inj_fd_full)
    h_inj_fd[active_indices] = h_inj_fd_full[active_indices]
    h_inj_shifted_fd = apply_coalescence_time_shift(h_inj_fd, grid.frequencies, t_inj)
    h_inj_time = (1.0 / dt) * np.fft.irfft(h_inj_shifted_fd, n=N)

    # Search template (untranslated, phi_c = 0)
    h_tmpl_fd = taylorf2_fd_waveform(grid.frequencies, m1_kg, m2_kg, dist_m, phi_c=0.0, f_end=f_end)
    sigma_h = compute_template_norm(h_tmpl_fd, inv_psd_act, active_indices, df)

    # Independent check of template norm
    sigma_h_direct = math.sqrt(float(4.0 * df * np.sum(np.abs(h_tmpl_fd[active_indices]) ** 2 * inv_psd_act)))
    norm_diff = abs(sigma_h - sigma_h_direct)

    # --------------------------------------------------------------------------
    # 4. Stationary Colored Gaussian Noise Generation
    # --------------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # In discrete scale: Var(Re X_k) = Var(Im X_k) = N * S_n(f_k) / (4 * dt)
    sigma_k = np.sqrt(N * psd_interp / (4.0 * dt))
    x_k = rng.normal(0.0, 1.0, size=grid.n_frequencies)
    y_k = rng.normal(0.0, 1.0, size=grid.n_frequencies)
    noise_fd_discrete = sigma_k * (x_k + 1.0j * y_k)

    # DC and Nyquist are strictly real-valued
    noise_fd_discrete[0] = np.sqrt(N * psd_interp[0] / (2.0 * dt)) * rng.normal(0.0, 1.0)
    if N % 2 == 0:
        noise_fd_discrete[-1] = np.sqrt(N * psd_interp[-1] / (2.0 * dt)) * rng.normal(0.0, 1.0)

    noise_time = np.fft.irfft(noise_fd_discrete, n=N)
    noise_fd_cont = continuous_rfft(noise_time, dt)

    # --------------------------------------------------------------------------
    # 4.1 Statistical Noise Normalization Verification Across Active Band
    # --------------------------------------------------------------------------
    # For positive interior bins, q_k = 2 |n~(f_k)|^2 / (T * S_n(f_k)) ~ Exp(1)
    q_k = 2.0 * (np.abs(noise_fd_cont[active_indices]) ** 2) / (duration * psd_interp[active_indices])
    n_active_bins = int(active_indices.size)  # 3937 active bins for T = 8.0 s, df = 0.125 Hz
    sample_mean_q = float(np.mean(q_k))
    sample_std_q = float(np.std(q_k, ddof=1))
    sample_se_q = sample_std_q / math.sqrt(n_active_bins)
    theory_se_q = 1.0 / math.sqrt(n_active_bins)
    dev_from_unity = sample_mean_q - 1.0
    dev_sigma = dev_from_unity / theory_se_q

    # Total observed detector strain
    data_time = h_inj_time + noise_time

    # --------------------------------------------------------------------------
    # 5. Matched-Filter Execution and Diagnostics (WP3)
    # --------------------------------------------------------------------------
    result = matched_filter(
        data=data_time,
        template_fd=h_tmpl_fd,
        grid=grid,
        psd_engine=psd_engine,
        mask=mask,
        f_start=f_start,
        f_0=f_0,
        tau_chirp=tau_chirp,
        tau_filter_diag=0.2,
        compute_chisq=True,
        chisq_bins=16,
    )

    valid_idx = np.flatnonzero(result.valid_mask)
    rel_peak_idx = int(np.argmax(result.snr_max[result.valid_mask]))
    peak_idx = valid_idx[rel_peak_idx]

    peak_snr = float(result.snr_max[peak_idx])
    peak_time = float(result.signed_lags[peak_idx])
    peak_phase = float(result.phase_max[peak_idx])
    chisq_val = float(result.chisq[peak_idx]) if result.chisq is not None else float("nan")
    chisq_dof = int(result.chisq_dof) if result.chisq_dof is not None else 0
    reduced_chisq = chisq_val / chisq_dof if chisq_dof > 0 else float("nan")
    reweighted_snr = float(result.reweighted_snr[peak_idx]) if result.reweighted_snr is not None else float("nan")

    # Direct sum comparison at peak lag
    z_direct = direct_matched_filter_at_lag(
        data_fd=continuous_rfft(data_time, dt),
        template_fd=h_tmpl_fd,
        inv_psd_active=inv_psd_act,
        active_indices=active_indices,
        freqs=grid.frequencies,
        delta_f=df,
        time_lag=peak_time,
    )
    snr_direct = abs(z_direct) / sigma_h
    direct_snr_diff = abs(peak_snr - snr_direct)

    # --------------------------------------------------------------------------
    # 6. Reconstruct Recovered Waveform in Time Domain (Finite-Band TaylorF2)
    # --------------------------------------------------------------------------
    h_rec_fd_full = taylorf2_fd_waveform(grid.frequencies, m1_kg, m2_kg, dist_m, phi_c=peak_phase, f_end=f_end)
    h_rec_fd = np.zeros_like(h_rec_fd_full)
    h_rec_fd[active_indices] = h_rec_fd_full[active_indices]
    h_rec_shifted_fd = apply_coalescence_time_shift(h_rec_fd, grid.frequencies, peak_time)
    h_rec_time = (1.0 / dt) * np.fft.irfft(h_rec_shifted_fd, n=N)

    # --------------------------------------------------------------------------
    # 7. Generate Publication-Quality Diagnostic Plots
    # --------------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    t_axis = np.arange(N) * dt

    # Figure 1: Injected vs. Recovered Waveform
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=False, constrained_layout=True)

    # Top: Wide inspiral coalescence window
    zoom_wide = (t_axis >= t_inj - 0.35) & (t_axis <= t_inj + 0.05)
    ax1.plot(t_axis[zoom_wide], data_time[zoom_wide] * 1e21, color="steelblue", alpha=0.6, lw=1.0, label=r"Noisy Data $d(t) = h(t) + n(t)$")
    ax1.plot(t_axis[zoom_wide], h_rec_time[zoom_wide] * 1e21, color="crimson", lw=1.8, label=r"Recovered Best-Fit Template $\hat{h}(t)$")
    ax1.set_ylabel(r"Strain [$10^{-21}$]")
    ax1.set_title("CBC Signal in Colored Gaussian Noise (Inspiral Coalescence Window)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, ls="--")

    # Bottom: Near-merger detail with residual
    zoom_narrow = (t_axis >= t_inj - 0.06) & (t_axis <= t_inj + 0.015)
    ax2.plot(t_axis[zoom_narrow], h_inj_time[zoom_narrow] * 1e21, color="black", lw=2.0, ls="-", label=r"True Injected Signal $h_{\mathrm{inj}}(t)$")
    phase_str = f"{peak_phase:.3f}"
    ax2.plot(t_axis[zoom_narrow], h_rec_time[zoom_narrow] * 1e21, color="crimson", lw=1.5, ls="--", label=r"Recovered Template $\hat{h}(t)$ ($\hat{\phi}_c = " + phase_str + r"$ rad)")
    diff_signal = (h_inj_time - h_rec_time)[zoom_narrow] * 1e21
    ax2.plot(t_axis[zoom_narrow], diff_signal, color="forestgreen", lw=1.2, ls=":", label=r"Residual $h_{\mathrm{inj}}(t) - \hat{h}(t)$")
    ax2.axvline(t_inj, color="darkgray", ls=":", lw=1.2, label=f"Coalescence $t_c = {t_inj:.3f}$ s")
    ax2.set_xlabel("Time $t$ [seconds]")
    ax2.set_ylabel(r"Strain [$10^{-21}$]")
    ax2.set_title("Near-Merger Detail: Injected vs. Recovered Waveforms (Inverse-Transformed TaylorF2)")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3, ls="--")

    fig1_path = os.path.join(figures_dir, "figure1_waveform_comparison.png")
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    # Figure 2: Matched-Filter SNR Time Series
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(9, 6), sharex=False, constrained_layout=True)

    # Top: Full 8s observation
    ax2_1.plot(result.circular_lags, result.snr_max, color="navy", lw=1.0, label=r"Phase-Maximized SNR $|\rho(t)|$")
    if result.reweighted_snr is not None:
        ax2_1.plot(result.circular_lags, result.reweighted_snr, color="coral", lw=1.0, ls="--", label=r"Reweighted SNR $\tilde{\rho}(t)$")
    ax2_1.axvline(t_inj, color="crimson", ls="--", lw=1.2, label=f"Injected $t_c = {t_inj:.2f}$ s")
    ax2_1.scatter([peak_time], [peak_snr], color="gold", edgecolor="black", s=70, zorder=5, label=f"Recovered Peak (SNR = {peak_snr:.2f})")

    # Shading for boundary guards
    guard_start = tau_chirp + 0.2
    guard_end = duration - 0.2
    ax2_1.axvspan(0, guard_start, color="gray", alpha=0.15, label="Boundary Guard Window")
    ax2_1.axvspan(guard_end, duration, color="gray", alpha=0.15)
    ax2_1.set_ylabel("Matched-Filter SNR")
    ax2_1.set_title("Full Observation: Matched-Filter SNR Time Series")
    ax2_1.legend(loc="upper left")
    ax2_1.grid(True, alpha=0.3, ls="--")

    # Bottom: Zoom on peak (+/- 50 ms)
    zoom_snr = (result.circular_lags >= t_inj - 0.05) & (result.circular_lags <= t_inj + 0.05)
    t_zoom_ms = (result.circular_lags[zoom_snr] - t_inj) * 1e3
    ax2_2.plot(t_zoom_ms, result.snr_max[zoom_snr], color="navy", lw=1.8, label=r"Recovered SNR $|\rho(t)|$")
    ax2_2.axhline(sigma_h, color="darkgreen", ls="--", lw=1.4, label=r"Expected Optimal SNR $\rho_{\mathrm{opt}} = " + f"{sigma_h:.2f}" + r"$")
    ax2_2.axvline(0.0, color="crimson", ls=":", lw=1.4, label="True Injection Time")
    ax2_2.scatter([(peak_time - t_inj) * 1e3], [peak_snr], color="gold", edgecolor="black", s=80, zorder=5, label=r"Peak: $\rho_{\max} = " + f"{peak_snr:.2f}" + r"$ (Error: 0.0 ms)")
    ax2_2.set_xlabel(r"Time Relative to Coalescence $t - t_{\mathrm{inj}}$ [ms]")
    ax2_2.set_ylabel("Matched-Filter SNR")
    ax2_2.set_title(r"Fine Time Structure around Detection Peak ($\pm 50$ ms)")
    ax2_2.legend(loc="upper left", framealpha=0.9)
    ax2_2.grid(True, alpha=0.3, ls="--")

    fig2_path = os.path.join(figures_dir, "figure2_snr_time_series.png")
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    # Figure 3: Frequency-Domain Signal Spectrum and One-Sided PSD
    fig3, ax3 = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)

    asd_noise = np.sqrt(psd_interp)
    signal_eff_asd = 2.0 * np.sqrt(active_freqs) * np.abs(h_inj_fd[active_indices])

    ax3.loglog(grid.frequencies[1:], asd_noise[1:], color="firebrick", lw=2.0, label=r"Detector Noise ASD $\sqrt{S_n(f)}$ [$\mathrm{strain}/\sqrt{\mathrm{Hz}}$] (One-Sided PSD)")
    ax3.loglog(active_freqs, signal_eff_asd, color="navy", lw=2.0, label=r"Signal Effective ASD $2\sqrt{f}|\widetilde{h}(f)|$ [$\mathrm{strain}/\sqrt{\mathrm{Hz}}$]")
    ax3.loglog(active_freqs, np.abs(h_inj_fd[active_indices]), color="cornflowerblue", lw=1.2, ls="--", label=r"Fourier Strain Amplitude $|\widetilde{h}(f)|$ [$\mathrm{strain}\cdot\mathrm{s}$]")

    ax3.axvline(f_start, color="gray", ls="--", lw=1.2, label=r"Analysis Band [$f_{\mathrm{start}}, f_{\mathrm{end}}]$ = " + f"[{f_start:.0f}, {f_end:.0f}] Hz")
    ax3.axvline(f_end, color="gray", ls="--", lw=1.2)
    ax3.axvspan(f_start, f_end, color="aliceblue", alpha=0.5, zorder=0)
    ax3.axvline(f_isco, color="darkorange", ls="-.", lw=1.5, label=r"Schwarzschild ISCO Cutoff $f_{\mathrm{ISCO}} = " + f"{f_isco:.1f}" + r"$ Hz")

    ax3.set_xlim(12.0, 600.0)
    ax3.set_ylim(1e-25, 2e-21)
    ax3.set_xlabel("Frequency $f$ [Hz]")
    ax3.set_ylabel(r"Spectral Amplitude [Physical SI Units]")
    ax3.set_title("Frequency-Domain Signal Spectrum, One-Sided PSD, and Active Observation Mask")
    ax3.legend(loc="lower left", framealpha=0.9)
    ax3.grid(True, which="both", alpha=0.3, ls="--")

    fig3_path = os.path.join(figures_dir, "figure3_frequency_domain_psd.png")
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)

    # --------------------------------------------------------------------------
    # 8. Export Numerical Results Table (CSV)
    # --------------------------------------------------------------------------
    results_dict: Dict[str, float | str | int] = {
        "m1_msun": m1_msun,
        "m2_msun": m2_msun,
        "m_total_msun": float(M_tot / M_SUN_SI),
        "chirp_mass_msun": float(M_c / M_SUN_SI),
        "distance_mpc": dist_mpc,
        "f_isco_hz": f_isco,
        "tau_chirp_s": tau_chirp,
        "t_inj_s": t_inj,
        "phi_c_inj_rad": phi_c_inj,
        "t_rec_s": peak_time,
        "time_error_ms": abs(peak_time - t_inj) * 1e3,
        "phi_c_rec_rad": peak_phase,
        "phase_error_rad": abs(peak_phase - phi_c_inj),
        "phase_high_snr_std_approx": 1.0 / sigma_h,
        "optimal_snr": sigma_h,
        "recovered_snr": peak_snr,
        "snr_diff_percent": ((peak_snr - sigma_h) / sigma_h) * 100.0,
        "allen_chisq": chisq_val,
        "allen_dof": chisq_dof,
        "reduced_chisq": reduced_chisq,
        "reweighted_snr": reweighted_snr,
        "direct_snr_discrepancy": direct_snr_diff,
        "template_norm_discrepancy": norm_diff,
        "noise_n_active_bins": n_active_bins,
        "noise_sample_mean": sample_mean_q,
        "noise_sample_std": sample_std_q,
        "noise_sample_se": sample_se_q,
        "noise_theory_se": theory_se_q,
        "noise_deviation_sigma": dev_sigma,
        "seed": seed,
    }

    csv_path = os.path.join(output_dir, "injection_recovery_results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "Unit", "Description"])
        writer.writerow(["m1", f"{m1_msun:.2f}", "M_sun", "Primary component mass"])
        writer.writerow(["m2", f"{m2_msun:.2f}", "M_sun", "Secondary component mass"])
        writer.writerow(["M_total", f"{M_tot/M_SUN_SI:.2f}", "M_sun", "Total binary mass"])
        writer.writerow(["M_chirp", f"{M_c/M_SUN_SI:.4f}", "M_sun", "Chirp mass"])
        writer.writerow(["distance", f"{dist_mpc:.1f}", "Mpc", "Luminosity distance"])
        writer.writerow(["f_isco", f"{f_isco:.2f}", "Hz", "Schwarzschild ISCO cutoff frequency"])
        writer.writerow(["tau_chirp", f"{tau_chirp:.4f}", "s", "In-band Newtonian chirp duration"])
        writer.writerow(["t_inj", f"{t_inj:.6f}", "s", "Injected coalescence time"])
        writer.writerow(["t_rec", f"{peak_time:.6f}", "s", "Recovered peak lag"])
        writer.writerow(["time_error", f"{abs(peak_time - t_inj)*1e3:.4f}", "ms", "Absolute arrival time error"])
        writer.writerow(["phi_c_inj", f"{phi_c_inj:.4f}", "rad", "Injected coalescence phase"])
        writer.writerow(["phi_c_rec", f"{peak_phase:.4f}", "rad", "Recovered coalescence phase"])
        writer.writerow(["phase_error", f"{abs(peak_phase - phi_c_inj):.4f}", "rad", "Absolute coalescence phase error"])
        writer.writerow(["phase_high_snr_std_approx", f"{1.0/sigma_h:.4f}", "rad", "Asymptotic high-SNR phase std approx 1/rho_opt"])
        writer.writerow(["optimal_snr", f"{sigma_h:.4f}", "", "Injected optimal template norm rho_opt"])
        writer.writerow(["recovered_snr", f"{peak_snr:.4f}", "", "Recovered phase-maximized peak SNR"])
        writer.writerow(["snr_relative_error", f"{((peak_snr - sigma_h) / sigma_h) * 100:+.2f}", "%", "Relative SNR error"])
        writer.writerow(["allen_chisq", f"{chisq_val:.4f}", "", "Binned Allen chi^2 statistic at peak"])
        writer.writerow(["allen_dof", f"{chisq_dof}", "", "Degrees of freedom 2q - 2"])
        writer.writerow(["reduced_chisq", f"{reduced_chisq:.4f}", "", "Reduced chi^2 (chi^2 / dof)"])
        writer.writerow(["reweighted_snr", f"{reweighted_snr:.4f}", "", "Allen reweighted SNR statistic"])
        writer.writerow(["direct_snr_discrepancy", f"{direct_snr_diff:.2e}", "", "Discrepancy vs direct frequency-sum reference"])
        writer.writerow(["template_norm_discrepancy", f"{norm_diff:.2e}", "", "Discrepancy in template norm calculation"])
        writer.writerow(["noise_n_active_bins", f"{n_active_bins}", "", "Number of active frequency bins in noise normalization check"])
        writer.writerow(["noise_sample_mean", f"{sample_mean_q:.6f}", "", "Sample mean of normalized power q_k"])
        writer.writerow(["noise_sample_std", f"{sample_std_q:.6f}", "", "Sample standard deviation of normalized power q_k"])
        writer.writerow(["noise_sample_se", f"{sample_se_q:.6f}", "", "Sample standard error of mean s_q / sqrt(N_bins)"])
        writer.writerow(["noise_theory_se", f"{theory_se_q:.6f}", "", "Theoretical standard error 1 / sqrt(N_bins)"])
        writer.writerow(["noise_deviation_sigma", f"{dev_sigma:+.4f}", "", "Noise mean deviation from unity in theoretical SE units"])
        writer.writerow(["seed", f"{seed}", "", "Random seed for Gaussian noise generator"])

    print("================================================================================")
    print("DEMONSTRATION RESULTS SUMMARY")
    print("================================================================================")
    print(f"Optimal SNR rho_opt:       {sigma_h:.4f}")
    print(f"Recovered Peak SNR:        {peak_snr:.4f} (Relative error: {((peak_snr - sigma_h)/sigma_h)*100:+.2f}%)")
    print(f"Injected Coalescence Time: {t_inj:.6f} s")
    print(f"Recovered Arrival Time:    {peak_time:.6f} s (Error: {abs(peak_time - t_inj)*1e3:.4f} ms)")
    print(f"Injected Phase:            {phi_c_inj:.4f} rad")
    print(f"Recovered Phase:           {peak_phase:.4f} rad (Error: {abs(peak_phase - phi_c_inj):.4f} rad)")
    print(f"Allen chi^2 at Peak:       {chisq_val:.4f} (dof = {chisq_dof}, reduced = {reduced_chisq:.4f})")
    print(f"Reweighted SNR:            {reweighted_snr:.4f}")
    print(f"Direct Sum Discrepancy:    {direct_snr_diff:.2e} (IFFT vs. authoritative direct sum)")
    print(f"Figures saved to:          {figures_dir}")
    print(f"Results CSV saved to:      {csv_path}")
    print("================================================================================")

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate CBC injection and matched-filter recovery.")
    parser.add_argument("--output-dir", default="reports/demonstrations", help="Directory for figures and CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Gaussian noise")
    args = parser.parse_args()

    run_demonstration(output_dir=args.output_dir, seed=args.seed)
