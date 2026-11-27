"""Generate publication-quality PDF figures for reports/archive/GWLab_Technical_Report.tex (and GWLab_Complete_Scientific_Report.tex).

Uses only deterministic gwlab.api.service calculations and saved read-only artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from gwlab.api.service import (
    compute_matched_filter_response,
    compute_noise_and_psd_response,
    get_experiments_explorer,
    get_inference_posterior,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def make_fig1_psd_asd() -> None:
    res = compute_noise_and_psd_response(
        psd_model="aligo_representative",
        f_min_hz=20.0,
        f_max_hz=512.0,
        sampling_rate_hz=2048.0,
        duration_obs_s=64.0,
        segment_duration_s=4.0,
        overlap_fraction=0.5,
        tukey_alpha=0.25,
        seed=7101,
    )
    f = np.array(res["frequencies"])
    asd_a = np.array(res["asd_analytic"])
    asd_w = np.array(res["asd_welch"])
    meta = res["metadata"]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.loglog(f, asd_w, color="#6b7280", lw=1.0, alpha=0.85, label=f"Welch ASD (64 s synthetic noise, seed={meta['seed']}, {meta['n_segments']} segs)")
    ax.loglog(f, asd_a, color="#b45309", lw=1.8, label="Analytical aLIGO-like PSD proxy $\\sqrt{S_n(f)}$")
    ax.axvspan(20.0, 512.0, color="#fef3c7", alpha=0.25, label="Active band $[20, 512]$ Hz")
    ax.set_xlabel("Frequency $f$ [Hz]")
    ax.set_ylabel("Amplitude Spectral Density $\\sqrt{S_n(f)}$ [$\\mathrm{Hz}^{-1/2}$]")
    ax.set_title("Analytical PSD Proxy vs. Synthetic Colored-Noise Welch Estimate")
    ax.set_xlim(18, 550)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_psd_asd.pdf")
    plt.close(fig)
    print(f"FIG1: n_bins={len(f)}, n_segments={meta['n_segments']}, kappa={meta['kappa_overlap']:.4f}, nu_eff={meta['nu_eff']:.2f}")


def make_fig2_matched_filter() -> None:
    matched = compute_matched_filter_response(delta_mc_msun=0.0, seed=7151)
    mismatched = compute_matched_filter_response(delta_mc_msun=2.5, seed=7151)

    t = np.array(matched["zoom_time_s"])
    snr_m = np.array(matched["zoom_abs"])
    snr_mm = np.array(mismatched["zoom_abs"])
    m_meta = matched["metadata"]
    mm_meta = mismatched["metadata"]

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.plot(t, snr_m, color="#1d4ed8", lw=1.6, label=f"Matched template ($\\Delta\\mathcal{{M}}_c = 0.0\\,M_\\odot$, peak={m_meta['recovered_peak_snr']:.2f})")
    ax.plot(t, snr_mm, color="#dc2626", lw=1.3, ls="--", label=f"Mismatched template ($\\Delta\\mathcal{{M}}_c = +2.5\\,M_\\odot$, peak={mm_meta['recovered_peak_snr']:.2f})")
    ax.axvline(m_meta["injected_tc_s"], color="#374151", lw=1.0, ls=":", label=f"Injected $t_c = {m_meta['injected_tc_s']:.2f}$ s ($\\rho_{{\\mathrm{{opt}}}}={m_meta['optimal_snr']:.2f}$)")
    ax.set_xlabel("Time $t$ [s]")
    ax.set_ylabel("Matched-Filter SNR $|\\rho(t)|$ [dimensionless]")
    ax.set_title("Matched-Filter SNR Time Series Around Coalescence (Synthetic Signal + Noise)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_matched_filter_snr.pdf")
    plt.close(fig)
    print(f"FIG2: opt_snr={m_meta['optimal_snr']:.4f}, peak_matched={m_meta['recovered_peak_snr']:.4f}, peak_mismatched={mm_meta['recovered_peak_snr']:.4f}")


def make_fig3_posteriors() -> None:
    post = get_inference_posterior(param_x="mc", param_y="eta", bins_1d=40, bins_2d=36)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8))
    order = ["mc", "eta", "tc", "phic"]
    titles = {
        "mc": r"Chirp Mass $\mathcal{M}_c$ [$M_\odot$]",
        "eta": r"Symmetric Mass Ratio $\eta$ [dimensionless]",
        "tc": r"Coalescence Time $t_c$ [s]",
        "phic": r"Coalescence Phase $\phi_c$ [rad]",
    }
    for ax, key in zip(axes.flat, order):
        m = post["marginals_1d"][key]
        x = np.array(m["bin_centers"])
        y = np.array(m["density"])
        s = m["summary"]
        ax.bar(x, y, width=m["bin_width"] * 0.92, color="#0f766e", alpha=0.75, edgecolor="#115e59", linewidth=0.4)
        ax.axvline(m["injected"], color="#dc2626", lw=1.3, ls="--", label=f"Injected: {m['injected']:.4f}")
        if key == "phic":
            # Circular mean for periodic phase phi_c in [0, 2*pi)
            d_npz = np.load("experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz", allow_pickle=False)
            phic_s, w_s = d_npz["samples"][:, 3], d_npz["weights"]
            c_bar = float(np.sum(w_s * np.cos(phic_s)))
            s_bar = float(np.sum(w_s * np.sin(phic_s)))
            circ_mean = float(np.arctan2(s_bar, c_bar) % (2.0 * np.pi))
            r_bar = float(np.hypot(c_bar, s_bar))
            ax.axvline(circ_mean, color="#1e293b", lw=1.2, ls="-", label=f"Circ. mean: {circ_mean:.4f} (R={r_bar:.2f})")
        else:
            ax.axvline(s["median"], color="#1e293b", lw=1.2, ls="-", label=f"Median: {s['median']:.4f}")
        ax.set_xlabel(titles[key])
        ax.set_ylabel("Posterior Density")
        ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.9)
    fig.suptitle(
        f"Stage 6C Run A Weighted 1D Posterior Marginals (N={post['metadata']['n_samples']:,}, "
        f"Kish $N_{{\\mathrm{{eff}}}}={post['metadata']['kish_n_eff']:.1f}$)\n"
        f"Upper-1% $\\eta$ prior-range mass = {post['metadata']['eta_top1_pct']:.2f}%",
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "fig3_posterior_marginals.pdf")
    plt.close(fig)
    print(f"FIG3: logZ={post['metadata']['log_evidence']:.4f}+/-{post['metadata']['log_evidence_err']:.4f}, eta_top1_pct={post['metadata']['eta_top1_pct']:.2f}%")


def make_fig4_two_seed_evidence() -> None:
    exp = get_experiments_explorer()
    runs = exp["multi_seed_comparison"]["runs"]
    comp = exp["multi_seed_comparison"]

    labels = [f"{r['run_id']}\n(Seed {r['sampler_seed']})" for r in runs]
    logz = [r["log_evidence"] for r in runs]
    errs = [r["log_evidence_err"] for r in runs]
    baseline = min(logz)
    rel_logz = [z - baseline for z in logz]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y_pos = np.arange(len(runs))
    ax.errorbar(rel_logz, y_pos, xerr=errs, fmt="o", color="#1d4ed8", ecolor="#1e3a8a", elinewidth=1.8, capsize=5, markersize=7)
    for idx, (rz, z, e) in enumerate(zip(rel_logz, logz, errs)):
        ax.text(rz, idx + 0.16, f"$\\ln Z = {z:.4f} \\pm {e:.4f}$ nats", ha="center", va="bottom", fontsize=8.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, 1.65)
    ax.set_xlim(-0.8, max(rel_logz) + 1.2)
    ax.set_xlabel(r"Relative Log-Evidence $\ln Z - \min(\ln Z)$ [nats]")
    ax.set_title(
        f"Stage 5G.2 Two-Seed Evidence Sensitivity\n"
        f"$|\\Delta \\ln Z| = {comp['delta_logz']:.4f}$ nats, "
        f"$\\sigma_\\Delta = {comp['sigma_delta']:.4f}$, ratio $= {comp['discrepancy_ratio']:.2f}$",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_two_seed_evidence.pdf")
    plt.close(fig)
    print(f"FIG4: delta_logz={comp['delta_logz']:.4f}, sigma_delta={comp['sigma_delta']:.4f}, ratio={comp['discrepancy_ratio']:.2f}")


def make_fig0_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    boxes = [
        (0.2, 1.7, 2.1, 1.1, "#eff6ff", "#1d4ed8", "1. Waveform & PSD\nTaylorF2 3.5PN + Proxy\n[20, 512] Hz"),
        (2.7, 1.7, 2.1, 1.1, "#fef3c7", "#b45309", "2. Noise & Filter\nColored Noise + Welch\nMatched Filter SNR(t)"),
        (5.2, 1.7, 2.1, 1.1, "#f0fdf4", "#15803d", "3. Whittle Likelihood\nModel 2 Gaussian\nExact Norm ln L_norm"),
        (7.7, 1.7, 2.1, 1.1, "#f5f3ff", "#6d28d9", "4. Saved Archives\nNested Sampling (8101)\n2-Seed Diagnostic"),
        (2.6, 0.15, 4.8, 0.95, "#f8fafc", "#334155", "GWLab Presentation Layer (FastAPI + Next.js 16.4.6)\nDeterministic Endpoints | Read-Only Archives | Zero Live Sampling"),
    ]
    for x, y, w, h, fc, ec, txt in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", facecolor=fc, edgecolor=ec, linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.6, color="#111827")

    for x0, x1 in [(2.32, 2.68), (4.82, 5.18), (7.32, 7.68)]:
        ax.annotate("", xy=(x1, 2.25), xytext=(x0, 2.25), arrowprops=dict(arrowstyle="->", lw=1.3, color="#374151"))

    ax.annotate("", xy=(5.0, 1.15), xytext=(3.75, 1.68), arrowprops=dict(arrowstyle="->", lw=1.2, ls="--", color="#475569"))
    ax.annotate("", xy=(5.2, 1.15), xytext=(8.75, 1.68), arrowprops=dict(arrowstyle="->", lw=1.2, ls="--", color="#475569"))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig0_architecture_pipeline.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_fig0_pipeline()
    make_fig1_psd_asd()
    make_fig2_matched_filter()
    make_fig3_posteriors()
    make_fig4_two_seed_evidence()
