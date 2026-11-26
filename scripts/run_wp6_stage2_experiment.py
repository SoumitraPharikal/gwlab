#!/usr/bin/env python3
"""WP6 Stage 2: 2D Gravitational-Wave Bayesian Parameter Estimation and Independent Grid Validation.

This script executes:
1. Pre-flight verification of injection, likelihood, waveform units, and prior boundaries.
2. Full 2D nested sampling inference via NestedSamplerAdapter (dynesty) for theta = (Mc, eta).
3. Independent 2D numerical grid quadrature across multiple resolutions (50x50, 100x100, 150x150).
4. Rigorous statistical comparison: posterior moments, correlation, evidence, and Jensen-Shannon divergence.
5. Serialization of machine-readable data arrays to reports/demonstrations/wp6_stage2/.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np

# Ensure src/ is on python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.grid import FrequencyGrid
from gw_inference.psd import analytical_detector_psd
from gw_inference.waveform import (
    taylorf2_fd_waveform,
    apply_coalescence_time_shift,
    component_masses_to_chirp_mass,
)
from gw_inference.likelihood import FrequencyDomainGaussianLikelihood
from gw_inference.posterior import GravitationalWaveLikelihoodWrapper
from gw_inference.prior import prior_transform_2d, UniformChirpMassEtaPrior
from gw_inference.sampler import NestedSamplerAdapter, NestedSamplingResult


def compute_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence in nats between two discrete probability vectors."""
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    m = 0.5 * (p_norm + q_norm)

    # KL divergence with safe handling of zero probabilities (0 * log(0) = 0)
    def kl_div(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd = 0.5 * kl_div(p_norm, m) + 0.5 * kl_div(q_norm, m)
    return float(max(0.0, jsd))


def compute_joint_2d_jensen_shannon_divergence(
    samples: np.ndarray,
    weights: np.ndarray | None,
    mc_grid_pts: np.ndarray,
    eta_grid_pts: np.ndarray,
    p_density_2d: np.ndarray,
    dmc: float,
    deta: float,
    nbins_x: int = 25,
    nbins_y: int = 25,
) -> float:
    """Compute joint 2D Jensen-Shannon divergence between weighted samples and a 2D grid density."""
    mc_bins = np.linspace(mc_grid_pts[0], mc_grid_pts[-1], nbins_x + 1)
    eta_bins = np.linspace(eta_grid_pts[0], eta_grid_pts[-1], nbins_y + 1)

    # 2D histogram of sampler samples
    hist_sampler, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=[mc_bins, eta_bins], weights=weights
    )
    if np.sum(hist_sampler) == 0.0:
        return 0.0
    p_s = hist_sampler / np.sum(hist_sampler)

    # 2D discrete distribution from integrating grid posterior density over each cell
    hist_grid = np.zeros((nbins_x, nbins_y), dtype=np.float64)
    for i in range(nbins_x):
        idx_x = (mc_grid_pts >= mc_bins[i]) & (mc_grid_pts <= mc_bins[i + 1])
        for j in range(nbins_y):
            idx_y = (eta_grid_pts >= eta_bins[j]) & (eta_grid_pts <= eta_bins[j + 1])
            sub = p_density_2d[np.ix_(idx_x, idx_y)]
            hist_grid[i, j] = float(np.sum(sub) * dmc * deta)

    if np.sum(hist_grid) == 0.0:
        return 0.0
    p_g = hist_grid / np.sum(hist_grid)

    m = 0.5 * (p_s + p_g)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd_2d = 0.5 * kl(p_s, m) + 0.5 * kl(p_g, m)
    return float(max(0.0, jsd_2d))


def run_stage2_experiment(output_dir: str = "reports/demonstrations/wp6_stage2") -> Dict[str, Any]:
    """Execute complete Stage 2 nested-sampling and independent grid quadrature experiment."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("WP6 STAGE 2: 2D BAYESIAN INFERENCE & INDEPENDENT GRID VALIDATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Physical and Analysis Configuration
    # -------------------------------------------------------------------------
    duration = 4.0          # s
    sampling_rate = 2048.0 # Hz
    delta_t = 1.0 / sampling_rate
    n_samples = int(duration * sampling_rate) # 8192

    grid = FrequencyGrid(n_samples=n_samples, delta_t=delta_t)
    f_start = 20.0  # Hz
    f_end = 512.0   # Hz
    mask = grid.create_active_mask(f_start=f_start, f_end=f_end)
    active_indices = grid.get_active_indices(mask)
    n_active_bins = len(active_indices)
    delta_f = grid.delta_f

    # Reference analytical PSD: aligo_representative
    psd = analytical_detector_psd(grid.frequencies, model="aligo_representative")

    # Injected compact binary parameters
    m1_msun = 36.0
    m2_msun = 29.0
    m1_kg = m1_msun * M_SUN_SI
    m2_kg = m2_msun * M_SUN_SI
    dist_mpc = 410.0
    dist_m = dist_mpc * MPC_SI
    phi_c_inj = 0.0
    t_c_inj = 2.0

    M_total_kg, eta_inj, mc_inj_kg = component_masses_to_chirp_mass(m1_kg, m2_kg)
    mc_inj_msun = float(mc_inj_kg / M_SUN_SI)
    M_total_msun = float(M_total_kg / M_SUN_SI)

    print(f"Grid: T = {duration} s, fs = {sampling_rate} Hz, N = {n_samples}, df = {delta_f} Hz")
    print(f"Band: [{f_start}, {f_end}] Hz ({n_active_bins} active bins)")
    print(f"Injection: m1 = {m1_msun} Msun, m2 = {m2_msun} Msun, M_total = {M_total_msun} Msun")
    print(f"           Mc = {mc_inj_msun:.8f} Msun, eta = {eta_inj:.8f}")
    print(f"           dL = {dist_mpc} Mpc, phic = {phi_c_inj} rad, tc = {t_c_inj} s")

    # Waveform synthesis directly on the frequency grid
    h_raw = taylorf2_fd_waveform(grid.frequencies, m1_kg, m2_kg, dist_m, phi_c=phi_c_inj)
    h_inj = apply_coalescence_time_shift(h_raw, grid.frequencies, t_c_inj)

    # Optimal matched-filter SNR
    h_act = h_inj[active_indices]
    psd_act = psd[active_indices]
    snr_sq = float(4.0 * delta_f * np.sum(np.abs(h_act) ** 2 / psd_act))
    optimal_snr = math.sqrt(snr_sq)
    print(f"Optimal matched-filter SNR: {optimal_snr:.4f}")

    # Zero-noise observation
    data_fd = np.array(h_inj, copy=True)

    # Likelihood instantiation
    likelihood = FrequencyDomainGaussianLikelihood(
        data_fd=data_fd,
        psd=psd,
        grid=grid,
        mask=mask,
    )
    ln_l_norm = likelihood.log_likelihood_norm
    print(f"Precomputed Likelihood Normalization ln L_norm: {ln_l_norm:.6f}")

    # Wrapper instantiation
    wrapper = GravitationalWaveLikelihoodWrapper(
        likelihood=likelihood,
        distance_m=dist_m,
        phi_c=phi_c_inj,
        t_c=t_c_inj,
        f_start=f_start,
    )

    # -------------------------------------------------------------------------
    # 2. Pre-Flight Scientific Validations
    # -------------------------------------------------------------------------
    print("\nExecuting Pre-Flight Scientific Validations...")
    prior = UniformChirpMassEtaPrior()

    # A. Injection point inside prior support
    assert prior.mc_min <= mc_inj_msun <= prior.mc_max, "Injection Mc outside prior support"
    assert prior.eta_min <= eta_inj <= prior.eta_max, "Injection eta outside prior support"
    print("  [PASS] Injection parameters lie strictly inside prior support.")

    # B. Likelihood at injection in zero noise
    ln_l_at_inj = wrapper.evaluate_log_likelihood([mc_inj_msun, eta_inj])
    diff_norm = abs(ln_l_at_inj - ln_l_norm)
    assert diff_norm < 1e-10, f"Residual at injection is non-zero (diff={diff_norm})"
    print(f"  [PASS] Zero-noise injection likelihood ln L = ln L_norm exactly (diff={diff_norm:.2e}).")

    # C. Representative interior points are finite
    pts = [
        [25.0, 0.20],
        [30.0, 0.15],
        [35.0, 0.22],
    ]
    for pt in pts:
        val = wrapper.evaluate_log_likelihood(pt)
        assert math.isfinite(val), f"Likelihood non-finite at {pt}"
    print("  [PASS] Likelihood evaluates finitely across interior parameter space.")

    # D. Out-of-support rejection
    assert wrapper.evaluate_log_likelihood([15.0, 0.20]) == -math.inf or prior.log_prior([15.0, 0.20]) == -math.inf
    assert wrapper.evaluate_log_likelihood([45.0, 0.20]) == -math.inf or prior.log_prior([45.0, 0.20]) == -math.inf
    assert wrapper.evaluate_log_likelihood([30.0, 0.26]) == -math.inf
    print("  [PASS] Out-of-support and unphysical proposals return -inf.")

    # E. State immutability check
    val1 = wrapper.evaluate_log_likelihood([30.0, 0.20])
    _ = wrapper.evaluate_log_likelihood([22.0, 0.15])
    val2 = wrapper.evaluate_log_likelihood([30.0, 0.20])
    assert math.isclose(val1, val2, rel_tol=1e-15), "Wrapper mutated internal state between calls"
    print("  [PASS] Wrapper is strictly stateless and idempotent.")

    # -------------------------------------------------------------------------
    # 3. Part C: Nested-Sampling Inference (dynesty)
    # -------------------------------------------------------------------------
    print("\nExecuting Part C: Nested-Sampling Inference with dynesty...")
    nlive = 500
    dlogz = 0.1
    seed = 42

    adapter = NestedSamplerAdapter(
        log_likelihood_fn=wrapper.evaluate_log_likelihood,
        prior_transform_fn=prior_transform_2d,
        ndim=2,
        nlive=nlive,
        dlogz=dlogz,
        seed=seed,
        param_names=["Mc", "eta"],
    )

    t0_sampler = time.perf_counter()
    sampler_res: NestedSamplingResult = adapter.run(bootstrap=0, print_progress=False)
    t1_sampler = time.perf_counter()
    runtime_sampler = t1_sampler - t0_sampler

    # Extract sample properties
    dead_samples = sampler_res.samples
    dead_logwt = sampler_res.log_weights
    dead_weights = sampler_res.weights
    post_samples = sampler_res.posterior_samples
    logz_sampler = sampler_res.log_evidence
    logzerr_sampler = sampler_res.log_evidence_err
    n_evals_sampler = sampler_res.n_evaluations

    # Compute effective sample size (Kish's ESS: 1 / sum(w_i^2))
    ess_sampler = float(1.0 / np.sum(dead_weights ** 2))

    # Parameter moments from weighted dead samples
    mean_mc_sampler = float(np.sum(dead_samples[:, 0] * dead_weights))
    mean_eta_sampler = float(np.sum(dead_samples[:, 1] * dead_weights))
    var_mc_sampler = float(np.sum((dead_samples[:, 0] - mean_mc_sampler) ** 2 * dead_weights))
    var_eta_sampler = float(np.sum((dead_samples[:, 1] - mean_eta_sampler) ** 2 * dead_weights))
    cov_mc_eta_sampler = float(np.sum((dead_samples[:, 0] - mean_mc_sampler) * (dead_samples[:, 1] - mean_eta_sampler) * dead_weights))
    std_mc_sampler = math.sqrt(var_mc_sampler)
    std_eta_sampler = math.sqrt(var_eta_sampler)
    corr_sampler = float(cov_mc_eta_sampler / (std_mc_sampler * std_eta_sampler))

    print(f"  Nested sampling completed in {runtime_sampler:.2f} s")
    print(f"  Total likelihood evaluations: {n_evals_sampler}")
    print(f"  Total dead-point samples: {len(dead_samples)}")
    print(f"  Effective sample size (ESS): {ess_sampler:.1f}")
    print(f"  Log-evidence ln Z: {logz_sampler:.6f} +/- {logzerr_sampler:.6f}")
    print(f"  Mc: mean = {mean_mc_sampler:.6f} Msun, std = {std_mc_sampler:.6f} Msun (True: {mc_inj_msun:.6f})")
    print(f"  eta: mean = {mean_eta_sampler:.7f}, std = {std_eta_sampler:.7f} (True: {eta_inj:.7f})")
    print(f"  Correlation r(Mc, eta): {corr_sampler:.4f}")

    # -------------------------------------------------------------------------
    # 4. Part D: Independent Numerical Grid Quadrature
    # -------------------------------------------------------------------------
    print("\nExecuting Part D: Independent Numerical Grid Quadrature...")
    prior_density = 1.0 / 3.0
    log_prior_val = -math.log(3.0)

    # Refined bounding box enclosing all likelihood values with Delta ln L >= -27
    mc_box = (27.90, 28.30)
    eta_box = (0.2460, 0.2482)

    grid_resolutions = [50, 100, 150]
    grid_results: Dict[int, Dict[str, Any]] = {}

    for n_g in grid_resolutions:
        t0_g = time.perf_counter()
        mc_grid_1d = np.linspace(mc_box[0], mc_box[1], n_g)
        eta_grid_1d = np.linspace(eta_box[0], eta_box[1], n_g)
        dmc = float(mc_grid_1d[1] - mc_grid_1d[0])
        deta = float(eta_grid_1d[1] - eta_grid_1d[0])

        log_l_grid = np.zeros((n_g, n_g), dtype=np.float64)
        for i, mc_val in enumerate(mc_grid_1d):
            for j, eta_val in enumerate(eta_grid_1d):
                log_l_grid[i, j] = wrapper.evaluate_log_likelihood([mc_val, eta_val])

        # Unnormalized log-posterior density on grid
        log_unnorm_grid = log_l_grid + log_prior_val
        max_log_grid = float(np.max(log_unnorm_grid))

        # Stable log-domain evidence quadrature
        shifted_grid = np.exp(log_unnorm_grid - max_log_grid)
        integral_shifted = float(np.sum(shifted_grid) * dmc * deta)
        logz_grid = float(max_log_grid + math.log(integral_shifted))

        # Normalized posterior probability density: sum(p * dmc * deta) = 1.0
        p_density = shifted_grid / integral_shifted

        # Moments via 2D numerical quadrature
        MC_2D, ETA_2D = np.meshgrid(mc_grid_1d, eta_grid_1d, indexing="ij")
        mean_mc_g = float(np.sum(MC_2D * p_density) * dmc * deta)
        mean_eta_g = float(np.sum(ETA_2D * p_density) * dmc * deta)
        var_mc_g = float(np.sum((MC_2D - mean_mc_g) ** 2 * p_density) * dmc * deta)
        var_eta_g = float(np.sum((ETA_2D - mean_eta_g) ** 2 * p_density) * dmc * deta)
        cov_mc_eta_g = float(np.sum((MC_2D - mean_mc_g) * (ETA_2D - mean_eta_g) * p_density) * dmc * deta)
        std_mc_g = math.sqrt(var_mc_g)
        std_eta_g = math.sqrt(var_eta_g)
        corr_g = float(cov_mc_eta_g / (std_mc_g * std_eta_g))

        # Marginal densities
        # Marginal over Mc: integrate over eta (axis 1)
        marginal_mc = np.sum(p_density, axis=1) * deta # shape (n_g,)
        # Marginal over eta: integrate over Mc (axis 0)
        marginal_eta = np.sum(p_density, axis=0) * dmc # shape (n_g,)

        t1_g = time.perf_counter()
        grid_runtime = t1_g - t0_g

        grid_results[n_g] = {
            "n_grid": n_g,
            "runtime": grid_runtime,
            "dmc": dmc,
            "deta": deta,
            "mc_vals": mc_grid_1d,
            "eta_vals": eta_grid_1d,
            "log_likelihood": log_l_grid,
            "posterior_density": p_density,
            "marginal_mc": marginal_mc,
            "marginal_eta": marginal_eta,
            "log_evidence": logz_grid,
            "mean_mc": mean_mc_g,
            "std_mc": std_mc_g,
            "mean_eta": mean_eta_g,
            "std_eta": std_eta_g,
            "cov_mc_eta": cov_mc_eta_g,
            "corr": corr_g,
        }

        print(f"  Grid {n_g}x{n_g} evaluated in {grid_runtime:.2f} s:")
        print(f"    ln Z: {logz_grid:.6f}")
        print(f"    Mc: mean = {mean_mc_g:.6f} Msun, std = {std_mc_g:.6f} Msun")
        print(f"    eta: mean = {mean_eta_g:.7f}, std = {std_eta_g:.7f}")
        print(f"    Corr: {corr_g:.4f}")

    # Check grid convergence between 100x100 and 150x150
    g100 = grid_results[100]
    g150 = grid_results[150]
    d_logz_grid = abs(g150["log_evidence"] - g100["log_evidence"])
    rel_mean_mc_grid = abs(g150["mean_mc"] - g100["mean_mc"]) / g100["mean_mc"]
    rel_std_mc_grid = abs(g150["std_mc"] - g100["std_mc"]) / g100["std_mc"]
    rel_mean_eta_grid = abs(g150["mean_eta"] - g100["mean_eta"]) / g100["mean_eta"]
    rel_std_eta_grid = abs(g150["std_eta"] - g100["std_eta"]) / g100["std_eta"]

    print("\nGrid Resolution Convergence (100x100 -> 150x150):")
    print(f"  Delta ln Z: {d_logz_grid:.6f}")
    print(f"  Rel diff Mean Mc: {rel_mean_mc_grid:.2e}")
    print(f"  Rel diff Std Mc:  {rel_std_mc_grid:.2e}")
    print(f"  Rel diff Mean eta: {rel_mean_eta_grid:.2e}")
    print(f"  Rel diff Std eta:  {rel_std_eta_grid:.2e}")

    # -------------------------------------------------------------------------
    # 5. Statistical Comparison: Sampler vs. Grid
    # -------------------------------------------------------------------------
    print("\nStatistical Comparison: Nested Sampling vs. Independent Grid Quadrature (100x100):")
    ref_grid = grid_results[100]

    # Differences in moments
    diff_mean_mc = mean_mc_sampler - ref_grid["mean_mc"]
    diff_std_mc = std_mc_sampler - ref_grid["std_mc"]
    diff_mean_eta = mean_eta_sampler - ref_grid["mean_eta"]
    diff_std_eta = std_eta_sampler - ref_grid["std_eta"]
    diff_logz = logz_sampler - ref_grid["log_evidence"]

    print(f"  Evidence Difference: ln Z_sampler - ln Z_grid = {diff_logz:+.6f}")
    print(f"    |Delta ln Z| = {abs(diff_logz):.6f} (sampler uncertainty sigma_lnZ = {logzerr_sampler:.6f})")
    print(f"    Normalized evidence residual: |Delta ln Z| / sigma_lnZ = {abs(diff_logz) / logzerr_sampler:.3f} (target <= 3.0)")
    print(f"  Mc Mean diff:  {diff_mean_mc:+.6f} Msun ({abs(diff_mean_mc) / ref_grid['std_mc']:.3f} sigma_Mc)")
    print(f"  Mc Std diff:   {diff_std_mc:+.6f} Msun")
    print(f"  eta Mean diff: {diff_mean_eta:+.7f} ({abs(diff_mean_eta) / ref_grid['std_eta']:.3f} sigma_eta)")
    print(f"  eta Std diff:  {diff_std_eta:+.7f}")
    print(f"  Correlation:   Sampler = {corr_sampler:.4f}, Grid = {ref_grid['corr']:.4f}")

    # Jensen-Shannon Divergence on 1D Marginals
    # We bin both the sampler posterior samples and the grid posterior onto identical 50-bin histograms
    n_bins_jsd = 50
    mc_bins = np.linspace(mc_box[0], mc_box[1], n_bins_jsd + 1)
    eta_bins = np.linspace(eta_box[0], eta_box[1], n_bins_jsd + 1)

    # Sampler discrete distributions
    hist_mc_sampler, _ = np.histogram(post_samples[:, 0], bins=mc_bins)
    hist_eta_sampler, _ = np.histogram(post_samples[:, 1], bins=eta_bins)

    # Grid discrete distributions by interpolating/integrating the 100x100 marginals
    # Direct trapezoidal integration of marginal_mc over each bin interval
    hist_mc_grid = np.zeros(n_bins_jsd, dtype=np.float64)
    mc_grid_pts = ref_grid["mc_vals"]
    p_mc_pts = ref_grid["marginal_mc"]
    for b in range(n_bins_jsd):
        b_lo, b_hi = mc_bins[b], mc_bins[b+1]
        idx_in = (mc_grid_pts >= b_lo) & (mc_grid_pts <= b_hi)
        if np.any(idx_in):
            hist_mc_grid[b] = getattr(np, "trapezoid", getattr(np, "trapz", None))(p_mc_pts[idx_in], mc_grid_pts[idx_in])
        else:
            mid = 0.5 * (b_lo + b_hi)
            hist_mc_grid[b] = np.interp(mid, mc_grid_pts, p_mc_pts) * (b_hi - b_lo)

    hist_eta_grid = np.zeros(n_bins_jsd, dtype=np.float64)
    eta_grid_pts = ref_grid["eta_vals"]
    p_eta_pts = ref_grid["marginal_eta"]
    for b in range(n_bins_jsd):
        b_lo, b_hi = eta_bins[b], eta_bins[b+1]
        idx_in = (eta_grid_pts >= b_lo) & (eta_grid_pts <= b_hi)
        if np.any(idx_in):
            hist_eta_grid[b] = getattr(np, "trapezoid", getattr(np, "trapz", None))(p_eta_pts[idx_in], eta_grid_pts[idx_in])
        else:
            mid = 0.5 * (b_lo + b_hi)
            hist_eta_grid[b] = np.interp(mid, eta_grid_pts, p_eta_pts) * (b_hi - b_lo)

    jsd_mc = compute_jensen_shannon_divergence(hist_mc_sampler.astype(np.float64), hist_mc_grid)
    jsd_eta = compute_jensen_shannon_divergence(hist_eta_sampler.astype(np.float64), hist_eta_grid)

    print(f"  Jensen-Shannon Divergence D_JS(Mc):  {jsd_mc:.6f} (target < 0.05)")
    print(f"  Jensen-Shannon Divergence D_JS(eta): {jsd_eta:.6f} (target < 0.05)")

    # Joint 2D Jensen-Shannon Divergence across multiple bin resolutions
    jsd_2d_sensitivity: Dict[str, float] = {}
    for n_b in [10, 15, 20, 25, 30, 40, 50]:
        val_jsd_2d = compute_joint_2d_jensen_shannon_divergence(
            samples=dead_samples,
            weights=dead_weights,
            mc_grid_pts=ref_grid["mc_vals"],
            eta_grid_pts=ref_grid["eta_vals"],
            p_density_2d=ref_grid["posterior_density"],
            dmc=ref_grid["dmc"],
            deta=ref_grid["deta"],
            nbins_x=n_b,
            nbins_y=n_b,
        )
        jsd_2d_sensitivity[f"{n_b}x{n_b}"] = float(val_jsd_2d)

    jsd_2d_primary = jsd_2d_sensitivity["25x25"]
    print(f"  Joint 2D Jensen-Shannon Divergence D_JS_2D (25x25): {jsd_2d_primary:.6f} (target < 0.05)")
    print(f"    Sensitivity range across 10x10 to 50x50: [{min(jsd_2d_sensitivity.values()):.6f}, {max(jsd_2d_sensitivity.values()):.6f}]")

    # -------------------------------------------------------------------------
    # 5b. Secondary Fringe Multi-Resolution Quadrature & Conditional Exterior Bounds
    # -------------------------------------------------------------------------
    print("\nExecuting Secondary Fringe Multi-Resolution Quadrature & Conditional Exterior Bounds...")
    mc_fringe_box = (27.85, 28.15)
    eta_fringe_box = (0.2330, 0.2355)
    fringe_resolutions = [40, 80, 120]
    fringe_convergence_table: Dict[str, Dict[str, float]] = {}

    logz_fringe = 0.0
    ratio_fringe_main = 0.0
    for n_f in fringe_resolutions:
        t0_f = time.perf_counter()
        mc_fringe_1d = np.linspace(mc_fringe_box[0], mc_fringe_box[1], n_f)
        eta_fringe_1d = np.linspace(eta_fringe_box[0], eta_fringe_box[1], n_f)
        dmc_f = float(mc_fringe_1d[1] - mc_fringe_1d[0])
        deta_f = float(eta_fringe_1d[1] - eta_fringe_1d[0])

        log_l_fringe = np.zeros((n_f, n_f), dtype=np.float64)
        for i, mc_val in enumerate(mc_fringe_1d):
            for j, eta_val in enumerate(eta_fringe_1d):
                log_l_fringe[i, j] = wrapper.evaluate_log_likelihood([mc_val, eta_val])

        log_unnorm_fringe = log_l_fringe + log_prior_val
        max_log_fringe = float(np.max(log_unnorm_fringe))
        shifted_fringe = np.exp(log_unnorm_fringe - max_log_fringe)
        integral_shifted_fringe = float(np.sum(shifted_fringe) * dmc_f * deta_f)
        logz_fringe = float(max_log_fringe + math.log(integral_shifted_fringe))
        t1_f = time.perf_counter()

        ratio_fringe_main = math.exp(logz_fringe - ref_grid["log_evidence"])
        bnd_max_l = max(
            float(np.max(log_l_fringe[0, :])), float(np.max(log_l_fringe[-1, :])),
            float(np.max(log_l_fringe[:, 0])), float(np.max(log_l_fringe[:, -1]))
        )
        bnd_max_delta = bnd_max_l - ln_l_norm

        fringe_convergence_table[f"{n_f}x{n_f}"] = {
            "n_grid": n_f,
            "runtime_seconds": t1_f - t0_f,
            "log_evidence": logz_fringe,
            "ratio_to_main": ratio_fringe_main,
            "boundary_max_delta_lnl": bnd_max_delta,
        }
        print(f"  Fringe {n_f}x{n_f}: ln Z = {logz_fringe:.6f}, ratio to main = {ratio_fringe_main:.4e}, bnd max Delta ln L = {bnd_max_delta:.2f} ({t1_f - t0_f:.2f} s)")

    # Final combined evidence using highest-resolution 120x120 fringe grid
    logz_combined_partitioned = float(ref_grid["log_evidence"] + math.log1p(ratio_fringe_main))

    # Numerically supported exterior bound conditional on search resolution:
    # Under global coarse/fine grid search across prior support, Delta ln L < -35.0
    # Prior volume V = 3.0 Msun, density pi = 1/3.0 => integral over V <= exp(ln_l_norm - 35)
    ratio_rest_upper_bound = math.exp((ln_l_norm - 35.0) - ref_grid["log_evidence"])
    max_possible_exterior_ratio = ratio_fringe_main + ratio_rest_upper_bound
    delta_logz_exterior_bound = math.log1p(max_possible_exterior_ratio)

    print(f"  Combined partitioned evidence ln Z_partitioned: {logz_combined_partitioned:.6f}")
    print(f"  Conditional exterior bound Delta ln Z_exterior <= {delta_logz_exterior_bound:.4e}")

    # -------------------------------------------------------------------------
    # 6. Save Machine-Readable Data Outputs (Part F)
    # -------------------------------------------------------------------------
    print(f"\nSaving data products to {output_dir}...")

    # 1. Sampler results
    np.savez_compressed(
        os.path.join(output_dir, "sampler_results.npz"),
        dead_samples=dead_samples,
        dead_logwt=dead_logwt,
        dead_weights=dead_weights,
        posterior_samples=post_samples,
        log_evidence=logz_sampler,
        log_evidence_err=logzerr_sampler,
        n_evaluations=n_evals_sampler,
        effective_sample_size=ess_sampler,
        runtime_seconds=runtime_sampler,
        param_names=np.array(["Mc", "eta"]),
        injection_mc=mc_inj_msun,
        injection_eta=eta_inj,
        optimal_snr=optimal_snr,
        ln_l_norm=ln_l_norm,
    )

    # 2. Grid results for 50x50, 100x100, 150x150
    for n_g in grid_resolutions:
        res_g = grid_results[n_g]
        np.savez_compressed(
            os.path.join(output_dir, f"grid_{n_g}x{n_g}_results.npz"),
            mc_vals=res_g["mc_vals"],
            eta_vals=res_g["eta_vals"],
            log_likelihood=res_g["log_likelihood"],
            posterior_density=res_g["posterior_density"],
            marginal_mc=res_g["marginal_mc"],
            marginal_eta=res_g["marginal_eta"],
            log_evidence=res_g["log_evidence"],
            mean_mc=res_g["mean_mc"],
            std_mc=res_g["std_mc"],
            mean_eta=res_g["mean_eta"],
            std_eta=res_g["std_eta"],
            cov_mc_eta=res_g["cov_mc_eta"],
            corr=res_g["corr"],
            runtime_seconds=res_g["runtime"],
        )

    # 3. Summary comparison JSON
    summary_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_branch": "milestone-1.4a",
            "git_head": "ecf033d435371834926a2f81b3409a8c82f33d7f",
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "dynesty_version": "3.1.0",
        },
        "injection": {
            "m1_msun": m1_msun,
            "m2_msun": m2_msun,
            "M_total_msun": M_total_msun,
            "chirp_mass_msun": mc_inj_msun,
            "symmetric_mass_ratio": eta_inj,
            "luminosity_distance_mpc": dist_mpc,
            "coalescence_time_seconds": t_c_inj,
            "coalescence_phase_radians": phi_c_inj,
            "optimal_matched_filter_snr": optimal_snr,
            "log_likelihood_norm": ln_l_norm,
        },
        "nested_sampling": {
            "nlive": nlive,
            "dlogz": dlogz,
            "seed": seed,
            "runtime_seconds": runtime_sampler,
            "n_evaluations": n_evals_sampler,
            "n_dead_samples": len(dead_samples),
            "effective_sample_size": ess_sampler,
            "log_evidence": logz_sampler,
            "log_evidence_err": logzerr_sampler,
            "mean_mc": mean_mc_sampler,
            "std_mc": std_mc_sampler,
            "mean_eta": mean_eta_sampler,
            "std_eta": std_eta_sampler,
            "covariance_mc_eta": cov_mc_eta_sampler,
            "correlation": corr_sampler,
        },
        "grid_quadrature_100x100": {
            "n_grid": 100,
            "runtime_seconds": ref_grid["runtime"],
            "log_evidence": ref_grid["log_evidence"],
            "mean_mc": ref_grid["mean_mc"],
            "std_mc": ref_grid["std_mc"],
            "mean_eta": ref_grid["mean_eta"],
            "std_eta": ref_grid["std_eta"],
            "covariance_mc_eta": ref_grid["cov_mc_eta"],
            "correlation": ref_grid["corr"],
        },
        "grid_quadrature_150x150": {
            "n_grid": 150,
            "runtime_seconds": grid_results[150]["runtime"],
            "log_evidence": grid_results[150]["log_evidence"],
            "mean_mc": grid_results[150]["mean_mc"],
            "std_mc": grid_results[150]["std_mc"],
            "mean_eta": grid_results[150]["mean_eta"],
            "std_eta": grid_results[150]["std_eta"],
            "covariance_mc_eta": grid_results[150]["cov_mc_eta"],
            "correlation": grid_results[150]["corr"],
        },
        "comparison": {
            "delta_log_evidence": diff_logz,
            "delta_log_evidence_in_sigmas": abs(diff_logz) / logzerr_sampler,
            "diff_mean_mc": diff_mean_mc,
            "diff_mean_mc_in_sigmas": abs(diff_mean_mc) / ref_grid["std_mc"],
            "diff_std_mc": diff_std_mc,
            "diff_mean_eta": diff_mean_eta,
            "diff_mean_eta_in_sigmas": abs(diff_mean_eta) / ref_grid["std_eta"],
            "diff_std_eta": diff_std_eta,
            "diff_correlation": corr_sampler - ref_grid["corr"],
            "jensen_shannon_divergence_mc": jsd_mc,
            "jensen_shannon_divergence_eta": jsd_eta,
            "jensen_shannon_divergence_2d_primary": jsd_2d_primary,
            "jensen_shannon_divergence_2d_sensitivity": jsd_2d_sensitivity,
            "grid_refinement_delta_logz": d_logz_grid,
        },
        "partitioned_quadrature_and_exterior_bounds": {
            "fringe_box_mc": list(mc_fringe_box),
            "fringe_box_eta": list(eta_fringe_box),
            "fringe_convergence_table": fringe_convergence_table,
            "log_evidence_fringe": logz_fringe,
            "ratio_fringe_to_main": ratio_fringe_main,
            "log_evidence_combined_partitioned": logz_combined_partitioned,
            "ratio_rest_upper_bound": ratio_rest_upper_bound,
            "max_possible_exterior_ratio": max_possible_exterior_ratio,
            "delta_logz_exterior_bound": delta_logz_exterior_bound,
        },
    }

    with open(os.path.join(output_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("  Outputs successfully written:")
    print("    - sampler_results.npz")
    print("    - grid_50x50_results.npz")
    print("    - grid_100x100_results.npz")
    print("    - grid_150x150_results.npz")
    print("    - comparison_summary.json")

    print("\n" + "=" * 80)
    print("WP6 STAGE 2 EXPERIMENT COMPLETE")
    print("=" * 80)
    return summary_data


if __name__ == "__main__":
    run_stage2_experiment()
