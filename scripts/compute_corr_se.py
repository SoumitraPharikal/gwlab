"""Compute standard errors for empirical lag correlation using 50 runs."""

from pathlib import Path
import numpy as np

def compute_se():
    results_path = Path("reports/demonstrations/wp6_stage5/stage_b/welch_stage_b_results.npz")
    npz_data = np.load(results_path)
    psd_realizations = npz_data["psd_realizations"]
    active_indices = npz_data["active_indices"]
    
    act_reals = psd_realizations[:, active_indices]  # (50, 1969)
    n_runs = act_reals.shape[0]  # 50
    n_bins = act_reals.shape[1]  # 1969
    
    # 1. Ensemble method (as in run_wp6_stage5_stage_b.py)
    # Centered by ensemble mean per bin
    act_devs = act_reals - np.mean(act_reals, axis=0, keepdims=True)
    cov_diag = np.sum(act_devs ** 2, axis=0) / (n_runs - 1.0)
    
    emp_rho_ensemble = np.zeros(11)
    emp_rho_ensemble[0] = 1.0
    for d in range(1, 11):
        v1 = act_devs[:, :-d]
        v2 = act_devs[:, d:]
        s1 = np.sqrt(cov_diag[:-d])
        s2 = np.sqrt(cov_diag[d:])
        cross_cov = np.sum(v1 * v2, axis=0) / (n_runs - 1.0)
        corr_pairs = cross_cov / (s1 * s2)
        emp_rho_ensemble[d] = float(np.mean(corr_pairs))
        
    # Jackknife across the 50 runs for the ensemble estimator
    jack_rhos = np.zeros((n_runs, 11))
    for i in range(n_runs):
        sub_reals = np.delete(act_reals, i, axis=0)  # (49, 1969)
        sub_devs = sub_reals - np.mean(sub_reals, axis=0, keepdims=True)
        sub_var = np.sum(sub_devs ** 2, axis=0) / 48.0
        jack_rhos[i, 0] = 1.0
        for d in range(1, 11):
            v1 = sub_devs[:, :-d]
            v2 = sub_devs[:, d:]
            s1 = np.sqrt(sub_var[:-d])
            s2 = np.sqrt(sub_var[d:])
            cross_cov = np.sum(v1 * v2, axis=0) / 48.0
            jack_rhos[i, d] = float(np.mean(cross_cov / (s1 * s2)))
            
    # Jackknife standard error formula: sqrt((n-1)/n * sum((theta_i - theta_bar)^2))
    jack_mean = np.mean(jack_rhos, axis=0)
    jack_se = np.sqrt(((n_runs - 1) / n_runs) * np.sum((jack_rhos - jack_mean) ** 2, axis=0))

    # 2. Per-realization autocorrelation (sample autocorrelation along frequency axis)
    # Centered by realization band mean
    per_run_rhos = np.zeros((n_runs, 11))
    for r in range(n_runs):
        row = act_reals[r]
        row_dev = row - np.mean(row)
        denom = np.sum(row_dev ** 2)
        per_run_rhos[r, 0] = 1.0
        for d in range(1, 11):
            per_run_rhos[r, d] = np.sum(row_dev[:-d] * row_dev[d:]) / denom
            
    per_run_mean = np.mean(per_run_rhos, axis=0)
    per_run_std = np.std(per_run_rhos, axis=0, ddof=1)
    per_run_se = per_run_std / np.sqrt(n_runs)

    theory_rho = npz_data["theoretical_autocorr_lags"][:11]

    print("Lag | Theory | Ens Est | Jackknife SE | Per-Run Mean | Per-Run Std | Per-Run SE")
    for d in range(11):
        print(f"{d:3d} | {theory_rho[d]*100:6.3f}% | {emp_rho_ensemble[d]*100:7.3f}% | {jack_se[d]*100:12.4f}% | {per_run_mean[d]*100:12.3f}% | {per_run_std[d]*100:11.4f}% | {per_run_se[d]*100:10.4f}%")

if __name__ == "__main__":
    compute_se()
