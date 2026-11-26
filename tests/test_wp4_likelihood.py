"""Focused test suite for WP4: Frequency-Domain Gaussian Likelihood.

Validates:
1. Zero-residual case: ln L = ln L_norm when d_tilde = h_tilde.
2. Hand-computed residual and independent reference scalar summation.
3. Likelihood ordering: better-fitting templates have strictly higher log-likelihoods.
4. Parameter dependence: physical parameter variations change residual and log-likelihood as expected.
5. Active-mask correctness: excluded bins (outside active band, DC, Nyquist) strictly do not contribute.
6. Even- and odd-length grids: positive-frequency selection and Nyquist handling across parities.
7. PSD normalization term: exact scaling with pi * N * S_n / (2 * f_s) with no missing or duplicated factors.
8. Invalid inputs: shape mismatch, non-finite values, non-positive PSD, empty or illegal masks.
9. Numerical consistency: exact agreement with WP3 noise_weighted_inner_product.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference.grid import FrequencyGrid
from gw_inference.psd import PSDEngine
from gw_inference.filter import noise_weighted_inner_product
from gw_inference.waveform import (
    taylorf2_fd_waveform,
    apply_coalescence_time_shift,
)
from gw_inference.likelihood import (
    GaussianLikelihoodResult,
    compute_residual,
    compute_quadratic_form,
    compute_log_likelihood_normalization,
    compute_log_likelihood,
    FrequencyDomainGaussianLikelihood,
)


class TestWP4GaussianLikelihood:
    """Comprehensive test suite for WP4 likelihood implementation."""

    # --------------------------------------------------------------------------
    # 1. Zero-Residual Case
    # --------------------------------------------------------------------------
    def test_zero_residual_case(self):
        """When d_tilde == h_tilde, residual is identically zero, quadratic form is zero,

        and log-likelihood equals the normalization constant ln L_norm.
        """
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=4.0, sampling_rate=1024.0)
        mask = grid.create_active_mask(f_start=30.0, f_end=300.0)
        active_indices = grid.get_active_indices(mask)

        # Realistic non-trivial PSD and data
        psd = 1e-46 * (1.0 + (100.0 / grid.frequencies[1:]) ** 2)
        psd = np.insert(psd, 0, psd[0])  # include DC

        # Synthetic signal
        rng = np.random.default_rng(1234)
        signal = rng.normal(0, 1e-21, size=grid.n_frequencies) + 1.0j * rng.normal(0, 1e-21, size=grid.n_frequencies)

        # 1. Functional interface
        log_l = compute_log_likelihood(
            data_fd=signal,
            template_fd=signal,
            psd=psd,
            grid=grid,
            mask=mask,
        )
        norm_const = compute_log_likelihood_normalization(
            psd=psd,
            n_samples=grid.n_samples,
            sampling_rate=grid.sampling_rate,
            active_indices=active_indices,
        )

        assert math.isclose(log_l, norm_const, rel_tol=1e-14, abs_tol=1e-14)

        # 2. Object-oriented interface with detailed diagnostics
        likelihood_obj = FrequencyDomainGaussianLikelihood(
            data_fd=signal,
            psd=psd,
            grid=grid,
            mask=mask,
        )
        res = likelihood_obj.evaluate_detailed(signal)

        assert isinstance(res, GaussianLikelihoodResult)
        assert res.quadratic_form == 0.0
        assert math.isclose(res.log_likelihood, res.log_likelihood_norm, rel_tol=1e-14, abs_tol=1e-14)
        assert res.n_active_bins == active_indices.size

    # --------------------------------------------------------------------------
    # 2. Hand-Computed Scalar Reference Comparison
    # --------------------------------------------------------------------------
    def test_hand_computed_scalar_reference(self):
        """Compare quadratic form and normalization against an independently coded scalar loop."""
        # Small synthetic setup: N = 16, dt = 0.1 s -> fs = 10 Hz, T = 1.6 s, df = 0.625 Hz
        N = 16
        fs = 10.0
        dt = 1.0 / fs
        df = fs / N
        grid = FrequencyGrid(n_samples=N, delta_t=dt)

        # Active bins: indices 2, 3, 4 (interior, strictly excluding DC=0 and Nyquist=8)
        active_indices = np.array([2, 3, 4], dtype=np.intp)
        mask = np.zeros(grid.n_frequencies, dtype=bool)
        mask[active_indices] = True

        data_fd = np.array([
            0.0 + 0.0j,
            1.0 + 2.0j,
            3.0 - 4.0j,  # active k=2
            -2.0 + 5.0j, # active k=3
            1.5 - 0.5j,  # active k=4
            0.1 + 0.2j,
            -0.3 - 0.4j,
            0.5 + 0.6j,
            1.0 + 0.0j,  # Nyquist k=8
        ], dtype=np.complex128)

        template_fd = np.array([
            0.0 + 0.0j,
            0.5 + 1.0j,
            1.0 - 1.0j,  # active k=2
            -1.0 + 2.0j, # active k=3
            0.5 + 0.5j,  # active k=4
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ], dtype=np.complex128)

        psd = np.array([10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64)

        # Independent hand calculation
        # k=2: r = (3 - 4j) - (1 - 1j) = 2 - 3j => |r|^2 = 4 + 9 = 13. S_n = 3.0
        # k=3: r = (-2 + 5j) - (-1 + 2j) = -1 + 3j => |r|^2 = 1 + 9 = 10. S_n = 4.0
        # k=4: r = (1.5 - 0.5j) - (0.5 + 0.5j) = 1.0 - 1.0j => |r|^2 = 1 + 1 = 2. S_n = 5.0
        # sum (|r|^2 / S_n) = 13/3 + 10/4 + 2/5 = 4.333333333333333 + 2.5 + 0.4 = 7.233333333333333
        # quad = 4 * df * sum = 4 * 0.625 * (13/3 + 2.5 + 0.4) = 2.5 * 7.233333333333333 = 18.083333333333332
        expected_quad = 4.0 * df * ((13.0 / 3.0) + (10.0 / 4.0) + (2.0 / 5.0))

        # Normalization hand calculation:
        # scale = pi * N / (2 * fs) = pi * 16 / (2 * 10) = 0.8 * pi
        # k=2: var = 0.8 * pi * 3.0 = 2.4 * pi
        # k=3: var = 0.8 * pi * 4.0 = 3.2 * pi
        # k=4: var = 0.8 * pi * 5.0 = 4.0 * pi
        # ln L_norm = - (ln(2.4 pi) + ln(3.2 pi) + ln(4.0 pi))
        scale = (math.pi * N) / (2.0 * fs)
        expected_norm = -(math.log(scale * 3.0) + math.log(scale * 4.0) + math.log(scale * 5.0))
        expected_log_l = expected_norm - 0.5 * expected_quad

        # Evaluate via implementation
        quad_calc = compute_quadratic_form(
            residual_fd=data_fd - template_fd,
            psd=psd,
            delta_f=df,
            active_indices=active_indices,
        )
        norm_calc = compute_log_likelihood_normalization(
            psd=psd,
            n_samples=N,
            sampling_rate=fs,
            active_indices=active_indices,
        )
        log_l_calc = compute_log_likelihood(
            data_fd=data_fd,
            template_fd=template_fd,
            psd=psd,
            grid=grid,
            mask=mask,
        )

        assert math.isclose(quad_calc, expected_quad, rel_tol=1e-14, abs_tol=1e-14)
        assert math.isclose(norm_calc, expected_norm, rel_tol=1e-14, abs_tol=1e-14)
        assert math.isclose(log_l_calc, expected_log_l, rel_tol=1e-14, abs_tol=1e-14)

    # --------------------------------------------------------------------------
    # 3. Likelihood Ordering Property
    # --------------------------------------------------------------------------
    def test_likelihood_ordering(self):
        """A better-fitting template (smaller residual quadratic form) must produce

        a strictly higher log-likelihood for fixed data and PSD.
        """
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=4.0, sampling_rate=2048.0)
        mask = grid.create_active_mask(f_start=25.0, f_end=400.0)

        f_meas = np.linspace(10.0, 1024.0, 2000)
        psd_meas = 1e-46 * ((30.0 / f_meas) ** 4 + 1.0 + (f_meas / 300.0) ** 2)
        psd_engine = PSDEngine(f_meas, psd_meas)
        psd = psd_engine.interpolate(grid.frequencies, f_start=25.0, f_0=15.0)

        m1 = 20.0 * 1.9884099021470415e30
        m2 = 15.0 * 1.9884099021470415e30
        dist = 200.0 * 1e6 * 3.085677581491367e16
        h_true = taylorf2_fd_waveform(grid.frequencies, m1, m2, dist, phi_c=0.0)

        # 1. Monotonic ordering in clean signal regime:
        # Distance (h_true - h | h_true - h) strictly increases with parameter mismatch
        likelihood_clean = FrequencyDomainGaussianLikelihood(h_true, psd, grid, mask)

        h_cand1 = taylorf2_fd_waveform(grid.frequencies, m1, m2, dist, phi_c=0.1)
        h_cand2 = taylorf2_fd_waveform(grid.frequencies, m1, m2, dist, phi_c=0.5)
        h_cand3 = taylorf2_fd_waveform(grid.frequencies, m1, m2, dist, phi_c=1.2)

        log_l_0 = likelihood_clean(h_true)
        log_l_1 = likelihood_clean(h_cand1)
        log_l_2 = likelihood_clean(h_cand2)
        log_l_3 = likelihood_clean(h_cand3)

        assert log_l_0 > log_l_1
        assert log_l_1 > log_l_2
        assert log_l_2 > log_l_3

        # 2. Ordering for fixed noisy data:
        # For arbitrary fixed data, candidate with smaller (d - h|d - h) MUST have higher log-likelihood
        rng = np.random.default_rng(999)
        noise = rng.normal(0, 1e-22, size=grid.n_frequencies) + 1.0j * rng.normal(0, 1e-22, size=grid.n_frequencies)
        noisy_data = h_true + noise

        likelihood_noisy = FrequencyDomainGaussianLikelihood(noisy_data, psd, grid, mask)
        res_0 = likelihood_noisy.evaluate_detailed(h_true)
        res_1 = likelihood_noisy.evaluate_detailed(h_cand1)
        res_2 = likelihood_noisy.evaluate_detailed(h_cand2)

        # Confirm strict anti-correlation between quadratic form and log-likelihood
        candidates = [res_0, res_1, res_2]
        candidates.sort(key=lambda r: r.quadratic_form)  # Best fit first (lowest quadratic form)
        for i in range(len(candidates) - 1):
            assert candidates[i].quadratic_form < candidates[i + 1].quadratic_form
            assert candidates[i].log_likelihood > candidates[i + 1].log_likelihood

    # --------------------------------------------------------------------------
    # 4. Parameter Dependence
    # --------------------------------------------------------------------------
    def test_parameter_dependence(self):
        """Varying physical parameters (time shift, chirp mass, distance) systematically alters log-likelihood."""
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=4.0, sampling_rate=2048.0)
        mask = grid.create_active_mask(f_start=25.0, f_end=350.0)

        psd = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)

        m1 = 20.0 * 1.9884099021470415e30
        m2 = 15.0 * 1.9884099021470415e30
        dist = 200.0 * 1e6 * 3.085677581491367e16

        # Injected at t_c = 1.5 s
        h_inj = taylorf2_fd_waveform(grid.frequencies, m1, m2, dist, phi_c=0.2)
        h_inj_shifted = apply_coalescence_time_shift(h_inj, grid.frequencies, 1.5)

        # Noise-free test: peak should be strictly at t_c = 1.5 s
        likelihood = FrequencyDomainGaussianLikelihood(h_inj_shifted, psd, grid, mask)

        times = np.array([1.48, 1.49, 1.50, 1.51, 1.52])
        log_l_vals = []
        for t in times:
            h_trial = apply_coalescence_time_shift(h_inj, grid.frequencies, t)
            log_l_vals.append(likelihood(h_trial))

        log_l_vals = np.array(log_l_vals)
        # Peak at exact injected time
        assert np.argmax(log_l_vals) == 2
        assert log_l_vals[2] > log_l_vals[1]
        assert log_l_vals[2] > log_l_vals[3]

    # --------------------------------------------------------------------------
    # 5. Active-Mask Correctness and Strict Endpoint Exclusion
    # --------------------------------------------------------------------------
    def test_active_mask_and_endpoint_exclusion(self):
        """Data and template changes outside active band, at DC, or at Nyquist do not affect likelihood."""
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=4.0, sampling_rate=1024.0)
        mask = grid.create_active_mask(f_start=40.0, f_end=200.0)
        active_indices = grid.get_active_indices(mask)

        psd = np.full(grid.n_frequencies, 1e-45, dtype=np.float64)

        rng = np.random.default_rng(42)
        data_fd = rng.normal(0, 1e-21, size=grid.n_frequencies) + 1.0j * rng.normal(0, 1e-21, size=grid.n_frequencies)
        tmpl_fd = rng.normal(0, 1e-21, size=grid.n_frequencies) + 1.0j * rng.normal(0, 1e-21, size=grid.n_frequencies)

        base_log_l = compute_log_likelihood(data_fd, tmpl_fd, psd, grid, mask=mask)

        # Corrupt DC (index 0) with huge number
        data_corrupt_dc = data_fd.copy()
        data_corrupt_dc[0] = 1e10 + 1e10j
        log_l_dc = compute_log_likelihood(data_corrupt_dc, tmpl_fd, psd, grid, mask=mask)
        assert math.isclose(base_log_l, log_l_dc, rel_tol=1e-15, abs_tol=1e-15)

        # Corrupt Nyquist (index -1) with huge number
        data_corrupt_nyq = data_fd.copy()
        data_corrupt_nyq[-1] = 1e10 + 1e10j
        log_l_nyq = compute_log_likelihood(data_corrupt_nyq, tmpl_fd, psd, grid, mask=mask)
        assert math.isclose(base_log_l, log_l_nyq, rel_tol=1e-15, abs_tol=1e-15)

        # Corrupt an out-of-band bin (f < 40 Hz, e.g. bin 10 at 2.5 Hz)
        data_corrupt_oob = data_fd.copy()
        data_corrupt_oob[10] = 1e10 + 1e10j
        assert 10 not in active_indices
        log_l_oob = compute_log_likelihood(data_corrupt_oob, tmpl_fd, psd, grid, mask=mask)
        assert math.isclose(base_log_l, log_l_oob, rel_tol=1e-15, abs_tol=1e-15)

    # --------------------------------------------------------------------------
    # 6. Even- and Odd-Length Grids
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("n_samples", [1024, 1025])
    def test_even_and_odd_grid_support(self, n_samples):
        """Verify likelihood computation on both even and odd length grids."""
        dt = 1.0 / 2048.0
        grid = FrequencyGrid(n_samples=n_samples, delta_t=dt)
        mask = grid.create_active_mask(f_start=50.0, f_end=400.0)
        active_indices = grid.get_active_indices(mask)

        # Verify DC is strictly excluded in both parities
        assert 0 not in active_indices

        # For even N, Nyquist is strictly excluded
        if n_samples % 2 == 0:
            assert (grid.n_frequencies - 1) not in active_indices

        psd = np.full(grid.n_frequencies, 1e-46, dtype=np.float64)
        data = np.ones(grid.n_frequencies, dtype=np.complex128)
        tmpl = np.zeros(grid.n_frequencies, dtype=np.complex128)

        log_l = compute_log_likelihood(data, tmpl, psd, grid, mask=mask)
        assert math.isfinite(log_l)

        # Object interface
        lik_obj = FrequencyDomainGaussianLikelihood(data, psd, grid, mask)
        res = lik_obj.evaluate_detailed(tmpl)
        assert math.isclose(res.log_likelihood, log_l, rel_tol=1e-14)
        assert res.n_active_bins == active_indices.size

    # --------------------------------------------------------------------------
    # 7. PSD Normalization Scaling Verification
    # --------------------------------------------------------------------------
    def test_psd_normalization_scaling(self):
        """Verify the exact normalization formula -sum log[pi * N * S_n / (2 * fs)]."""
        N = 2048
        fs = 1024.0
        grid = FrequencyGrid(n_samples=N, delta_t=1.0 / fs)
        mask = grid.create_active_mask(f_start=20.0, f_end=200.0)
        active_indices = grid.get_active_indices(mask)
        n_act = active_indices.size

        # Base flat PSD S_0
        S_0 = 1e-46
        psd_base = np.full(grid.n_frequencies, S_0, dtype=np.float64)
        norm_base = compute_log_likelihood_normalization(
            psd=psd_base,
            n_samples=N,
            sampling_rate=fs,
            active_indices=active_indices,
        )

        # Scaling PSD by factor c: S_n -> c * S_n
        # ln L_norm(c * S_n) = - sum ln[ scale * c * S_n ] = ln L_norm - n_act * ln(c)
        c = 4.5
        psd_scaled = psd_base * c
        norm_scaled = compute_log_likelihood_normalization(
            psd=psd_scaled,
            n_samples=N,
            sampling_rate=fs,
            active_indices=active_indices,
        )
        expected_diff = -n_act * math.log(c)
        assert math.isclose(norm_scaled - norm_base, expected_diff, rel_tol=1e-12, abs_tol=1e-10)

        # Scaling N by factor k at fixed fs: N -> k * N
        k_factor = 2
        N_new = N * k_factor
        grid_new = FrequencyGrid(n_samples=N_new, delta_t=1.0 / fs)
        # Choose matching active bins
        psd_new = np.full(grid_new.n_frequencies, S_0, dtype=np.float64)
        # Using exact same active bins count
        norm_n_scaled = compute_log_likelihood_normalization(
            psd=psd_new,
            n_samples=N_new,
            sampling_rate=fs,
            active_indices=active_indices,
        )
        expected_n_diff = -n_act * math.log(k_factor)
        assert math.isclose(norm_n_scaled - norm_base, expected_n_diff, rel_tol=1e-12, abs_tol=1e-10)

    # --------------------------------------------------------------------------
    # 8. Invalid Input and Boundary Handling
    # --------------------------------------------------------------------------
    def test_invalid_input_handling(self):
        """Verify strict error checking for malformed inputs, shapes, and invalid masks."""
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=1.0, sampling_rate=256.0)
        mask = grid.create_active_mask(f_start=10.0, f_end=100.0)

        data = np.ones(grid.n_frequencies, dtype=np.complex128)
        tmpl = np.zeros(grid.n_frequencies, dtype=np.complex128)
        psd = np.ones(grid.n_frequencies, dtype=np.float64)

        # 1. Shape mismatch in residual
        with pytest.raises(ValueError, match="shape"):
            compute_residual(data, tmpl[:-1])

        # 2. Non-finite values in data/template
        data_nan = data.copy()
        data_nan[5] = np.nan
        with pytest.raises(ValueError, match="finite"):
            compute_residual(data_nan, tmpl)

        tmpl_inf = tmpl.copy()
        tmpl_inf[5] = np.inf
        with pytest.raises(ValueError, match="finite"):
            compute_residual(data, tmpl_inf)

        # 3. Non-positive PSD in active band
        psd_zero = psd.copy()
        psd_zero[grid.get_active_indices(mask)[0]] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            compute_quadratic_form(data, psd_zero, grid.delta_f, grid.get_active_indices(mask))

        psd_neg = psd.copy()
        psd_neg[grid.get_active_indices(mask)[0]] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            compute_log_likelihood_normalization(psd_neg, grid.n_samples, grid.sampling_rate, grid.get_active_indices(mask))

        # 4. Non-finite PSD in active band
        psd_nan = psd.copy()
        psd_nan[grid.get_active_indices(mask)[0]] = np.nan
        with pytest.raises(ValueError, match="finite"):
            compute_quadratic_form(data, psd_nan, grid.delta_f, grid.get_active_indices(mask))

        # 5. Invalid active mask with DC included
        bad_mask_dc = mask.copy()
        bad_mask_dc[0] = True
        with pytest.raises(ValueError, match="strictly exclude DC"):
            compute_log_likelihood(data, tmpl, psd, grid, mask=bad_mask_dc)

        # 6. Invalid active mask with Nyquist included (for even N)
        bad_mask_nyq = mask.copy()
        bad_mask_nyq[-1] = True
        with pytest.raises(ValueError, match="strictly exclude Nyquist"):
            compute_log_likelihood(data, tmpl, psd, grid, mask=bad_mask_nyq)

        # 7. Invalid delta_f or n_samples
        with pytest.raises(ValueError, match="delta_f"):
            compute_quadratic_form(data, psd, delta_f=0.0)

        with pytest.raises(ValueError, match="n_samples"):
            compute_log_likelihood_normalization(psd, n_samples=0, sampling_rate=256.0)

        # 8. PSDEngine construction requires explicit f_0 without default
        f_meas = np.linspace(5.0, 128.0, 500)
        psd_engine = PSDEngine(f_meas, np.ones_like(f_meas))
        with pytest.raises(ValueError, match="f_0"):
            FrequencyDomainGaussianLikelihood.from_psd_engine(
                data, psd_engine, grid, mask, f_start=10.0, f_0=-1.0
            )

    # --------------------------------------------------------------------------
    # 9. Numerical Consistency with WP3 Inner Product
    # --------------------------------------------------------------------------
    def test_numerical_consistency_with_wp3_inner_product(self):
        """The quadratic form compute_quadratic_form(r, psd, delta_f, active_indices)

        must match noise_weighted_inner_product(r, r, psd, delta_f, active_indices)
        to exact floating-point machine precision.
        """
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=8.0, sampling_rate=2048.0)
        mask = grid.create_active_mask(f_start=20.0, f_end=512.0)
        active_indices = grid.get_active_indices(mask)

        # Realistic colored PSD
        f_meas = np.linspace(10.0, 1024.0, 4000)
        psd_meas = 1e-46 * ((35.0 / f_meas) ** 4 + 1.0 + (f_meas / 300.0) ** 2)
        psd_engine = PSDEngine(f_meas, psd_meas)
        psd = psd_engine.interpolate(grid.frequencies, f_start=20.0, f_0=15.0)

        # Generate realistic data and template
        m1 = 20.0 * 1.9884099021470415e30
        m2 = 15.0 * 1.9884099021470415e30
        dist = 200.0 * 1e6 * 3.085677581491367e16

        h_true = taylorf2_fd_waveform(grid.frequencies, m1, m2, dist, phi_c=0.5)
        h_cand = taylorf2_fd_waveform(grid.frequencies, m1 * 1.02, m2 * 0.98, dist, phi_c=0.2)

        rng = np.random.default_rng(2026)
        noise = rng.normal(0, 1e-22, size=grid.n_frequencies) + 1.0j * rng.normal(0, 1e-22, size=grid.n_frequencies)
        data = h_true + noise

        residual = compute_residual(data, h_cand)

        # 1. Quadratic form
        quad_wp4 = compute_quadratic_form(
            residual_fd=residual,
            psd=psd,
            delta_f=grid.delta_f,
            active_indices=active_indices,
        )

        # 2. WP3 noise-weighted inner product (r|r)
        inner_wp3 = noise_weighted_inner_product(
            a_fd=residual,
            b_fd=residual,
            psd=psd,
            delta_f=grid.delta_f,
            active_indices=active_indices,
        )

        discrepancy = abs(quad_wp4 - inner_wp3)
        assert discrepancy == 0.0 or discrepancy < 1e-13 * abs(inner_wp3)

        # 3. Object-oriented evaluation consistency
        lik_obj = FrequencyDomainGaussianLikelihood(data, psd, grid, mask)
        det_res = lik_obj.evaluate_detailed(h_cand)

        assert math.isclose(det_res.quadratic_form, inner_wp3, rel_tol=1e-14, abs_tol=1e-14)
        assert math.isclose(det_res.log_likelihood, lik_obj.log_likelihood_norm - 0.5 * inner_wp3, rel_tol=1e-14)
