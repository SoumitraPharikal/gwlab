"""WP1 Unit and Integration Test Suite: Frequency Grid, Active Mask, and PSD Engine.

Validates:
1. FrequencyGrid construction from (N, dt) or (T, fs).
2. Static positive-frequency active observation mask K_+ on [f_start, f_end].
3. Strict exclusion of DC (k=0) and Nyquist (k=N/2) from active mask.
4. Mask invariance: static across calls.
5. Operational low-frequency continuation below f_start (Group A functional form).
6. C^0 continuity at f_start and configurable f_0.
7. PSDEngine interpolation, positivity preservation, rejection of non-positive values,
   and rejection of high-frequency extrapolation beyond measured band.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gw_inference.grid import FrequencyGrid
from gw_inference.psd import PSDEngine, low_frequency_psd_continuation, welch_psd
from gw_inference.noise import generate_white_noise


class TestFrequencyGrid:
    """Tests for discrete frequency grid abstraction and static active masks."""

    def test_grid_initialization_and_properties(self):
        """Verify grid properties derived from N=4096, fs=2048 Hz."""
        N = 4096
        fs = 2048.0
        dt = 1.0 / fs
        grid = FrequencyGrid(n_samples=N, delta_t=dt)

        assert grid.n_samples == N
        assert grid.delta_t == dt
        assert grid.sampling_rate == fs
        assert grid.duration == N * dt  # 2.0 seconds
        assert grid.delta_f == 0.5  # 1 / 2.0 s = 0.5 Hz
        assert grid.nyquist == 1024.0  # fs / 2
        assert grid.n_frequencies == N // 2 + 1  # 2049
        assert grid.frequencies[0] == 0.0  # DC
        assert math.isclose(grid.frequencies[-1], 1024.0, rel_tol=1e-14)  # Nyquist

    def test_from_duration_and_sampling_rate(self):
        """Verify constructor from duration T and sampling rate fs."""
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=4.0, sampling_rate=4096.0)
        assert grid.n_samples == 16384
        assert grid.delta_f == 0.25
        assert grid.nyquist == 2048.0

    def test_active_mask_gw150914_restricted_inspiral(self):
        """Verify static active mask for GW150914 restricted inspiral: [30, 63] Hz."""
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=4.0, sampling_rate=2048.0)
        # delta_f = 0.25 Hz
        mask = grid.create_active_mask(f_start=30.0, f_end=63.0)

        # 1. Endpoints must be strictly excluded
        assert not mask[0], "DC bin (k=0) must be False"
        assert not mask[-1], "Nyquist bin (k=N/2) must be False"

        # 2. Extract active frequencies
        active_f = grid.get_active_frequencies(mask)
        assert np.all(active_f >= 30.0)
        assert np.all(active_f <= 63.0)
        assert active_f[0] == 30.0
        assert active_f[-1] == 63.0

        # Number of bins from 30.0 to 63.0 with df=0.25 is (63 - 30) / 0.25 + 1 = 133
        assert len(active_f) == 133

    def test_active_mask_strict_endpoint_exclusion(self):
        """Verify that even if f_start=0 or f_end=f_Nyquist is requested, endpoints are excluded."""
        grid = FrequencyGrid(n_samples=512, delta_t=1.0 / 1024.0)  # Nyquist = 512 Hz
        with pytest.raises(ValueError, match="positive finite float"):
            grid.create_active_mask(f_start=0.0, f_end=100.0)

        # When f_end = Nyquist exactly:
        mask = grid.create_active_mask(f_start=10.0, f_end=512.0)
        assert not mask[0]
        assert not mask[-1], "Nyquist bin must remain strictly False"
        active_f = grid.get_active_frequencies(mask)
        assert active_f[-1] < 512.0

    def test_mask_invariance(self):
        """Verify that mask generation is deterministic and immutable."""
        grid = FrequencyGrid(n_samples=1024, delta_t=1.0 / 2048.0)
        m1 = grid.create_active_mask(f_start=20.0, f_end=100.0)
        m2 = grid.create_active_mask(f_start=20.0, f_end=100.0)
        np.testing.assert_array_equal(m1, m2)

    def test_odd_sample_count_grid_properties(self):
        """Verify grid properties for an odd sample count N."""
        N = 101
        dt = 0.001  # fs = 1000.0 Hz
        grid = FrequencyGrid(n_samples=N, delta_t=dt)

        assert grid.n_samples == 101
        assert grid.n_frequencies == 101 // 2 + 1  # 51 bins: k = 0, ..., 50
        assert grid.delta_f == 1.0 / (101 * 0.001)
        assert grid.nyquist == 500.0
        assert grid.frequencies[0] == 0.0  # DC

        # Highest frequency is k=50: f_50 = 50 / 0.101 ~ 495.0495 Hz < 500 Hz (Nyquist)
        f_highest = 50.0 / (101.0 * 0.001)
        assert math.isclose(grid.frequencies[-1], f_highest, rel_tol=1e-14)
        assert grid.frequencies[-1] < grid.nyquist

    def test_odd_sample_count_mask_retention_and_parity_contrast(self):
        """Verify that odd-N retains highest positive frequency bin, while even-N excludes Nyquist."""
        dt = 0.001  # fs = 1000.0 Hz

        # 1. Odd N = 101: highest frequency is ~495.05 Hz
        grid_odd = FrequencyGrid(n_samples=101, delta_t=dt)
        mask_odd = grid_odd.create_active_mask(f_start=10.0, f_end=500.0)

        assert not mask_odd[0], "DC bin must always be False for odd N"
        assert mask_odd[-1], "Highest positive frequency bin must be True for odd N"
        active_f_odd = grid_odd.get_active_frequencies(mask_odd)
        assert active_f_odd[-1] == grid_odd.frequencies[-1]

        # 2. Even N = 100: highest frequency is 500.0 Hz (Nyquist)
        grid_even = FrequencyGrid(n_samples=100, delta_t=dt)
        mask_even = grid_even.create_active_mask(f_start=10.0, f_end=500.0)

        assert not mask_even[0], "DC bin must always be False for even N"
        assert not mask_even[-1], "Nyquist bin must strictly be False for even N"
        active_f_even = grid_even.get_active_frequencies(mask_even)
        assert active_f_even[-1] == grid_even.frequencies[-2]
        assert active_f_even[-1] < 500.0

    def test_odd_sample_count_single_highest_bin_band(self):
        """Verify active mask covering a narrow band containing ONLY the highest bin for odd N."""
        N = 101
        dt = 0.001
        grid = FrequencyGrid(n_samples=N, delta_t=dt)
        f_highest = grid.frequencies[-1]  # ~495.0495 Hz

        # Narrow band enclosing only f_highest
        mask = grid.create_active_mask(f_start=495.0, f_end=496.0)
        assert np.sum(mask) == 1
        assert mask[-1]
        assert not mask[0]
        active_f = grid.get_active_frequencies(mask)
        assert len(active_f) == 1
        assert active_f[0] == f_highest

        # Contrast with even N=100: band covering only Nyquist [499, 500] Hz raises ValueError
        grid_even = FrequencyGrid(n_samples=100, delta_t=dt)
        with pytest.raises(ValueError, match="zero interior frequency bins"):
            grid_even.create_active_mask(f_start=499.0, f_end=500.0)

    def test_odd_sample_count_from_duration_and_sampling_rate(self):
        """Verify from_duration_and_sampling_rate supports odd sample count without error."""
        grid = FrequencyGrid.from_duration_and_sampling_rate(duration=0.101, sampling_rate=1000.0)
        assert grid.n_samples == 101
        assert grid.n_frequencies == 51


class TestPSDEngine:
    """Tests for low-frequency continuation, interpolation, and PSDEngine."""

    def test_low_frequency_continuation_functional_form(self):
        """Verify Group A functional form: S_n(f) = S_n(f_start) * [(f_start^2 + f_0^2)/(f^2 + f_0^2)]^2."""
        freqs = np.linspace(0.0, 100.0, 1001)  # df = 0.1 Hz
        # Synthetic flat PSD = 1e-46
        psd_raw = np.full_like(freqs, 1e-46)
        f_start = 20.0
        f_0 = 10.0

        psd_cont = low_frequency_psd_continuation(freqs, psd_raw, f_start=f_start, f_0=f_0)

        # C^0 value continuity at f_start
        idx_start = np.where(freqs == f_start)[0][0]
        assert math.isclose(psd_cont[idx_start], 1e-46, rel_tol=1e-15)

        # Above f_start: unchanged
        idx_above = freqs >= f_start
        np.testing.assert_array_equal(psd_cont[idx_above], psd_raw[idx_above])

        # Below f_start: check at f = 0 Hz
        # Ratio at 0 Hz: (20^2 + 10^2) / (0 + 10^2) = 500 / 100 = 5.0
        # PSD at 0 Hz: 1e-46 * 5.0^2 = 25.0 * 1e-46
        assert math.isclose(psd_cont[0], 25.0 * 1e-46, rel_tol=1e-14)

        # Check at f = 10 Hz
        # Ratio at 10 Hz: 500 / (100 + 100) = 500 / 200 = 2.5
        # PSD at 10 Hz: 1e-46 * 2.5^2 = 6.25 * 1e-46
        idx_10 = np.where(freqs == 10.0)[0][0]
        assert math.isclose(psd_cont[idx_10], 6.25 * 1e-46, rel_tol=1e-14)

    def test_low_frequency_continuation_configurable_f0(self):
        """Verify configurable f_0 parameter behavior (Decision D8)."""
        freqs = np.linspace(0.0, 50.0, 501)
        psd_raw = np.full_like(freqs, 2.0e-46)
        f_start = 30.0

        for f_0_val in [5.0, 10.0, 15.0, 20.0]:
            psd_cont = low_frequency_psd_continuation(freqs, psd_raw, f_start=f_start, f_0=f_0_val)
            # Value at f_start must match exactly
            idx_start = np.where(freqs == f_start)[0][0]
            assert math.isclose(psd_cont[idx_start], 2.0e-46, rel_tol=1e-15)
            # Value at 0 Hz
            expected_0 = 2.0e-46 * ((f_start**2 + f_0_val**2) / (f_0_val**2)) ** 2
            assert math.isclose(psd_cont[0], expected_0, rel_tol=1e-14)

    def test_psd_engine_interpolation_and_positivity(self):
        """Verify PSDEngine log-log interpolation and strict positivity preservation."""
        # Simulated measured PSD on a coarse grid
        meas_f = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
        # Power law S_n(f) ~ f^-2
        meas_p = 1e-44 * (meas_f / 100.0) ** (-2.0)

        engine = PSDEngine(frequencies=meas_f, psd=meas_p)

        # Target fine grid between 5 Hz and 800 Hz
        target_f = np.linspace(0.0, 800.0, 1601)  # includes sub-band f < 10 Hz
        interp_p = engine.interpolate(target_f, f_start=20.0, f_0=10.0)

        # 1. Must be strictly positive everywhere
        assert np.all(interp_p > 0.0)
        assert np.all(np.isfinite(interp_p))

        # 2. In-band power law: since meas_p is exact power law, log-log interpolation is exact
        # to within numerical precision of transcendentals exp(ln(...)) (~1e-14 relative).
        idx_in = (target_f >= 20.0) & (target_f <= 1000.0)
        expected_in = 1e-44 * (target_f[idx_in] / 100.0) ** (-2.0)
        np.testing.assert_allclose(interp_p[idx_in], expected_in, rtol=1e-13)

    def test_psd_engine_rejects_non_positive_and_non_finite(self):
        """Verify that PSDEngine raises ValueError on non-positive or non-finite PSD."""
        f = np.array([10.0, 20.0, 30.0])

        # Zero value
        with pytest.raises(ValueError, match="strictly positive"):
            PSDEngine(frequencies=f, psd=np.array([1e-46, 0.0, 1e-46]))

        # Negative value
        with pytest.raises(ValueError, match="strictly positive"):
            PSDEngine(frequencies=f, psd=np.array([1e-46, -1e-47, 1e-46]))

        # NaN / Inf
        with pytest.raises(ValueError, match="finite"):
            PSDEngine(frequencies=f, psd=np.array([1e-46, float("nan"), 1e-46]))

    def test_psd_engine_rejects_high_frequency_extrapolation(self):
        """Verify that requesting frequencies above f_max raises ValueError."""
        meas_f = np.array([10.0, 20.0, 100.0])
        meas_p = np.array([1e-45, 1e-46, 1e-47])
        engine = PSDEngine(frequencies=meas_f, psd=meas_p)

        # 105 Hz > 100 Hz
        with pytest.raises(ValueError, match="exceeds maximum measured PSD frequency"):
            engine.interpolate(np.array([50.0, 105.0]), f_start=15.0, f_0=10.0)

    def test_inverse_psd_on_active_bins(self):
        """Verify 1 / S_n(f_k) calculation on active bins."""
        grid = FrequencyGrid(n_samples=2048, delta_t=1.0 / 2048.0)
        mask = grid.create_active_mask(f_start=30.0, f_end=150.0)

        meas_f = np.linspace(10.0, 1024.0, 1015)
        meas_p = np.full_like(meas_f, 1e-46)
        engine = PSDEngine(frequencies=meas_f, psd=meas_p)

        inv_psd = engine.get_inverse_psd_on_active_bins(
            grid.frequencies, mask, f_start=30.0, f_0=10.0
        )

        assert len(inv_psd) == np.sum(mask)
        assert np.all(inv_psd > 0.0)
        np.testing.assert_allclose(inv_psd, 1.0 / 1e-46, rtol=1e-14)

    def test_f0_is_strictly_mandatory(self):
        """Verify that calling low_frequency_psd_continuation, PSDEngine.interpolate,
        or get_inverse_psd_on_active_bins without explicit f_0 raises TypeError.
        This guarantees Decision D8 is not bypassed with an implicit default.
        """
        freqs = np.array([10.0, 20.0, 30.0])
        psd = np.array([1e-46, 1e-46, 1e-46])
        engine = PSDEngine(frequencies=freqs, psd=psd)
        grid = FrequencyGrid(n_samples=512, delta_t=1.0 / 1024.0)
        mask = grid.create_active_mask(f_start=15.0, f_end=25.0)

        # 1. low_frequency_psd_continuation without f_0
        with pytest.raises(TypeError):
            low_frequency_psd_continuation(freqs, psd, f_start=15.0)  # type: ignore

        # 2. PSDEngine.interpolate without f_0
        with pytest.raises(TypeError):
            engine.interpolate(freqs, f_start=15.0)  # type: ignore

        # 3. PSDEngine.get_inverse_psd_on_active_bins without f_0
        with pytest.raises(TypeError):
            engine.get_inverse_psd_on_active_bins(grid.frequencies, mask, f_start=15.0)  # type: ignore
