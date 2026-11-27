"""Offline unit tests for the GW150914 external-data conditioning and off-source PSD adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure scripts/ is importable without altering package layout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_gw150914_gwosc import estimate_offsource_psd  # noqa: E402


def test_offsource_psd_excludes_event_and_returns_finite_positive() -> None:
    """Verify off-source Welch PSD ignores a large transient injected at t=15.4 s."""
    fs = 2048.0
    duration = 32.0
    rng = np.random.default_rng(1234)
    conditioned = rng.normal(0.0, 1e-21, size=int(duration * fs))

    # Inject a huge unphysical spike right at t_event = 15.4 s (inside [13, 18] s guard)
    spike_idx = int(15.4 * fs)
    conditioned[spike_idx - 100 : spike_idx + 100] += 1e-15

    freqs, psd = estimate_offsource_psd(conditioned, fs=fs, t_event_rel=15.4)
    assert freqs.ndim == 1 and psd.ndim == 1
    assert freqs.shape == psd.shape
    assert np.all(np.isfinite(psd))
    assert np.all(psd > 0.0)
    # PSD level should remain at ~1e-45 (noise floor), completely uncontaminated by the 1e-15 spike
    assert float(np.median(psd[freqs > 20.0])) < 1e-40


def test_gw150914_active_and_exclusion_bin_counts() -> None:
    """Derive exact active-mask, template-support, and 52.25-53.25 Hz exclusion bin counts."""
    from gw_inference.constants import M_SUN_SI, MPC_SI
    from gw_inference.grid import FrequencyGrid
    from gw_inference.waveform import taylorf2_fd_waveform

    grid = FrequencyGrid.from_duration_and_sampling_rate(duration=4.0, sampling_rate=2048.0)
    mask = grid.create_active_mask(f_start=20.0, f_end=512.0)
    tmpl = taylorf2_fd_waveform(
        grid.frequencies,
        m1_kg=35.0 * M_SUN_SI,
        m2_kg=30.0 * M_SUN_SI,
        distance_m=410.0 * MPC_SI,
        phi_c=0.0,
        f_end=512.0,
    )
    support_mask = mask & (np.abs(tmpl) > 0)
    notch = (grid.frequencies >= 52.25) & (grid.frequencies <= 53.25)

    assert int(np.sum(mask)) == 1969
    assert int(np.sum(support_mask)) == 191
    assert int(np.sum(notch & support_mask)) == 5
