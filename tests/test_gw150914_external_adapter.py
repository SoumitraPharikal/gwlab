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
