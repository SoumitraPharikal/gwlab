"""Tests for GWLab Phase 4C Noise & PSD service and FastAPI endpoint."""

import math
import numpy as np
from fastapi.testclient import TestClient

from gwlab.api.app import app
from gwlab.api.service import compute_noise_and_psd_response

client = TestClient(app)


def test_noise_and_psd_frozen_stage5b_defaults():
    """Verify frozen Stage 5B Welch setup and analytical PSD/ASD consistency."""
    res = compute_noise_and_psd_response()

    freqs = np.asarray(res["frequencies"], dtype=np.float64)
    psd_a = np.asarray(res["psd_analytic"], dtype=np.float64)
    psd_w = np.asarray(res["psd_welch"], dtype=np.float64)
    asd_a = np.asarray(res["asd_analytic"], dtype=np.float64)
    asd_w = np.asarray(res["asd_welch"], dtype=np.float64)

    assert freqs.size > 0
    assert freqs.shape == psd_a.shape == psd_w.shape == asd_a.shape == asd_w.shape
    assert freqs[0] >= 20.0
    assert freqs[-1] <= 512.0
    assert np.all(np.isfinite(psd_a)) and np.all(psd_a > 0)
    assert np.all(np.isfinite(psd_w)) and np.all(psd_w >= 0)
    assert np.allclose(asd_a, np.sqrt(psd_a), rtol=1e-12)
    assert np.allclose(asd_w, np.sqrt(psd_w), rtol=1e-12)

    meta = res["metadata"]
    assert meta["psd_model"] == "aligo_representative"
    assert meta["n_segments"] == 31
    assert math.isclose(meta["kappa_overlap"], 1157.0 / 837.0, rel_tol=1e-9)
    assert math.isclose(meta["nu_eff"], 2.0 * 31.0 / (1157.0 / 837.0), rel_tol=1e-9)
    assert meta["delta_f_hz"] == 0.25
    assert meta["n_active_bins"] == 1969
    assert meta["units"]["psd"] == "strain^2/Hz"
    assert meta["units"]["asd"] == "strain/sqrt(Hz)"


def test_noise_and_psd_deterministic_seed():
    """Verify identical seed produces identical synthetic noise and Welch PSD."""
    r1 = compute_noise_and_psd_response(seed=7101)
    r2 = compute_noise_and_psd_response(seed=7101)
    r3 = compute_noise_and_psd_response(seed=7102)

    assert np.array_equal(r1["strain_preview"], r2["strain_preview"])
    assert np.array_equal(r1["psd_welch"], r2["psd_welch"])
    assert not np.array_equal(r1["strain_preview"], r3["strain_preview"])
    assert not np.array_equal(r1["psd_welch"], r3["psd_welch"])


def test_api_noise_analyze_endpoint():
    """Verify POST /api/noise/analyze schema and flat model support."""
    resp = client.post(
        "/api/noise/analyze",
        json={
            "psd_model": "flat",
            "f_min_hz": 30.0,
            "f_max_hz": 256.0,
            "sampling_rate_hz": 2048.0,
            "duration_obs_s": 32.0,
            "segment_duration_s": 4.0,
            "overlap_fraction": 0.5,
            "tukey_alpha": 0.25,
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["psd_model"] == "flat"
    assert data["metadata"]["seed"] == 42
    assert data["frequencies"][0] >= 30.0
    assert data["frequencies"][-1] <= 256.0
    assert np.allclose(data["psd_analytic"], 1.0e-46, rtol=1e-12)
    assert len(data["time_preview_s"]) == len(data["strain_preview"])
    assert len(data["time_preview_s"]) > 0


def test_api_noise_analyze_invalid_inputs():
    """Verify clear HTTP 400/422 errors for invalid parameters."""
    # f_min >= f_max
    r1 = client.post("/api/noise/analyze", json={"f_min_hz": 200.0, "f_max_hz": 100.0})
    assert r1.status_code == 400

    # f_max > Nyquist (sampling_rate_hz = 512 -> Nyquist = 256)
    r2 = client.post(
        "/api/noise/analyze",
        json={"sampling_rate_hz": 512.0, "f_max_hz": 512.0},
    )
    assert r2.status_code == 400

    # Invalid psd_model (Pydantic 422)
    r3 = client.post("/api/noise/analyze", json={"psd_model": "invalid_model"})
    assert r3.status_code == 422
