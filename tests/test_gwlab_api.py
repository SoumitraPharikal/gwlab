"""Tests for the GWLab FastAPI service boundary."""

import pytest
from fastapi.testclient import TestClient

from gwlab.api.app import app

client = TestClient(app)


def test_api_status():
    """Verify health and configuration status endpoint."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["mode"] == "Reference archive"
    assert data["detector"] == "LIGO Hanford (H1)"
    assert data["f_lower_hz"] == 20.0
    assert data["f_upper_hz"] == 512.0


def test_api_benchmarks_overview():
    """Verify overview benchmarks endpoint returns valid authoritative artifacts."""
    response = client.get("/api/benchmarks/overview")
    assert response.status_code == 200
    data = response.json()

    # Stage 6C checks
    assert "stage6c" in data
    s6c = data["stage6c"]
    assert s6c["run_id"] == "run_a_anchor"
    assert s6c["status"] == "SUCCESS"
    assert abs(s6c["log_evidence"] - 201646.1413) < 0.01
    assert abs(s6c["log_evidence_err"] - 0.2374) < 0.01
    assert s6c["n_evaluations"] == 500657
    assert abs(s6c["wall_time_s"] - 514.83) < 0.1
    assert abs(s6c["cpu_time_s"] - 473.16) < 0.5
    assert s6c["boundary_concentration_pct"] == 38.57

    # Ledger checks
    assert "ledger" in data
    led = data["ledger"]
    assert led["cumulative_calls"] == 487407
    assert led["remaining_calls"] == 712593
    assert led["total_budget_calls"] == 1200000
    assert led["consumption_pct"] == 40.6

    # Seed comparison checks
    assert "seed_comparison" in data
    sc = data["seed_comparison"]
    assert sc["seed_1"] == 8101
    assert sc["seed_2"] == 8201
    assert abs(sc["delta_logz"] - 2.9720) < 0.01
    assert abs(sc["sigma_delta"] - 0.3331) < 0.01
    assert sc["z_ratio_formatted"] == "8.92"
    assert "Observed evidence discrepancy" in sc["qualification"]

    # Provenance matrix checks
    assert "provenance_matrix" in data
    assert len(data["provenance_matrix"]) == 4


def test_api_waveform_generate():
    """Verify deterministic waveform generation endpoint."""
    payload = {
        "mc_msun": 28.095555795460438,
        "eta": 0.24710059171597634,
        "distance_mpc": 410.0,
        "tc_s": 0.0,
        "phic_rad": 0.0,
        "duration_s": 4.0,
        "sampling_rate_hz": 2048.0,
        "f_end_hz": 512.0,
    }
    response = client.post("/api/waveforms/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "frequencies" in data
    assert "amplitude" in data
    assert "unwrapped_phase" in data
    assert "wrapped_phase" in data
    assert "real" in data
    assert "imag" in data
    assert len(data["frequencies"]) > 0
    assert len(data["frequencies"]) == len(data["amplitude"])
    assert data["m1_msun"] >= data["m2_msun"]
    assert data["f_isco"] > 0
    assert data["f_term"] <= 512.0


def test_api_chirp_time():
    """Verify chirp time calculation endpoint."""
    payload = {
        "mc_msun": 28.095555795460438,
        "eta": 0.24710059171597634,
    }
    response = client.post("/api/waveforms/chirp-time", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "frequencies" in data
    assert "tau_newtonian" in data
    assert "tau_35pn" in data
    assert len(data["frequencies"]) > 0
    assert len(data["tau_newtonian"]) == len(data["frequencies"])
    assert len(data["tau_35pn"]) == len(data["frequencies"])
