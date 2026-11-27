"""Tests for GWLab Phase 4E Bayesian Posterior Explorer service and FastAPI endpoint."""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gwlab.api.app import app
from gwlab.api.service import get_inference_posterior

client = TestClient(app)


def test_inference_posterior_artifact_and_metadata():
    """Verify Stage 6C Run A artifact identity, parameter order, weights, and summaries."""
    res = get_inference_posterior(param_x="mc", param_y="eta")

    assert res["parameter_order"] == ["mc", "eta", "tc", "phic"]
    meta = res["metadata"]
    assert meta["run_id"] == "run_a_anchor"
    assert meta["seed"] == 8101
    assert meta["n_samples"] == 11951
    assert meta["n_dims"] == 4
    assert meta["guard_evaluations_completed"] == 487407
    assert math.isclose(meta["log_evidence"], 201646.1413147189, rel_tol=1e-9)
    assert math.isclose(meta["weight_sum"], 1.0, abs_tol=1e-6)
    assert math.isclose(meta["eta_top1_pct"], 38.5718, abs_tol=0.05)
    assert math.isclose(meta["kish_n_eff"], 2867.45, abs_tol=1.0)

    # Verify units and weighted density normalization for all 4 marginals
    expected_units = {"mc": "M_sun", "eta": "dimensionless", "tc": "s", "phic": "rad"}
    for key, unit in expected_units.items():
        m = res["marginals_1d"][key]
        assert m["unit"] == unit
        integral = float(np.sum(np.asarray(m["density"]) * m["bin_width"]))
        assert math.isclose(integral, 1.0, abs_tol=0.02)
        s = m["summary"]
        assert s["p5"] < s["median"] < s["p95"]

    # Verify 2D joint grid
    j2 = res["joint_2d"]
    assert j2["param_x"] == "mc"
    assert j2["param_y"] == "eta"
    grid = np.asarray(j2["grid_normalized"], dtype=np.float64)
    assert grid.shape == (36, 36)
    assert np.all(np.isfinite(grid))
    assert math.isclose(float(np.max(grid)), 1.0, rel_tol=1e-12)


def test_inference_posterior_deterministic_and_api_endpoint():
    """Verify deterministic responses and FastAPI GET /api/inference/posterior."""
    r1 = client.get("/api/inference/posterior?param_x=tc&param_y=phic&bins_1d=30&bins_2d=24")
    r2 = client.get("/api/inference/posterior?param_x=tc&param_y=phic&bins_1d=30&bins_2d=24")
    assert r1.status_code == 200
    assert r1.json() == r2.json()

    data = r1.json()
    assert data["joint_2d"]["param_x"] == "tc"
    assert data["joint_2d"]["param_y"] == "phic"
    assert len(data["joint_2d"]["grid_normalized"]) == 24
    assert len(data["marginals_1d"]["mc"]["bin_centers"]) == 30


def test_inference_posterior_error_handling(tmp_path, monkeypatch):
    """Verify missing/corrupt artifact handling and invalid query parameter errors."""
    # Invalid same parameter pair -> HTTP 400
    r_same = client.get("/api/inference/posterior?param_x=mc&param_y=mc")
    assert r_same.status_code == 400

    # Invalid parameter name -> HTTP 422
    r_bad_param = client.get("/api/inference/posterior?param_x=unknown&param_y=eta")
    assert r_bad_param.status_code == 422

    # Missing NPZ artifact -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        get_inference_posterior(npz_rel_path="experiments/wp6_stage6c_anchor/nonexistent.npz")

    # Corrupt NPZ (unnormalized weights) -> ValueError
    bad_npz = tmp_path / "bad_samples.npz"
    np.savez(
        bad_npz,
        samples=np.ones((10, 4), dtype=np.float64),
        weights=np.ones(10, dtype=np.float64),  # sums to 10 != 1
    )
    monkeypatch.setattr(
        "gwlab.api.service.resolve_safe_path",
        lambda rel: bad_npz if rel.endswith(".npz") else bad_npz.parent / "res.json",
    )
    (tmp_path / "res.json").write_text('{"run_id": "test"}', encoding="utf-8")
    with pytest.raises(ValueError, match="do not sum to unity"):
        get_inference_posterior(npz_rel_path="bad.npz", result_rel_path="res.json")
