"""Tests for GWLab Phase 4F Saved-Results Experiment Explorer."""

import json
import math
import pytest
from fastapi.testclient import TestClient

from gwlab.api.app import app
from gwlab.api.service import get_experiments_explorer

client = TestClient(app)


def test_experiments_explorer_valid_artifacts_and_comparison():
    """Verify valid Stage 5G.2, 6C, and 6D loading, provenance, and two-seed comparison."""
    res = get_experiments_explorer()

    comp = res["multi_seed_comparison"]
    runs = comp["runs"]
    assert len(runs) == 2
    assert runs[0]["run_id"] == "base_01_plug_inst_s1"
    assert runs[0]["sampler_seed"] == 8101
    assert runs[1]["run_id"] == "base_01_plug_inst_s2"
    assert runs[1]["sampler_seed"] == 8201

    assert math.isclose(comp["delta_logz"], 2.972018634, rel_tol=1e-6)
    assert math.isclose(comp["sigma_delta"], 0.333210178, rel_tol=1e-5)
    assert math.isclose(comp["discrepancy_ratio"], 8.91935, rel_tol=1e-4)

    # Verify campaign status labels strictly separate completed vs unexecuted/proposed/frozen
    stages = {s["stage_id"]: s for s in res["campaign_stages"]}
    assert stages["stage_5g2"]["status"] == "Completed / Historical"
    assert stages["stage_6c_run_a"]["status"] == "Completed / Historical"
    assert stages["stage_6d_closeout"]["status"] == "Completed / Historical"
    assert stages["stage_6c_run_b"]["status"] == "Unexecuted"
    assert stages["seed_8301"]["status"] == "Proposed"
    assert stages["model_3_student_t"]["status"] == "Frozen / Specified"


def test_experiments_explorer_api_endpoint():
    """Verify GET /api/experiments/explorer returns 200 OK with expected top-level keys."""
    resp = client.get("/api/experiments/explorer")
    assert resp.status_code == 200
    data = resp.json()
    assert "multi_seed_comparison" in data
    assert "stage6c_anchor" in data
    assert "ledger" in data
    assert "campaign_stages" in data
    assert "limitations" in data
    assert data["stage6c_anchor"]["run_id"] == "run_a_anchor"


def test_experiments_explorer_missing_malformed_and_optional_fields(tmp_path, monkeypatch):
    """Verify defensive handling of missing files, malformed JSON, inconsistent provenance, and missing optional fields."""
    # 1. Missing artifact
    with pytest.raises(FileNotFoundError):
        get_experiments_explorer(s1_rel_path="reports/nonexistent_s1.json")

    # 2. Malformed JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr("gwlab.api.service.resolve_safe_path", lambda rel: bad_json)
    with pytest.raises(ValueError, match="Corrupt JSON"):
        get_experiments_explorer(s1_rel_path="bad.json")

    # 3. Missing optional fields handled cleanly without fabricating defaults
    s1_min = tmp_path / "s1.json"
    s2_min = tmp_path / "s2.json"
    s6c_min = tmp_path / "s6c.json"
    led_min = tmp_path / "led.json"
    s1_min.write_text(
        json.dumps(
            {
                "run_id": "s1_test",
                "status": "SUCCESS",
                "model": "plug",
                "realization": 1,
                "sampler_seed": 8101,
                "log_evidence": 100.0,
                "log_evidence_err": 0.3,
            }
        ),
        encoding="utf-8",
    )
    s2_min.write_text(
        json.dumps(
            {
                "run_id": "s2_test",
                "status": "SUCCESS",
                "model": "plug",
                "realization": 1,
                "sampler_seed": 8201,
                "log_evidence": 98.0,
                "log_evidence_err": 0.4,
            }
        ),
        encoding="utf-8",
    )
    s6c_min.write_text(
        json.dumps({"run_id": "run_a_test", "status": "SUCCESS", "log_evidence": 100.0, "log_evidence_err": 0.3}),
        encoding="utf-8",
    )
    led_min.write_text(json.dumps({"cumulative_calls": 10, "remaining_calls": 90}), encoding="utf-8")

    mapping = {"s1.json": s1_min, "s2.json": s2_min, "s6c.json": s6c_min, "led.json": led_min}
    monkeypatch.setattr("gwlab.api.service.resolve_safe_path", lambda rel: mapping[rel])

    res = get_experiments_explorer(
        s1_rel_path="s1.json",
        s2_rel_path="s2.json",
        stage6c_rel_path="s6c.json",
        ledger_rel_path="led.json",
    )
    assert res["multi_seed_comparison"]["runs"][0]["n_eff"] is None
    assert res["multi_seed_comparison"]["runs"][0]["mc_median"] is None
    assert math.isclose(res["multi_seed_comparison"]["delta_logz"], 2.0, rel_tol=1e-12)
    assert math.isclose(res["multi_seed_comparison"]["sigma_delta"], 0.5, rel_tol=1e-12)

    # 4. Inconsistent realization raises ValueError
    s2_mismatch = tmp_path / "s2_mismatch.json"
    s2_mismatch.write_text(
        json.dumps(
            {
                "run_id": "s2_test",
                "status": "SUCCESS",
                "model": "plug",
                "realization": 2,
                "sampler_seed": 8201,
                "log_evidence": 98.0,
                "log_evidence_err": 0.4,
            }
        ),
        encoding="utf-8",
    )
    mapping["s2.json"] = s2_mismatch
    with pytest.raises(ValueError, match="Inconsistent realization"):
        get_experiments_explorer(
            s1_rel_path="s1.json",
            s2_rel_path="s2.json",
            stage6c_rel_path="s6c.json",
            ledger_rel_path="led.json",
        )
