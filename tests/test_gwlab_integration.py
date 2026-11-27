"""Phase 6A Frontend/Backend integration and error-handling tests for GWLab FastAPI."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gwlab.api.app import app

client = TestClient(app)


def _assert_finite_list(values: list[float], min_len: int = 1) -> None:
    assert isinstance(values, list)
    assert len(values) >= min_len
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class TestFrontendBackendContract:
    """Validate exact JSON schemas matching frontend/src/lib/api.ts."""

    def test_status_schema_contract(self) -> None:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        expected_keys = {
            "status",
            "version",
            "mode",
            "detector",
            "duration_s",
            "sampling_rate_hz",
            "f_lower_hz",
            "f_upper_hz",
            "waveform_approximant",
            "governance",
        }
        assert expected_keys.issubset(data.keys())
        assert data["status"] == "online"

    def test_overview_benchmarks_schema_contract(self) -> None:
        resp = client.get("/api/benchmarks/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert {"stage6c", "ledger", "seed_comparison", "provenance_matrix"} <= set(data.keys())
        assert {"mc", "eta", "tc", "phic"} <= set(data["stage6c"]["parameters"].keys())
        for item in data["provenance_matrix"]:
            assert {"artifact", "category", "path", "scope", "status"} <= set(item.keys())

    def test_noise_analyze_contract_and_psd_models(self) -> None:
        for model in ("aligo_representative", "flat"):
            resp = client.post(
                "/api/noise/analyze",
                json={"psd_model": model, "duration_obs_s": 16.0, "segment_duration_s": 4.0, "seed": 7101},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert {
                "frequencies",
                "psd_analytic",
                "psd_welch",
                "asd_analytic",
                "asd_welch",
                "time_preview_s",
                "strain_preview",
                "metadata",
            } <= set(data.keys())
            _assert_finite_list(data["frequencies"], min_len=10)
            _assert_finite_list(data["psd_analytic"], min_len=10)
            _assert_finite_list(data["asd_analytic"], min_len=10)
            assert len(data["frequencies"]) == len(data["psd_analytic"]) == len(data["psd_welch"])

            # ASD == sqrt(PSD) invariant
            psd = np.array(data["psd_analytic"])
            asd = np.array(data["asd_analytic"])
            assert np.allclose(asd, np.sqrt(psd), rtol=1e-9, atol=0.0)
            assert data["metadata"]["psd_model"] == model
            assert {"frequency", "psd", "asd", "strain", "time"} <= set(data["metadata"]["units"].keys())

    def test_matched_filter_contract_and_mismatch_invariant(self) -> None:
        matched = client.post(
            "/api/matched-filter/evaluate",
            json={"delta_mc_msun": 0.0, "psd_model": "aligo_representative", "seed": 7151},
        )
        mismatched = client.post(
            "/api/matched-filter/evaluate",
            json={"delta_mc_msun": 2.5, "psd_model": "aligo_representative", "seed": 7151},
        )
        flat_resp = client.post(
            "/api/matched-filter/evaluate",
            json={"delta_mc_msun": 0.0, "psd_model": "flat", "seed": 7151},
        )
        assert matched.status_code == 200
        assert mismatched.status_code == 200
        assert flat_resp.status_code == 200

        m_data = matched.json()
        mm_data = mismatched.json()
        assert {
            "snr_time_s",
            "snr_abs",
            "snr_real",
            "zoom_time_s",
            "zoom_abs",
            "zoom_real",
            "frequencies",
            "data_amp",
            "injection_amp",
            "template_amp",
            "metadata",
        } <= set(m_data.keys())
        _assert_finite_list(m_data["snr_abs"], min_len=100)
        _assert_finite_list(m_data["zoom_abs"], min_len=10)
        assert m_data["metadata"]["optimal_snr"] > 0
        assert m_data["metadata"]["recovered_peak_snr"] > mm_data["metadata"]["recovered_peak_snr"]

    def test_inference_posterior_contract_and_invariants(self) -> None:
        resp = client.get("/api/inference/posterior?param_x=mc&param_y=eta&bins_1d=30&bins_2d=24")
        assert resp.status_code == 200
        data = resp.json()
        assert {"parameter_order", "marginals_1d", "joint_2d", "metadata"} <= set(data.keys())
        assert data["parameter_order"] == ["mc", "eta", "tc", "phic"]
        for key in data["parameter_order"]:
            m = data["marginals_1d"][key]
            assert m["summary"]["p5"] <= m["summary"]["median"] <= m["summary"]["p95"]
            _assert_finite_list(m["density"], min_len=15)
        meta = data["metadata"]
        assert abs(meta["weight_sum"] - 1.0) < 1e-4
        assert 0.0 <= meta["eta_top1_pct"] <= 100.0

    def test_experiments_explorer_contract(self) -> None:
        resp = client.get("/api/experiments/explorer")
        assert resp.status_code == 200
        data = resp.json()
        assert {
            "multi_seed_comparison",
            "stage6c_anchor",
            "ledger",
            "campaign_stages",
            "limitations",
        } <= set(data.keys())
        assert len(data["multi_seed_comparison"]["runs"]) == 2
        assert len(data["campaign_stages"]) >= 1
        assert len(data["limitations"]) >= 1


class TestApiErrorHandling:
    """Validate HTTP 400/404/422/500 error branches in app.py and service.py."""

    def test_waveform_and_chirp_time_error_responses(self) -> None:
        # Pydantic validation failure -> 422
        bad_eta = client.post("/api/waveforms/generate", json={"mc_msun": 28.0, "eta": 0.30, "distance_mpc": 410.0})
        assert bad_eta.status_code == 422

        # Service exception -> 400
        with patch("gwlab.api.app.compute_waveform_response", side_effect=ValueError("bad waveform input")):
            resp = client.post("/api/waveforms/generate", json={"mc_msun": 28.0, "eta": 0.24, "distance_mpc": 410.0})
            assert resp.status_code == 400
            assert "bad waveform input" in resp.json()["detail"]

        with patch("gwlab.api.app.compute_chirp_time_response", side_effect=ValueError("bad chirp input")):
            resp = client.post("/api/waveforms/chirp-time", json={"mc_msun": 28.0, "eta": 0.24})
            assert resp.status_code == 400
            assert "bad chirp input" in resp.json()["detail"]

    def test_noise_analyze_validation_errors(self) -> None:
        # f_min >= f_max
        r1 = client.post("/api/noise/analyze", json={"f_min_hz": 100.0, "f_max_hz": 80.0})
        assert r1.status_code == 400

        # f_max > nyquist (sampling_rate=512 -> nyquist=256)
        r2 = client.post("/api/noise/analyze", json={"sampling_rate_hz": 512.0, "f_max_hz": 300.0})
        assert r2.status_code == 400

        # duration_obs_s < segment_duration_s
        r3 = client.post("/api/noise/analyze", json={"duration_obs_s": 8.0, "segment_duration_s": 16.0})
        assert r3.status_code == 400

    def test_matched_filter_validation_errors(self) -> None:
        # f_min >= f_max
        r1 = client.post("/api/matched-filter/evaluate", json={"f_min_hz": 120.0, "f_max_hz": 100.0})
        assert r1.status_code == 400

        # Pydantic boundary on tc_s
        r2 = client.post("/api/matched-filter/evaluate", json={"tc_s": 0.05})
        assert r2.status_code == 422

    def test_inference_posterior_error_responses(self) -> None:
        # Same parameter for X and Y -> 400
        r_same = client.get("/api/inference/posterior?param_x=mc&param_y=mc")
        assert r_same.status_code == 400

        with patch("gwlab.api.app.get_inference_posterior", side_effect=FileNotFoundError("missing npz")):
            r_404 = client.get("/api/inference/posterior")
            assert r_404.status_code == 404

        with patch("gwlab.api.app.get_inference_posterior", side_effect=RuntimeError("unexpected crash")):
            r_500 = client.get("/api/inference/posterior")
            assert r_500.status_code == 500

    def test_overview_and_experiments_error_responses(self) -> None:
        with patch("gwlab.api.app.get_overview_benchmarks", side_effect=RuntimeError("overview crash")):
            r_ov = client.get("/api/benchmarks/overview")
            assert r_ov.status_code == 500

        with patch("gwlab.api.app.get_experiments_explorer", side_effect=FileNotFoundError("missing json")):
            r_404 = client.get("/api/experiments/explorer")
            assert r_404.status_code == 404

        with patch("gwlab.api.app.get_experiments_explorer", side_effect=ValueError("corrupt json")):
            r_400 = client.get("/api/experiments/explorer")
            assert r_400.status_code == 400

        with patch("gwlab.api.app.get_experiments_explorer", side_effect=RuntimeError("unexpected")):
            r_500 = client.get("/api/experiments/explorer")
            assert r_500.status_code == 500
