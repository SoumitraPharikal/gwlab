"""Tests for GWLab Phase 4D Matched Filtering service and FastAPI endpoint."""

import math
import numpy as np
from fastapi.testclient import TestClient

from gw_inference.constants import M_SUN_SI, MPC_SI
from gw_inference.filter import compute_template_norm
from gw_inference.grid import FrequencyGrid
from gw_inference.psd import analytical_detector_psd
from gw_inference.waveform import taylorf2_fd_waveform
from gwlab.api.app import app
from gwlab.api.service import compute_matched_filter_response
from gwlab.physics import mchirp_eta_to_component_masses

client = TestClient(app)


def test_matched_filter_optimal_snr_and_inner_product_consistency():
    """Verify optimal_snr matches sqrt((h|h)_{K_+}) and canonical benchmark 45.3153."""
    res = compute_matched_filter_response()
    meta = res["metadata"]

    # Check exact optimal SNR against direct gw_inference.filter.compute_template_norm
    grid = FrequencyGrid.from_duration_and_sampling_rate(duration=4.0, sampling_rate=2048.0)
    mask = grid.create_active_mask(f_start=20.0, f_end=512.0)
    active_idx = grid.get_active_indices(mask)
    psd_grid = analytical_detector_psd(grid.frequencies, model="aligo_representative")
    inv_psd_act = 1.0 / psd_grid[active_idx]

    m1, m2, _ = mchirp_eta_to_component_masses(
        28.095555795460438, 0.24710059171597634
    )
    h_fd = taylorf2_fd_waveform(
        grid.frequencies,
        m1 * M_SUN_SI,
        m2 * M_SUN_SI,
        410.0 * MPC_SI,
        phi_c=0.0,
        f_end=512.0,
    )
    sigma_h_direct = compute_template_norm(h_fd, inv_psd_act, active_idx, grid.delta_f)

    assert math.isclose(meta["optimal_snr"], sigma_h_direct, rel_tol=1e-12)
    assert math.isclose(meta["optimal_snr"], 45.31529969318099, rel_tol=1e-12)
    assert meta["n_active_bins"] == 1969
    assert abs(meta["recovered_peak_time_s"] - 2.0) < 0.01
    assert meta["recovered_peak_snr"] > 35.0
    assert np.all(np.isfinite(res["snr_abs"]))
    assert np.all(np.isfinite(res["zoom_abs"]))


def test_matched_filter_deterministic_seed_and_mismatch():
    """Verify deterministic seed behavior and SNR degradation under chirp-mass mismatch."""
    r1 = compute_matched_filter_response(seed=7151, delta_mc_msun=0.0)
    r2 = compute_matched_filter_response(seed=7151, delta_mc_msun=0.0)
    r3 = compute_matched_filter_response(seed=7152, delta_mc_msun=0.0)
    r_mismatch = compute_matched_filter_response(seed=7151, delta_mc_msun=1.5)

    assert np.array_equal(r1["snr_abs"], r2["snr_abs"])
    assert r1["metadata"]["recovered_peak_snr"] == r2["metadata"]["recovered_peak_snr"]
    assert r1["metadata"]["recovered_peak_snr"] != r3["metadata"]["recovered_peak_snr"]
    assert r_mismatch["metadata"]["recovered_peak_snr"] < r1["metadata"]["recovered_peak_snr"]


def test_api_matched_filter_endpoint_and_validation():
    """Verify POST /api/matched-filter/evaluate schema, band limits, and error handling."""
    resp = client.post(
        "/api/matched-filter/evaluate",
        json={
            "f_min_hz": 30.0,
            "f_max_hz": 256.0,
            "tc_s": 1.5,
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["frequencies"][0] >= 30.0
    assert data["frequencies"][-1] <= 256.0
    assert abs(data["metadata"]["recovered_peak_time_s"] - 1.5) < 0.02
    assert len(data["snr_time_s"]) == len(data["snr_abs"]) == len(data["snr_real"])
    assert len(data["zoom_time_s"]) == len(data["zoom_abs"]) == len(data["zoom_real"])
    assert len(data["frequencies"]) == len(data["data_amp"]) == len(data["template_amp"])

    # Invalid frequency band (f_min >= f_max)
    r_bad_band = client.post(
        "/api/matched-filter/evaluate",
        json={"f_min_hz": 150.0, "f_max_hz": 100.0},
    )
    assert r_bad_band.status_code == 400

    # Invalid tc_s (Pydantic 422)
    r_bad_tc = client.post("/api/matched-filter/evaluate", json={"tc_s": 5.0})
    assert r_bad_tc.status_code == 422
