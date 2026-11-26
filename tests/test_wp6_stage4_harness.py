"""Unit and integration test suite for WP6 Stage 4 campaign execution harness."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Dict

import numpy as np
import pytest

import scripts.run_wp6_stage4_campaign as harness


class TestHarnessSpecifications:
    """Test campaign specifications and canonical constants."""

    def test_tier1_specifications_structure(self) -> None:
        specs = harness.get_tier1_specifications()
        assert len(specs) == 3
        assert "run_1_01" in specs
        assert "run_1_02" in specs
        assert "run_1_03" in specs

        # run_1_01 check
        r1 = specs["run_1_01"]
        assert r1["noise_type"] == "zero_noise"
        assert r1["noise_seed"] is None
        assert r1["sampler_seed"] == 42
        assert r1["nlive"] == 1000
        assert r1["dlogz"] == 0.05
        assert np.isclose(r1["mc_msun"], 28.095556, atol=1e-5)
        assert np.isclose(r1["eta"], 0.247101, atol=1e-5)

        # run_1_02 check
        r2 = specs["run_1_02"]
        assert r2["noise_type"] == "colored_noise"
        assert r2["noise_seed"] == 4001
        assert r2["sampler_seed"] == 4001

        # run_1_03 check
        r3 = specs["run_1_03"]
        assert r3["noise_type"] == "colored_noise"
        assert r3["noise_seed"] == 4002
        assert r3["sampler_seed"] == 4002

    def test_tier2_specifications_structure(self) -> None:
        specs = harness.get_tier2_specifications()
        assert len(specs) == 12

        # Verify all 4 parts are represented
        part_a = [f"run_2A_{i:02d}" for i in range(1, 5)]
        part_b = [f"run_2B_{i:02d}" for i in range(1, 5)]
        part_c = ["run_2C_01", "run_2C_02"]
        part_d = ["run_2D_01", "run_2D_02"]

        for rid in part_a + part_b + part_c + part_d:
            assert rid in specs
            assert specs[rid]["tier"] == "tier2"
            assert specs[rid]["nlive"] == 500
            assert specs[rid]["dlogz"] == 0.10

        # Part C perturbations
        assert specs["run_2C_01"]["psd_scale_factor"] == 1.20
        assert specs["run_2C_02"]["psd_tilt_delta"] == 0.10

        # Part D mismatches
        assert specs["run_2D_01"]["recovery_f_end"] == 50.0
        assert specs["run_2D_02"]["recovery_pn_order"] == 3.0


class TestGate41Evaluation:
    """Test Gate 4.1 validation logic."""

    def test_gate_4_1_incomplete_runs(self) -> None:
        incomplete_results = {"run_1_01": {}}
        eval_dict = harness.evaluate_gate_4_1(incomplete_results)
        assert eval_dict["all_3_runs_completed"] is False
        assert eval_dict["overall_gate_4_1_passed"] is False

    def test_gate_4_1_passing_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_dummy = os.path.join(tmpdir, "dummy.npz")
            np.savez(npz_dummy, a=np.zeros(100))

            mock_results = {
                "run_1_01": {
                    "log_evidence": 203628.126,
                    "log_evidence_err": 0.15,
                    "metadata": {"npz_path": npz_dummy},
                    "recovered_parameters": {
                        "chirp_mass_msun": {
                            "type": "linear",
                            "linear_moments": {"mean": 28.092703},
                        },
                        "symmetric_mass_ratio": {
                            "type": "linear",
                            "linear_moments": {"mean": 0.246884},
                        },
                        "coalescence_time_s": {
                            "type": "linear",
                            "linear_moments": {"mean": 2.000017},
                        },
                        "coalescence_phase_rad": {
                            "type": "circular",
                            "circular_moments": {"circular_mean": 6.193538},
                        },
                    },
                },
                "run_1_02": {"metadata": {"npz_path": npz_dummy}},
                "run_1_03": {"metadata": {"npz_path": npz_dummy}},
            }

            eval_dict = harness.evaluate_gate_4_1(mock_results)
            assert eval_dict["all_3_runs_completed"] is True
            assert eval_dict["zero_noise_evidence_agreement"] is True
            assert eval_dict["zero_noise_parameter_means_agreement"] is True
            assert eval_dict["serialization_valid"] is True
            assert eval_dict["overall_gate_4_1_passed"] is True

    def test_gate_4_1_failing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_dummy = os.path.join(tmpdir, "dummy.npz")
            np.savez(npz_dummy, a=np.zeros(100))

            mock_results = {
                "run_1_01": {
                    "log_evidence": 203627.50,  # Delta = 0.626 > 0.15
                    "log_evidence_err": 0.15,
                    "metadata": {"npz_path": npz_dummy},
                    "recovered_parameters": {
                        "chirp_mass_msun": {
                            "type": "linear",
                            "linear_moments": {"mean": 28.092703},
                        },
                        "symmetric_mass_ratio": {
                            "type": "linear",
                            "linear_moments": {"mean": 0.246884},
                        },
                        "coalescence_time_s": {
                            "type": "linear",
                            "linear_moments": {"mean": 2.000017},
                        },
                        "coalescence_phase_rad": {
                            "type": "circular",
                            "circular_moments": {"circular_mean": 6.193538},
                        },
                    },
                },
                "run_1_02": {"metadata": {"npz_path": npz_dummy}},
                "run_1_03": {"metadata": {"npz_path": npz_dummy}},
            }

            eval_dict = harness.evaluate_gate_4_1(mock_results)
            assert eval_dict["zero_noise_evidence_agreement"] is False
            assert eval_dict["overall_gate_4_1_passed"] is False


class TestHarnessExecutionFastSmoke:
    """Fast smoke test executing end-to-end inference with minimal live points."""

    def test_fast_execution_and_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = harness.get_tier1_specifications()
            spec = specs["run_1_01"]

            # Run with tiny nlive=20 and dlogz=2.0 for fast test
            diag_dict = harness.execute_single_run(
                run_id="test_smoke_01",
                spec=spec,
                output_dir=tmpdir,
                nlive_override=20,
                dlogz_override=2.0,
                sampler_seed_override=42,
                force=True,
            )

            assert "run_id" in diag_dict
            assert diag_dict["run_id"] == "test_smoke_01"
            assert "log_evidence" in diag_dict
            assert "recovered_parameters" in diag_dict

            # Check files on disk
            npz_file = os.path.join(tmpdir, "test_smoke_01_results.npz")
            json_file = os.path.join(tmpdir, "test_smoke_01_diagnostics.json")

            assert os.path.isfile(npz_file)
            assert os.path.isfile(json_file)

            with np.load(npz_file) as npz_data:
                assert "samples" in npz_data
                assert "weights" in npz_data
                assert "posterior_samples" in npz_data
                assert "injected_params" in npz_data
                assert len(npz_data["injected_params"]) == 4
