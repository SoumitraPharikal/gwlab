"""Mocked integration test suite for WP6 Stage 6C production runner and controller.

This suite verifies the complete production wiring of Stage 6C Run A and its
supervising controller using strictly mocked and fake components.

STRICT SAFETY MANDATE:
----------------------
- ZERO scientific execution of Run A or Run B.
- ZERO dynesty invocations on real scientific likelihoods.
- ZERO Whittle likelihood evaluations or waveform simulations.
- ZERO noise or PSD realizations.
- All tests replace data preparation, likelihood construction, and sampler execution
  with fast, deterministic mocks.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(WORKSPACE_DIR / "src"))

from gw_inference.prior import prior_transform_4d
from gw_inference.resource_control import (
    AggregateResourceLedger,
    LikelihoodBudgetExceededError,
    LikelihoodBudgetGuard,
    Stage6CConfig,
    Stage6CController,
    WorkerStatus,
    atomic_write_json,
)
from scripts.run_wp6_stage6c_anchor import (
    RUN_A_FROZEN_CONFIG,
    run_anchor_run_a_worker,
    run_controller_anchor,
)

SCRIPT_PATH = WORKSPACE_DIR / "scripts" / "run_wp6_stage6c_anchor.py"


# ==============================================================================
# Mock Sampler Result and Adapter Helpers
# ==============================================================================


class FakeSamplerResult:
    """Mock container representing dynesty NestedSamplingResult."""

    def __init__(
        self,
        log_evidence: float = -1234.56,
        log_evidence_err: float = 0.15,
        n_evaluations: int = 1500,
        samples: Optional[np.ndarray] = None,
    ) -> None:
        self.log_evidence = log_evidence
        self.log_evidence_err = log_evidence_err
        self.n_evaluations = n_evaluations
        self.samples = (
            samples
            if samples is not None
            else np.ones((50, 4), dtype=np.float64) * [28.1, 0.24, 2.0, 1.0]
        )


class FakeSamplerAdapter:
    """Mock NestedSamplerAdapter that simulates sampler execution without real sampling."""

    captured_init_kwargs: Dict[str, Any] = {}
    captured_run_kwargs: Dict[str, Any] = {}
    evaluations_to_perform: int = 20
    fail_with_budget_error: bool = False
    fail_with_runtime_error: bool = False

    def __init__(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        prior_transform_fn: Callable[[np.ndarray], np.ndarray],
        ndim: int,
        nlive: int,
        dlogz: float,
        seed: Optional[int] = None,
        param_names: Optional[tuple] = None,
        **kwargs: Any,
    ) -> None:
        FakeSamplerAdapter.captured_init_kwargs = {
            "log_likelihood_fn": log_likelihood_fn,
            "prior_transform_fn": prior_transform_fn,
            "ndim": ndim,
            "nlive": nlive,
            "dlogz": dlogz,
            "seed": seed,
            "param_names": param_names,
            **kwargs,
        }
        self.logl = log_likelihood_fn
        self.ptform = prior_transform_fn
        self.ndim = ndim
        self.nlive = nlive
        self.dlogz = dlogz
        self.seed = seed

    def run(self, **dynesty_kwargs: Any) -> FakeSamplerResult:
        FakeSamplerAdapter.captured_run_kwargs = dict(dynesty_kwargs)

        if FakeSamplerAdapter.fail_with_runtime_error:
            raise RuntimeError("Simulated internal sampler crash during run_nested")

        # Simulate evaluations through the provided log_likelihood_fn
        for i in range(FakeSamplerAdapter.evaluations_to_perform):
            u = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
            th = self.ptform(u)
            if FakeSamplerAdapter.fail_with_budget_error and i >= 5:
                # Trigger budget exceeded error directly through the guard or simulate it
                raise LikelihoodBudgetExceededError(
                    f"Likelihood evaluation cap exceeded: attempted=6, cap=5"
                )
            self.logl(th)

        return FakeSamplerResult(
            log_evidence=-203500.0,
            log_evidence_err=0.12,
            n_evaluations=FakeSamplerAdapter.evaluations_to_perform,
        )


# ==============================================================================
# Test Suite: 10 Required Verification Points
# ==============================================================================


class TestWP6Stage6CProductionRunnerMocked:
    """Mocked integration verification of Stage 6C production runner and controller."""

    @pytest.fixture(autouse=True)
    def reset_fakes(self) -> None:
        """Reset fake sampler class variables before each test."""
        FakeSamplerAdapter.captured_init_kwargs = {}
        FakeSamplerAdapter.captured_run_kwargs = {}
        FakeSamplerAdapter.evaluations_to_perform = 20
        FakeSamplerAdapter.fail_with_budget_error = False
        FakeSamplerAdapter.fail_with_runtime_error = False

    # --------------------------------------------------------------------------
    # 1. Missing authorization prevents all worker activity
    # --------------------------------------------------------------------------
    def test_worker_missing_authorization_fails_closed(self, tmp_path: Path) -> None:
        """Missing authorization flag must fail closed immediately before data prep or likelihood."""
        mock_data_prep = MagicMock()
        mock_likelihood_factory = MagicMock()

        # Direct function call check
        with pytest.raises(RuntimeError, match="CRITICAL SECURITY GATE.*STRICTLY UNAUTHORIZED"):
            run_anchor_run_a_worker(
                run_id="unauthorized_test",
                output_dir=tmp_path,
                authorized=False,
                data_prep_fn=mock_data_prep,
                likelihood_factory_fn=mock_likelihood_factory,
            )

        # Invariant: Neither data prep nor likelihood construction was invoked
        mock_data_prep.assert_not_called()
        mock_likelihood_factory.assert_not_called()

        # CLI subprocess invocation check (without --authorize-stage6c-execution)
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "worker",
            "--mode",
            "anchor",
            "--target",
            "run_a",
            "--run-id",
            "unauthorized_cli_test",
            "--output-dir",
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "CRITICAL SECURITY GATE" in proc.stderr
        assert "STRICTLY UNAUTHORIZED" in proc.stderr

    # --------------------------------------------------------------------------
    # 2. Controller missing authorization prevents process launch
    # --------------------------------------------------------------------------
    def test_controller_missing_authorization_fails_closed(self, tmp_path: Path) -> None:
        """Controller must fail closed before launching worker if authorization is omitted."""
        mock_controller_cls = MagicMock()

        with pytest.raises(RuntimeError, match="CRITICAL SECURITY GATE.*controller cannot launch"):
            run_controller_anchor(
                target="run_a",
                run_id="ctrl_unauth_test",
                output_dir=tmp_path,
                authorized=False,
                controller_cls=mock_controller_cls,
            )

        # Invariant: Controller never spawned or supervised any process
        mock_controller_cls.assert_not_called()

        # CLI subprocess invocation check
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "controller",
            "--target",
            "run_a",
            "--run-id",
            "ctrl_unauth_cli_test",
            "--output-dir",
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "CRITICAL SECURITY GATE" in proc.stderr

    # --------------------------------------------------------------------------
    # 3. Authorized worker wiring passes expected frozen configuration
    # --------------------------------------------------------------------------
    def test_authorized_worker_passes_frozen_configuration(self, tmp_path: Path) -> None:
        """Authorized worker passes exact frozen Realization 01 and sampler configuration."""
        captured_sched = {}
        captured_model = ""

        def mock_data_prep(sched: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal captured_sched
            captured_sched = dict(sched)
            return {"fake": "realization_data"}

        def mock_likelihood_factory(model: str, real_data: Dict[str, Any]) -> Callable:
            nonlocal captured_model
            captured_model = model

            def mock_logl(theta: np.ndarray) -> float:
                return -100.0

            return mock_logl

        run_anchor_run_a_worker(
            run_id="frozen_config_test",
            output_dir=tmp_path,
            call_cap=600_000,
            authorized=True,
            data_prep_fn=mock_data_prep,
            likelihood_factory_fn=mock_likelihood_factory,
            sampler_adapter_cls=FakeSamplerAdapter,
            prior_transform_fn=prior_transform_4d,
        )

        # Verify Realization 01 schedule
        assert captured_sched["realization"] == 1
        assert captured_sched["seed_off"] == 7101
        assert captured_sched["seed_on"] == 7151
        assert captured_sched["sampler_seeds"] == [8101, 8201]

        # Verify Likelihood model is Model 2 plug-in Gaussian
        assert captured_model == "plug"

        # Verify Sampler adapter init parameters
        init_kwargs = FakeSamplerAdapter.captured_init_kwargs
        assert init_kwargs["ndim"] == 4
        assert init_kwargs["nlive"] == 500
        assert init_kwargs["dlogz"] == 0.10
        assert init_kwargs["seed"] == 8101
        assert init_kwargs["param_names"] == ["mc", "eta", "tc", "phic"]

        # Verify Sampler adapter run parameters
        run_kwargs = FakeSamplerAdapter.captured_run_kwargs
        assert run_kwargs["bound"] == "multi"
        assert run_kwargs["sample"] == "rwalk"
        assert run_kwargs["walks"] == 50
        assert run_kwargs["periodic"] == [3]
        assert run_kwargs["print_progress"] is False

    # --------------------------------------------------------------------------
    # 4. Sampler receives the guarded likelihood callable, not raw likelihood
    # --------------------------------------------------------------------------
    def test_sampler_receives_guarded_likelihood(self, tmp_path: Path) -> None:
        """The log_likelihood_fn passed to the sampler adapter must be a LikelihoodBudgetGuard."""
        raw_called_count = 0

        def raw_likelihood(theta: np.ndarray) -> float:
            nonlocal raw_called_count
            raw_called_count += 1
            return -50.0

        def mock_likelihood_factory(model: str, data: Dict[str, Any]) -> Callable:
            return raw_likelihood

        FakeSamplerAdapter.evaluations_to_perform = 15

        run_anchor_run_a_worker(
            run_id="guarded_logl_test",
            output_dir=tmp_path,
            call_cap=600_000,
            authorized=True,
            data_prep_fn=lambda sched: {"dummy": True},
            likelihood_factory_fn=mock_likelihood_factory,
            sampler_adapter_cls=FakeSamplerAdapter,
            prior_transform_fn=prior_transform_4d,
        )

        passed_logl = FakeSamplerAdapter.captured_init_kwargs["log_likelihood_fn"]
        # Invariant: Passed callable is an instance of LikelihoodBudgetGuard
        assert isinstance(passed_logl, LikelihoodBudgetGuard)
        assert passed_logl.call_cap == 600_000
        # Invariant: Guard counts match raw invocations
        assert passed_logl.n_completed == 15
        assert raw_called_count == 15

    # --------------------------------------------------------------------------
    # 5. Prior transform and parameter dimension are correct
    # --------------------------------------------------------------------------
    def test_prior_transform_and_dimension(self, tmp_path: Path) -> None:
        """Verify 4D unit-cube mapping to physical parameter bounds."""
        run_anchor_run_a_worker(
            run_id="ptform_dim_test",
            output_dir=tmp_path,
            call_cap=600_000,
            authorized=True,
            data_prep_fn=lambda sched: {},
            likelihood_factory_fn=lambda m, d: lambda th: -10.0,
            sampler_adapter_cls=FakeSamplerAdapter,
            prior_transform_fn=prior_transform_4d,
        )

        ptform = FakeSamplerAdapter.captured_init_kwargs["prior_transform_fn"]
        ndim = FakeSamplerAdapter.captured_init_kwargs["ndim"]
        assert ndim == 4

        # Test lower boundary mapping u = [0, 0, 0, 0]
        u_min = np.array([0.0, 0.0, 0.0, 0.0])
        th_min = ptform(u_min)
        assert pytest.approx(th_min[0], rel=1e-7) == 20.0
        assert pytest.approx(th_min[1], rel=1e-7) == 0.10
        assert pytest.approx(th_min[2], rel=1e-7) == 1.90
        assert pytest.approx(th_min[3], rel=1e-7) == 0.0

        # Test upper boundary mapping u = [1, 1, 1, 1]
        u_max = np.array([1.0, 1.0, 1.0, 1.0])
        th_max = ptform(u_max)
        assert pytest.approx(th_max[0], rel=1e-7) == 40.0
        assert pytest.approx(th_max[1], rel=1e-7) == 0.25
        assert pytest.approx(th_max[2], rel=1e-7) == 2.10
        assert pytest.approx(th_max[3], rel=1e-7) == 2.0 * math.pi

    # --------------------------------------------------------------------------
    # 6. Worker records guard counters and correct completion status
    # --------------------------------------------------------------------------
    def test_worker_records_guard_counters_and_status(self, tmp_path: Path) -> None:
        """Worker status record derives authoritative evaluation counts from the guard."""
        FakeSamplerAdapter.evaluations_to_perform = 42

        run_anchor_run_a_worker(
            run_id="status_record_test",
            output_dir=tmp_path,
            call_cap=600_000,
            authorized=True,
            data_prep_fn=lambda sched: {},
            likelihood_factory_fn=lambda m, d: lambda th: -5.0,
            sampler_adapter_cls=FakeSamplerAdapter,
            prior_transform_fn=prior_transform_4d,
        )

        status_file = tmp_path / "status_record_test_status.json"
        assert status_file.is_file()
        with open(status_file, "r", encoding="utf-8") as f:
            status_data = json.load(f)

        assert status_data["run_id"] == "status_record_test"
        assert status_data["status"] == WorkerStatus.COMPLETED.value
        assert status_data["is_complete"] is True
        assert status_data["n_eval_started"] == 42
        assert status_data["n_eval_completed"] == 42
        assert status_data["n_eval_failed"] == 0
        assert status_data["call_cap"] == 600_000

        # Verify telemetry record exists and is consistent
        telem_file = tmp_path / "status_record_test_telemetry.json"
        assert telem_file.is_file()
        with open(telem_file, "r", encoding="utf-8") as f:
            telem_data = json.load(f)
        assert len(telem_data) >= 1
        assert telem_data[-1]["n_eval_completed"] == 42

    # --------------------------------------------------------------------------
    # 7. Exceptions and budget termination cannot produce is_complete=true
    # --------------------------------------------------------------------------
    def test_exceptions_and_budget_termination_cannot_produce_is_complete(
        self, tmp_path: Path
    ) -> None:
        """Terminated or crashed runs must strictly have is_complete=False."""
        # Case A: Budget exceeded termination
        FakeSamplerAdapter.fail_with_budget_error = True
        run_anchor_run_a_worker(
            run_id="budget_exceeded_test",
            output_dir=tmp_path,
            call_cap=5,
            authorized=True,
            data_prep_fn=lambda sched: {},
            likelihood_factory_fn=lambda m, d: lambda th: -1.0,
            sampler_adapter_cls=FakeSamplerAdapter,
            prior_transform_fn=prior_transform_4d,
        )

        with open(tmp_path / "budget_exceeded_test_status.json", "r") as f:
            status_data = json.load(f)
        assert status_data["status"] == WorkerStatus.TERMINATED_BUDGET_CALLS.value
        assert status_data["is_complete"] is False

        # Case B: Sampler unhandled runtime exception
        FakeSamplerAdapter.fail_with_budget_error = False
        FakeSamplerAdapter.fail_with_runtime_error = True
        run_anchor_run_a_worker(
            run_id="runtime_crash_test",
            output_dir=tmp_path,
            call_cap=600_000,
            authorized=True,
            data_prep_fn=lambda sched: {},
            likelihood_factory_fn=lambda m, d: lambda th: -1.0,
            sampler_adapter_cls=FakeSamplerAdapter,
            prior_transform_fn=prior_transform_4d,
        )

        with open(tmp_path / "runtime_crash_test_status.json", "r") as f:
            status_data = json.load(f)
        assert status_data["status"] == WorkerStatus.FAILED_RUNTIME.value
        assert status_data["is_complete"] is False

    # --------------------------------------------------------------------------
    # 8. Missing or corrupt final result artifacts cannot produce COMPLETED
    # --------------------------------------------------------------------------
    def test_missing_or_corrupt_result_artifact_cannot_produce_completed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing or corrupted final result file must downgrade run to FAILED_RUNTIME."""
        from scripts import run_wp6_stage6c_anchor
        original_atomic_write = run_wp6_stage6c_anchor.atomic_write_json

        # Case A: Corrupted result artifact written on disk
        run_id = "corrupt_artifact_test"

        def mock_corrupt_result_write(path: Path, data: Any) -> None:
            if "_result.json" in str(path):
                Path(path).write_text("CORRUPTED_NON_JSON_DATA", encoding="utf-8")
            else:
                original_atomic_write(path, data)

        monkeypatch.setattr(run_wp6_stage6c_anchor, "atomic_write_json", mock_corrupt_result_write)

        run_anchor_run_a_worker(
            run_id=run_id,
            output_dir=tmp_path,
            call_cap=600_000,
            authorized=True,
            data_prep_fn=lambda sched: {},
            likelihood_factory_fn=lambda m, d: lambda th: -1.0,
            sampler_adapter_cls=FakeSamplerAdapter,
            prior_transform_fn=prior_transform_4d,
        )

        with open(tmp_path / f"{run_id}_status.json", "r") as f:
            status_data = json.load(f)
        assert status_data["status"] == WorkerStatus.FAILED_RUNTIME.value
        assert status_data["is_complete"] is False

        # Case B: Missing result artifact on disk
        run_id_missing = "missing_artifact_test"

        def mock_missing_result_write(path: Path, data: Any) -> None:
            if "_result.json" in str(path):
                pass  # Do not write result artifact
            else:
                original_atomic_write(path, data)

        monkeypatch.setattr(run_wp6_stage6c_anchor, "atomic_write_json", mock_missing_result_write)

        run_anchor_run_a_worker(
            run_id=run_id_missing,
            output_dir=tmp_path,
            call_cap=600_000,
            authorized=True,
            data_prep_fn=lambda sched: {},
            likelihood_factory_fn=lambda m, d: lambda th: -1.0,
            sampler_adapter_cls=FakeSamplerAdapter,
            prior_transform_fn=prior_transform_4d,
        )

        with open(tmp_path / f"{run_id_missing}_status.json", "r") as f:
            status_data_missing = json.load(f)
        assert status_data_missing["status"] == WorkerStatus.FAILED_RUNTIME.value
        assert status_data_missing["is_complete"] is False

        # Case C: Controller-level reconciliation if worker claims COMPLETED but artifact is missing
        monkeypatch.undo()
        ctrl_output_dir = tmp_path / "ctrl_corrupt_test"
        ctrl_output_dir.mkdir(parents=True, exist_ok=True)
        fake_ctrl_run_id = "ctrl_fake_run"

        class MockController:
            def __init__(self, config: Any) -> None:
                self.config = config

            def run_worker_process(self, **kwargs: Any) -> Dict[str, Any]:
                return {
                    "run_id": fake_ctrl_run_id,
                    "status": WorkerStatus.COMPLETED.value,
                    "is_complete": True,
                    "n_eval_completed": 100,
                    "cpu_time_s": 10.0,
                    "wall_time_s": 12.0,
                    "authoritative_counts_available": True,
                }

        # Do not create fake_ctrl_run_id_result.json, so controller detects missing artifact
        reconciled = run_controller_anchor(
            target="run_a",
            run_id=fake_ctrl_run_id,
            output_dir=ctrl_output_dir,
            authorized=True,
            controller_cls=MockController,
        )

        assert reconciled["reconciled_status"]["status"] == WorkerStatus.FAILED_RUNTIME.value
        assert reconciled["reconciled_status"]["is_complete"] is False
        assert "Missing" in reconciled["reconciled_status"]["termination_reason"]

    # --------------------------------------------------------------------------
    # 9. Controller launches expected worker command and applies resource limits
    # --------------------------------------------------------------------------
    def test_controller_launches_expected_command_and_limits(self, tmp_path: Path) -> None:
        """Controller formats exact worker CLI command and propagates budgets."""
        captured_kwargs: Dict[str, Any] = {}

        class MockControllerCapturing:
            def __init__(self, config: Any) -> None:
                self.config = config

            def run_worker_process(self, **kwargs: Any) -> Dict[str, Any]:
                nonlocal captured_kwargs
                captured_kwargs = dict(kwargs)
                # Write a minimal valid result file so reconciliation succeeds
                res_file = Path(kwargs["output_dir"]) / f"{kwargs['run_id']}_result.json"
                atomic_write_json(
                    res_file,
                    {"status": "SUCCESS", "log_evidence": -100.0, "log_evidence_err": 0.1},
                )
                return {
                    "run_id": kwargs["run_id"],
                    "status": WorkerStatus.COMPLETED.value,
                    "is_complete": True,
                    "n_eval_completed": 1000,
                    "cpu_time_s": 50.0,
                    "wall_time_s": 60.0,
                    "authoritative_counts_available": True,
                }

        run_controller_anchor(
            target="run_a",
            run_id="ctrl_command_test",
            output_dir=tmp_path,
            call_cap=600_000,
            cpu_budget=900.0,
            wall_budget=1500.0,
            authorized=True,
            controller_cls=MockControllerCapturing,
        )

        cmd = captured_kwargs["command"]
        assert cmd[1] == str(SCRIPT_PATH)
        assert cmd[2] == "worker"
        assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "anchor"
        assert "--target" in cmd and cmd[cmd.index("--target") + 1] == "run_a"
        assert "--run-id" in cmd and cmd[cmd.index("--run-id") + 1] == "ctrl_command_test"
        assert "--call-cap" in cmd and cmd[cmd.index("--call-cap") + 1] == "600000"
        assert "--authorize-stage6c-execution" in cmd

        # Verify resource limits passed to run_worker_process
        assert captured_kwargs["custom_max_calls"] == 600_000
        assert captured_kwargs["custom_max_cpu_s"] == 900.0
        assert captured_kwargs["custom_max_wall_s"] == 1500.0

    # --------------------------------------------------------------------------
    # 10. Run B cannot be selected or dispatched in this milestone
    # --------------------------------------------------------------------------
    def test_run_b_cannot_be_selected_or_dispatched(self, tmp_path: Path) -> None:
        """Attempting to select or dispatch Run B must be strictly blocked at all interfaces."""
        # Controller interface block
        with pytest.raises(RuntimeError, match="Target 'run_b' is strictly prohibited"):
            run_controller_anchor(
                target="run_b",
                run_id="illegal_run_b",
                output_dir=tmp_path,
                authorized=True,
            )

        # Worker CLI interface block
        cmd_worker = [
            sys.executable,
            str(SCRIPT_PATH),
            "worker",
            "--mode",
            "anchor",
            "--target",
            "run_b",
            "--run-id",
            "illegal_run_b",
            "--output-dir",
            str(tmp_path),
            "--authorize-stage6c-execution",
        ]
        proc_worker = subprocess.run(cmd_worker, capture_output=True, text=True)
        assert proc_worker.returncode != 0
        assert "invalid choice" in proc_worker.stderr or "strictly prohibited" in proc_worker.stderr

        # Controller CLI interface block
        cmd_ctrl = [
            sys.executable,
            str(SCRIPT_PATH),
            "controller",
            "--target",
            "run_b",
            "--run-id",
            "illegal_run_b",
            "--output-dir",
            str(tmp_path),
            "--authorize-stage6c-execution",
        ]
        proc_ctrl = subprocess.run(cmd_ctrl, capture_output=True, text=True)
        assert proc_ctrl.returncode != 0
        assert "invalid choice" in proc_ctrl.stderr or "strictly prohibited" in proc_ctrl.stderr
