"""Isolated unit and integration tests for WP6 Stage 6C resource-control infrastructure.

These tests verify:
1. Likelihood call cap enforcement (before excess evaluation, boundary cases 0, 1, N).
2. CPU watchdog monitoring and detection.
3. Wall-clock timeout watchdog monitoring and detection.
4. Graceful shutdown signaling and recording.
5. Forced process-tree termination fallback (`taskkill` / kill).
6. Partial output classification guarantee (never classified as complete).
7. Three-resource Run B dispatch gate rule across all required scenarios.
8. Aggregate ledger ceiling bounds.
9. Worker crash and malformed telemetry recovery.
10. Repeated test run identity separation.

MANDATORY SAFETY BOUNDARY:
--------------------------
These tests use strictly synthetic dummy workloads. Zero production CBC waveforms,
zero scientific likelihoods, and zero nested samplers are invoked.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from gw_inference.resource_control import (
    AggregateResourceLedger,
    LikelihoodBudgetExceededError,
    LikelihoodBudgetGuard,
    Stage6CConfig,
    Stage6CController,
    WorkerStatus,
    atomic_write_json,
    get_process_tree_cpu,
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = WORKSPACE_DIR / "scripts" / "run_wp6_stage6c_anchor.py"


# ==============================================================================
# 1. LikelihoodBudgetGuard Unit Tests (In-Path Counter & Invariant)
# ==============================================================================


class TestLikelihoodBudgetGuard:
    """Tests for in-path LikelihoodBudgetGuard counter enforcement and invariance."""

    def test_call_cap_zero_blocks_first_evaluation(self) -> None:
        """Call cap 0 must block the very first evaluation before entering target_fn."""
        eval_count = 0

        def dummy_fn(x: float) -> float:
            nonlocal eval_count
            eval_count += 1
            return x * 2.0

        guard = LikelihoodBudgetGuard(dummy_fn, call_cap=0)

        with pytest.raises(LikelihoodBudgetExceededError) as exc_info:
            guard(42.0)

        assert "Likelihood call budget reached" in str(exc_info.value)
        assert eval_count == 0  # Function was never entered
        assert guard.n_attempted == 1
        assert guard.n_started == 0
        assert guard.n_completed == 0
        assert guard.n_failed == 0
        assert guard.n_completed <= guard.n_started <= guard.call_cap

    def test_call_cap_one_permits_single_evaluation(self) -> None:
        """Call cap 1 permits exactly one evaluation and blocks the second."""
        eval_count = 0

        def dummy_fn(x: float) -> float:
            nonlocal eval_count
            eval_count += 1
            return x * 2.0

        guard = LikelihoodBudgetGuard(dummy_fn, call_cap=1)

        # Call 1: Must succeed
        res = guard(10.0)
        assert res == 20.0
        assert eval_count == 1
        assert guard.n_attempted == 1
        assert guard.n_started == 1
        assert guard.n_completed == 1

        # Call 2: Must be blocked immediately before entering
        with pytest.raises(LikelihoodBudgetExceededError):
            guard(20.0)

        assert eval_count == 1  # Function was not entered a second time
        assert guard.n_attempted == 2
        assert guard.n_started == 1
        assert guard.n_completed == 1
        assert guard.n_completed <= guard.n_started <= guard.call_cap

    def test_call_cap_small_positive_and_boundary(self) -> None:
        """Test cap=5: exactly 5 succeed, 6th is blocked before entering."""
        eval_count = 0

        def dummy_fn(x: float) -> float:
            nonlocal eval_count
            eval_count += 1
            return x + 1.0

        guard = LikelihoodBudgetGuard(dummy_fn, call_cap=5)

        for i in range(5):
            res = guard(float(i))
            assert res == float(i + 1)
            assert guard.n_started == i + 1
            assert guard.n_completed == i + 1

        # 6th call must fail
        with pytest.raises(LikelihoodBudgetExceededError):
            guard(99.0)

        assert eval_count == 5
        assert guard.n_attempted == 6
        assert guard.n_started == 5
        assert guard.n_completed == 5
        assert guard.n_completed <= guard.n_started <= guard.call_cap

    def test_underlying_exception_not_misclassified(self) -> None:
        """Exceptions from target_fn must NOT be masked as budget exhaustion."""

        def failing_fn(x: float) -> float:
            raise ValueError("Physical domain parameter error")

        guard = LikelihoodBudgetGuard(failing_fn, call_cap=10)

        with pytest.raises(ValueError) as exc_info:
            guard(1.0)

        assert "Physical domain parameter error" in str(exc_info.value)
        assert guard.n_attempted == 1
        assert guard.n_started == 1
        assert guard.n_completed == 0
        assert guard.n_failed == 1

    def test_thread_safety_concurrency(self) -> None:
        """Verify thread-safety of invocation counters under concurrent evaluation."""
        eval_count = 0
        lock = threading.Lock()

        def slow_fn(x: float) -> float:
            nonlocal eval_count
            time.sleep(0.001)
            with lock:
                eval_count += 1
            return x * 2.0

        cap = 25
        guard = LikelihoodBudgetGuard(slow_fn, call_cap=cap)
        threads = []
        n_threads = 10
        calls_per_thread = 20

        def worker_target() -> None:
            for _ in range(calls_per_thread):
                try:
                    guard(1.0)
                except LikelihoodBudgetExceededError:
                    pass

        for _ in range(n_threads):
            t = threading.Thread(target=worker_target)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert guard.n_started == cap
        assert guard.n_completed == cap
        assert eval_count == cap
        assert guard.n_attempted == n_threads * calls_per_thread
        assert guard.n_completed <= guard.n_started <= guard.call_cap


# ==============================================================================
# 2. Configuration Validation Tests
# ==============================================================================


class TestStage6CConfig:
    """Tests for Stage6CConfig validation."""

    def test_default_config_matches_preflight_spec(self) -> None:
        """Verify default limits match the approved Stage 6C preflight specifications."""
        cfg = Stage6CConfig()
        assert cfg.max_calls_per_run == 600_000
        assert cfg.max_cpu_per_run_s == 900.0
        assert cfg.max_wall_per_run_s == 1_500.0
        assert cfg.max_aggregate_calls == 1_200_000
        assert cfg.max_aggregate_cpu_s == 1_800.0
        assert cfg.max_aggregate_wall_s == 3_000.0
        assert cfg.required_calls_run_b == 530_000
        assert cfg.required_cpu_run_b_s == 795.0
        assert cfg.required_wall_run_b_s == 1_300.0
        assert cfg.watchdog_interval_s == 1.0
        assert cfg.shutdown_grace_period_s == 5.0

    def test_invalid_config_raises_error(self) -> None:
        """Invalid configurations must raise ValueError without silent default fallback."""
        with pytest.raises(ValueError, match="max_calls_per_run"):
            Stage6CConfig(max_calls_per_run=0)

        with pytest.raises(ValueError, match="max_calls_per_run.*cannot exceed max_aggregate_calls"):
            Stage6CConfig(max_calls_per_run=1_500_000, max_aggregate_calls=1_200_000)

        with pytest.raises(ValueError, match="required_cpu_run_b_s.*cannot exceed max_cpu_per_run_s"):
            Stage6CConfig(required_cpu_run_b_s=950.0, max_cpu_per_run_s=900.0)


# ==============================================================================
# 3. Aggregate Ledger and Run B Gate Tests
# ==============================================================================


class TestAggregateResourceLedger:
    """Tests for AggregateResourceLedger and Run B dispatch feasibility gate."""

    def test_run_b_gate_all_scenarios(self) -> None:
        """Evaluate Run B gate across all 7 scenarios specified in preflight."""
        cfg = Stage6CConfig()

        # Scenario 1: Run A consumes minimal resources
        ledger1 = AggregateResourceLedger(cfg)
        ledger1.record_run("run_a", calls=5_000, cpu_s=10.0, wall_s=25.0, status=WorkerStatus.COMPLETED)
        gate1 = ledger1.evaluate_run_b_gate()
        assert gate1["authorized"] is True
        assert gate1["pass_calls"] is True
        assert gate1["pass_cpu"] is True
        assert gate1["pass_wall"] is True

        # Scenario 2: Run A reaches exactly expected planning allocation
        ledger2 = AggregateResourceLedger(cfg)
        ledger2.record_run("run_a", calls=530_000, cpu_s=795.0, wall_s=1_300.0, status=WorkerStatus.COMPLETED)
        gate2 = ledger2.evaluate_run_b_gate()
        assert gate2["authorized"] is True
        assert gate2["cap_calls"] == 600_000  # min(600k, 1200k - 530k = 670k) = 600k >= 530k
        assert gate2["cap_cpu_s"] == 900.0  # min(900, 1800 - 795 = 1005) = 900 >= 795
        assert gate2["cap_wall_s"] == 1_500.0  # min(1500, 3000 - 1300 = 1700) = 1500 >= 1300

        # Scenario 3: Run A exceeds expected allocation but remains within aggregate limits
        ledger3 = AggregateResourceLedger(cfg)
        ledger3.record_run("run_a", calls=580_000, cpu_s=850.0, wall_s=1_400.0, status=WorkerStatus.COMPLETED)
        gate3 = ledger3.evaluate_run_b_gate()
        assert gate3["authorized"] is True
        assert gate3["cap_calls"] == 600_000  # min(600k, 620k) = 600k >= 530k
        assert gate3["cap_cpu_s"] == 900.0  # min(900, 950) = 900 >= 795
        assert gate3["cap_wall_s"] == 1_500.0  # min(1500, 1600) = 1500 >= 1300

        # Scenario 4: Run A leaves insufficient likelihood calls
        ledger4 = AggregateResourceLedger(cfg)
        ledger4.record_run("run_a", calls=700_000, cpu_s=500.0, wall_s=800.0, status=WorkerStatus.COMPLETED)
        gate4 = ledger4.evaluate_run_b_gate()
        assert gate4["authorized"] is False
        assert gate4["pass_calls"] is False  # min(600k, 1200k - 700k = 500k) = 500k < 530k
        assert gate4["pass_cpu"] is True
        assert gate4["pass_wall"] is True

        # Scenario 5: Run A leaves insufficient process CPU
        ledger5 = AggregateResourceLedger(cfg)
        ledger5.record_run("run_a", calls=500_000, cpu_s=1_100.0, wall_s=1_200.0, status=WorkerStatus.COMPLETED)
        gate5 = ledger5.evaluate_run_b_gate()
        assert gate5["authorized"] is False
        assert gate5["pass_calls"] is True
        assert gate5["pass_cpu"] is False  # min(900, 1800 - 1100 = 700) = 700 < 795
        assert gate5["pass_wall"] is True

        # Scenario 6: Run A leaves insufficient wall-clock time
        ledger6 = AggregateResourceLedger(cfg)
        ledger6.record_run("run_a", calls=400_000, cpu_s=600.0, wall_s=1_800.0, status=WorkerStatus.COMPLETED)
        gate6 = ledger6.evaluate_run_b_gate()
        assert gate6["authorized"] is False
        assert gate6["pass_calls"] is True
        assert gate6["pass_cpu"] is True
        assert gate6["pass_wall"] is False  # min(1500, 3000 - 1800 = 1200) = 1200 < 1300

        # Scenario 7: Run A reaches aggregate ceiling
        ledger7 = AggregateResourceLedger(cfg)
        ledger7.record_run("run_a", calls=1_200_000, cpu_s=1_800.0, wall_s=3_000.0, status=WorkerStatus.COMPLETED)
        gate7 = ledger7.evaluate_run_b_gate()
        assert gate7["authorized"] is False
        assert gate7["pass_calls"] is False
        assert gate7["pass_cpu"] is False
        assert gate7["pass_wall"] is False

    def test_run_b_gate_blocked_when_no_run_a(self) -> None:
        """Run B must be strictly blocked if Run A has not been recorded."""
        ledger = AggregateResourceLedger()
        gate = ledger.evaluate_run_b_gate()
        assert gate["authorized"] is False
        assert gate["gate_blocked_reason"] == "NO_BASELINE_RUN_RECORDED"
        assert "No baseline run" in gate["error"]

    def test_run_b_gate_blocked_when_run_a_accounting_invalid(self) -> None:
        """Run B must be blocked if Run A resource accounting is missing, corrupted, or unmetered."""
        ledger = AggregateResourceLedger()
        ledger.record_run(
            run_id="run_a",
            calls=None,  # Missing/corrupted call accounting
            cpu_s=100.0,
            wall_s=200.0,
            status=WorkerStatus.FAILED_RUNTIME,
            has_valid_accounting=False,
        )
        gate = ledger.evaluate_run_b_gate()
        assert gate["authorized"] is False
        assert gate["gate_blocked_reason"] == "RUN_A_ACCOUNTING_INVALID_OR_MISSING"
        assert "Run A accounting is missing" in gate["error"]

    def test_run_b_gate_blocked_on_negative_raw_remaining(self) -> None:
        """Negative remaining resources must strictly block Run B dispatch."""
        ledger = AggregateResourceLedger()
        ledger.record_run(
            run_id="run_a",
            calls=1_300_000,  # Exceeds 1.2M ceiling by 100k
            cpu_s=500.0,
            wall_s=800.0,
            status=WorkerStatus.COMPLETED,
        )
        gate = ledger.evaluate_run_b_gate()
        assert gate["authorized"] is False
        assert gate["pass_calls"] is False

    def test_run_b_gate_records_scientific_acceptance_disclaimer_for_incomplete_run_a(self) -> None:
        """Resource gate feasibility must explicitly record Run A completion and scientific disclaimer."""
        ledger = AggregateResourceLedger()
        ledger.record_run(
            run_id="run_a",
            calls=400_000,
            cpu_s=600.0,
            wall_s=900.0,
            status=WorkerStatus.TERMINATED_BUDGET_CALLS,
            is_complete=False,
            has_valid_accounting=True,
        )
        gate = ledger.evaluate_run_b_gate()
        assert gate["authorized"] is False  # Incomplete Run A strictly blocks Run B
        assert gate["gate_blocked_reason"] == "RUN_A_TERMINATED_BUDGET_CALLS"
        assert gate["run_a_complete"] is False
        assert gate["run_a_status"] == WorkerStatus.TERMINATED_BUDGET_CALLS.value
        assert "does not constitute scientific acceptance" in gate["scientific_acceptance_disclaimer"]

    def test_run_b_gate_blocked_when_run_a_terminated_by_call_budget(self) -> None:
        """Run A terminated by call budget must strictly block Run B even with ample remaining resources."""
        ledger = AggregateResourceLedger()
        ledger.record_run(
            run_id="run_a",
            calls=300_000,
            cpu_s=400.0,
            wall_s=600.0,
            status=WorkerStatus.TERMINATED_BUDGET_CALLS,
            is_complete=False,
            has_valid_accounting=True,
        )
        gate = ledger.evaluate_run_b_gate()
        assert gate["authorized"] is False
        assert gate["gate_blocked_reason"] == "RUN_A_TERMINATED_BUDGET_CALLS"
        assert "strictly BLOCKED" in gate["error"]

    def test_run_b_gate_blocked_when_run_a_terminated_by_cpu_or_wall_budget(self) -> None:
        """Run A terminated by CPU or wall budget must strictly block Run B."""
        # CPU budget termination
        ledger_cpu = AggregateResourceLedger()
        ledger_cpu.record_run(
            run_id="run_a",
            calls=200_000,
            cpu_s=900.0,
            wall_s=1_000.0,
            status=WorkerStatus.TERMINATED_BUDGET_CPU,
            is_complete=False,
            has_valid_accounting=True,
        )
        gate_cpu = ledger_cpu.evaluate_run_b_gate()
        assert gate_cpu["authorized"] is False
        assert gate_cpu["gate_blocked_reason"] == "RUN_A_TERMINATED_BUDGET_CPU"

        # Wall budget termination
        ledger_wall = AggregateResourceLedger()
        ledger_wall.record_run(
            run_id="run_a",
            calls=200_000,
            cpu_s=300.0,
            wall_s=1_500.0,
            status=WorkerStatus.TERMINATED_BUDGET_WALL,
            is_complete=False,
            has_valid_accounting=True,
        )
        gate_wall = ledger_wall.evaluate_run_b_gate()
        assert gate_wall["authorized"] is False
        assert gate_wall["gate_blocked_reason"] == "RUN_A_TERMINATED_BUDGET_WALL"

    def test_run_b_gate_blocked_when_run_a_crashed_or_cancelled(self) -> None:
        """Run A crashed or cancelled by controller must strictly block Run B."""
        # Crashed / FAILED_RUNTIME
        ledger_crash = AggregateResourceLedger()
        ledger_crash.record_run(
            run_id="run_a",
            calls=100_000,
            cpu_s=150.0,
            wall_s=250.0,
            status=WorkerStatus.FAILED_RUNTIME,
            is_complete=False,
            has_valid_accounting=True,
        )
        gate_crash = ledger_crash.evaluate_run_b_gate()
        assert gate_crash["authorized"] is False
        assert gate_crash["gate_blocked_reason"] == "RUN_A_CRASHED_OR_FAILED_RUNTIME"

        # Cancelled by controller
        ledger_cancel = AggregateResourceLedger()
        ledger_cancel.record_run(
            run_id="run_a",
            calls=50_000,
            cpu_s=80.0,
            wall_s=120.0,
            status=WorkerStatus.CANCELLED_BY_CONTROLLER,
            is_complete=False,
            has_valid_accounting=True,
        )
        gate_cancel = ledger_cancel.evaluate_run_b_gate()
        assert gate_cancel["authorized"] is False
        assert gate_cancel["gate_blocked_reason"] == "RUN_A_CANCELLED"

    def test_run_b_gate_blocked_when_run_a_complete_but_accounting_invalid(self) -> None:
        """Run A marked complete with COMPLETED status but invalid accounting must block Run B."""
        ledger = AggregateResourceLedger()
        ledger.record_run(
            run_id="run_a",
            calls=None,  # Invalid / unmetered calls
            cpu_s=300.0,
            wall_s=500.0,
            status=WorkerStatus.COMPLETED,
            is_complete=True,
            has_valid_accounting=False,
        )
        gate = ledger.evaluate_run_b_gate()
        assert gate["authorized"] is False
        assert gate["gate_blocked_reason"] == "RUN_A_ACCOUNTING_INVALID_OR_MISSING"

    def test_run_b_gate_passes_when_cleanly_completed_with_valid_accounting_and_resources(self) -> None:
        """Fully completed Run A with valid accounting and sufficient resources authorizes Run B."""
        ledger = AggregateResourceLedger()
        ledger.record_run(
            run_id="run_a",
            calls=500_000,
            cpu_s=700.0,
            wall_s=1_100.0,
            status=WorkerStatus.COMPLETED,
            is_complete=True,
            has_valid_accounting=True,
        )
        gate = ledger.evaluate_run_b_gate()
        assert gate["authorized"] is True
        assert gate["gate_blocked_reason"] == "NONE"
        assert gate["pass_calls"] is True
        assert gate["pass_cpu"] is True
        assert gate["pass_wall"] is True


# ==============================================================================
# 4. Controller & Watchdog Isolated Subprocess Tests (Synthetic Workloads)
# ==============================================================================


class TestStage6CControllerSubprocess:
    """Isolated subprocess tests using synthetic workers to verify watchdog and safety."""

    @pytest.fixture
    def test_output_dir(self, tmp_path: Path) -> Path:
        """Create a dedicated temporary directory for test artifacts."""
        p = tmp_path / "stage6c_test_runs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def test_call_cap_enforced_in_subprocess(self, test_output_dir: Path) -> None:
        """Verify worker stops and records TERMINATED_BUDGET_CALLS when cap is reached."""
        controller = Stage6CController()
        run_id = "syn_test_call_cap"
        cap = 12

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "worker",
            "--run-id",
            run_id,
            "--output-dir",
            str(test_output_dir),
            "--worker-type",
            "call_bound",
            "--call-cap",
            str(cap),
            "--target-iterations",
            "100",
        ]

        result = controller.run_worker_process(
            command=cmd,
            run_id=run_id,
            output_dir=test_output_dir,
            custom_max_calls=cap,
        )

        assert result["status"] == WorkerStatus.TERMINATED_BUDGET_CALLS.value
        assert result["is_complete"] is False
        assert result["n_eval_started"] == cap
        assert result["n_eval_completed"] == cap
        assert result["forced_kill_applied"] is False

    def test_cpu_watchdog_detects_cpu_intensive_worker(self, test_output_dir: Path) -> None:
        """Verify watchdog detects process-tree CPU threshold and terminates cleanly."""
        controller = Stage6CController(
            config=Stage6CConfig(
                watchdog_interval_s=0.2,
                shutdown_grace_period_s=2.0,
            )
        )
        run_id = "syn_test_cpu_watchdog"
        cpu_limit = 1.0  # Low budget for fast test execution

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "worker",
            "--run-id",
            run_id,
            "--output-dir",
            str(test_output_dir),
            "--worker-type",
            "cpu_intensive",
            "--call-cap",
            "1_000_000",
            "--target-iterations",
            "500",
        ]

        result = controller.run_worker_process(
            command=cmd,
            run_id=run_id,
            output_dir=test_output_dir,
            custom_max_cpu_s=cpu_limit,
            custom_max_wall_s=30.0,
        )

        assert result["status"] == WorkerStatus.TERMINATED_BUDGET_CPU.value
        assert result["is_complete"] is False
        assert result["cpu_time_s"] >= cpu_limit

    def test_wall_watchdog_detects_sleeping_worker(self, test_output_dir: Path) -> None:
        """Verify watchdog detects wall-clock timeout on sleeping worker."""
        controller = Stage6CController(
            config=Stage6CConfig(
                watchdog_interval_s=0.2,
                shutdown_grace_period_s=1.5,
            )
        )
        run_id = "syn_test_wall_watchdog"
        wall_limit = 1.0  # Fast timeout for test

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "worker",
            "--run-id",
            run_id,
            "--output-dir",
            str(test_output_dir),
            "--worker-type",
            "sleep",
            "--call-cap",
            "1_000",
            "--param-val",
            "10.0",  # Sleeps 10s, but wall limit is 1.0s
        ]

        result = controller.run_worker_process(
            command=cmd,
            run_id=run_id,
            output_dir=test_output_dir,
            custom_max_cpu_s=30.0,
            custom_max_wall_s=wall_limit,
        )

        assert result["status"] == WorkerStatus.TERMINATED_BUDGET_WALL.value
        assert result["is_complete"] is False
        assert result["wall_time_s"] >= wall_limit

    def test_forced_termination_fallback_on_uncooperative_worker(self, test_output_dir: Path) -> None:
        """Verify fallback forced kill terminates uncooperative process after grace period."""
        controller = Stage6CController(
            config=Stage6CConfig(
                watchdog_interval_s=0.2,
                shutdown_grace_period_s=1.0,  # Short grace period
            )
        )
        run_id = "syn_test_uncooperative"

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "worker",
            "--run-id",
            run_id,
            "--output-dir",
            str(test_output_dir),
            "--worker-type",
            "uncooperative",
            "--call-cap",
            "1_000",
        ]

        result = controller.run_worker_process(
            command=cmd,
            run_id=run_id,
            output_dir=test_output_dir,
            custom_max_cpu_s=30.0,
            custom_max_wall_s=0.8,  # Timeout quickly
        )

        assert result["status"] == WorkerStatus.TERMINATED_BUDGET_WALL.value
        assert result["is_complete"] is False
        assert result["forced_kill_applied"] is True

    def test_worker_crash_handling(self, test_output_dir: Path) -> None:
        """Verify worker exceptions are safely caught and recorded as FAILED_RUNTIME."""
        controller = Stage6CController()
        run_id = "syn_test_crash"

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "worker",
            "--run-id",
            run_id,
            "--output-dir",
            str(test_output_dir),
            "--worker-type",
            "crash",
            "--call-cap",
            "1_000",
        ]

        result = controller.run_worker_process(
            command=cmd,
            run_id=run_id,
            output_dir=test_output_dir,
        )

        assert result["status"] == WorkerStatus.FAILED_RUNTIME.value
        assert result["is_complete"] is False

    def test_safety_gate_blocks_anchor_mode(self) -> None:
        """Attempting to run --mode anchor in Stage 6C must raise an immediate RuntimeError."""
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "worker",
            "--run-id",
            "illegal_anchor_attempt",
            "--output-dir",
            ".",
            "--mode",
            "anchor",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "CRITICAL SECURITY GATE" in proc.stderr
        assert "STRICTLY UNAUTHORIZED" in proc.stderr

    def test_repeated_runs_unique_identities(self, test_output_dir: Path) -> None:
        """Repeated test runs must preserve unique identities and avoid file overwrites."""
        controller = Stage6CController()

        for idx in range(3):
            run_id = f"syn_repeat_{idx}"
            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "worker",
                "--run-id",
                run_id,
                "--output-dir",
                str(test_output_dir),
                "--worker-type",
                "call_bound",
                "--call-cap",
                "5",
                "--target-iterations",
                "5",
            ]
            res = controller.run_worker_process(
                command=cmd,
                run_id=run_id,
                output_dir=test_output_dir,
            )
            assert res["run_id"] == run_id
            assert res["is_complete"] is True
            assert (test_output_dir / f"{run_id}_status.json").is_file()


# ==============================================================================
# 5. Checkpoint Integrity and Failure Handling Tests
# ==============================================================================


class TestCheckpointIntegrityAndFailureHandling:
    """Rigorous audit tests for checkpoint persistence, atomic replacement, and failure states."""

    def test_atomic_write_preserves_previous_file_on_write_failure(self, tmp_path: Path) -> None:
        """Atomic write must preserve existing valid target file if replacement fails before os.replace."""
        target_file = tmp_path / "protected_status.json"
        initial_data = {"run_id": "initial_valid", "status": "COMPLETED", "n_eval": 500}
        atomic_write_json(target_file, initial_data)
        assert target_file.is_file()

        # Attempt to write an un-serializable circular object which raises ValueError in json.dump
        bad_data: Dict[str, Any] = {}
        bad_data["self"] = bad_data
        with pytest.raises((ValueError, TypeError)):
            atomic_write_json(target_file, bad_data)

        # Verify original file was completely preserved and uncorrupted
        with open(target_file, "r", encoding="utf-8") as f:
            recovered = json.load(f)
        assert recovered == initial_data

        # Verify no orphaned temporary files remained
        tmp_files = list(tmp_path.glob("*.tmp.*"))
        assert len(tmp_files) == 0

    def test_malformed_status_json_detected_and_not_treated_as_success(self, tmp_path: Path) -> None:
        """Malformed or truncated status JSON must be marked FAILED_RUNTIME and is_complete=False."""
        controller = Stage6CController()
        status_file = tmp_path / "corrupt_run_status.json"

        # Write truncated/corrupt JSON content
        with open(status_file, "w", encoding="utf-8") as f:
            f.write('{"status": "COMPLETED", "is_complete": tr')  # Truncated JSON

        result = controller._reconcile_run_status(
            run_id="corrupt_run",
            status_file=status_file,
            exit_code=0,  # Process exited with code 0, but status is corrupt
            elapsed_wall=1.5,
            last_cpu_time=0.5,
            max_calls=600_000,
            max_cpu=900.0,
            max_wall=1500.0,
            shutdown_requested=False,
            shutdown_status=WorkerStatus.COMPLETED,
            termination_reason="PROCESS_COMPLETED",
            forced_kill_applied=False,
        )

        assert result["status"] == WorkerStatus.FAILED_RUNTIME.value
        assert result["is_complete"] is False
        assert result["telemetry_integrity"] == "CORRUPTED"
        assert result["authoritative_counts_available"] is False
        assert result["n_eval_completed"] is None
        assert "Malformed" in result["termination_reason"]

    def test_worker_crash_before_status_produces_incomplete_recovery(self, tmp_path: Path) -> None:
        """Worker crash with no status file produced must yield an incomplete recovery record."""
        controller = Stage6CController()
        status_file = tmp_path / "crashed_run_status.json"
        assert not status_file.is_file()

        result = controller._reconcile_run_status(
            run_id="crashed_run",
            status_file=status_file,
            exit_code=1,  # Non-zero exit
            elapsed_wall=2.0,
            last_cpu_time=0.8,
            max_calls=600_000,
            max_cpu=900.0,
            max_wall=1500.0,
            shutdown_requested=False,
            shutdown_status=WorkerStatus.COMPLETED,
            termination_reason="PROCESS_COMPLETED",
            forced_kill_applied=False,
        )

        assert result["status"] == WorkerStatus.FAILED_RUNTIME.value
        assert result["is_complete"] is False
        assert result["telemetry_integrity"] == "MISSING"
        assert result["authoritative_counts_available"] is False
        assert result["n_eval_completed"] is None
        assert status_file.is_file()  # Reconciled status was saved

    def test_forced_termination_overrides_worker_completed_flag(self, tmp_path: Path) -> None:
        """Forced termination or watchdog shutdown must strictly override any worker is_complete=True."""
        controller = Stage6CController()
        status_file = tmp_path / "forced_run_status.json"

        # Worker wrote status claiming COMPLETED before being killed
        worker_record = {
            "run_id": "forced_run",
            "status": "COMPLETED",
            "is_complete": True,
            "n_eval_started": 500,
            "n_eval_completed": 500,
            "cpu_time_s": 50.0,
        }
        atomic_write_json(status_file, worker_record)

        result = controller._reconcile_run_status(
            run_id="forced_run",
            status_file=status_file,
            exit_code=-1,
            elapsed_wall=10.0,
            last_cpu_time=910.0,
            max_calls=600_000,
            max_cpu=900.0,
            max_wall=1500.0,
            shutdown_requested=True,
            shutdown_status=WorkerStatus.TERMINATED_BUDGET_CPU,
            termination_reason="Watchdog detected CPU threshold",
            forced_kill_applied=True,
        )

        assert result["status"] == WorkerStatus.TERMINATED_BUDGET_CPU.value
        assert result["is_complete"] is False  # MUST be overridden to False
        assert result["forced_kill_applied"] is True

    def test_distinguish_partial_checkpoints_from_final_outputs(self, tmp_path: Path) -> None:
        """Partial progress telemetry and checkpoints must be clearly distinguishable from final output."""
        run_id = "partial_test"
        checkpoint_file = tmp_path / f"{run_id}_checkpoint.json"
        status_file = tmp_path / f"{run_id}_status.json"

        # Write intermediate checkpoint
        intermediate_checkpoint = {
            "run_id": run_id,
            "is_final": False,
            "stage": "INTERMEDIATE_CHECKPOINT",
            "iteration": 250,
            "live_points_recorded": 500,
            "dead_points_recorded": 1200,
        }
        atomic_write_json(checkpoint_file, intermediate_checkpoint)

        # Write completed status
        final_status = {
            "run_id": run_id,
            "status": "COMPLETED",
            "is_complete": True,
            "n_eval_completed": 50_000,
        }
        atomic_write_json(status_file, final_status)

        with open(checkpoint_file, "r") as f:
            cp = json.load(f)
        with open(status_file, "r") as f:
            st = json.load(f)

        assert cp["is_final"] is False
        assert cp["stage"] == "INTERMEDIATE_CHECKPOINT"
        assert st["is_complete"] is True
        assert cp.get("is_complete") is not True

    def test_missing_telemetry_recorded_in_ledger_blocks_run_b(self, tmp_path: Path) -> None:
        """When Run A status is missing or corrupt, ledger records unmetered calls and blocks Run B."""
        controller = Stage6CController()
        status_file = tmp_path / "unmetered_a_status.json"
        # Simulate reconcile with missing status file
        res_a = controller._reconcile_run_status(
            run_id="run_a",
            status_file=status_file,
            exit_code=1,
            elapsed_wall=5.0,
            last_cpu_time=2.0,
            max_calls=600_000,
            max_cpu=900.0,
            max_wall=1500.0,
            shutdown_requested=False,
            shutdown_status=WorkerStatus.COMPLETED,
            termination_reason="Crash before status write",
            forced_kill_applied=False,
        )

        assert res_a["n_eval_completed"] is None
        assert res_a["telemetry_integrity"] == "MISSING"

        # Evaluate gate
        gate = controller.ledger.evaluate_run_b_gate()
        assert gate["authorized"] is False
        assert gate["gate_blocked_reason"] == "RUN_A_ACCOUNTING_INVALID_OR_MISSING"

    def test_stale_or_mismatched_telemetry_rejected_and_blocks_run_b(self, tmp_path: Path) -> None:
        """Simulate telemetry from a different run ID or expired timestamp; must be rejected and block Run B."""
        # Case 1: Telemetry with mismatched run_id
        controller1 = Stage6CController()
        status_file1 = tmp_path / "mismatched_run_status.json"
        mismatched_telemetry = {
            "run_id": "different_previous_run_id",
            "status": "COMPLETED",
            "is_complete": True,
            "n_eval_started": 50_000,
            "n_eval_completed": 50_000,
            "cpu_time_s": 100.0,
            "wall_time_s": 200.0,
            "timestamp": time.time(),
        }
        atomic_write_json(status_file1, mismatched_telemetry)

        res_mismatch = controller1._reconcile_run_status(
            run_id="current_active_run",
            status_file=status_file1,
            exit_code=0,
            elapsed_wall=5.0,
            last_cpu_time=2.0,
            max_calls=600_000,
            max_cpu=900.0,
            max_wall=1500.0,
            shutdown_requested=False,
            shutdown_status=WorkerStatus.COMPLETED,
            termination_reason="PROCESS_COMPLETED",
            forced_kill_applied=False,
        )

        assert res_mismatch["telemetry_integrity"] == "STALE_OR_INVALID"
        assert res_mismatch["authoritative_counts_available"] is False
        assert res_mismatch["is_complete"] is False
        assert "Mismatched run_id" in res_mismatch["termination_reason"]

        gate1 = controller1.ledger.evaluate_run_b_gate()
        assert gate1["authorized"] is False
        assert gate1["gate_blocked_reason"] == "RUN_A_ACCOUNTING_INVALID_OR_MISSING"

        # Case 2: Telemetry with expired timestamp (older than run launch epoch)
        controller2 = Stage6CController()
        status_file2 = tmp_path / "expired_run_status.json"
        expired_telemetry = {
            "run_id": "current_run_2",
            "status": "COMPLETED",
            "is_complete": True,
            "n_eval_started": 50_000,
            "n_eval_completed": 50_000,
            "cpu_time_s": 100.0,
            "wall_time_s": 200.0,
            "timestamp": 1_000_000.0,  # Far in the past
        }
        atomic_write_json(status_file2, expired_telemetry)

        res_expired = controller2._reconcile_run_status(
            run_id="current_run_2",
            status_file=status_file2,
            exit_code=0,
            elapsed_wall=5.0,
            last_cpu_time=2.0,
            max_calls=600_000,
            max_cpu=900.0,
            max_wall=1500.0,
            shutdown_requested=False,
            shutdown_status=WorkerStatus.COMPLETED,
            termination_reason="PROCESS_COMPLETED",
            forced_kill_applied=False,
            min_timestamp=2_000_000.0,  # Run launch epoch after file was written
        )

        assert res_expired["telemetry_integrity"] == "STALE_OR_INVALID"
        assert res_expired["authoritative_counts_available"] is False
        assert res_expired["is_complete"] is False
        assert "Expired timestamp" in res_expired["termination_reason"]

        gate2 = controller2.ledger.evaluate_run_b_gate()
        assert gate2["authorized"] is False
        assert gate2["gate_blocked_reason"] == "RUN_A_ACCOUNTING_INVALID_OR_MISSING"
