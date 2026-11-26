"""Resource control and safety infrastructure for WP6 Stage 6C execution.

This module provides budget-aware likelihood wrapping, process-tree CPU and wall-clock
watchdog monitoring, atomic checkpoint persistence, and aggregate campaign budgeting
for the WP6 Stage 6C two-run anchor experiment.

Key Components:
---------------
1. `LikelihoodBudgetGuard`: Thread-safe, in-path likelihood evaluation wrapper maintaining
   authoritative invocation counters (attempted, started, completed) and enforcing a hard cap
   BEFORE entering the underlying evaluation function.
2. `Stage6CConfig`: Validated configuration dataclass preserving Stage 6C resource boundaries.
3. `AggregateResourceLedger`: Controller-side resource ledger tracking cumulative usage and
   enforcing the three-resource pre-dispatch gate rule for Run B.
4. `WorkerStatus`: Enumeration of distinct termination and execution statuses.
5. `Stage6CController`: Single-worker process manager with active watchdog monitoring,
   graceful shutdown signaling, fallback process-tree termination, and reconciliation.
"""

from __future__ import annotations

import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil


class WorkerStatus(str, Enum):
    """Execution and termination statuses for Stage 6C worker runs."""

    COMPLETED = "COMPLETED"
    TERMINATED_BUDGET_CALLS = "TERMINATED_BUDGET_CALLS"
    TERMINATED_BUDGET_CPU = "TERMINATED_BUDGET_CPU"
    TERMINATED_BUDGET_WALL = "TERMINATED_BUDGET_WALL"
    FAILED_IMPLEMENTATION = "FAILED_IMPLEMENTATION"
    FAILED_RUNTIME = "FAILED_RUNTIME"
    CANCELLED_BY_CONTROLLER = "CANCELLED_BY_CONTROLLER"


class ResourceControlError(RuntimeError):
    """Base exception for resource control operations."""


class BudgetExceededError(ResourceControlError):
    """Base exception when an operational resource budget is crossed."""


class LikelihoodBudgetExceededError(BudgetExceededError):
    """Raised inside the likelihood evaluation path when the call cap is reached."""


class CpuBudgetExceededError(BudgetExceededError):
    """Raised when process-tree CPU consumption exceeds the allocated cap."""


class WallBudgetExceededError(BudgetExceededError):
    """Raised when wall-clock time exceeds the allocated timeout."""


class LikelihoodBudgetGuard:
    """Thread-safe, budget-aware in-path likelihood evaluation wrapper.

    Maintains authoritative invocation counts strictly inside the worker process:
    - n_attempted: total invocations of `__call__`
    - n_started: evaluations that passed the budget check and entered target_fn
    - n_completed: evaluations that completed successfully and returned a result
    - n_failed: evaluations where target_fn raised an unhandled exception

    Safety Invariant:
    -----------------
    n_completed <= n_started <= call_cap

    The budget check is strictly performed BEFORE entering `target_fn`.
    When `n_started >= call_cap`, an attempt to evaluate immediately raises
    `LikelihoodBudgetExceededError` without invoking `target_fn`.
    """

    def __init__(self, target_fn: Callable[..., Any], call_cap: int) -> None:
        if not callable(target_fn):
            raise TypeError(f"target_fn must be callable, got {type(target_fn)}")
        if not isinstance(call_cap, (int, np.integer)) or call_cap < 0:
            raise ValueError(f"call_cap must be a non-negative integer, got {call_cap}")

        self._fn = target_fn
        self._call_cap = int(call_cap)
        self._lock = threading.Lock()
        self._n_attempted = 0
        self._n_started = 0
        self._n_completed = 0
        self._n_failed = 0

    @property
    def call_cap(self) -> int:
        """Configured maximum number of started evaluations allowed."""
        return self._call_cap

    @property
    def n_attempted(self) -> int:
        """Total number of evaluation attempts made on the wrapper."""
        with self._lock:
            return self._n_attempted

    @property
    def n_started(self) -> int:
        """Total number of evaluations that cleared the budget gate."""
        with self._lock:
            return self._n_started

    @property
    def n_completed(self) -> int:
        """Total number of evaluations that returned successfully."""
        with self._lock:
            return self._n_completed

    @property
    def n_failed(self) -> int:
        """Total number of evaluations where target_fn raised an exception."""
        with self._lock:
            return self._n_failed

    def get_counts(self) -> Dict[str, int]:
        """Return a snapshot of all authoritative invocation counters."""
        with self._lock:
            return {
                "call_cap": self._call_cap,
                "attempted": self._n_attempted,
                "started": self._n_started,
                "completed": self._n_completed,
                "failed": self._n_failed,
            }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._n_attempted += 1
            if self._n_started >= self._call_cap:
                raise LikelihoodBudgetExceededError(
                    f"Likelihood call budget reached: started={self._n_started}, cap={self._call_cap}"
                )
            self._n_started += 1

        # Evaluate target function outside lock to prevent serializing multi-threaded callers
        try:
            result = self._fn(*args, **kwargs)
        except Exception:
            with self._lock:
                self._n_failed += 1
            raise
        else:
            with self._lock:
                self._n_completed += 1
            return result


@dataclass(frozen=True)
class Stage6CConfig:
    """Validated configuration parameters for Stage 6C resource boundaries.

    All limits preserve the approved Stage 6C preflight specifications.
    """

    max_calls_per_run: int = 600_000
    max_cpu_per_run_s: float = 900.0
    max_wall_per_run_s: float = 1_500.0
    max_aggregate_calls: int = 1_200_000
    max_aggregate_cpu_s: float = 1_800.0
    max_aggregate_wall_s: float = 3_000.0
    required_calls_run_b: int = 530_000
    required_cpu_run_b_s: float = 795.0
    required_wall_run_b_s: float = 1_300.0
    watchdog_interval_s: float = 1.0
    shutdown_grace_period_s: float = 5.0

    def __post_init__(self) -> None:
        """Enforce strict validation on all resource boundary parameters."""
        for name, val in [
            ("max_calls_per_run", self.max_calls_per_run),
            ("max_aggregate_calls", self.max_aggregate_calls),
            ("required_calls_run_b", self.required_calls_run_b),
        ]:
            if not isinstance(val, (int, np.integer)) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")

        for name, val in [
            ("max_cpu_per_run_s", self.max_cpu_per_run_s),
            ("max_wall_per_run_s", self.max_wall_per_run_s),
            ("max_aggregate_cpu_s", self.max_aggregate_cpu_s),
            ("max_aggregate_wall_s", self.max_aggregate_wall_s),
            ("required_cpu_run_b_s", self.required_cpu_run_b_s),
            ("required_wall_run_b_s", self.required_wall_run_b_s),
            ("watchdog_interval_s", self.watchdog_interval_s),
            ("shutdown_grace_period_s", self.shutdown_grace_period_s),
        ]:
            if (
                not isinstance(val, (int, float, np.floating))
                or not math.isfinite(val)
                or val <= 0
            ):
                raise ValueError(f"{name} must be a positive finite float, got {val}")

        if self.max_calls_per_run > self.max_aggregate_calls:
            raise ValueError(
                f"max_calls_per_run ({self.max_calls_per_run}) cannot exceed max_aggregate_calls ({self.max_aggregate_calls})"
            )
        if self.max_cpu_per_run_s > self.max_aggregate_cpu_s:
            raise ValueError(
                f"max_cpu_per_run_s ({self.max_cpu_per_run_s}) cannot exceed max_aggregate_cpu_s ({self.max_aggregate_cpu_s})"
            )
        if self.max_wall_per_run_s > self.max_aggregate_wall_s:
            raise ValueError(
                f"max_wall_per_run_s ({self.max_wall_per_run_s}) cannot exceed max_aggregate_wall_s ({self.max_aggregate_wall_s})"
            )
        if self.required_calls_run_b > self.max_calls_per_run:
            raise ValueError(
                f"required_calls_run_b ({self.required_calls_run_b}) cannot exceed max_calls_per_run ({self.max_calls_per_run})"
            )
        if self.required_cpu_run_b_s > self.max_cpu_per_run_s:
            raise ValueError(
                f"required_cpu_run_b_s ({self.required_cpu_run_b_s}) cannot exceed max_cpu_per_run_s ({self.max_cpu_per_run_s})"
            )
        if self.required_wall_run_b_s > self.max_wall_per_run_s:
            raise ValueError(
                f"required_wall_run_b_s ({self.required_wall_run_b_s}) cannot exceed max_wall_per_run_s ({self.max_wall_per_run_s})"
            )


class AggregateResourceLedger:
    """Controller-side aggregate ledger tracking cumulative usage and Run B dispatch feasibility."""

    def __init__(self, config: Optional[Stage6CConfig] = None) -> None:
        self.config = config if config is not None else Stage6CConfig()
        self.runs: List[Dict[str, Any]] = []
        self.cumulative_calls: int = 0
        self.cumulative_cpu_s: float = 0.0
        self.cumulative_wall_s: float = 0.0

    @property
    def remaining_calls(self) -> int:
        """Remaining likelihood evaluations under the aggregate campaign ceiling."""
        return max(0, self.config.max_aggregate_calls - self.cumulative_calls)

    @property
    def remaining_cpu_s(self) -> float:
        """Remaining process CPU seconds under the aggregate campaign ceiling."""
        return max(0.0, self.config.max_aggregate_cpu_s - self.cumulative_cpu_s)

    @property
    def remaining_wall_s(self) -> float:
        """Remaining wall-clock seconds under the aggregate campaign timeout."""
        return max(0.0, self.config.max_aggregate_wall_s - self.cumulative_wall_s)

    def record_run(
        self,
        run_id: str,
        calls: Optional[int],
        cpu_s: float,
        wall_s: float,
        status: Union[str, WorkerStatus],
        is_complete: Optional[bool] = None,
        has_valid_accounting: bool = True,
    ) -> None:
        """Record completed or terminated run resource consumption."""
        status_str = status.value if isinstance(status, WorkerStatus) else str(status)
        cpu_flt = float(cpu_s)
        wall_flt = float(wall_s)

        if is_complete is None:
            is_comp = bool(status_str == WorkerStatus.COMPLETED.value)
        else:
            is_comp = bool(is_complete)

        # Invariant: If status is not COMPLETED, is_complete MUST be False
        if status_str != WorkerStatus.COMPLETED.value:
            is_comp = False

        valid_calls = True
        if calls is None or not isinstance(calls, (int, np.integer)) or calls < 0 or not has_valid_accounting:
            valid_calls = False
            calls_val: Optional[int] = None
        else:
            calls_val = int(calls)
            self.cumulative_calls += calls_val

        self.cumulative_cpu_s += cpu_flt
        self.cumulative_wall_s += wall_flt

        self.runs.append(
            {
                "run_id": str(run_id),
                "calls": calls_val,
                "cpu_s": cpu_flt,
                "wall_s": wall_flt,
                "status": status_str,
                "is_complete": is_comp,
                "has_valid_accounting": valid_calls,
                "recorded_at": time.time(),
            }
        )

    def evaluate_run_b_gate(self) -> Dict[str, Any]:
        """Evaluate the pre-dispatch gate rule for Run B.

        Safety Verifications:
        ---------------------
        1. Run A has been executed and recorded.
        2. Run A status is exactly COMPLETED.
        3. Run A has is_complete == True.
        4. Run A has valid, authoritative resource accounting.
        5. Negative remaining resources strictly block dispatch.
        6. Remaining aggregate and per-run call, CPU, and wall budgets satisfy preflight limits.
        7. Confirms that resource gate feasibility does NOT imply scientific acceptance of Run A.

        Returns
        -------
        Dict[str, Any]
            Structured evaluation record including `authorized` boolean and individual checks.
        """
        # Verification 1: Has Run A been recorded?
        if not self.runs:
            return {
                "authorized": False,
                "gate_blocked_reason": "NO_BASELINE_RUN_RECORDED",
                "error": "Run B dispatch strictly BLOCKED: No baseline run (Run A) has been recorded in the ledger.",
                "pass_calls": False,
                "pass_cpu": False,
                "pass_wall": False,
                "cap_calls": 0,
                "cap_cpu_s": 0.0,
                "cap_wall_s": 0.0,
                "required_calls": self.config.required_calls_run_b,
                "required_cpu_s": self.config.required_cpu_run_b_s,
                "required_wall_s": self.config.required_wall_run_b_s,
                "remaining_aggregate_calls": self.remaining_calls,
                "remaining_aggregate_cpu_s": self.remaining_cpu_s,
                "remaining_aggregate_wall_s": self.remaining_wall_s,
                "scientific_acceptance_disclaimer": (
                    "The resource gate assesses operational budget feasibility only; "
                    "it does not constitute scientific acceptance or convergence validation of Run A."
                ),
            }

        run_a = self.runs[0]
        status_val = str(run_a.get("status"))
        is_complete_val = bool(run_a.get("is_complete", False))
        has_valid_accounting = bool(run_a.get("has_valid_accounting", False)) and run_a.get("calls") is not None

        # Verification 2: Is Run A accounting authoritative and valid?
        if not has_valid_accounting:
            return {
                "authorized": False,
                "gate_blocked_reason": "RUN_A_ACCOUNTING_INVALID_OR_MISSING",
                "error": "Run B dispatch strictly BLOCKED: Run A accounting is missing, corrupted, or unmetered.",
                "pass_calls": False,
                "pass_cpu": False,
                "pass_wall": False,
                "cap_calls": 0,
                "cap_cpu_s": 0.0,
                "cap_wall_s": 0.0,
                "required_calls": self.config.required_calls_run_b,
                "required_cpu_s": self.config.required_cpu_run_b_s,
                "required_wall_s": self.config.required_wall_run_b_s,
                "remaining_aggregate_calls": self.remaining_calls,
                "remaining_aggregate_cpu_s": self.remaining_cpu_s,
                "remaining_aggregate_wall_s": self.remaining_wall_s,
                "run_a_status": status_val,
                "run_a_complete": is_complete_val,
                "scientific_acceptance_disclaimer": (
                    "The resource gate assesses operational budget feasibility only; "
                    "it does not constitute scientific acceptance or convergence validation of Run A."
                ),
            }

        # Verification 3: Is Run A status exactly COMPLETED?
        if status_val != WorkerStatus.COMPLETED.value:
            if status_val == WorkerStatus.TERMINATED_BUDGET_CALLS.value:
                blocked_reason = "RUN_A_TERMINATED_BUDGET_CALLS"
            elif status_val == WorkerStatus.TERMINATED_BUDGET_CPU.value:
                blocked_reason = "RUN_A_TERMINATED_BUDGET_CPU"
            elif status_val == WorkerStatus.TERMINATED_BUDGET_WALL.value:
                blocked_reason = "RUN_A_TERMINATED_BUDGET_WALL"
            elif status_val == WorkerStatus.FAILED_RUNTIME.value:
                blocked_reason = "RUN_A_CRASHED_OR_FAILED_RUNTIME"
            elif status_val == WorkerStatus.CANCELLED_BY_CONTROLLER.value:
                blocked_reason = "RUN_A_CANCELLED"
            else:
                blocked_reason = f"RUN_A_INCOMPLETE_STATUS_{status_val}"

            return {
                "authorized": False,
                "gate_blocked_reason": blocked_reason,
                "error": f"Run B dispatch strictly BLOCKED: Run A status is '{status_val}', not '{WorkerStatus.COMPLETED.value}'.",
                "pass_calls": False,
                "pass_cpu": False,
                "pass_wall": False,
                "cap_calls": 0,
                "cap_cpu_s": 0.0,
                "cap_wall_s": 0.0,
                "required_calls": self.config.required_calls_run_b,
                "required_cpu_s": self.config.required_cpu_run_b_s,
                "required_wall_s": self.config.required_wall_run_b_s,
                "remaining_aggregate_calls": self.remaining_calls,
                "remaining_aggregate_cpu_s": self.remaining_cpu_s,
                "remaining_aggregate_wall_s": self.remaining_wall_s,
                "run_a_status": status_val,
                "run_a_complete": is_complete_val,
                "scientific_acceptance_disclaimer": (
                    "The resource gate assesses operational budget feasibility only; "
                    "it does not constitute scientific acceptance or convergence validation of Run A."
                ),
            }

        # Verification 4: Is Run A marked is_complete == True?
        if not is_complete_val:
            return {
                "authorized": False,
                "gate_blocked_reason": "RUN_A_IS_COMPLETE_FALSE",
                "error": "Run B dispatch strictly BLOCKED: Run A is_complete flag is False despite COMPLETED status.",
                "pass_calls": False,
                "pass_cpu": False,
                "pass_wall": False,
                "cap_calls": 0,
                "cap_cpu_s": 0.0,
                "cap_wall_s": 0.0,
                "required_calls": self.config.required_calls_run_b,
                "required_cpu_s": self.config.required_cpu_run_b_s,
                "required_wall_s": self.config.required_wall_run_b_s,
                "remaining_aggregate_calls": self.remaining_calls,
                "remaining_aggregate_cpu_s": self.remaining_cpu_s,
                "remaining_aggregate_wall_s": self.remaining_wall_s,
                "run_a_status": status_val,
                "run_a_complete": False,
                "scientific_acceptance_disclaimer": (
                    "The resource gate assesses operational budget feasibility only; "
                    "it does not constitute scientific acceptance or convergence validation of Run A."
                ),
            }

        # Verification 5: Evaluate raw remaining resources (negative remaining cannot authorize)
        raw_remaining_calls = self.config.max_aggregate_calls - self.cumulative_calls
        raw_remaining_cpu = self.config.max_aggregate_cpu_s - self.cumulative_cpu_s
        raw_remaining_wall = self.config.max_aggregate_wall_s - self.cumulative_wall_s

        cap_calls = min(self.config.max_calls_per_run, max(0, raw_remaining_calls))
        cap_cpu = min(self.config.max_cpu_per_run_s, max(0.0, raw_remaining_cpu))
        cap_wall = min(self.config.max_wall_per_run_s, max(0.0, raw_remaining_wall))

        pass_calls = bool(raw_remaining_calls >= self.config.required_calls_run_b and cap_calls >= self.config.required_calls_run_b)
        pass_cpu = bool(raw_remaining_cpu >= self.config.required_cpu_run_b_s and cap_cpu >= self.config.required_cpu_run_b_s)
        pass_wall = bool(raw_remaining_wall >= self.config.required_wall_run_b_s and cap_wall >= self.config.required_wall_run_b_s)

        authorized = bool(pass_calls and pass_cpu and pass_wall)

        reasons = []
        if not pass_calls:
            reasons.append(f"Insufficient calls: available={cap_calls} < required={self.config.required_calls_run_b}")
        if not pass_cpu:
            reasons.append(f"Insufficient CPU: available={cap_cpu:.1f} s < required={self.config.required_cpu_run_b_s:.1f} s")
        if not pass_wall:
            reasons.append(f"Insufficient wall: available={cap_wall:.1f} s < required={self.config.required_wall_run_b_s:.1f} s")

        return {
            "authorized": authorized,
            "gate_blocked_reason": "; ".join(reasons) if reasons else "NONE",
            "cap_calls": cap_calls,
            "cap_cpu_s": cap_cpu,
            "cap_wall_s": cap_wall,
            "required_calls": self.config.required_calls_run_b,
            "required_cpu_s": self.config.required_cpu_run_b_s,
            "required_wall_s": self.config.required_wall_run_b_s,
            "pass_calls": pass_calls,
            "pass_cpu": pass_cpu,
            "pass_wall": pass_wall,
            "remaining_aggregate_calls": self.remaining_calls,
            "remaining_aggregate_cpu_s": self.remaining_cpu_s,
            "remaining_aggregate_wall_s": self.remaining_wall_s,
            "run_a_status": run_a.get("status"),
            "run_a_complete": run_a.get("is_complete", False),
            "scientific_acceptance_disclaimer": (
                "The resource gate assesses operational budget feasibility only; "
                "it does not constitute scientific acceptance or convergence validation of Run A."
            ),
        }


def atomic_write_json(file_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Atomically write JSON data by writing to a unique temporary file then renaming.

    Ensures that process abortion or interruption never leaves a corrupted, half-written file.
    If serialization or write fails, the temporary file is safely unlinked and the original
    target file remains completely untouched.
    """
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.is_file():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise


def get_process_tree_cpu(pid: int) -> float:
    """Measure cumulative process-tree CPU time (user + system) using psutil.

    Technical Limitations:
    ----------------------
    1. psutil samples processes at discrete query time. Short-lived descendants that
       spawn and terminate between watchdog polling ticks cannot be detected.
    2. Returns 0.0 if the process has already exited or cannot be accessed.
    """
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0

    total_cpu = 0.0
    try:
        times = parent.cpu_times()
        total_cpu += times.user + times.system
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    try:
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []

    for child in children:
        try:
            times = child.cpu_times()
            total_cpu += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return total_cpu


def force_kill_process_tree(pid: int) -> None:
    """Forcefully terminate a process and all its descendants.

    On Windows, invokes `taskkill /F /T /PID <pid>` as an external fallback mechanism,
    followed by psutil process-tree kill.

    Important Qualification:
    ------------------------
    `taskkill /F /T` is an external fallback termination mechanism to halt uncooperative
    processes; it is NOT a hard kernel resource quota.
    """
    if platform.system() == "Windows":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
            )
        except Exception:
            pass

    # Ensure termination across all platforms using psutil
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        parent.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


class Stage6CController:
    """Parent controller running a single worker process with active watchdog supervision.

    Enforces:
    - Single concurrency (max_workers=1).
    - Monitored soft CPU budget (with discrete polling lag).
    - Monitored wall-clock timeout.
    - Graceful shutdown signaling (via signal/abort flag) with shutdown grace period.
    - Fallback forced process-tree termination.
    - Lossless status reconciliation and artifact persistence.
    """

    def __init__(
        self,
        config: Optional[Stage6CConfig] = None,
        ledger: Optional[AggregateResourceLedger] = None,
    ) -> None:
        self.config = config if config is not None else Stage6CConfig()
        self.ledger = ledger if ledger is not None else AggregateResourceLedger(self.config)
        self._active_worker: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def run_worker_process(
        self,
        command: List[str],
        run_id: str,
        output_dir: Union[str, Path],
        extra_env: Optional[Dict[str, str]] = None,
        custom_max_calls: Optional[int] = None,
        custom_max_cpu_s: Optional[float] = None,
        custom_max_wall_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Launch, monitor, and supervise a single child worker process.

        Parameters
        ----------
        command : List[str]
            Subprocess command arguments (e.g. [sys.executable, script_path, ...]).
        run_id : str
            Unique identifier for the worker run.
        output_dir : Union[str, Path]
            Dedicated directory where worker outputs, telemetry, and status are stored.
        extra_env : Optional[Dict[str, str]]
            Environment variables to inject (e.g. single-threaded library settings).
        custom_max_calls : Optional[int]
            Per-run likelihood call cap (defaults to config.max_calls_per_run).
        custom_max_cpu_s : Optional[float]
            Per-run process CPU limit (defaults to config.max_cpu_per_run_s).
        custom_max_wall_s : Optional[float]
            Per-run wall-clock timeout (defaults to config.max_wall_per_run_s).

        Returns
        -------
        Dict[str, Any]
            Reconciled run execution status record.
        """
        with self._lock:
            if self._active_worker is not None and self._active_worker.poll() is None:
                raise RuntimeError("A worker process is already running. Concurrency cap is 1.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        status_file = out_path / f"{run_id}_status.json"
        abort_file = out_path / f"{run_id}.abort_signal"
        if abort_file.is_file():
            abort_file.unlink(missing_ok=True)

        max_calls = custom_max_calls if custom_max_calls is not None else self.config.max_calls_per_run
        max_cpu = custom_max_cpu_s if custom_max_cpu_s is not None else self.config.max_cpu_per_run_s
        max_wall = custom_max_wall_s if custom_max_wall_s is not None else self.config.max_wall_per_run_s

        # Prepare single-threaded execution environment
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        if extra_env:
            env.update(extra_env)

        # Platform-specific process group creation
        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        t0_wall = time.perf_counter()
        t0_epoch = time.time()
        proc = subprocess.Popen(
            command,
            env=env,
            creationflags=creationflags,
            cwd=str(out_path),
        )

        with self._lock:
            self._active_worker = proc

        watchdog_interval = self.config.watchdog_interval_s
        grace_period = self.config.shutdown_grace_period_s

        termination_reason = "PROCESS_COMPLETED"
        forced_kill_applied = False
        shutdown_requested = False
        shutdown_status = WorkerStatus.COMPLETED

        # Active Watchdog Monitoring Loop
        last_cpu_time = 0.0
        try:
            while True:
                # Check if process exited on its own
                retcode = proc.poll()
                if retcode is not None:
                    break

                elapsed_wall = time.perf_counter() - t0_wall
                current_cpu = get_process_tree_cpu(proc.pid)
                if current_cpu > 0.0:
                    last_cpu_time = max(last_cpu_time, current_cpu)

                # Check CPU Budget Limit
                if current_cpu >= max_cpu and not shutdown_requested:
                    shutdown_requested = True
                    shutdown_status = WorkerStatus.TERMINATED_BUDGET_CPU
                    termination_reason = (
                        f"Watchdog detected process-tree CPU ({current_cpu:.2f} s) >= limit ({max_cpu:.2f} s)"
                    )
                    self._signal_graceful_shutdown(abort_file)

                # Check Wall-Clock Timeout
                elif elapsed_wall >= max_wall and not shutdown_requested:
                    shutdown_requested = True
                    shutdown_status = WorkerStatus.TERMINATED_BUDGET_WALL
                    termination_reason = (
                        f"Watchdog detected wall-clock time ({elapsed_wall:.2f} s) >= timeout ({max_wall:.2f} s)"
                    )
                    self._signal_graceful_shutdown(abort_file)

                # If shutdown was requested, monitor grace period
                if shutdown_requested:
                    try:
                        proc.wait(timeout=grace_period)
                        break
                    except subprocess.TimeoutExpired:
                        # Grace period expired: worker failed to exit gracefully, apply forced termination
                        forced_kill_applied = True
                        force_kill_process_tree(proc.pid)
                        try:
                            proc.wait(timeout=3.0)
                        except subprocess.TimeoutExpired:
                            pass
                        break

                time.sleep(watchdog_interval)

        finally:
            with self._lock:
                self._active_worker = None
            if proc.poll() is None:
                force_kill_process_tree(proc.pid)

        elapsed_wall = time.perf_counter() - t0_wall

        # Clean up abort signal file if present
        if abort_file.is_file():
            abort_file.unlink(missing_ok=True)

        # Reconcile status and metadata
        return self._reconcile_run_status(
            run_id=run_id,
            status_file=status_file,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            elapsed_wall=elapsed_wall,
            last_cpu_time=last_cpu_time,
            max_calls=max_calls,
            max_cpu=max_cpu,
            max_wall=max_wall,
            shutdown_requested=shutdown_requested,
            shutdown_status=shutdown_status,
            termination_reason=termination_reason,
            forced_kill_applied=forced_kill_applied,
            min_timestamp=t0_epoch,
        )

    def _signal_graceful_shutdown(self, abort_file: Path) -> None:
        """Signal child process for graceful termination via abort signal file.

        Cooperative workers poll this abort flag at each iteration, flush their
        checkpoints and dead points, and exit cleanly before grace period expiry.
        """
        try:
            abort_file.write_text(f"ABORT_REQUESTED at {time.time()}\n", encoding="utf-8")
        except Exception:
            pass

    def _reconcile_run_status(
        self,
        run_id: str,
        status_file: Path,
        exit_code: int,
        elapsed_wall: float,
        last_cpu_time: float,
        max_calls: int,
        max_cpu: float,
        max_wall: float,
        shutdown_requested: bool,
        shutdown_status: WorkerStatus,
        termination_reason: str,
        forced_kill_applied: bool,
        min_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Reconcile worker's on-disk status with parent watchdog measurements."""
        worker_data: Optional[Dict[str, Any]] = None
        status_file_corrupt = False
        if status_file.is_file():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        worker_data = data
                    else:
                        status_file_corrupt = True
            except Exception:
                worker_data = None
                status_file_corrupt = True

        # Check for stale or mismatched telemetry
        is_stale_or_mismatched = False
        stale_reason = ""
        if worker_data is not None:
            # Check 1: run_id mismatch
            worker_run_id = worker_data.get("run_id")
            if worker_run_id is not None and str(worker_run_id) != str(run_id):
                is_stale_or_mismatched = True
                stale_reason = f"Mismatched run_id: expected '{run_id}', found '{worker_run_id}'"

            # Check 2: expired/stale timestamp
            telemetry_ts = worker_data.get("timestamp", worker_data.get("recorded_at", worker_data.get("created_at")))
            if telemetry_ts is not None and min_timestamp is not None:
                try:
                    if float(telemetry_ts) < float(min_timestamp):
                        is_stale_or_mismatched = True
                        stale_reason = f"Expired timestamp: {telemetry_ts} < run launch {min_timestamp}"
                except (ValueError, TypeError):
                    is_stale_or_mismatched = True
                    stale_reason = "Malformed telemetry timestamp"

            # Check 3: explicit stale flag
            if worker_data.get("is_stale", False):
                is_stale_or_mismatched = True
                stale_reason = "Telemetry explicitly flagged as stale"

        if worker_data is not None and not is_stale_or_mismatched:
            # Worker wrote an authoritative status record
            status_val = worker_data.get("status", WorkerStatus.FAILED_RUNTIME.value)
            calls_started = worker_data.get("n_eval_started", 0)
            calls_completed = worker_data.get("n_eval_completed", calls_started)
            is_complete = bool(worker_data.get("is_complete", False))

            # Safety enforcement:
            # 1. If shutdown was requested or forced kill applied, watchdog initiated termination;
            # the run cannot be complete, and status must reflect the watchdog termination reason.
            if shutdown_requested or forced_kill_applied:
                is_complete = False
                status_val = shutdown_status.value
            elif exit_code != 0:
                # 2. Non-zero exit code without watchdog intervention means worker failed or crashed
                is_complete = False
                if status_val == WorkerStatus.COMPLETED.value:
                    status_val = WorkerStatus.FAILED_RUNTIME.value

            # 3. Invariant: If status is anything other than COMPLETED, is_complete MUST be False
            if status_val != WorkerStatus.COMPLETED.value:
                is_complete = False

            reconciled = {
                "run_id": run_id,
                "status": status_val,
                "is_complete": is_complete,
                "exit_code": exit_code,
                "n_eval_attempted": worker_data.get("n_eval_attempted", calls_started),
                "n_eval_started": calls_started,
                "n_eval_completed": calls_completed,
                "cpu_time_s": max(float(worker_data.get("cpu_time_s", 0.0)), float(last_cpu_time)),
                "wall_time_s": max(float(worker_data.get("wall_time_s", 0.0)), float(elapsed_wall)),
                "max_calls": max_calls,
                "max_cpu_s": max_cpu,
                "max_wall_s": max_wall,
                "termination_reason": worker_data.get("termination_reason", termination_reason),
                "forced_kill_applied": forced_kill_applied,
                "telemetry_integrity": "VALID",
                "authoritative_counts_available": True,
                "reconciled_by": "WORKER_AUTH_AND_PARENT",
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "python_version": sys.version,
                },
            }
            has_valid_accounting = True
            calls_for_ledger = calls_completed
        else:
            # Worker crashed, status was missing, corrupt, or stale/mismatched
            final_status = (
                shutdown_status.value
                if shutdown_requested
                else WorkerStatus.FAILED_RUNTIME.value
            )
            if is_stale_or_mismatched:
                integrity = "STALE_OR_INVALID"
                term_reason = f"Telemetry rejected as stale/invalid: {stale_reason}"
            elif status_file_corrupt:
                integrity = "CORRUPTED"
                term_reason = "Malformed or unreadable worker status file"
            elif shutdown_requested:
                integrity = "MISSING"
                term_reason = termination_reason
            else:
                integrity = "MISSING"
                term_reason = f"Worker process crashed or exited abnormally with code {exit_code} before writing status"

            reconciled = {
                "run_id": run_id,
                "status": final_status,
                "is_complete": False,  # Strict guarantee: NEVER complete when worker status is missing, corrupt, or stale
                "exit_code": exit_code,
                "n_eval_attempted": None,
                "n_eval_started": None,
                "n_eval_completed": None,
                "cpu_time_s": last_cpu_time,
                "wall_time_s": elapsed_wall,
                "max_calls": max_calls,
                "max_cpu_s": max_cpu,
                "max_wall_s": max_wall,
                "termination_reason": term_reason,
                "forced_kill_applied": forced_kill_applied,
                "telemetry_integrity": integrity,
                "authoritative_counts_available": False,
                "reconciled_by": "PARENT_WATCHDOG_ONLY",
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "python_version": sys.version,
                },
            }
            has_valid_accounting = False
            calls_for_ledger = None

        # Atomically persist reconciled status file
        atomic_write_json(status_file, reconciled)

        # Record in ledger
        self.ledger.record_run(
            run_id=run_id,
            calls=calls_for_ledger,
            cpu_s=reconciled["cpu_time_s"],
            wall_s=reconciled["wall_time_s"],
            status=reconciled["status"],
            is_complete=reconciled["is_complete"],
            has_valid_accounting=has_valid_accounting,
        )

        return reconciled
