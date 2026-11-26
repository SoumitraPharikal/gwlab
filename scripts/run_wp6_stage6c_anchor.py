#!/usr/bin/env python3
"""WP6 Stage 6C: Resource-Controlled Runner Script.

This script implements the standalone entry points for the Stage 6C two-run anchor
comparison and its isolated synthetic verification harnesses.

Strict Execution Safety Policy:
-------------------------------
Stage 6C scientific sampler runs (Run A and Run B) remain STRICTLY UNAUTHORIZED.
Any attempt to launch real scientific inference without explicit authorization flags
raises an immediate fatal RuntimeError. Only synthetic test modes are authorized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure workspace paths are on sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(WORKSPACE_DIR / "src"))

from gw_inference.resource_control import (
    AggregateResourceLedger,
    LikelihoodBudgetExceededError,
    LikelihoodBudgetGuard,
    Stage6CConfig,
    Stage6CController,
    WorkerStatus,
    atomic_write_json,
)

# Frozen Run A scientific configuration specification
RUN_A_FROZEN_CONFIG = {
    "target": "run_a",
    "coordinates": "(Mc, eta, tc, phi_c)",
    "prior_transform": "prior_transform_4d",
    "realization": 1,
    "model": "plug",
    "seed_off": 7101,
    "seed_on": 7151,
    "sampler_seed": 8101,
    "t_off": 64.0,
    "t_on": 4.0,
    "fs": 2048.0,
    "tukey_alpha": 0.25,
    "f_start": 20.0,
    "f_end": 512.0,
    "k_start": 80,
    "k_end": 2048,
    "nlive": 500,
    "bound": "multi",
    "sample": "rwalk",
    "walks": 50,
    "periodic": [3],
    "dlogz": 0.10,
    "call_cap": 600_000,
    "cpu_budget_s": 900.0,
    "wall_budget_s": 1500.0,
    "canonical_mc": 28.095555795460438,
    "canonical_eta": 0.24710059171597634,
    "canonical_tc": 2.0,
    "canonical_phic": 0.0,
    "canonical_dist_mpc": 410.0,
}


def run_synthetic_worker(
    worker_type: str,
    run_id: str,
    output_dir: Path,
    call_cap: int,
    target_iterations: int = 100,
    param_val: float = 0.0,
) -> None:
    """Execute an isolated synthetic worker without invoking production waveforms or likelihoods.

    Parameters
    ----------
    worker_type : str
        Type of synthetic behavior:
        - 'call_bound': Performs simple arithmetic evaluations up to call_cap or target.
        - 'cpu_intensive': Runs a dense CPU mathematical loop to test CPU watchdog.
        - 'sleep': Sleeps to test wall-clock timeout watchdog.
        - 'uncooperative': Catches termination signals and refuses to exit to test forced kill.
        - 'crash': Raises an unhandled exception to test crash handling.
        - 'failing_call': Evaluates a callable that fails on a specific evaluation index.
    run_id : str
        Unique run identifier.
    output_dir : Path
        Directory for status and telemetry output.
    call_cap : int
        Authoritative in-path call cap.
    target_iterations : int
        Desired number of iterations.
    param_val : float
        Auxiliary parameter (e.g. sleep duration, fail index).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    status_file = output_dir / f"{run_id}_status.json"
    telem_file = output_dir / f"{run_id}_telemetry.json"
    abort_file = output_dir / f"{run_id}.abort_signal"

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    # Define synthetic evaluation callable
    external_call_counter = 0

    def dummy_likelihood(x: float) -> float:
        nonlocal external_call_counter
        external_call_counter += 1
        if worker_type == "failing_call" and external_call_counter == int(param_val):
            raise ValueError(f"Synthetic underlying likelihood failure at evaluation {external_call_counter}")
        return math.cos(x) * math.exp(-0.1 * abs(x))

    guard = LikelihoodBudgetGuard(dummy_likelihood, call_cap=call_cap)
    status = WorkerStatus.COMPLETED
    is_complete = False
    termination_reason = "Completed all target iterations"
    telemetry_records = []

    # Handle OS signals for graceful shutdown on Windows and POSIX
    shutdown_requested = False

    def handle_signal(sig: int, frame: Any) -> None:
        nonlocal shutdown_requested
        if worker_type == "uncooperative":
            # Intentionally ignore signal to test fallback forced termination
            return
        shutdown_requested = True

    try:
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_signal)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, handle_signal)
    except Exception:
        pass

    try:
        if worker_type == "crash":
            raise RuntimeError("Synthetic simulated worker crash before evaluation")

        if worker_type == "cpu_intensive":
            # Tight CPU arithmetic loop designed to consume CPU rapidly
            val = 1.0001
            for it in range(target_iterations):
                if shutdown_requested or abort_file.is_file():
                    status = WorkerStatus.TERMINATED_BUDGET_CPU
                    termination_reason = "Shutdown signaled by watchdog during CPU workload"
                    break
                for _ in range(500_000):
                    val = math.sin(val) + math.cos(val)
                guard(val)
            else:
                is_complete = True

        elif worker_type == "sleep":
            # Cooperative sleeping worker that checks abort flag
            sleep_duration = param_val if param_val > 0 else 5.0
            end_time = time.time() + sleep_duration
            while time.time() < end_time:
                if shutdown_requested or abort_file.is_file():
                    status = WorkerStatus.TERMINATED_BUDGET_WALL
                    termination_reason = "Shutdown signaled by watchdog during sleep workload"
                    break
                time.sleep(0.05)
            else:
                is_complete = True

        elif worker_type == "uncooperative":
            # Ignores abort signals and file flags, forcing watchdog to invoke fallback forced kill
            while True:
                time.sleep(0.2)

        else:
            # Standard or failing call evaluations
            for it in range(target_iterations):
                if shutdown_requested or abort_file.is_file():
                    status = WorkerStatus.CANCELLED_BY_CONTROLLER
                    termination_reason = "Shutdown signaled by controller"
                    break

                try:
                    res = guard(float(it))
                except LikelihoodBudgetExceededError as e:
                    status = WorkerStatus.TERMINATED_BUDGET_CALLS
                    termination_reason = str(e)
                    break

                if it % 10 == 0:
                    rec = {
                        "iteration": it,
                        "n_eval_started": guard.n_started,
                        "n_eval_completed": guard.n_completed,
                        "cpu_s": time.process_time() - t0_cpu,
                        "wall_s": time.perf_counter() - t0_wall,
                    }
                    telemetry_records.append(rec)
            else:
                is_complete = True

    except Exception as exc:
        if not isinstance(exc, LikelihoodBudgetExceededError):
            status = WorkerStatus.FAILED_RUNTIME
            termination_reason = f"Unhandled worker exception: {exc}"
        else:
            status = WorkerStatus.TERMINATED_BUDGET_CALLS
            termination_reason = str(exc)

    elapsed_wall = time.perf_counter() - t0_wall
    elapsed_cpu = time.process_time() - t0_cpu

    # Write telemetry atomically
    if telemetry_records:
        atomic_write_json(telem_file, telemetry_records)

    # Write final authoritative status record atomically
    status_record = {
        "run_id": run_id,
        "worker_type": worker_type,
        "status": status.value,
        "is_complete": is_complete,
        "n_eval_attempted": guard.n_attempted,
        "n_eval_started": guard.n_started,
        "n_eval_completed": guard.n_completed,
        "n_eval_failed": guard.n_failed,
        "call_cap": guard.call_cap,
        "cpu_time_s": elapsed_cpu,
        "wall_time_s": elapsed_wall,
        "termination_reason": termination_reason,
        "timestamp": time.time(),
    }
    atomic_write_json(status_file, status_record)


def run_anchor_run_a_worker(
    run_id: str,
    output_dir: Path,
    call_cap: int = 600_000,
    authorized: bool = False,
    data_prep_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    likelihood_factory_fn: Optional[Callable[[str, Dict[str, Any]], Callable[[Any], float]]] = None,
    sampler_adapter_cls: Optional[Any] = None,
    prior_transform_fn: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Execute the production Stage 6C Run A nested sampling worker under resource control.

    Strict Authorization Gate:
    --------------------------
    Execution requires explicit authorization via `authorized=True`. If False,
    execution aborts immediately before data preparation or likelihood construction.

    Parameters
    ----------
    run_id : str
        Unique run identifier (e.g. 'run_a_anchor').
    output_dir : Path
        Directory for status, telemetry, and result artifacts.
    call_cap : int
        Authoritative in-path likelihood evaluation cap (default 600,000).
    authorized : bool
        Explicit execution authorization flag.
    data_prep_fn : Optional[Callable]
        Dependency injection hook for realization data preparation.
    likelihood_factory_fn : Optional[Callable]
        Dependency injection hook for likelihood construction.
    sampler_adapter_cls : Optional[Any]
        Dependency injection hook for sampler adapter class.
    prior_transform_fn : Optional[Callable]
        Dependency injection hook for unit-cube prior transform.
    """
    if not authorized:
        raise RuntimeError(
            "CRITICAL SECURITY GATE: Stage 6C scientific sampler runs (Run A) "
            "are STRICTLY UNAUTHORIZED without the explicit --authorize-stage6c-execution flag."
        )

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    status_file = out_path / f"{run_id}_status.json"
    telem_file = out_path / f"{run_id}_telemetry.json"
    result_file = out_path / f"{run_id}_result.json"
    abort_file = out_path / f"{run_id}.abort_signal"

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    shutdown_requested = False

    def handle_signal(sig: int, frame: Any) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    try:
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_signal)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, handle_signal)
    except Exception:
        pass

    if abort_file.is_file():
        status_record = {
            "run_id": run_id,
            "target": "run_a",
            "worker_type": "anchor_run_a",
            "status": WorkerStatus.CANCELLED_BY_CONTROLLER.value,
            "is_complete": False,
            "n_eval_attempted": 0,
            "n_eval_started": 0,
            "n_eval_completed": 0,
            "n_eval_failed": 0,
            "call_cap": call_cap,
            "cpu_time_s": time.process_time() - t0_cpu,
            "wall_time_s": time.perf_counter() - t0_wall,
            "termination_reason": "Abort signal present before worker start",
            "timestamp": time.time(),
        }
        atomic_write_json(status_file, status_record)
        return

    # Resolve production dependencies lazily if not injected
    if data_prep_fn is None:
        from scripts.run_wp6_stage5_stage_g1 import prepare_realization_data
        data_prep_fn = prepare_realization_data
    if likelihood_factory_fn is None:
        from scripts.run_wp6_stage5_stage_g1 import make_likelihood_fn
        likelihood_factory_fn = make_likelihood_fn
    if sampler_adapter_cls is None:
        from gw_inference.sampler import NestedSamplerAdapter
        sampler_adapter_cls = NestedSamplerAdapter
    if prior_transform_fn is None:
        from gw_inference.prior import prior_transform_4d
        prior_transform_fn = prior_transform_4d

    # Frozen Realization 01 configuration
    sched = {
        "realization": 1,
        "seed_off": 7101,
        "seed_on": 7151,
        "sampler_seeds": [8101, 8201],
    }

    # Data preparation and likelihood factory
    real_data = data_prep_fn(sched)
    raw_logl = likelihood_factory_fn("plug", real_data)

    # In-path budget guard wrapping the production likelihood
    guard = LikelihoodBudgetGuard(raw_logl, call_cap=call_cap)

    # Sampler adapter configured with guarded likelihood and frozen settings
    adapter = sampler_adapter_cls(
        log_likelihood_fn=guard,
        prior_transform_fn=prior_transform_fn,
        ndim=4,
        nlive=500,
        dlogz=0.10,
        seed=8101,
        param_names=["mc", "eta", "tc", "phic"],
    )

    status = WorkerStatus.COMPLETED
    is_complete = False
    termination_reason = "Run completed successfully"

    try:
        # Note on Abort Semantics: dynesty does not expose a per-iteration abort callback
        # during run_nested. Execution runs synchronously; controller-side process watchdog
        # handles CPU/wall termination externally via graceful abort signal then forced process kill.
        res = adapter.run(
            bound="multi",
            sample="rwalk",
            walks=50,
            periodic=[3],
            print_progress=False,
        )

        # Extract result metadata
        if hasattr(res, "log_evidence"):
            logz = float(res.log_evidence)
            logzerr = float(res.log_evidence_err)
            n_evals = int(res.n_evaluations)
        elif isinstance(res, dict):
            logz = float(res.get("log_evidence", 0.0))
            logzerr = float(res.get("log_evidence_err", 0.0))
            n_evals = int(res.get("n_evaluations", guard.n_completed))
        else:
            raise ValueError(f"Unrecognized sampler result type: {type(res)}")

        result_payload = {
            "run_id": run_id,
            "target": "run_a",
            "status": "SUCCESS",
            "log_evidence": logz,
            "log_evidence_err": logzerr,
            "n_evaluations": n_evals,
            "guard_evaluations_completed": guard.n_completed,
            "guard_evaluations_started": guard.n_started,
            "wall_time_s": time.perf_counter() - t0_wall,
            "cpu_time_s": time.process_time() - t0_cpu,
            "timestamp": time.time(),
        }
        if hasattr(res, "summary"):
            try:
                result_payload["summary"] = res.summary()
            except Exception:
                pass
        if hasattr(res, "samples"):
            try:
                import numpy as np
                npz_file = out_path / f"{run_id}_samples.npz"
                save_kw = {"samples": np.asarray(res.samples)}
                if hasattr(res, "weights"):
                    save_kw["weights"] = np.asarray(res.weights)
                if hasattr(res, "posterior_samples"):
                    save_kw["posterior_samples"] = np.asarray(res.posterior_samples)
                np.savez_compressed(npz_file, **save_kw)
                result_payload["samples_npz"] = npz_file.name
            except Exception:
                pass

        atomic_write_json(result_file, result_payload)

        # Integrity verification: Missing or corrupt final result artifacts cannot produce COMPLETED
        if result_file.is_file() and os.path.getsize(result_file) > 0:
            try:
                with open(result_file, "r", encoding="utf-8") as rf:
                    parsed = json.load(rf)
                if isinstance(parsed, dict) and parsed.get("status") == "SUCCESS" and "log_evidence" in parsed:
                    is_complete = True
                    status = WorkerStatus.COMPLETED
                else:
                    is_complete = False
                    status = WorkerStatus.FAILED_RUNTIME
                    termination_reason = "Final result artifact validation failed"
            except Exception as ex:
                is_complete = False
                status = WorkerStatus.FAILED_RUNTIME
                termination_reason = f"Final result artifact unreadable: {ex}"
        else:
            is_complete = False
            status = WorkerStatus.FAILED_RUNTIME
            termination_reason = "Final result artifact missing or empty"

    except LikelihoodBudgetExceededError as exc:
        is_complete = False
        status = WorkerStatus.TERMINATED_BUDGET_CALLS
        termination_reason = str(exc)
    except Exception as exc:
        is_complete = False
        status = WorkerStatus.FAILED_RUNTIME
        termination_reason = f"Sampler execution failed: {exc}"

    if abort_file.is_file() or shutdown_requested:
        is_complete = False
        if status == WorkerStatus.COMPLETED:
            status = WorkerStatus.CANCELLED_BY_CONTROLLER
            termination_reason = "Shutdown signaled by controller/watchdog"

    elapsed_wall = time.perf_counter() - t0_wall
    elapsed_cpu = time.process_time() - t0_cpu

    # Write telemetry
    telem_record = {
        "run_id": run_id,
        "n_eval_attempted": guard.n_attempted,
        "n_eval_started": guard.n_started,
        "n_eval_completed": guard.n_completed,
        "n_eval_failed": guard.n_failed,
        "cpu_s": elapsed_cpu,
        "wall_s": elapsed_wall,
        "timestamp": time.time(),
    }
    atomic_write_json(telem_file, [telem_record])

    # Write authoritative status record
    status_record = {
        "run_id": run_id,
        "target": "run_a",
        "worker_type": "anchor_run_a",
        "status": status.value,
        "is_complete": is_complete,
        "n_eval_attempted": guard.n_attempted,
        "n_eval_started": guard.n_started,
        "n_eval_completed": guard.n_completed,
        "n_eval_failed": guard.n_failed,
        "call_cap": guard.call_cap,
        "cpu_time_s": elapsed_cpu,
        "wall_time_s": elapsed_wall,
        "termination_reason": termination_reason,
        "timestamp": time.time(),
    }
    atomic_write_json(status_file, status_record)


def run_controller_anchor(
    target: str,
    run_id: str,
    output_dir: Path,
    call_cap: int = 600_000,
    cpu_budget: float = 900.0,
    wall_budget: float = 1500.0,
    authorized: bool = False,
    controller_cls: Optional[Any] = None,
    ledger_cls: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute the Stage 6C controller supervising a single anchor run.

    Strict Authorization Gate:
    --------------------------
    Execution requires explicit authorization via `authorized=True`. If False,
    execution aborts immediately before worker process launch.

    Target Restriction:
    -------------------
    Only target='run_a' is permitted in this milestone. Any request for 'run_b'
    or other targets is immediately rejected.
    """
    if not authorized:
        raise RuntimeError(
            "CRITICAL SECURITY GATE: Stage 6C controller cannot launch scientific runs "
            "without the explicit --authorize-stage6c-execution flag."
        )

    if target != "run_a":
        raise RuntimeError(
            f"Target '{target}' is strictly prohibited and not implemented in this milestone. "
            "Only 'run_a' is permitted."
        )

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    config = Stage6CConfig(
        max_calls_per_run=call_cap,
        max_cpu_per_run_s=cpu_budget,
        max_wall_per_run_s=wall_budget,
    )
    controller = (controller_cls or Stage6CController)(config=config)
    ledger = (ledger_cls or AggregateResourceLedger)(config=config)

    worker_script = Path(__file__).resolve()
    command = [
        sys.executable,
        str(worker_script),
        "worker",
        "--mode", "anchor",
        "--target", "run_a",
        "--run-id", str(run_id),
        "--output-dir", str(out_path),
        "--call-cap", str(call_cap),
        "--authorize-stage6c-execution",
    ]

    reconciled_status = controller.run_worker_process(
        command=command,
        run_id=run_id,
        output_dir=out_path,
        custom_max_calls=call_cap,
        custom_max_cpu_s=cpu_budget,
        custom_max_wall_s=wall_budget,
    )

    # Post-execution result verification: missing or corrupt result artifact cannot produce COMPLETED
    if reconciled_status.get("status") == WorkerStatus.COMPLETED.value:
        result_file = out_path / f"{run_id}_result.json"
        is_valid_result = False
        if result_file.is_file() and os.path.getsize(result_file) > 0:
            try:
                with open(result_file, "r", encoding="utf-8") as rf:
                    res_json = json.load(rf)
                if isinstance(res_json, dict) and res_json.get("status") == "SUCCESS" and "log_evidence" in res_json:
                    is_valid_result = True
            except Exception:
                is_valid_result = False

        if not is_valid_result:
            reconciled_status["status"] = WorkerStatus.FAILED_RUNTIME.value
            reconciled_status["is_complete"] = False
            reconciled_status["termination_reason"] = "Missing, unreadable, or corrupt final result artifact"

    calls_completed = reconciled_status.get("n_eval_completed")
    cpu_time_s = reconciled_status.get("cpu_time_s", 0.0)
    wall_time_s = reconciled_status.get("wall_time_s", 0.0)
    status_str = reconciled_status.get("status", WorkerStatus.FAILED_RUNTIME.value)
    is_comp = bool(reconciled_status.get("is_complete", False))
    has_valid_accounting = bool(reconciled_status.get("authoritative_counts_available", False))

    ledger.record_run(
        run_id=run_id,
        calls=calls_completed,
        cpu_s=cpu_time_s,
        wall_s=wall_time_s,
        status=status_str,
        is_complete=is_comp,
        has_valid_accounting=has_valid_accounting,
    )

    ledger_file = out_path / "stage6c_aggregate_ledger.json"
    ledger_data = {
        "runs": ledger.runs,
        "cumulative_calls": ledger.cumulative_calls,
        "cumulative_cpu_s": ledger.cumulative_cpu_s,
        "cumulative_wall_s": ledger.cumulative_wall_s,
        "remaining_calls": ledger.remaining_calls,
        "remaining_cpu_s": ledger.remaining_cpu_s,
        "remaining_wall_s": ledger.remaining_wall_s,
        "run_b_gate": ledger.evaluate_run_b_gate(),
    }
    atomic_write_json(ledger_file, ledger_data)

    return {
        "reconciled_status": reconciled_status,
        "ledger_file": str(ledger_file),
        "run_b_gate": ledger.evaluate_run_b_gate(),
    }


def main() -> None:
    """CLI entry point for Stage 6C runner."""
    parser = argparse.ArgumentParser(description="WP6 Stage 6C Resource-Controlled Runner")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Worker subcommand
    worker_p = subparsers.add_parser("worker", help="Run a worker process")
    worker_p.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    worker_p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    worker_p.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "anchor"],
        help="Execution mode (anchor requires explicit --authorize-stage6c-execution)",
    )
    worker_p.add_argument(
        "--target",
        type=str,
        default="run_a",
        choices=["run_a"],
        help="Scientific target (only run_a is permitted)",
    )
    worker_p.add_argument(
        "--worker-type",
        type=str,
        default="call_bound",
        choices=["call_bound", "cpu_intensive", "sleep", "uncooperative", "crash", "failing_call"],
        help="Synthetic worker behavior type",
    )
    worker_p.add_argument("--call-cap", type=int, default=600_000, help="Likelihood call cap")
    worker_p.add_argument("--target-iterations", type=int, default=100, help="Target iterations")
    worker_p.add_argument("--param-val", type=float, default=0.0, help="Auxiliary parameter")
    worker_p.add_argument(
        "--authorize-stage6c-execution",
        action="store_true",
        default=False,
        help="Explicit authorization flag for Stage 6C scientific execution",
    )

    # Controller subcommand
    ctrl_p = subparsers.add_parser("controller", help="Run the Stage 6C controller")
    ctrl_p.add_argument(
        "--target",
        type=str,
        default="run_a",
        choices=["run_a"],
        help="Scientific target to supervise (only run_a is permitted in this milestone)",
    )
    ctrl_p.add_argument("--run-id", type=str, default="run_a_anchor", help="Unique run identifier")
    ctrl_p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    ctrl_p.add_argument("--call-cap", type=int, default=600_000, help="Likelihood call cap")
    ctrl_p.add_argument("--cpu-budget", type=float, default=900.0, help="Process CPU budget (s)")
    ctrl_p.add_argument("--wall-budget", type=float, default=1500.0, help="Wall-clock timeout (s)")
    ctrl_p.add_argument(
        "--authorize-stage6c-execution",
        action="store_true",
        default=False,
        help="Explicit authorization flag for Stage 6C scientific execution",
    )

    args = parser.parse_args()

    if args.subcommand == "worker":
        if args.mode == "anchor":
            if not args.authorize_stage6c_execution:
                raise RuntimeError(
                    "CRITICAL SECURITY GATE: Stage 6C scientific sampler runs (Run A) "
                    "are STRICTLY UNAUTHORIZED in this milestone without the explicit "
                    "--authorize-stage6c-execution flag."
                )
            if args.target != "run_a":
                raise RuntimeError(
                    f"Target '{args.target}' is strictly prohibited in this milestone. Only 'run_a' is permitted."
                )
            run_anchor_run_a_worker(
                run_id=args.run_id,
                output_dir=Path(args.output_dir),
                call_cap=args.call_cap,
                authorized=args.authorize_stage6c_execution,
            )
        else:
            run_synthetic_worker(
                worker_type=args.worker_type,
                run_id=args.run_id,
                output_dir=Path(args.output_dir),
                call_cap=args.call_cap,
                target_iterations=args.target_iterations,
                param_val=args.param_val,
            )

    elif args.subcommand == "controller":
        if not args.authorize_stage6c_execution:
            raise RuntimeError(
                "CRITICAL SECURITY GATE: Stage 6C controller cannot launch scientific runs "
                "without the explicit --authorize-stage6c-execution flag."
            )
        if args.target != "run_a":
            raise RuntimeError(
                f"Target '{args.target}' is strictly prohibited in this milestone. Only 'run_a' is permitted."
            )
        run_controller_anchor(
            target=args.target,
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
            call_cap=args.call_cap,
            cpu_budget=args.cpu_budget,
            wall_budget=args.wall_budget,
            authorized=args.authorize_stage6c_execution,
        )


if __name__ == "__main__":
    main()
