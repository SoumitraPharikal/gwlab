"""Cross-platform local launcher for the GWLab FastAPI + Next.js application.

Checks dependencies, read-only scientific artifacts, and port availability
before starting the backend and frontend servers. Never launches samplers or
recomputes scientific results.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"

REQUIRED_PYTHON_MODULES = (
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "scipy",
)

REQUIRED_ARTIFACTS = (
    "experiments/wp6_stage6c_anchor/run_a_anchor_result.json",
    "experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz",
    "experiments/wp6_stage6c_anchor/stage6c_aggregate_ledger.json",
    "reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s1.json",
    "reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s2.json",
)


def is_port_in_use(host: str, port: int) -> bool:
    """Return True if a TCP listener is already bound on host:port."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe_host, port)) == 0


def run_preflight(api_host: str, api_port: int, frontend_port: int, dev_mode: bool) -> List[str]:
    """Validate dependencies, artifacts, frontend build, and port availability."""
    errors: List[str] = []

    for mod in REQUIRED_PYTHON_MODULES:
        if importlib.util.find_spec(mod) is None:
            errors.append(f"Missing Python dependency: '{mod}'. Install via `pip install -e \".[api]\"`.")

    if shutil.which("node") is None or shutil.which("npm") is None:
        errors.append("Node.js and npm must be installed and available on PATH.")

    data_root = Path(os.environ.get("GWLAB_DATA_ROOT", str(REPO_ROOT))).resolve()
    for rel_path in REQUIRED_ARTIFACTS:
        if not (data_root / rel_path).is_file():
            errors.append(f"Missing required read-only artifact: {data_root / rel_path}")

    if not FRONTEND_DIR.is_dir():
        errors.append(f"Frontend directory not found: {FRONTEND_DIR}")
    elif not (FRONTEND_DIR / "node_modules").is_dir():
        errors.append("Frontend dependencies not installed. Run `cd frontend && npm ci`.")
    elif not dev_mode and not (FRONTEND_DIR / ".next").is_dir():
        errors.append("Next.js production build missing (.next/). Run `cd frontend && npm run build` or pass `--dev`.")

    if is_port_in_use(api_host, api_port):
        errors.append(f"Port {api_port} is already in use (cannot start duplicate FastAPI server).")
    if is_port_in_use("127.0.0.1", frontend_port):
        errors.append(f"Port {frontend_port} is already in use (cannot start duplicate Next.js server).")

    return errors


def wait_for_http(url: str, timeout_s: float = 20.0) -> bool:
    """Poll an HTTP endpoint until it returns < 400 or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 400:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    return False


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch GWLab FastAPI backend and Next.js frontend.")
    parser.add_argument("--api-host", default=os.environ.get("GWLAB_API_HOST", "127.0.0.1"))
    parser.add_argument("--api-port", type=int, default=int(os.environ.get("GWLAB_API_PORT", "8000")))
    parser.add_argument("--frontend-port", type=int, default=int(os.environ.get("GWLAB_FRONTEND_PORT", "3000")))
    parser.add_argument("--dev", action="store_true", help="Run Next.js in development mode instead of `next start`.")
    parser.add_argument("--check-only", action="store_true", help="Run preflight checks and exit without starting servers.")
    args = parser.parse_args(argv)

    errors = run_preflight(args.api_host, args.api_port, args.frontend_port, args.dev)
    if errors:
        print("GWLab preflight failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if args.check_only:
        print("GWLab preflight passed: dependencies, artifacts, build, and ports are ready.")
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    env["GWLAB_API_HOST"] = args.api_host
    env["GWLAB_API_PORT"] = str(args.api_port)
    env.setdefault("NEXT_PUBLIC_API_URL", f"http://localhost:{args.api_port}")

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "gwlab.api.app:app",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
    ]
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
    frontend_cmd = (
        [npx_cmd, "next", "dev", "-p", str(args.frontend_port)]
        if args.dev
        else [npx_cmd, "next", "start", "-p", str(args.frontend_port)]
    )

    print(f"Starting FastAPI backend on http://{args.api_host}:{args.api_port} ...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(REPO_ROOT), env=env)
    frontend_proc: subprocess.Popen[bytes] | None = None

    try:
        health_url = f"http://127.0.0.1:{args.api_port}/api/status"
        if not wait_for_http(health_url, timeout_s=20.0):
            print(f"ERROR: FastAPI backend failed health check at {health_url}.", file=sys.stderr)
            return 1

        print(f"Starting Next.js frontend on http://localhost:{args.frontend_port} ...")
        frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR), env=env)
        print("GWLab is running. Press Ctrl+C to stop both services.")

        while True:
            if backend_proc.poll() is not None:
                print(f"FastAPI backend exited unexpectedly with code {backend_proc.returncode}.", file=sys.stderr)
                return 1
            if frontend_proc.poll() is not None:
                print(f"Next.js frontend exited unexpectedly with code {frontend_proc.returncode}.", file=sys.stderr)
                return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down GWLab services...")
        return 0
    finally:
        for proc in (frontend_proc, backend_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
