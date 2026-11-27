"""Application configuration, constants, and page registry for GWLab."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

# Root workspace path resolution
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8502",
    "http://127.0.0.1:8502",
)


@dataclass(frozen=True)
class ApiSettings:
    """Centralized runtime settings for the GWLab FastAPI service."""

    env: str
    host: str
    port: int
    data_root: Path
    cors_origins: List[str]


def get_data_root() -> Path:
    """Return the configured read-only artifact root directory."""
    raw = os.environ.get("GWLAB_DATA_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return WORKSPACE_ROOT


def get_api_settings() -> ApiSettings:
    """Resolve FastAPI runtime settings from environment variables with safe local defaults."""
    env = os.environ.get("GWLAB_ENV", "development").strip() or "development"
    host = os.environ.get("GWLAB_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    raw_port = os.environ.get("GWLAB_API_PORT", "8000").strip()
    try:
        port = int(raw_port)
    except ValueError as err:
        raise ValueError(f"GWLAB_API_PORT must be an integer, got '{raw_port}'.") from err
    if not (1 <= port <= 65535):
        raise ValueError(f"GWLAB_API_PORT must be in [1, 65535], got {port}.")

    raw_cors = os.environ.get("GWLAB_CORS_ORIGINS", "").strip()
    if raw_cors:
        cors_origins = [o.strip() for o in raw_cors.split(",") if o.strip()]
    else:
        cors_origins = list(DEFAULT_CORS_ORIGINS)

    return ApiSettings(
        env=env,
        host=host,
        port=port,
        data_root=get_data_root(),
        cors_origins=cors_origins,
    )

# Application Metadata
APP_TITLE = "GWLab"
APP_SUBTITLE = "Gravitational-Wave Analysis Laboratory"
APP_VERSION = "0.1.0"
APP_ICON = None

# Execution & Safety Boundaries
EXECUTION_MODE = "Reference archive"
SAFETY_NOTICE = (
    "Read-only reference archive. Live parameter estimation, Dynesty nested sampling, "
    "and likelihood evaluations are disabled in this interface. "
    "Displays reflect verified saved artifacts from the gw-inference pipeline."
)

# Instrument Metadata
DETECTOR_NAME = "LIGO Hanford (H1)"
DETECTOR_SPECS = "4 km Dual-Recycled Fabry-Pérot Michelson Interferometer"
OBSERVING_RUN = "Design Sensitivity Baseline (aLIGO Zero-Detuned High Power)"

# Page Registry Definition (Visitor Destinations)
PAGE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "overview",
        "title": "Overview",
        "icon": None,
        "file": "pages/1_overview.py",
        "description": "Mission dashboard, pipeline architecture, and reference execution state.",
    },
    {
        "id": "waveforms",
        "title": "Waveforms",
        "icon": None,
        "file": "pages/2_signal_lab.py",
        "description": "TaylorF2 3.5PN frequency-domain waveform inspection and chirp dynamics.",
    },
    {
        "id": "noise_and_psd",
        "title": "Noise and PSD",
        "icon": None,
        "file": "pages/3_noise_and_psd.py",
        "description": "Spectral density estimation, Welch periodograms, and variance inflation.",
    },
    {
        "id": "matched_filtering",
        "title": "Matched filtering",
        "icon": None,
        "file": "pages/4_matched_filtering.py",
        "description": "Matched-filter SNR time series, time-lag recovery, and chi-squared veto.",
    },
    {
        "id": "inference",
        "title": "Inference",
        "icon": None,
        "file": "pages/5_bayesian_inference.py",
        "description": "Stage 6C Run A posterior forensics, eta-boundary accumulation, and circular phase.",
    },
    {
        "id": "experiments",
        "title": "Experiments",
        "icon": None,
        "file": "pages/6_experiment_explorer.py",
        "description": "Cross-run comparative workbench, seed repeatability, and Model 2 vs Model 3.",
    },
    {
        "id": "methods_and_provenance",
        "title": "Methods and provenance",
        "icon": None,
        "file": "pages/7_research_notes.py",
        "description": "Mathematical specifications, derivations, and research report archive.",
    },
]
