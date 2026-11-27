"""FastAPI application for GWLab Scientific Backend.

Provides a clean, typed REST API for:
- System and engine status
- Authoritative benchmark artifacts (Overview)
- Deterministic waveform generation (TaylorF2 3.5PN)
- Chirp time calculations
"""

from __future__ import annotations

from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gwlab.config import get_api_settings
from gwlab.api.service import (
    get_overview_benchmarks,
    compute_waveform_response,
    compute_chirp_time_response,
    compute_noise_and_psd_response,
    compute_matched_filter_response,
    get_inference_posterior,
    get_experiments_explorer,
)

app = FastAPI(
    title="GWLab Scientific Engine API",
    description="Deterministic gravitational-wave waveform generation and reference benchmark server.",
    version="1.0.0",
)

_settings = get_api_settings()

# Enable CORS for Next.js frontend (configurable via GWLAB_CORS_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WaveformRequest(BaseModel):
    mc_msun: float = Field(28.095555795460438, ge=5.0, le=100.0, description="Chirp mass in solar masses")
    eta: float = Field(0.24710059171597634, gt=0.0, le=0.25, description="Symmetric mass ratio")
    distance_mpc: float = Field(410.0, ge=10.0, le=5000.0, description="Luminosity distance in Mpc")
    tc_s: float = Field(0.0, ge=-2.0, le=2.0, description="Coalescence arrival time shift in seconds")
    phic_rad: float = Field(0.0, ge=-6.2831853, le=6.2831853, description="Coalescence phase in radians")
    duration_s: float = Field(4.0, ge=1.0, le=16.0, description="Segment duration in seconds")
    sampling_rate_hz: float = Field(2048.0, ge=512.0, le=8192.0, description="Sampling rate in Hz")
    f_end_hz: float = Field(512.0, ge=30.0, le=2048.0, description="Cutoff frequency in Hz")


class ChirpTimeRequest(BaseModel):
    mc_msun: float = Field(28.095555795460438, ge=5.0, le=100.0)
    eta: float = Field(0.24710059171597634, gt=0.0, le=0.25)
    f_isco: Optional[float] = Field(None, ge=10.0, le=4000.0)


class NoiseAnalysisRequest(BaseModel):
    psd_model: Literal["aligo_representative", "flat"] = Field(
        "aligo_representative",
        description="Analytical reference PSD proxy model",
    )
    f_min_hz: float = Field(20.0, ge=1.0, le=200.0, description="Lower active frequency in Hz")
    f_max_hz: float = Field(512.0, ge=50.0, le=1024.0, description="Upper active frequency in Hz")
    sampling_rate_hz: float = Field(2048.0, ge=512.0, le=4096.0, description="Sampling rate in Hz")
    duration_obs_s: float = Field(64.0, ge=8.0, le=128.0, description="Off-source observation duration in s")
    segment_duration_s: float = Field(4.0, ge=1.0, le=16.0, description="Welch segment duration in s")
    overlap_fraction: float = Field(0.5, ge=0.0, le=0.75, description="Welch segment overlap fraction")
    tukey_alpha: float = Field(0.25, ge=0.0, le=1.0, description="Tukey window cosine taper fraction")
    seed: int = Field(7101, ge=0, le=2147483647, description="Deterministic RNG seed for synthetic noise")


class MatchedFilterRequest(BaseModel):
    mc_msun: float = Field(28.095555795460438, ge=10.0, le=50.0, description="Injected chirp mass in M_sun")
    eta: float = Field(0.24710059171597634, gt=0.05, le=0.25, description="Symmetric mass ratio")
    distance_mpc: float = Field(410.0, ge=50.0, le=2000.0, description="Luminosity distance in Mpc")
    tc_s: float = Field(2.0, ge=0.2, le=3.8, description="Injected coalescence time in s")
    phic_rad: float = Field(0.0, ge=-6.2831853, le=6.2831853, description="Injected coalescence phase in rad")
    delta_mc_msun: float = Field(0.0, ge=-3.0, le=3.0, description="Template chirp mass mismatch in M_sun")
    psd_model: Literal["aligo_representative", "flat"] = Field("aligo_representative")
    f_min_hz: float = Field(20.0, ge=5.0, le=200.0, description="Lower active frequency in Hz")
    f_max_hz: float = Field(512.0, ge=50.0, le=1024.0, description="Upper active frequency in Hz")
    seed: int = Field(7151, ge=0, le=2147483647, description="Deterministic synthetic noise seed")


@app.get("/api/status")
def get_status():
    """Health check and scientific engine configuration status."""
    return {
        "status": "online",
        "version": "1.0.0",
        "mode": "Reference archive",
        "detector": "LIGO Hanford (H1)",
        "duration_s": 4.0,
        "sampling_rate_hz": 2048.0,
        "f_lower_hz": 20.0,
        "f_upper_hz": 512.0,
        "waveform_approximant": "TaylorF2 (3.5PN phase, Newtonian amplitude)",
        "governance": "Deterministic execution only; samplers disabled in presentation layer.",
    }


@app.get("/api/benchmarks/overview")
def get_benchmarks():
    """Retrieve verified benchmarks, independent-seed comparison, and provenance matrix."""
    try:
        return get_overview_benchmarks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark artifacts: {str(e)}")


@app.post("/api/waveforms/generate")
def generate_waveform(req: WaveformRequest):
    """Generate pure deterministic TaylorF2 frequency-domain waveform."""
    try:
        return compute_waveform_response(
            mc_msun=req.mc_msun,
            eta=req.eta,
            distance_mpc=req.distance_mpc,
            tc_s=req.tc_s,
            phic_rad=req.phic_rad,
            duration_s=req.duration_s,
            sampling_rate_hz=req.sampling_rate_hz,
            f_end_hz=req.f_end_hz,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Waveform generation failed: {str(e)}")


@app.post("/api/waveforms/chirp-time")
def generate_chirp_time(req: ChirpTimeRequest):
    """Generate Newtonian and 3.5PN time to coalescence series."""
    try:
        return compute_chirp_time_response(
            mc_msun=req.mc_msun,
            eta=req.eta,
            f_isco=req.f_isco,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chirp time calculation failed: {str(e)}")


@app.post("/api/noise/analyze")
def analyze_noise_and_psd(req: NoiseAnalysisRequest):
    """Generate deterministic synthetic colored Gaussian noise and compute analytical + Welch PSD/ASD."""
    try:
        return compute_noise_and_psd_response(
            psd_model=req.psd_model,
            f_min_hz=req.f_min_hz,
            f_max_hz=req.f_max_hz,
            sampling_rate_hz=req.sampling_rate_hz,
            duration_obs_s=req.duration_obs_s,
            segment_duration_s=req.segment_duration_s,
            overlap_fraction=req.overlap_fraction,
            tukey_alpha=req.tukey_alpha,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Noise and PSD analysis failed: {str(e)}")


@app.post("/api/matched-filter/evaluate")
def evaluate_matched_filter(req: MatchedFilterRequest):
    """Evaluate matched-filter SNR time series on synthetic signal + colored Gaussian noise."""
    try:
        return compute_matched_filter_response(
            mc_msun=req.mc_msun,
            eta=req.eta,
            distance_mpc=req.distance_mpc,
            tc_s=req.tc_s,
            phic_rad=req.phic_rad,
            delta_mc_msun=req.delta_mc_msun,
            psd_model=req.psd_model,
            f_min_hz=req.f_min_hz,
            f_max_hz=req.f_max_hz,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Matched filter evaluation failed: {str(e)}")


@app.get("/api/inference/posterior")
def get_posterior_explorer(
    param_x: Literal["mc", "eta", "tc", "phic"] = Query("mc"),
    param_y: Literal["mc", "eta", "tc", "phic"] = Query("eta"),
    bins_1d: int = Query(40, ge=15, le=80),
    bins_2d: int = Query(36, ge=12, le=60),
):
    """Retrieve weighted 1D marginal and 2D joint posterior histograms from Stage 6C Run A."""
    try:
        return get_inference_posterior(
            param_x=param_x,
            param_y=param_y,
            bins_1d=bins_1d,
            bins_2d=bins_2d,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load posterior artifact: {str(e)}")


@app.get("/api/experiments/explorer")
def get_experiments_catalog():
    """Retrieve saved Stage 5G.2, 6C, and 6D experiment comparison and campaign status."""
    try:
        return get_experiments_explorer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load experiment artifacts: {str(e)}")
