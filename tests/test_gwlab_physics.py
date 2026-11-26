"""Tests for GWLab physical parameter mappings and deterministic waveform helpers."""

import math
import numpy as np
import pytest

from gwlab.physics import (
    STAGE6C_INJECTION,
    mchirp_eta_to_component_masses,
    component_masses_to_mchirp_eta,
    generate_taylorf2_strain,
    compute_chirp_time_series,
)
from gw_inference.constants import C_ISCO_SI, M_SUN_SI, MPC_SI


class TestPhysicalParameterMappings:
    """Test physical mass transformations and boundary handling."""

    def test_stage6c_injection_mass_conversion(self):
        """Verify mass conversion for Stage 6C injection benchmark."""
        mc = STAGE6C_INJECTION["mc_msun"]
        eta = STAGE6C_INJECTION["eta"]

        m1, m2, M = mchirp_eta_to_component_masses(mc, eta)

        # Expected GW150914-like parameters: m1 ~ 36 M_sun, m2 ~ 29 M_sun, M ~ 65 M_sun
        assert math.isclose(M, 65.0, rel_tol=1e-12)
        assert math.isclose(m1, 36.0, rel_tol=1e-12)
        assert math.isclose(m2, 29.0, rel_tol=1e-12)
        assert m1 >= m2

        # Round-trip inverse consistency
        mc_rec, eta_rec, M_rec = component_masses_to_mchirp_eta(m1, m2)
        assert math.isclose(mc_rec, mc, rel_tol=1e-14)
        assert math.isclose(eta_rec, eta, rel_tol=1e-14)
        assert math.isclose(M_rec, M, rel_tol=1e-14)

    def test_equal_mass_boundary_eta_025(self):
        """Verify exact equal-mass boundary eta = 0.25 handling without numerical error."""
        mc = 30.0 * (0.25**0.6)  # Total mass M = 30 M_sun
        eta = 0.25

        m1, m2, M = mchirp_eta_to_component_masses(mc, eta)

        assert math.isclose(M, 30.0, rel_tol=1e-14)
        assert math.isclose(m1, 15.0, rel_tol=1e-14)
        assert math.isclose(m2, 15.0, rel_tol=1e-14)
        assert m1 == m2

        # Inverse
        mc_rec, eta_rec, M_rec = component_masses_to_mchirp_eta(m1, m2)
        assert math.isclose(mc_rec, mc, rel_tol=1e-14)
        assert math.isclose(eta_rec, 0.25, rel_tol=1e-14)

    def test_invalid_parameters_rejected(self):
        """Verify that unphysical parameters raise ValueError."""
        # Non-positive chirp mass
        with pytest.raises(ValueError, match="Chirp mass must be a positive finite float"):
            mchirp_eta_to_component_masses(0.0, 0.25)
        with pytest.raises(ValueError, match="Chirp mass must be a positive finite float"):
            mchirp_eta_to_component_masses(-10.0, 0.25)
        with pytest.raises(ValueError, match="Chirp mass must be a positive finite float"):
            mchirp_eta_to_component_masses(float("nan"), 0.25)

        # Unphysical eta > 0.25 or <= 0
        with pytest.raises(ValueError, match="eta must be in"):
            mchirp_eta_to_component_masses(28.0, 0.251)
        with pytest.raises(ValueError, match="eta must be in"):
            mchirp_eta_to_component_masses(28.0, 0.0)
        with pytest.raises(ValueError, match="eta must be in"):
            mchirp_eta_to_component_masses(28.0, -0.1)

        # Invalid component masses
        with pytest.raises(ValueError, match="m1_msun must be a positive finite float"):
            component_masses_to_mchirp_eta(0.0, 30.0)
        with pytest.raises(ValueError, match="m2_msun must be a positive finite float"):
            component_masses_to_mchirp_eta(30.0, -5.0)


class TestTaylorF2StrainGeneration:
    """Test deterministic waveform generation, shapes, finiteness, and ISCO cutoff."""

    def test_deterministic_waveform_output(self):
        """Verify that identical inputs produce bit-for-bit identical complex strains."""
        w1 = generate_taylorf2_strain(
            mc_msun=28.095555795460438,
            eta=0.24710059171597634,
            distance_mpc=410.0,
            phic_rad=0.0,
            tc_s=2.0,
        )
        w2 = generate_taylorf2_strain(
            mc_msun=28.095555795460438,
            eta=0.24710059171597634,
            distance_mpc=410.0,
            phic_rad=0.0,
            tc_s=2.0,
        )

        np.testing.assert_array_equal(w1["h_tilde"], w2["h_tilde"])
        np.testing.assert_array_equal(w1["amplitude"], w2["amplitude"])
        assert w1["f_isco"] == w2["f_isco"]

    def test_frequency_grid_and_array_shapes(self):
        """Verify frequency grid consistency: N=8192, f_s=2048 Hz -> n_freqs=4097."""
        res = generate_taylorf2_strain(
            mc_msun=28.0,
            eta=0.24,
            distance_mpc=410.0,
            duration_s=4.0,
            sampling_rate_hz=2048.0,
        )

        assert res["frequencies"].shape == (4097,)
        assert res["h_tilde"].shape == (4097,)
        assert res["amplitude"].shape == (4097,)
        assert res["frequencies"][0] == 0.0
        assert res["frequencies"][-1] == 1024.0
        assert res["delta_f"] == 0.25

    def test_finiteness_over_valid_range(self):
        """Verify amplitude and phase are strictly finite over valid frequency band."""
        res = generate_taylorf2_strain(
            mc_msun=STAGE6C_INJECTION["mc_msun"],
            eta=STAGE6C_INJECTION["eta"],
            distance_mpc=STAGE6C_INJECTION["distance_mpc"],
            tc_s=STAGE6C_INJECTION["tc_s"],
        )

        assert np.all(np.isfinite(res["amplitude"]))
        assert np.all(np.isfinite(res["wrapped_phase"]))
        assert np.all(np.isfinite(res["unwrapped_phase"]))

        # Nonzero bins should have strictly positive amplitude
        active_amp = res["amplitude"][res["nonzero_mask"]]
        assert len(active_amp) > 0
        assert np.all(active_amp > 0)

    def test_isco_cutoff_enforcement(self):
        """Verify that bins with f > f_term are identically zero (inclusive ISCO convention)."""
        res = generate_taylorf2_strain(
            mc_msun=STAGE6C_INJECTION["mc_msun"],
            eta=STAGE6C_INJECTION["eta"],
            distance_mpc=STAGE6C_INJECTION["distance_mpc"],
        )

        f_term = res["f_term"]
        freqs = res["frequencies"]
        h_tilde = res["h_tilde"]

        # DC bin must be 0
        assert h_tilde[0] == 0.0 + 0.0j

        # All bins with f > f_term must be identically 0
        cutoff_mask = freqs > f_term
        assert np.all(h_tilde[cutoff_mask] == 0.0 + 0.0j)

        # Last active bin must satisfy f <= f_term
        active_freqs = freqs[res["nonzero_mask"]]
        assert np.max(active_freqs) <= f_term

    def test_chirp_time_series_evaluation(self):
        """Verify Newtonian and 3.5PN chirp time series calculation."""
        mc = STAGE6C_INJECTION["mc_msun"]
        eta = STAGE6C_INJECTION["eta"]
        m1, m2, M = mchirp_eta_to_component_masses(mc, eta)
        f_isco = C_ISCO_SI / M

        freqs = np.linspace(20.0, 60.0, 41)
        eval_f, tau_n, tau_pn = compute_chirp_time_series(freqs, m1, m2, mc, f_isco)

        assert len(eval_f) == 41
        assert np.all(np.isfinite(tau_n))
        assert np.all(np.isfinite(tau_pn))
        assert np.all(tau_n > 0.0)
        assert np.all(tau_pn > 0.0)
        # Chirp time should decrease monotonically as frequency increases
        assert np.all(np.diff(tau_n) < 0)
        assert np.all(np.diff(tau_pn) < 0)


class TestSafetyBoundaries:
    """Verify that importing physics and running waveforms never imports dynesty or samplers."""

    def test_no_dynesty_in_modules(self):
        """Verify dynesty is never imported during physics operations."""
        import sys
        import importlib
        for mod in list(sys.modules.keys()):
            if "dynesty" in mod:
                del sys.modules[mod]
        import gwlab.physics
        importlib.reload(gwlab.physics)
        assert "dynesty" not in sys.modules

    def test_signal_lab_source_syntax_and_no_samplers(self):
        """Verify that Signal Lab parses cleanly without sampler or likelihood engine imports."""
        import ast
        from pathlib import Path
        from gwlab.config import WORKSPACE_ROOT

        sig_lab_path = WORKSPACE_ROOT / "src" / "gwlab" / "pages" / "2_signal_lab.py"
        assert sig_lab_path.is_file()

        code = sig_lab_path.read_text(encoding="utf-8")
        parsed = ast.parse(code)
        assert parsed is not None

        forbidden = [
            "dynesty",
            "gw_inference.sampler",
            "gw_inference.likelihood",
            "gw_inference.noise",
            "gw_inference.psd",
            "run_sampler",
        ]
        for term in forbidden:
            assert f"import {term}" not in code, f"Forbidden import found: {term}"
            assert f"from {term}" not in code, f"Forbidden from-import found: {term}"
