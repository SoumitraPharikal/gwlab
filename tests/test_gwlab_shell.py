"""Focused unit and safety tests for the GWLab Phase 1 application shell."""

import re
import sys
from pathlib import Path
import pytest


def test_gwlab_package_metadata():
    """Verify package import, version, and docstring."""
    import gwlab

    assert hasattr(gwlab, "__version__")
    assert gwlab.__version__ == "0.1.0"
    assert "GWLab" in gwlab.__doc__


def test_gwlab_config_and_page_registry():
    """Verify application configuration and the 7 page definitions."""
    import gwlab
    from gwlab.config import PAGE_DEFINITIONS, APP_TITLE, WORKSPACE_ROOT

    assert APP_TITLE == "GWLab"
    assert len(PAGE_DEFINITIONS) == 7

    # Verify each page file exists on disk
    pages_dir = Path(gwlab.__file__).parent / "pages"
    assert pages_dir.is_dir()

    for p in PAGE_DEFINITIONS:
        page_file = pages_dir / Path(p["file"]).name
        assert page_file.is_file(), f"Missing page file: {page_file}"
        assert page_file.stat().st_size > 0, f"Empty page file: {page_file}"


def test_gwlab_safety_boundary_no_dynesty():
    """Verify that importing GWLab shell modules does not load Dynesty or sampling adapters."""
    # Ensure dynesty is not pre-imported
    for mod in list(sys.modules.keys()):
        if "dynesty" in mod or "gw_inference.sampler" in mod:
            del sys.modules[mod]

    import gwlab.config
    import gwlab.theme
    import gwlab.components

    assert "dynesty" not in sys.modules, "Dynesty must not be imported by gwlab shell!"
    assert "gw_inference.sampler" not in sys.modules, "Sampler adapter must not be imported by gwlab shell!"


def test_gwlab_theme_tokens_structure():
    """Verify dark and light theme tokens and contrast color standards."""
    from gwlab.theme import THEME_TOKENS

    for theme_name in ["dark", "light"]:
        assert theme_name in THEME_TOKENS
        tokens = THEME_TOKENS[theme_name]
        for key in ["bg_primary", "bg_secondary", "text_primary", "text_secondary", "text_muted", "border"]:
            assert key in tokens, f"Missing token '{key}' in theme '{theme_name}'"
            assert tokens[key].startswith("#") or tokens[key].startswith("rgba"), f"Invalid token format for {key}"

    # Verify refined light theme text_muted is darkened for WCAG AA compliance (>= 4.5:1)
    assert THEME_TOKENS["light"]["text_muted"] in ["#68655F", "#6F6F6F"]


def test_no_sampler_launch_controls_in_pages():
    """Safety check: Verify no placeholder page exposes buttons to launch samplers or likelihoods."""
    pages_dir = Path(__file__).resolve().parent.parent / "src" / "gwlab" / "pages"
    assert pages_dir.is_dir()

    forbidden_patterns = [
        r"st\.button\(['\"].*run.*sampler",
        r"st\.button\(['\"].*start.*sampling",
        r"st\.button\(['\"].*execute.*likelihood",
        r"st\.button\(['\"].*launch.*dynesty",
    ]

    for py_file in pages_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8").lower()
        for pat in forbidden_patterns:
            match = re.search(pat, content)
            assert not match, f"Forbidden button found in {py_file.name}: {match.group(0)}"


def test_app_entrypoint_exists():
    """Verify that src/gwlab/app.py exists and configures navigation."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "gwlab" / "app.py"
    assert app_path.is_file()
    content = app_path.read_text(encoding="utf-8")
    assert "st.navigation" in content
    assert "set_page_config" in content
