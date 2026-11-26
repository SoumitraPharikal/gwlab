"""Focused configuration and artifact-path resolution tests for Phase 6B."""

from __future__ import annotations

from pathlib import Path

import pytest

from gwlab.config import DEFAULT_CORS_ORIGINS, WORKSPACE_ROOT, get_api_settings, get_data_root
from gwlab.api.service import resolve_safe_path


def test_default_data_root_and_api_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GWLAB_DATA_ROOT", raising=False)
    monkeypatch.delenv("GWLAB_API_HOST", raising=False)
    monkeypatch.delenv("GWLAB_API_PORT", raising=False)
    monkeypatch.delenv("GWLAB_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("GWLAB_ENV", raising=False)

    assert get_data_root() == WORKSPACE_ROOT
    settings = get_api_settings()
    assert settings.env == "development"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.data_root == WORKSPACE_ROOT
    assert settings.cors_origins == list(DEFAULT_CORS_ORIGINS)


def test_custom_data_root_and_cors_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sample_file = tmp_path / "experiments" / "wp6_stage6c_anchor" / "run_a_anchor_result.json"
    sample_file.parent.mkdir(parents=True)
    sample_file.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("GWLAB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("GWLAB_ENV", "production")
    monkeypatch.setenv("GWLAB_API_HOST", "0.0.0.0")
    monkeypatch.setenv("GWLAB_API_PORT", "8080")
    monkeypatch.setenv("GWLAB_CORS_ORIGINS", "http://frontend:3000, https://gwlab.local")

    settings = get_api_settings()
    assert settings.env == "production"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8080
    assert settings.data_root == tmp_path.resolve()
    assert settings.cors_origins == ["http://frontend:3000", "https://gwlab.local"]

    resolved = resolve_safe_path("experiments/wp6_stage6c_anchor/run_a_anchor_result.json")
    assert resolved == sample_file.resolve()

    with pytest.raises(ValueError, match="Directory traversal detected"):
        resolve_safe_path("../outside.json")


def test_invalid_api_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GWLAB_API_PORT", "not_an_int")
    with pytest.raises(ValueError, match="GWLAB_API_PORT must be an integer"):
        get_api_settings()

    monkeypatch.setenv("GWLAB_API_PORT", "70000")
    with pytest.raises(ValueError, match="GWLAB_API_PORT must be in"):
        get_api_settings()
