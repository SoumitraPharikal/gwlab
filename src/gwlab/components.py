"""Reusable workstation UI components for GWLab.

Follows the editorial and typographical conventions of Kerr Geodesics Lab:
- Clean serif titles (Newsreader / Georgia)
- Restrained hairline borders and muted badges (no emojis, no loud yellow banners)
- Minimalist theme switcher
"""

from __future__ import annotations

import streamlit as st
from gwlab.config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    DETECTOR_NAME,
    EXECUTION_MODE,
    SAFETY_NOTICE,
)
from gwlab.theme import is_dark_mode, get_current_tokens


def render_header() -> None:
    """Render the persistent top workstation header with metadata pills and theme switch."""
    tokens = get_current_tokens()

    col_title, col_meta, col_theme = st.columns([5.5, 4.0, 1.5], vertical_alignment="center")

    with col_title:
        st.markdown(
            f"""
            <div style="display: flex; align-items: baseline; gap: 0.6rem;">
                <span class="gwlab-brand">
                    {APP_TITLE}
                </span>
                <span class="gwlab-subtitle">
                    v{APP_VERSION} &middot; {APP_SUBTITLE}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_meta:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; gap: 0.5rem; align-items: center;">
                <span class="gwlab-badge" style="background-color: {tokens['bg_secondary']}; color: {tokens['text_secondary']}; border: 1px solid {tokens['border']};">
                    {DETECTOR_NAME}
                </span>
                <span class="gwlab-badge">
                    {EXECUTION_MODE}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_theme:
        current_dark = is_dark_mode()
        theme_label = "Dark" if current_dark else "Light"
        if st.button(theme_label, key="theme_toggle_btn", use_container_width=True, help="Toggle light / dark appearance"):
            st.session_state.dark_mode = not current_dark
            st.rerun()


def render_page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    """Render standardized page header with section title, description, and optional status pill."""
    tokens = get_current_tokens()
    badge_html = (
        f'<span class="gwlab-badge" style="background-color: {tokens["bg_secondary"]}; border: 1px solid {tokens["border"]}; color: {tokens["text_secondary"]};">{badge}</span>'
        if badge
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-top: 0.1rem; margin-bottom: 1.0rem; border-bottom: 1px solid {tokens['border_subtle']}; padding-bottom: 0.6rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h1 style="margin: 0; font-size: 1.85rem; font-weight: 500; color: {tokens['text_primary']};">
                    {title}
                </h1>
                {badge_html}
            </div>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.88rem; color: {tokens['text_secondary']}; font-family: 'IBM Plex Sans', sans-serif;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_archive_note() -> None:
    """Render a quiet, restrained archive status note."""
    tokens = get_current_tokens()
    st.markdown(
        f"""
        <div class="gwlab-archive-note">
            <span style="color: {tokens['accent']}; font-weight: 600;">Reference archive:</span>
            {SAFETY_NOTICE}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_safety_banner() -> None:
    """Backwards-compatible wrapper that renders the quiet archive note instead of a loud yellow banner."""
    render_archive_note()


def render_placeholder_card(
    title: str,
    scope_description: str,
    planned_features: list[str],
    candidate_artifacts: list[str],
    target_phase: str,
) -> None:
    """Render standardized placeholder card explaining future implementation scope and candidate datasets."""
    tokens = get_current_tokens()

    features_html = "".join([f"<li>{feat}</li>" for feat in planned_features])
    artifacts_html = "".join([f"<li><code>{art}</code></li>" for art in candidate_artifacts])

    st.markdown(
        f"""
        <div class="gwlab-card">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem; border-bottom: 1px solid {tokens['border_subtle']}; padding-bottom: 0.4rem;">
                <span class="gwlab-card-title" style="margin: 0;">{title}</span>
                <span class="gwlab-badge" style="background-color: {tokens['accent_subtle']}; color: {tokens['accent']}; border: 1px solid {tokens['accent_border']};">
                    Target: {target_phase}
                </span>
            </div>
            <div class="gwlab-card-desc">
                <p style="margin-bottom: 0.6rem;">{scope_description}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-top: 0.6rem;">
                    <div>
                        <div style="color: {tokens['text_primary']}; font-size: 0.82rem; font-weight: 500; margin-bottom: 0.25rem;">Planned Interactive Capabilities:</div>
                        <ul style="margin: 0; padding-left: 1.1rem; font-size: 0.80rem; color: {tokens['text_secondary']}; line-height: 1.45;">
                            {features_html}
                        </ul>
                    </div>
                    <div>
                        <div style="color: {tokens['text_primary']}; font-size: 0.82rem; font-weight: 500; margin-bottom: 0.25rem;">Registered Repository Artifacts for Ingestion:</div>
                        <ul style="margin: 0; padding-left: 1.1rem; font-size: 0.80rem; color: {tokens['text_secondary']}; line-height: 1.45;">
                            {artifacts_html}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
