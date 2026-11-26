"""Page 3: Noise & PSD — Spectral Estimation & Detector Calibration."""

import streamlit as st
from gwlab.components import (
    render_header,
    render_page_header,
    render_safety_banner,
    render_placeholder_card,
)
from gwlab.theme import inject_custom_css, get_current_tokens

inject_custom_css()
render_header()
render_page_header(
    title="Noise and PSD",
    subtitle="Analytical detector sensitivity models, empirical Welch estimation, and variance inflation.",
    badge="Reference design",
)
render_safety_banner()

tokens = get_current_tokens()

col_spec, col_diag = st.columns([1, 1])

with col_spec:
    st.markdown(
        f"""
        <div class="gwlab-card">
            <div class="gwlab-card-title">Detector sensitivity models</div>
            <p style="font-size: 0.8rem; color: {tokens['text_secondary']};">
                Analytical sensitivity curves and calibrated Welch periodogram configurations.
            </p>
            <table class="param-table">
                <tr><th>Model / Configuration</th><th>Value</th></tr>
                <tr><td>Target Detector</td><td>aLIGO Zero-Detuned High Power</td></tr>
                <tr><td>Sampling Frequency f_s</td><td>2048 Hz</td></tr>
                <tr><td>Analysis Band</td><td>[20.0, 512.0] Hz</td></tr>
                <tr><td>Welch Segment Duration</td><td>4.0 s (50% Overlap)</td></tr>
                <tr><td>Tukey Window Alpha</td><td>0.25 (Flat-top cosine taper)</td></tr>
                <tr><td>Effective DoF (nu_eff)</td><td>44.85 (Welch variance inflation)</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_diag:
    render_placeholder_card(
        title="Noise & PSD Implementation Scope",
        scope_description=(
            "The Noise & PSD page will enable interactive exploration of detector noise spectra. "
            "It will display analytical one-sided power spectral density curves S_n(f), amplitude spectral "
            "densities sqrt(S_n(f)), and Stage 5B empirical Welch periodograms computed on synthetic off-source data."
        ),
        planned_features=[
            "Analytical aLIGO ZDHP vs Advanced Virgo ASD curve comparisons",
            "Interactive Welch periodogram explorer showing window attenuation and segment averaging",
            "Stage 5B empirical covariance matrix and off-diagonal spectral leakage diagnostics",
            "Student-t effective degrees of freedom (nu_eff) calibration visualization",
        ],
        candidate_artifacts=[
            "reports/demonstrations/wp6_stage5/stage_b/welch_stage_b_results.npz",
            "reports/demonstrations/wp6_stage5/stage_b/welch_stage_b_summary.csv",
            "reports/demonstrations/figures/figure3_frequency_domain_psd.png",
        ],
        target_phase="Phase 3",
    )
