"""Page 7: Research Notes — Mathematical Specifications & Research Reports."""

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
    title="Methods and provenance",
    subtitle="Mathematical derivations, likelihood formulations, and interactive research report browser.",
    badge="Reference design",
)
render_safety_banner()

tokens = get_current_tokens()

col_math, col_docs = st.columns([1, 1])

with col_math:
    st.markdown(
        f"""
        <div class="gwlab-card">
            <div class="gwlab-card-title">Core mathematical formulations</div>
            <p style="font-size: 0.8rem; color: {tokens['text_secondary']};">
                Key equations underpinning the gw-inference pipeline.
            </p>
            <div style="font-size: 0.82rem; color: {tokens['text_primary']}; line-height: 1.5;">
                <p><strong>1. Whittle Frequency-Domain Likelihood:</strong></p>
                <code style="display: block; padding: 0.4rem; background: {tokens['bg_primary']}; border: 1px solid {tokens['border']}; margin-bottom: 0.6rem;">
                    ln L_Whittle = -0.5 * sum_k [ 4 * |d_k - h_k|^2 / (T * S_n(f_k)) + ln(pi * T * S_n(f_k) / 2) ]
                </code>
                <p><strong>2. Marginalized Student-t Likelihood:</strong></p>
                <code style="display: block; padding: 0.4rem; background: {tokens['bg_primary']}; border: 1px solid {tokens['border']}; margin-bottom: 0.6rem;">
                    ln L_Student_t = const - ((nu + 2)/2) * sum_k ln[ 1 + (4 * |d_k - h_k|^2) / (nu * T * S_n_hat(f_k)) ]
                </code>
                <p><strong>3. Circular Phase Harmonics:</strong></p>
                <code style="display: block; padding: 0.4rem; background: {tokens['bg_primary']}; border: 1px solid {tokens['border']};">
                    C_m = sum_k w_k cos(m * phi_k),  S_m = sum_k w_k sin(m * phi_k),  R_m = sqrt(C_m^2 + S_m^2)
                </code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_docs:
    render_placeholder_card(
        title="Research Notes Implementation Scope",
        scope_description=(
            "The Research Notes page will provide an in-canvas documentation reader. "
            "It will render Markdown research reports directly within the application, "
            "allowing reviewers and researchers to inspect stage design memos, preflights, "
            "and closeout reports without leaving the laboratory workstation interface."
        ),
        planned_features=[
            "Interactive in-canvas Markdown reader for all repository reports in reports/",
            "Table of contents auto-generation and intra-document anchor linking",
            "KaTeX mathematical formula rendering for analytical waveform derivations",
            "Downloadable PDF / Markdown exports of verified scientific reports",
        ],
        candidate_artifacts=[
            "reports/WP6_Stage6D_Sampler_Repeatability_Design_and_Resource_Review.md",
            "reports/WP6_Stage6C_Run_A_Anchor_Execution_Report.md",
            "reports/WP6_Stage6C_Run_A_Posterior_Forensic_Analysis.md",
            "reports/WP6_Stage6C_Closeout_and_Next_Research_Decision.md",
            "reports/archive/phases/GWLab_Phase0_Architecture_and_Repository_Audit.md",
        ],
        target_phase="Phase 5",
    )
