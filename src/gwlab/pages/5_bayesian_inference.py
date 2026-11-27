"""Page 5: Bayesian Inference — Stage 6C Anchor Posterior Forensics."""

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
    title="Inference",
    subtitle="Verified posterior distributions, 1D/2D credible intervals, eta-boundary density, and circular phase topology.",
    badge="Reference design",
)
render_safety_banner()

tokens = get_current_tokens()

col_post, col_forensic = st.columns([1, 1])

with col_post:
    st.markdown(
        f"""
        <div class="gwlab-card">
            <div class="gwlab-card-title">Stage 6C Run A summary</div>
            <p style="font-size: 0.8rem; color: {tokens['text_secondary']};">
                Frozen Realization 01, Model 2 (Whittle Gaussian), Seed 8101, 11,951 samples.
            </p>
            <table class="param-table">
                <tr><th>Parameter / Diagnostic</th><th>Authoritative Value</th></tr>
                <tr><td>Log-Evidence ln(Z)</td><td>201,646.1413 ± 0.2374 nats</td></tr>
                <tr><td>Likelihood Evaluations</td><td>487,407 calls (Guard Reconciled)</td></tr>
                <tr><td>Chirp Mass M_c (median)</td><td>28.1968 [28.0786, 28.2951] M_sun</td></tr>
                <tr><td>Symmetric Mass Ratio eta</td><td>0.2481 [0.2429, 0.2497]</td></tr>
                <tr><td>Coalescence Time t_c</td><td>1.9994 [1.9985, 2.0004] s</td></tr>
                <tr><td>Coalescence Phase phi_c</td><td>0.5805 [0.0821, 6.1828] rad</td></tr>
                <tr><td>Top 1% eta Boundary Mass</td><td>38.57% of total posterior mass</td></tr>
                <tr><td>Circular Mean Resultant R_1</td><td>0.6991 (Dominant phase concentration)</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_forensic:
    render_placeholder_card(
        title="Bayesian Inference Implementation Scope",
        scope_description=(
            "The Bayesian Inference page will provide deep interactive forensics into the Stage 6C "
            "production anchor run. Users can explore 1D marginal histograms with kernel density estimates, "
            "2D joint posterior contours with 50% and 90% credible regions, and investigate physical parameter correlations."
        ),
        planned_features=[
            "Interactive corner plot explorer for (M_c, eta, t_c, phi_c) with injection markers",
            "Fine-binned eta-boundary pile-up density analyzer ([0.2485, 0.2500] region)",
            "Polar circular histogram showing circular phase distribution and first/second harmonics",
            "Summary table of weighted medians, credible intervals, and circular moments",
        ],
        candidate_artifacts=[
            "experiments/wp6_stage6c_anchor/run_a_anchor_result.json",
            "experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz",
            "experiments/wp6_stage6c_anchor/posterior_forensics/fig1_phase_circular_distribution.png",
            "experiments/wp6_stage6c_anchor/posterior_forensics/fig2_eta_boundary_analysis.png",
            "experiments/wp6_stage6c_anchor/posterior_forensics/fig3_mass_parameter_correlations.png",
        ],
        target_phase="Phase 5",
    )
