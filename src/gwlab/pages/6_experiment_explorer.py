"""Page 6: Experiment Explorer — Multi-Run Comparative Scientific Workbench."""

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
    title="Experiments",
    subtitle="Cross-run comparison, independent-seed repeatability analysis, and likelihood model evaluation.",
    badge="Reference design",
)
render_safety_banner()

tokens = get_current_tokens()

col_comp, col_plan = st.columns([1, 1])

with col_comp:
    st.markdown(
        f"""
        <div class="gwlab-card">
            <div class="gwlab-card-title">Independent-seed comparison</div>
            <p style="font-size: 0.8rem; color: {tokens['text_secondary']};">
                Frozen Realization 01: Seed 8101 vs Seed 8201 (Model 2 Whittle).
            </p>
            <table class="param-table">
                <tr><th>Metric / Parameter</th><th>Seed 8101 (Run A)</th><th>Seed 8201 (Base 01)</th></tr>
                <tr><td>Log-Evidence ln(Z)</td><td>201,646.1413 ± 0.2374</td><td>201,643.1693 ± 0.2338</td></tr>
                <tr><td>Observed Evidence Diff</td><td colspan="2">|Delta ln Z| = 2.9720 nats</td></tr>
                <tr><td>Combined Uncertainty</td><td colspan="2">sigma_Delta = 0.3332 nats</td></tr>
                <tr><td>Tension Ratio (z_rep)</td><td colspan="2">8.92 (Observed Discrepancy)</td></tr>
                <tr><td>Chirp Mass M_c (median)</td><td>28.1965 M_sun</td><td>28.0945 M_sun</td></tr>
                <tr><td>Symmetric Mass Ratio eta</td><td>0.2481</td><td>0.2381</td></tr>
                <tr><td>Top 1% eta Boundary Mass</td><td>38.57%</td><td>0.35%</td></tr>
            </table>
            <p style="font-size: 0.72rem; color: {tokens['accent_amber']}; margin-top: 0.5rem; line-height: 1.3;">
                ⚠️ <em>Scientific Caveat:</em> This ratio is an observed discrepancy relative to the runs' reported
                internal uncertainties under the frozen inference problem, not a calibrated frequentist significance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_plan:
    render_placeholder_card(
        title="Experiment Explorer Implementation Scope",
        scope_description=(
            "The Experiment Explorer will provide a multi-run comparative workbench across the repository's "
            "experimental campaigns. Researchers can select pairs or cohorts of runs, compare posterior "
            "distributions, evaluate sampler resolution scaling (N_live = 500 vs 1000), and inspect nested sampling diagnostics."
        ),
        planned_features=[
            "Side-by-side seed repeatability workbench with comparative 1D/2D posterior overlays",
            "Likelihood model evaluation console: Model 2 (Whittle) vs Model 3 (Student-t)",
            "Resolution scaling inspector: N_live = 500 vs 1000 and random walk vs slice sampling",
            "Nested sampling diagnostic viewer: evidence accumulation curves and prior volume shrinkage",
        ],
        candidate_artifacts=[
            "reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s1.json",
            "reports/demonstrations/wp6_stage5/stage_g2/base_01_plug_inst_s2.json",
            "reports/demonstrations/wp6_stage5/stage_g2/base_01_stud_inst_s1.json",
            "reports/demonstrations/wp6_stage5/stage_g2/diag_01_plug_nl1000_s1.json",
            "reports/demonstrations/wp6_stage5/stage_g2/diag_01_plug_rslice_s1.json",
        ],
        target_phase="Phase 5",
    )
