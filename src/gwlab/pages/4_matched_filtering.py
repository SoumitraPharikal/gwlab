"""Page 4: Matched Filtering — Filter Inner Product & Signal Consistency Veto."""

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
    title="Matched filtering",
    subtitle="Frequency-domain inner products, complex SNR time series, and signal-consistency testing.",
    badge="Reference design",
)
render_safety_banner()

tokens = get_current_tokens()

col_snr, col_veto = st.columns([1, 1])

with col_snr:
    st.markdown(
        f"""
        <div class="gwlab-card">
            <div class="gwlab-card-title">Detection statistics</div>
            <p style="font-size: 0.8rem; color: {tokens['text_secondary']};">
                Key detection benchmarks from the canonical Realization 01 injection.
            </p>
            <table class="param-table">
                <tr><th>Filter Metric</th><th>Nominal Value</th></tr>
                <tr><td>Injected Optimal SNR</td><td>45.3153</td></tr>
                <tr><td>Expected Peak Time t_c</td><td>2.0000 s</td></tr>
                <tr><td>Recovered Peak SNR</td><td>45.2891 (Realization 01)</td></tr>
                <tr><td>Time Offset Delta t</td><td>-0.0006 s</td></tr>
                <tr><td>Reduced Chi^2 (rchi^2)</td><td>1.024 (Consistent with Gaussian noise)</td></tr>
                <tr><td>Reweighted SNR (rho_new)</td><td>45.2891 (No glitch penalty)</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_veto:
    render_placeholder_card(
        title="Matched Filtering Implementation Scope",
        scope_description=(
            "The Matched Filtering page will provide an interactive detection console. "
            "Users can inspect the complex SNR time series rho(t), investigate peak extraction "
            "around the coalescence time, and evaluate the binned chi-squared veto statistic "
            "designed to discriminate true gravitational-wave chirps from non-Gaussian instrumental transients."
        ),
        planned_features=[
            "High-resolution Plotly visualization of complex SNR magnitude |rho(t)| and phase",
            "Time-lag recovery slider demonstrating phase alignment at merger",
            "Binned chi-squared veto time series and reduced chi_r^2 distribution",
            "Template mismatch demonstration showing SNR degradation under delta M_c",
        ],
        candidate_artifacts=[
            "reports/demonstrations/figures/figure2_snr_time_series.png",
            "reports/demonstrations/injection_recovery_results.csv",
            "src/gw_inference/filter.py (Matched filter core engine)",
        ],
        target_phase="Phase 4",
    )
