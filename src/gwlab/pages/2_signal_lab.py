"""Page 2: Waveforms — Deterministic TaylorF2 Waveform Synthesizer & Inspection."""

from __future__ import annotations

import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from gwlab.components import (
    render_header,
    render_page_header,
    render_archive_note,
)
from gwlab.theme import inject_custom_css, get_current_tokens, get_plotly_layout, is_dark_mode
from gwlab.format import (
    format_float,
    format_scientific,
    format_with_unit,
)
from gwlab.physics import (
    STAGE6C_INJECTION,
    mchirp_eta_to_component_masses,
    generate_taylorf2_strain,
    compute_chirp_time_series,
)

# -----------------------------------------------------------------------------
# WORKSTATION SHELL & THEME
# -----------------------------------------------------------------------------
inject_custom_css()
render_header()
render_page_header(
    title="Waveforms",
    subtitle="Interactive deterministic synthesis and inspection of TaylorF2 3.5PN frequency-domain gravitational waveforms.",
    badge="Deterministic synthesis",
)

tokens = get_current_tokens()
is_dark = is_dark_mode()
base_layout = get_plotly_layout()

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION (STAGE 6C INJECTION DEFAULTS)
# -----------------------------------------------------------------------------
if "siglab_mc" not in st.session_state:
    st.session_state.siglab_mc = float(STAGE6C_INJECTION["mc_msun"])
if "siglab_eta" not in st.session_state:
    st.session_state.siglab_eta = float(STAGE6C_INJECTION["eta"])
if "siglab_dist" not in st.session_state:
    st.session_state.siglab_dist = float(STAGE6C_INJECTION["distance_mpc"])
if "siglab_tc" not in st.session_state:
    st.session_state.siglab_tc = float(STAGE6C_INJECTION["tc_s"])
if "siglab_phic" not in st.session_state:
    st.session_state.siglab_phic = float(STAGE6C_INJECTION["phic_rad"])


def reset_to_stage6c_injection():
    """Reset parameter controls to the exact Stage 6C Run A injection benchmark."""
    st.session_state.siglab_mc = float(STAGE6C_INJECTION["mc_msun"])
    st.session_state.siglab_eta = float(STAGE6C_INJECTION["eta"])
    st.session_state.siglab_dist = float(STAGE6C_INJECTION["distance_mpc"])
    st.session_state.siglab_tc = float(STAGE6C_INJECTION["tc_s"])
    st.session_state.siglab_phic = float(STAGE6C_INJECTION["phic_rad"])


# -----------------------------------------------------------------------------
# TWO-COLUMN EDITORIAL LAYOUT (Visual Center >60%)
# -----------------------------------------------------------------------------
col_ctrl, col_viz = st.columns([1.1, 2.9], gap="medium")

with col_ctrl:
    st.markdown(
        f"""
        <div class="gwlab-card" style="padding: 0.9rem 1.0rem; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid {tokens['border_subtle']}; padding-bottom: 0.35rem; margin-bottom: 0.5rem;">
                <span class="gwlab-card-title" style="font-size: 1.05rem; margin: 0;">Parameters</span>
                <span class="gwlab-badge">3.5PN SPA</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Reset to injection benchmark", on_click=reset_to_stage6c_injection, use_container_width=True):
        pass

    # Chirp mass Mc
    mc_val = st.number_input(
        "Chirp mass M_c (M_sun)",
        min_value=10.0,
        max_value=60.0,
        value=float(st.session_state.siglab_mc),
        step=0.01,
        format="%.4f",
        key="siglab_mc",
        help="Detector-frame chirp mass M_c = M * eta^(3/5). Injection: 28.0956 M_sun.",
    )

    # Symmetric mass ratio eta
    eta_val = st.number_input(
        "Symmetric mass ratio eta",
        min_value=0.10,
        max_value=0.25,
        value=float(st.session_state.siglab_eta),
        step=0.001,
        format="%.4f",
        key="siglab_eta",
        help="Symmetric mass ratio eta in (0, 0.25]. Equal-mass limit is 0.25. Injection: 0.2471.",
    )

    # Luminosity distance d_L
    dist_val = st.number_input(
        "Luminosity distance d_L (Mpc)",
        min_value=50.0,
        max_value=2500.0,
        value=float(st.session_state.siglab_dist),
        step=10.0,
        format="%.1f",
        key="siglab_dist",
        help="Luminosity distance in Megaparsecs. Strain amplitude scales inversely with d_L. Injection: 410.0 Mpc.",
    )

    # Coalescence reference phase phi_c
    phic_val = st.number_input(
        "Coalescence phase phi_c (rad)",
        min_value=0.0,
        max_value=float(2.0 * math.pi),
        value=float(st.session_state.siglab_phic),
        step=0.05,
        format="%.2f",
        key="siglab_phic",
        help="Orbital reference phase at coalescence. Injection: 0.00 rad.",
    )

    # Coalescence time shift t_c
    tc_val = st.number_input(
        "Coalescence time t_c (s)",
        min_value=0.0,
        max_value=4.0,
        value=float(st.session_state.siglab_tc),
        step=0.01,
        format="%.2f",
        key="siglab_tc",
        help="Coalescence arrival time relative to segment start. Induces linear phase slope -2pi f t_c. Injection: 2.00 s.",
    )

    with st.expander("Grid and analysis options", expanded=False):
        duration_val = st.selectbox(
            "Segment duration T (s)",
            options=[2.0, 4.0, 8.0, 16.0],
            index=1,
            help="Observation duration in seconds. Determines delta_f = 1 / T.",
        )
        srate_val = st.selectbox(
            "Sampling rate f_s (Hz)",
            options=[1024.0, 2048.0, 4096.0],
            index=1,
            help="Sampling rate in Hz. Nyquist limit is f_Nyq = f_s / 2.",
        )
        pn_val = st.radio(
            "Phase PN order",
            options=[3.5, 3.0],
            index=0,
            horizontal=True,
            help="Post-Newtonian phase expansion order.",
        )
        show_analysis_band = st.checkbox(
            "Highlight [20, 512] Hz analysis band",
            value=True,
            help="Overlays the Stage 6C/6D active likelihood frequency band.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Compute deterministic waveform & derived quantities
    wf = generate_taylorf2_strain(
        mc_msun=mc_val,
        eta=eta_val,
        distance_mpc=dist_val,
        phic_rad=phic_val,
        tc_s=tc_val,
        duration_s=float(duration_val),
        sampling_rate_hz=float(srate_val),
        pn_order=float(pn_val),
    )

    # Derived Physical Quantities Summary Card
    st.markdown(
        f"""
        <div class="gwlab-card" style="padding: 0.9rem 1.0rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid {tokens['border_subtle']}; padding-bottom: 0.35rem; margin-bottom: 0.4rem;">
                <span class="gwlab-card-title" style="font-size: 1.05rem; margin: 0;">Derived quantities</span>
            </div>
            <table class="param-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Primary mass <em>m</em><sub>1</sub></td><td>{format_float(wf['m1_msun'], 3)}&thinsp;M<sub>&odot;</sub></td></tr>
                <tr><td>Secondary mass <em>m</em><sub>2</sub></td><td>{format_float(wf['m2_msun'], 3)}&thinsp;M<sub>&odot;</sub></td></tr>
                <tr><td>Total mass <em>M</em></td><td>{format_float(wf['m_total_msun'], 3)}&thinsp;M<sub>&odot;</sub></td></tr>
                <tr><td>Mass ratio <em>q</em></td><td>{format_float(wf['q'], 3)}</td></tr>
                <tr><td>ISCO frequency <em>f</em><sub>ISCO</sub></td><td>{format_float(wf['f_isco'], 1)}&thinsp;Hz</td></tr>
                <tr><td>Cutoff <em>f</em><sub>term</sub></td><td>{format_float(wf['f_term'], 1)}&thinsp;Hz</td></tr>
                <tr><td>Grid step &Delta;<em>f</em></td><td>{format_float(wf['delta_f'], 3)}&thinsp;Hz</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_viz:
    tab_amp, tab_phase, tab_chirp, tab_quad = st.tabs([
        "Strain amplitude",
        "Fourier phase",
        "Chirp time",
        "Quadratures",
    ])

    mask = wf["nonzero_mask"]
    active_freqs = wf["frequencies"][mask]
    active_amp = wf["amplitude"][mask]
    active_unwrapped = wf["unwrapped_phase"][mask]
    active_wrapped = wf["wrapped_phase"][mask]
    active_htilde = wf["h_tilde"][mask]

    # -------------------------------------------------------------------------
    # TAB 1: AMPLITUDE SPECTRUM
    # -------------------------------------------------------------------------
    with tab_amp:
        st.latex(r"|\tilde{h}(f)| = \mathcal{A} f^{-7/6} \Theta(f_{\mathrm{ISCO}} - f)")

        fig_amp = go.Figure()

        # TaylorF2 Amplitude Trace
        fig_amp.add_trace(
            go.Scatter(
                x=active_freqs,
                y=active_amp,
                mode="lines",
                name="TaylorF2 3.5PN",
                line=dict(color=tokens["accent"], width=2.0),
                hovertemplate="f = %{x:.1f} Hz<br>|h~(f)| = %{y:.3e} Hz⁻¹<extra></extra>",
            )
        )

        # ISCO Frequency Cutoff Line
        fig_amp.add_vline(
            x=wf["f_isco"],
            line_dash="dash",
            line_color=tokens["text_secondary"],
            line_width=1.5,
            annotation_text=f"f_ISCO = {format_float(wf['f_isco'], 1)} Hz",
            annotation_position="top right",
            annotation_font=dict(color=tokens["text_secondary"], size=10, family="IBM Plex Sans"),
        )

        # Highlight Analysis Band [20, 512] Hz
        if show_analysis_band:
            fig_amp.add_vrect(
                x0=20.0,
                x1=min(512.0, wf["f_isco"]),
                fillcolor=tokens["accent_subtle"],
                opacity=0.5,
                layer="below",
                line_width=0,
                annotation_text="Analysis band [20, 512] Hz",
                annotation_position="top left",
                annotation_font=dict(color=tokens["accent"], size=10, family="IBM Plex Sans"),
            )

        layout_amp = dict(base_layout)
        layout_amp.update({
            "xaxis": {
                **base_layout["xaxis"],
                "type": "log",
                "title": {"text": "Frequency f (Hz)", "font": base_layout["xaxis"]["title"]["font"]},
                "tickmode": "array",
                "tickvals": [20, 50, 100, 200, 500, 1000],
                "ticktext": ["20", "50", "100", "200", "500", "1k"],
                "range": [math.log10(15.0), math.log10(max(100.0, wf["f_isco"] * 1.15))],
            },
            "yaxis": {
                **base_layout["yaxis"],
                "type": "log",
                "title": {"text": "Strain spectral amplitude |h~(f)| (Hz⁻¹)", "font": base_layout["yaxis"]["title"]["font"]},
            },
            "height": 410,
            "showlegend": True,
            "legend": {
                **base_layout["legend"],
                "x": 0.03,
                "y": 0.08,
            },
        })
        fig_amp.update_layout(layout_amp)
        st.plotly_chart(fig_amp, use_container_width=True)

        st.caption(
            f"Frequency-domain strain amplitude follows the stationary-phase approximation power law "
            f"|h~(f)| &prop; f⁻⁷/⁶ with sharp cutoff at Schwarzschild ISCO (f_ISCO = {format_float(wf['f_isco'], 1)} Hz). "
            f"Above f_ISCO, strain is identically zero."
        )

    # -------------------------------------------------------------------------
    # TAB 2: FOURIER PHASE TRAJECTORY
    # -------------------------------------------------------------------------
    with tab_phase:
        st.latex(r"\Psi(f) = 2\pi f t_c - \phi_c - \frac{\pi}{4} + \frac{3}{128\eta}(\pi M f)^{-5/3} \sum_{i=0}^7 \alpha_i (\pi M f)^{i/3}")

        fig_phase = go.Figure()
        fig_phase.add_trace(
            go.Scatter(
                x=active_freqs,
                y=active_unwrapped,
                mode="lines",
                name="Unwrapped phase",
                line=dict(color=tokens["accent"], width=2.0),
                hovertemplate="f = %{x:.1f} Hz<br>Ψ(f) = %{y:.1f} rad<extra></extra>",
            )
        )

        layout_phase = dict(base_layout)
        layout_phase.update({
            "xaxis": {
                **base_layout["xaxis"],
                "title": {"text": "Frequency f (Hz)", "font": base_layout["xaxis"]["title"]["font"]},
            },
            "yaxis": {
                **base_layout["yaxis"],
                "title": {"text": "Unwrapped Fourier phase Ψ(f) (rad)", "font": base_layout["yaxis"]["title"]["font"]},
            },
            "height": 410,
            "showlegend": False,
        })
        fig_phase.update_layout(layout_phase)
        st.plotly_chart(fig_phase, use_container_width=True)

        st.caption(
            f"Continuous Fourier phase evolution. The slope dΨ/df incorporates both the negative post-Newtonian "
            f"inspiral chirp evolution and the linear arrival time gradient -2π t_c "
            f"(slope = {format_float(-2.0 * math.pi * tc_val, 2)} rad/Hz at t_c = {format_float(tc_val, 2)} s)."
        )

    # -------------------------------------------------------------------------
    # TAB 3: INSPIRAL CHIRP TIME DIAGNOSTIC
    # -------------------------------------------------------------------------
    with tab_chirp:
        st.latex(r"\tau(f) = t_c - t(f) = -\frac{1}{2\pi} \frac{\mathrm{d}\Psi_{\mathrm{int}}}{\mathrm{d}f} \approx \frac{5}{256} \mathcal{M}_c^{-5/3} (\pi f)^{-8/3}")

        tau_grid = np.linspace(20.0, min(float(wf["f_isco"]) - 0.5, 300.0), 120)
        eval_f, tau_newton, tau_35pn = compute_chirp_time_series(
            tau_grid, wf["m1_msun"], wf["m2_msun"], mc_val, wf["f_isco"]
        )

        if eval_f.size > 0:
            fig_chirp = go.Figure()
            fig_chirp.add_trace(
                go.Scatter(
                    x=eval_f,
                    y=tau_newton,
                    mode="lines",
                    name="Newtonian leading order (f⁻⁸/³)",
                    line=dict(color=tokens["text_secondary"], dash="dash", width=1.8),
                    hovertemplate="f = %{x:.1f} Hz<br>τ_N = %{y:.4f} s<extra></extra>",
                )
            )
            fig_chirp.add_trace(
                go.Scatter(
                    x=eval_f,
                    y=tau_35pn,
                    mode="lines",
                    name="Full 3.5PN inspiral time",
                    line=dict(color=tokens["accent"], width=2.0),
                    hovertemplate="f = %{x:.1f} Hz<br>τ_3.5PN = %{y:.4f} s<extra></extra>",
                )
            )

            layout_chirp = dict(base_layout)
            layout_chirp.update({
                "xaxis": {
                    **base_layout["xaxis"],
                    "title": {"text": "Frequency f (Hz)", "font": base_layout["xaxis"]["title"]["font"]},
                },
                "yaxis": {
                    **base_layout["yaxis"],
                    "title": {"text": "Time to coalescence τ(f) (s)", "font": base_layout["yaxis"]["title"]["font"]},
                },
                "height": 410,
                "legend": {
                    **base_layout["legend"],
                    "x": 0.45,
                    "y": 0.95,
                },
            })
            fig_chirp.update_layout(layout_chirp)
            st.plotly_chart(fig_chirp, use_container_width=True)

            st.caption(
                f"Time remaining until coalescence from frequency f up to ISCO. "
                f"At f = 20 Hz, remaining inspiral duration is τ_3.5PN = {format_float(tau_35pn[0], 3)} s "
                f"(Newtonian: {format_float(tau_newton[0], 3)} s)."
            )
        else:
            st.warning("ISCO cutoff is below the active evaluation range.")

    # -------------------------------------------------------------------------
    # TAB 4: REAL AND IMAGINARY QUADRATURES
    # -------------------------------------------------------------------------
    with tab_quad:
        st.latex(r"\tilde{h}(f) = \mathrm{Re}[\tilde{h}(f)] + i\,\mathrm{Im}[\tilde{h}(f)]")

        quad_mask = (active_freqs >= 25.0) & (active_freqs <= min(60.0, float(wf["f_isco"])))
        q_freqs = active_freqs[quad_mask]
        q_real = np.real(active_htilde[quad_mask])
        q_imag = np.imag(active_htilde[quad_mask])

        fig_quad = go.Figure()
        fig_quad.add_trace(
            go.Scatter(
                x=q_freqs,
                y=q_real,
                mode="lines",
                name="Cosine quadrature Re[h~(f)]",
                line=dict(color=tokens["accent"], width=1.8),
            )
        )
        fig_quad.add_trace(
            go.Scatter(
                x=q_freqs,
                y=q_imag,
                mode="lines",
                name="Sine quadrature Im[h~(f)]",
                line=dict(color=tokens["text_secondary"], dash="dot", width=1.5),
            )
        )

        layout_quad = dict(base_layout)
        layout_quad.update({
            "xaxis": {
                **base_layout["xaxis"],
                "title": {"text": "Frequency f (Hz)", "font": base_layout["xaxis"]["title"]["font"]},
            },
            "yaxis": {
                **base_layout["yaxis"],
                "title": {"text": "Strain component (Hz⁻¹)", "font": base_layout["yaxis"]["title"]["font"]},
            },
            "height": 410,
            "legend": {
                **base_layout["legend"],
                "x": 0.03,
                "y": 0.95,
            },
        })
        fig_quad.update_layout(layout_quad)
        st.plotly_chart(fig_quad, use_container_width=True)

        st.caption(
            "Cosine and sine strain quadratures over the [25, 60] Hz interval. "
            "Interference oscillations reflect rapid phase accumulation prior to coalescence."
        )

# -----------------------------------------------------------------------------
# SECTION 3: MODEL BOUNDARIES & ASSUMPTIONS
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="gwlab-card" style="margin-top: 1.0rem;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid {tokens['border_subtle']}; padding-bottom: 0.35rem; margin-bottom: 0.5rem;">
            <span class="gwlab-card-title" style="margin: 0;">Physical Model Assumptions & Boundaries</span>
            <span class="gwlab-badge">TaylorF2 SPA</span>
        </div>
        <div style="font-size: 0.82rem; color: {tokens['text_secondary']}; line-height: 1.5;">
            Evaluates <code>src/gw_inference/waveform.py</code> in pure deterministic mode.
            Assumes quasi-circular, nonspinning compact binary coalescence under the stationary-phase approximation.
            Terminated abruptly at Schwarzschild ISCO (<em>f</em><sub>ISCO</sub> = 4397.17 / <em>M</em> Hz);
            merger and ringdown phenomenological physics are excluded.
            No likelihood evaluations, sampling engines, or noise processes are invoked.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
