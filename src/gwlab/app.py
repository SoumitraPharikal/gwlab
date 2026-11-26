"""GWLab — Research-Grade Gravitational-Wave Interactive Analysis Laboratory.

Application entry point and navigation router.
"""

from pathlib import Path
import streamlit as st

# Application Configuration
st.set_page_config(
    page_title="GWLab — Gravitational-Wave Analysis Laboratory",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)

PAGES_DIR = Path(__file__).parent / "pages"

# Define Navigation Structure using Native st.Page with Sentence-Case Titles and No Emojis
nav_sections = {
    "Laboratory": [
        st.Page(
            str(PAGES_DIR / "1_overview.py"),
            title="Overview",
            default=True,
        ),
        st.Page(
            str(PAGES_DIR / "2_signal_lab.py"),
            title="Waveforms",
        ),
        st.Page(
            str(PAGES_DIR / "3_noise_and_psd.py"),
            title="Noise and PSD",
        ),
    ],
    "Analysis": [
        st.Page(
            str(PAGES_DIR / "4_matched_filtering.py"),
            title="Matched filtering",
        ),
        st.Page(
            str(PAGES_DIR / "5_bayesian_inference.py"),
            title="Inference",
        ),
        st.Page(
            str(PAGES_DIR / "6_experiment_explorer.py"),
            title="Experiments",
        ),
    ],
    "Documentation": [
        st.Page(
            str(PAGES_DIR / "7_research_notes.py"),
            title="Methods and provenance",
        ),
    ],
}

# Sidebar metadata branding - Minimalist Editorial Wordmark
with st.sidebar:
    st.markdown(
        """
        <div style="padding-bottom: 0.6rem; border-bottom: 1px solid rgba(128, 128, 128, 0.2); margin-bottom: 0.8rem;">
            <div style="font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem; font-weight: 500; letter-spacing: -0.01em;">
                GWLab
            </div>
            <div style="font-size: 0.76rem; color: #7E7B74; font-family: 'IBM Plex Mono', monospace; margin-top: 0.15rem;">
                v0.1.0 &middot; Reference archive
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Run Selected Page
current_page = st.navigation(nav_sections)
current_page.run()
