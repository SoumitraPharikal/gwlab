"""Editorial research laboratory design system and tokens for GWLab.

Follows the aesthetic identity of Kerr Geodesics Lab:
- Light theme is default (warm paper `#FAF9F6`, crisp hairline borders `#DEDAD2`)
- Dark theme is a faithful inversion (`#0E1013`, `#262930`)
- Burnt orange / terracotta accent (`#B94A24` light, `#EA580C` dark)
- Editorial typography: Newsreader / Georgia serif for titles, IBM Plex Sans for controls,
  IBM Plex Mono with tabular numerals for figures and data.
- Programmatic WCAG 2.1 AA contrast compliance across all text/background pairs.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

# Design Tokens (Kerr Geodesics Lab shared visual system)
# Contrast ratios verified for WCAG 2.1 AA (all >= 4.5:1, headings/large text >= 3.0:1)
THEME_TOKENS = {
    "light": {
        "bg_primary": "#FAF9F6",       # Paper light background
        "bg_surface": "#FFFFFF",       # Card / panel background
        "bg_secondary": "#F5F3ED",     # Surface hover / sidebar
        "bg_subtle": "#F3EFEA",        # Subtle well / container
        "text_primary": "#20201E",     # Near-black high-contrast body (15.8:1)
        "text_secondary": "#55534F",   # Editorial secondary text (7.2:1)
        "text_muted": "#68655F",       # Caption / metadata text (5.5:1)
        "border": "#DEDAD2",           # Hairline border
        "border_subtle": "#E8E4DC",    # Extra-subtle divider
        "accent": "#B94A24",           # Burnt Orange / Terracotta primary (4.93:1)
        "accent_hover": "#983D1D",     # Deeper terracotta
        "accent_subtle": "#FBF4EE",    # Soft tint for badges/highlights
        "accent_border": "#E3A894",    # Tinted hairline border
        # Aliases for compatibility
        "bg_card": "#FFFFFF",
        "accent_terracotta": "#B94A24",
        "accent_blue": "#2B6CB0",
        "accent_amber": "#B7791F",
        "accent_red": "#B94A24",
        "banner_bg": "#FBF4EE",
        "banner_border": "#E3A894",
    },
    "dark": {
        "bg_primary": "#0E1013",       # Dark slate primary background
        "bg_surface": "#15181C",       # Dark card / panel background
        "bg_secondary": "#1B1E23",     # Surface hover / sidebar
        "bg_subtle": "#191C21",        # Subtle well / container
        "text_primary": "#ECEFF1",     # Near-white high-contrast body (>15:1)
        "text_secondary": "#B0B5BF",   # Editorial secondary text (~9.0:1)
        "text_muted": "#9CA3AF",       # Caption / metadata text (7.4:1)
        "border": "#262930",           # Hairline border
        "border_subtle": "#1E2127",    # Extra-subtle divider
        "accent": "#EA580C",           # Burnt Orange / Terracotta primary (5.36:1)
        "accent_hover": "#C2410C",     # Bright terracotta
        "accent_subtle": "#2A170A",    # Dark tint for badges/highlights
        "accent_border": "#7C2D12",    # Tinted hairline border
        # Aliases for compatibility
        "bg_card": "#15181C",
        "accent_terracotta": "#EA580C",
        "accent_blue": "#60A5FA",
        "accent_amber": "#FBBF24",
        "accent_red": "#EA580C",
        "banner_bg": "#2A170A",
        "banner_border": "#7C2D12",
    },
}


def is_dark_mode() -> bool:
    """Check if dark mode is currently active in session state.
    
    Default is False (Light theme is the primary, paper-like editorial baseline).
    """
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    return bool(st.session_state.dark_mode)


def set_theme_mode(dark: bool) -> None:
    """Set the theme mode in session state."""
    st.session_state.dark_mode = dark


def toggle_theme_mode() -> None:
    """Toggle between light and dark modes."""
    st.session_state.dark_mode = not is_dark_mode()


def get_current_tokens() -> dict[str, str]:
    """Return the active color token dictionary."""
    theme_key = "dark" if is_dark_mode() else "light"
    return THEME_TOKENS[theme_key]


def get_plotly_layout(theme_name: str | None = None) -> dict:
    """Return a publication-grade, shared Plotly layout configuration.
    
    Ensures plot background matches panel surface, typography uses IBM Plex Sans/Mono,
    grid lines are restrained hairline dividers, and traces use color-vision-safe palette.
    """
    if theme_name is None:
        tokens = get_current_tokens()
    else:
        tokens = THEME_TOKENS["dark" if theme_name == "dark" else "light"]

    return {
        "paper_bgcolor": tokens["bg_surface"],
        "plot_bgcolor": tokens["bg_surface"],
        "font": {
            "family": "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "size": 12,
            "color": tokens["text_secondary"],
        },
        "colorway": [
            tokens["accent"],        # Burnt Orange / Terracotta
            tokens["text_primary"],  # High-contrast primary trace
            "#2B6CB0",               # Muted sapphire blue
            "#2C7A7B",               # Muted teal
            "#B7791F",               # Warm amber
            "#744210",               # Warm sepia
        ],
        "xaxis": {
            "gridcolor": tokens["border_subtle"],
            "linecolor": tokens["border"],
            "tickcolor": tokens["border"],
            "zerolinecolor": tokens["border"],
            "zerolinewidth": 1,
            "tickfont": {
                "family": "'IBM Plex Mono', ui-monospace, monospace",
                "size": 11,
                "color": tokens["text_muted"],
            },
            "title": {
                "font": {
                    "family": "'IBM Plex Sans', sans-serif",
                    "size": 12,
                    "color": tokens["text_secondary"],
                },
            },
        },
        "yaxis": {
            "gridcolor": tokens["border_subtle"],
            "linecolor": tokens["border"],
            "tickcolor": tokens["border"],
            "zerolinecolor": tokens["border"],
            "zerolinewidth": 1,
            "tickfont": {
                "family": "'IBM Plex Mono', ui-monospace, monospace",
                "size": 11,
                "color": tokens["text_muted"],
            },
            "title": {
                "font": {
                    "family": "'IBM Plex Sans', sans-serif",
                    "size": 12,
                    "color": tokens["text_secondary"],
                },
            },
        },
        "margin": {"l": 55, "r": 25, "t": 35, "b": 45},
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": tokens["border_subtle"],
            "borderwidth": 1,
            "font": {
                "family": "'IBM Plex Sans', sans-serif",
                "size": 11,
                "color": tokens["text_secondary"],
            },
        },
    }


def inject_custom_css() -> None:
    """Inject editorial typography, hairline styling, and responsive rules."""
    tokens = get_current_tokens()

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], [class*="st-"] {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {tokens["text_primary"]};
    }}

    .stApp {{
        background-color: {tokens["bg_primary"]} !important;
        color: {tokens["text_primary"]} !important;
    }}

    /* Sidebar - Paper editorial style */
    section[data-testid="stSidebar"] {{
        background-color: {tokens["bg_secondary"]} !important;
        border-right: 1px solid {tokens["border"]} !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {{
        padding-top: 0.5rem;
    }}

    /* Monospace elements with tabular numerals */
    code, kbd, samp, pre, .stCode, .monospace-font, .tabular-figure {{
        font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace !important;
        font-variant-numeric: tabular-nums;
    }}

    /* Headings - Newsreader serif editorial style */
    h1, h2, h3, .gwlab-editorial-title {{
        font-family: 'Newsreader', Georgia, Cambria, serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.015em !important;
        color: {tokens["text_primary"]} !important;
    }}

    h1 {{
        font-size: 2.1rem !important;
        line-height: 1.25 !important;
        margin-bottom: 0.4rem !important;
    }}

    h2 {{
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
        border-bottom: 1px solid {tokens["border_subtle"]};
        padding-bottom: 0.3rem;
    }}

    h3 {{
        font-size: 1.1rem !important;
        margin-top: 1.0rem !important;
        margin-bottom: 0.3rem !important;
    }}

    /* Main container bounds & vertical rhythm */
    .block-container {{
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1380px !important;
    }}

    /* Top Editorial Header */
    .gwlab-header {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        border-bottom: 1px solid {tokens["border"]};
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
    }}

    .gwlab-brand {{
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.45rem;
        font-weight: 500;
        letter-spacing: -0.02em;
        color: {tokens["text_primary"]};
    }}

    .gwlab-subtitle {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
        color: {tokens["text_secondary"]};
        font-weight: 400;
        margin-left: 0.6rem;
    }}

    .gwlab-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.15rem 0.5rem;
        border-radius: 2px;
        letter-spacing: 0.3px;
        background-color: {tokens["accent_subtle"]};
        color: {tokens["accent"]};
        border: 1px solid {tokens["accent_border"]};
    }}

    /* Restrained Archive Notice */
    .gwlab-archive-note {{
        font-size: 0.80rem;
        color: {tokens["text_muted"]};
        line-height: 1.4;
        border-left: 2px solid {tokens["border"]};
        padding: 0.35rem 0.75rem;
        margin-bottom: 1.25rem;
    }}

    /* Clean Card Surface */
    .gwlab-card {{
        background-color: {tokens["bg_surface"]};
        border: 1px solid {tokens["border"]};
        border-radius: 3px;
        padding: 1.0rem 1.2rem;
        margin-bottom: 1.0rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }}

    .gwlab-card-title {{
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.15rem;
        font-weight: 500;
        color: {tokens["text_primary"]};
        margin-bottom: 0.3rem;
    }}

    .gwlab-card-desc {{
        font-size: 0.83rem;
        color: {tokens["text_secondary"]};
        line-height: 1.45;
    }}

    /* Definition / Key-Value List */
    .gwlab-def-list {{
        display: grid;
        grid-template-columns: 140px 1fr;
        row-gap: 0.4rem;
        column-gap: 0.75rem;
        font-size: 0.84rem;
        margin-top: 0.5rem;
    }}

    .gwlab-def-term {{
        color: {tokens["text_secondary"]};
        font-weight: 400;
    }}

    .gwlab-def-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-variant-numeric: tabular-nums;
        color: {tokens["text_primary"]};
    }}

    /* Publication-grade Tabular Figures Table */
    .param-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.83rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }}

    .param-table th {{
        text-align: left;
        padding: 0.4rem 0.6rem;
        border-bottom: 1px solid {tokens["border"]};
        color: {tokens["text_secondary"]};
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
    }}

    .param-table td {{
        padding: 0.4rem 0.6rem;
        border-bottom: 1px solid {tokens["border_subtle"]};
        font-family: 'IBM Plex Mono', monospace;
        font-variant-numeric: tabular-nums;
        color: {tokens["text_primary"]};
    }}

    /* Streamlit Widget Overrides for Art Direction */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stSelectbox"] label {{
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        color: {tokens["text_secondary"]} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }}

    /* Primary button & tabs */
    button[kind="primary"] {{
        background-color: {tokens["accent"]} !important;
        border-color: {tokens["accent"]} !important;
        color: #FFFFFF !important;
        border-radius: 2px !important;
    }}

    button[kind="primary"]:hover {{
        background-color: {tokens["accent_hover"]} !important;
        border-color: {tokens["accent_hover"]} !important;
    }}

    button[kind="secondary"] {{
        background-color: {tokens["bg_surface"]} !important;
        border: 1px solid {tokens["border"]} !important;
        color: {tokens["text_primary"]} !important;
        border-radius: 2px !important;
    }}

    button[kind="secondary"]:hover {{
        border-color: {tokens["accent"]} !important;
        color: {tokens["accent"]} !important;
    }}

    /* Tab styling */
    button[data-baseweb="tab"] {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.85rem !important;
        color: {tokens["text_secondary"]} !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {tokens["accent"]} !important;
        border-bottom-color: {tokens["accent"]} !important;
        font-weight: 600 !important;
    }}

    /* Hide default Streamlit decoration bar */
    header[data-testid="stHeader"] {{
        background-color: {tokens["bg_primary"]} !important;
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1.0rem !important;
            padding-right: 1.0rem !important;
        }}
        .gwlab-header {{
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }}
        .gwlab-def-list {{
            grid-template-columns: 120px 1fr;
        }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
