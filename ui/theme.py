"""
theme.py — Injects custom CSS + Google Fonts into the Streamlit app
for a polished, light/dark-mode-aware research dashboard aesthetic.
"""
import streamlit as st


def get_dark_theme_css():
    """Return CSS for dark theme"""
    return """
    /* ── Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Overrides ────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* ── Main Canvas ─────────────────────────────────────── */
    [data-testid="stAppViewContainer"] {
        background: #0d1117 !important;
        color: #e6edf3 !important;
    }
    
    .main {
        background: #0d1117 !important;
        color: #e6edf3 !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: #0d1117 !important;
        color: #e6edf3 !important;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #161b22 100%) !important;
        border-right: 1px solid #21262d !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h5 {
        color: #e6edf3 !important;
    }

    /* ── Metric Cards ────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b2280, #1c222b80) !important;
        border: 1px solid #30363d !important;
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-weight: 700 !important;
    }

    /* ── Expander styling ────────────────────────────────── */
    details[data-testid="stExpander"] {
        background: #161b2260;
        border: 1px solid #30363d;
        border-radius: 10px;
    }
    details[data-testid="stExpander"] summary {
        font-weight: 600;
    }

    /* ── Container / Card blocks ─────────────────────────── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        border: 1px solid #30363d;
        border-radius: 12px;
        background: #0d111780;
    }

    /* ── Progress bars ───────────────────────────────────── */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #238636, #3fb950) !important;
        border-radius: 6px;
    }

    /* ── Section headers ─────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .section-header .icon {
        font-size: 1.4rem;
        background: linear-gradient(135deg, #58a6ff20, #58a6ff10);
        border: 1px solid #58a6ff30;
        border-radius: 10px;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .section-header .title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e6edf3;
        letter-spacing: -0.3px;
    }
    .section-header .subtitle {
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 0.3px;
    }

    /* ── Score ring ───────────────────────────────────────── */
    .score-ring {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
    }
    .score-ring .value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #e6edf3;
        line-height: 1;
    }
    .score-ring .label {
        font-size: 0.65rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Paper cards ──────────────────────────────────────── */
    .paper-card {
        background: linear-gradient(135deg, #161b2280, #1c222b80);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
        transition: border-color 0.2s ease;
    }
    .paper-card:hover {
        border-color: #58a6ff;
    }
    .paper-card .paper-title {
        font-weight: 700;
        font-size: 0.9rem;
        color: #e6edf3;
        margin-bottom: 6px;
    }
    .paper-card .paper-meta {
        font-size: 0.72rem;
        color: #8b949e;
        margin-bottom: 8px;
    }
    .paper-card .paper-meta .venue-badge {
        background: #58a6ff15;
        color: #58a6ff;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.68rem;
    }
    .paper-card .paper-abstract {
        font-size: 0.82rem;
        color: #c9d1d9;
        line-height: 1.65;
        border-left: 3px solid #30363d;
        padding-left: 12px;
    }

    /* ── Output field card ────────────────────────────────── */
    .output-field {
        background: #0d111780;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .output-field .field-name {
        font-size: 0.7rem;
        font-weight: 700;
        color: #58a6ff;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .output-field .field-value {
        font-size: 0.88rem;
        color: #c9d1d9;
        line-height: 1.6;
    }

    /* ── Empty state ─────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 6rem 2rem;
    }
    .empty-state .icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    .empty-state .heading {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 0.5rem;
    }
    .empty-state .body {
        font-size: 0.9rem;
        color: #8b949e;
        max-width: 420px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Score badge (inline) ────────────────────────────── */
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .score-badge.excellent { background: #23863620; color: #3fb950; border: 1px solid #23863640; }
    .score-badge.medium { background: #d2992220; color: #d29922; border: 1px solid #d2992240; }
    .score-badge.poor { background: #da363420; color: #f85149; border: 1px solid #da363440; }
    """


def get_bright_theme_css():
    """Return CSS for bright theme"""
    return """
    /* ── Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Overrides ────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* ── Main Canvas ─────────────────────────────────────── */
    [data-testid="stAppViewContainer"] {
        background: #ffffff !important;
        color: #24292f !important;
    }
    
    .main {
        background: #ffffff !important;
        color: #24292f !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: #ffffff !important;
        color: #24292f !important;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f6f8fb 100%) !important;
        border-right: 1px solid #d0d7de !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h5 {
        color: #24292f !important;
    }

    /* ── Metric Cards ────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f6f8fb80, #eaeef280) !important;
        border: 1px solid #d0d7de !important;
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #0969da;
    }
    div[data-testid="stMetric"] label {
        color: #57606a !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #24292f !important;
        font-weight: 700 !important;
    }

    /* ── Expander styling ────────────────────────────────── */
    details[data-testid="stExpander"] {
        background: #f6f8fb60;
        border: 1px solid #d0d7de;
        border-radius: 10px;
    }
    details[data-testid="stExpander"] summary {
        font-weight: 600;
        color: #24292f;
    }

    /* ── Container / Card blocks ─────────────────────────── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        border: 1px solid #d0d7de;
        border-radius: 12px;
        background: #ffffff80;
    }

    /* ── Progress bars ───────────────────────────────────── */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #1a7f37, #2da44e) !important;
        border-radius: 6px;
    }

    /* ── Section headers ─────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .section-header .icon {
        font-size: 1.4rem;
        background: linear-gradient(135deg, #0969da20, #0969da10);
        border: 1px solid #0969da30;
        border-radius: 10px;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .section-header .title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #24292f;
        letter-spacing: -0.3px;
    }
    .section-header .subtitle {
        font-size: 0.75rem;
        color: #57606a;
        letter-spacing: 0.3px;
    }

    /* ── Score ring ───────────────────────────────────────── */
    .score-ring {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
    }
    .score-ring .value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #24292f;
        line-height: 1;
    }
    .score-ring .label {
        font-size: 0.65rem;
        color: #57606a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Paper cards ──────────────────────────────────────── */
    .paper-card {
        background: linear-gradient(135deg, #f6f8fb80, #eaeef280);
        border: 1px solid #d0d7de;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
        transition: border-color 0.2s ease;
    }
    .paper-card:hover {
        border-color: #0969da;
    }
    .paper-card .paper-title {
        font-weight: 700;
        font-size: 0.9rem;
        color: #24292f;
        margin-bottom: 6px;
    }
    .paper-card .paper-meta {
        font-size: 0.72rem;
        color: #57606a;
        margin-bottom: 8px;
    }
    .paper-card .paper-meta .venue-badge {
        background: #0969da15;
        color: #0969da;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.68rem;
    }
    .paper-card .paper-abstract {
        font-size: 0.82rem;
        color: #424a53;
        line-height: 1.65;
        border-left: 3px solid #d0d7de;
        padding-left: 12px;
    }

    /* ── Output field card ────────────────────────────────── */
    .output-field {
        background: #ffffff80;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .output-field .field-name {
        font-size: 0.7rem;
        font-weight: 700;
        color: #0969da;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .output-field .field-value {
        font-size: 0.88rem;
        color: #424a53;
        line-height: 1.6;
    }

    /* ── Empty state ─────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 6rem 2rem;
    }
    .empty-state .icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    .empty-state .heading {
        font-size: 1.4rem;
        font-weight: 700;
        color: #24292f;
        margin-bottom: 0.5rem;
    }
    .empty-state .body {
        font-size: 0.9rem;
        color: #57606a;
        max-width: 420px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Score badge (inline) ────────────────────────────── */
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .score-badge.excellent { background: #1a7f3720; color: #1a7f37; border: 1px solid #1a7f3740; }
    .score-badge.medium { background: #9e613220; color: #9e6132; border: 1px solid #9e613240; }
    .score-badge.poor { background: #da363420; color: #cf222e; border: 1px solid #da363440; }
    """


def inject_theme(theme_mode="bright"):
    """Inject theme CSS based on theme_mode parameter"""
    if theme_mode == "dark":
        css = get_dark_theme_css()
    else:
        css = get_bright_theme_css()
    
    # Use unique ID to force CSS refresh on theme change
    theme_id = f"theme-style-{theme_mode}"
    st.markdown(
        f'<style id="{theme_id}">{css}</style>',
        unsafe_allow_html=True
    )
