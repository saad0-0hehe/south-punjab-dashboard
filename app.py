"""Punjab Districts — MICS Longitudinal Development Analysis.

MICS-only dashboard: household-level microdata from three MICS Punjab rounds
(2011, 2014, 2017-18) covering 185,303 households across 36 districts.

Census-based pages (Overview, District Profiles, Indicators, Trends 2011–2023,
Budget Accountability, Poverty Co-Movement) are archived in archive_census2023/
and can be revived as a separate project.
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.theme import masthead
from src import theme
from src import mics_page

# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Punjab MICS · Longitudinal Development Study",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design system: CSS + chart templates (single source of truth in src/theme.py)
theme.inject_theme()
theme.register_plotly_template()


# ── Helper: Custom Metric Card ───────────────────────────────────────────────

def metric_card(label, value, delta=None, delta_type="neutral", color="blue"):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-delta {delta_type}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 1.2rem 0.75rem 0.9rem; border-bottom: 1px solid rgba(246,241,231,0.12); margin-bottom: 0.75rem;">
        <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #F6F1E7 !important; line-height: 1.2;">
            Uneven Progress
        </div>
        <div style="font-size: 0.68rem; font-weight: 600; color: #BC4B26 !important; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px;">
            Punjab · Development Atlas
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["MICS Longitudinal Study", "About & Sources"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # MICS-specific stats block
    st.markdown("""
    <div style="padding: 0 0.75rem;">
        <div style="font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.12em; color: #8A8172 !important; margin-bottom: 0.6rem; font-weight: 700;">
            The Study
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: rgba(246,241,231,0.12); border: 1px solid rgba(246,241,231,0.12);">
            <div style="background: #211B12; padding: 0.55rem 0.7rem;">
                <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #F6F1E7 !important;">3</div>
                <div style="font-size: 0.62rem; color: #8A8172 !important; text-transform: uppercase; letter-spacing: 0.06em;">MICS Rounds</div>
            </div>
            <div style="background: #211B12; padding: 0.55rem 0.7rem;">
                <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #E0784F !important;">185K</div>
                <div style="font-size: 0.62rem; color: #8A8172 !important; text-transform: uppercase; letter-spacing: 0.06em;">Households</div>
            </div>
            <div style="background: #211B12; padding: 0.55rem 0.7rem;">
                <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #F6F1E7 !important;">36</div>
                <div style="font-size: 0.62rem; color: #8A8172 !important; text-transform: uppercase; letter-spacing: 0.06em;">Districts</div>
            </div>
            <div style="background: #211B12; padding: 0.55rem 0.7rem;">
                <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #F6F1E7 !important;">2011–18</div>
                <div style="font-size: 0.62rem; color: #8A8172 !important; text-transform: uppercase; letter-spacing: 0.06em;">Survey Years</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 0.75rem; font-size: 0.64rem; color: #8A8172 !important; line-height: 1.6;">
        Source: UNICEF MICS Punjab · 2011 · 2014 · 2017-18
    </div>
    """, unsafe_allow_html=True)


# ── PAGE: MICS LONGITUDINAL STUDY (default) ──────────────────────────────────

if page == "MICS Longitudinal Study":
    mics_page.render()


# ── PAGE: ABOUT & SOURCES ────────────────────────────────────────────────────

elif page == "About & Sources":
    masthead(
        "Methodology & Provenance",
        "About This Project",
        "A data science portfolio project analyzing human development dynamics in "
        "Punjab, built on UNICEF MICS Punjab household microdata across three survey rounds.",
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, icon, name, desc in [
        (col1, "", "Python",      "Core Language"),
        (col2, "", "Pandas",      "Data Processing"),
        (col3, "", "Plotly",      "Visualizations"),
        (col4, "", "XGBoost",     "ML / SHAP"),
        (col5, "", "Streamlit",   "Dashboard"),
    ]:
        with col:
            st.markdown(f"""
            <div class="about-card">
                <div class="icon">{icon}</div>
                <h4>{name}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Objective")
        st.markdown("""
        - Construct a household-level HDI from three MICS Punjab rounds (2011, 2014, 2017-18)
        - Validate the composite index against Naveed & Gordon (2024) published district rankings
        - Test for unconditional β-convergence across 36 Punjab districts
        - Measure within-district and between-district inequality trajectories
        - Identify deprivation drivers via SHAP (corrected, exogenous-features-only model)
        - Present findings in an interactive, explorable dashboard
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### South Punjab Districts (11)")
        st.markdown("""
        | Division | Districts |
        |----------|-----------|
        | **Multan** | Multan, Lodhran, Khanewal, Vehari |
        | **Bahawalpur** | Bahawalpur, Bahawalnagar, Rahim Yar Khan |
        | **DG Khan** | DG Khan, Muzaffargarh, Layyah, Rajanpur |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Data Sources")
        st.markdown("""
        | Source | Data | Year |
        |--------|------|------|
        | **UNICEF MICS Punjab 2011** | Household microdata: assets, education, child health | 2011 |
        | **UNICEF MICS Punjab 2014** | Household microdata: assets, education, child health | 2014 |
        | **UNICEF MICS Punjab 2017-18** | Household microdata: assets, education, child health | 2017-18 |
        | **Naveed & Gordon (2024)** | Published district HDI rankings (external validation) | 2024 |
        | **PDHS 2017-18** | Pakistan Demographic & Health Survey (immunization cross-check) | 2018 |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Methodology Notes")
        st.markdown("""
        - **HDI construction**: mean of asset, education, and child-health sub-indices per household
        - **Scope**: child-bearing households (primary metric) and all households (secondary)
        - **Validation**: Spearman ρ = 0.969, Kendall τ = 0.879 vs Naveed & Gordon (2024)
        - **Convergence**: β = −0.054 (p < 0.001), λ ≈ 5.5%/yr, half-life ≈ 13 years
        - **ML model**: XGBoost on strictly exogenous features; honest R² = 0.42–0.48 (corrected from inflated 0.97)
        - **Sample**: 185,303 households across 36 Punjab districts, three survey rounds
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Author")
        st.markdown("""
        **M Saad Sadaf**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
