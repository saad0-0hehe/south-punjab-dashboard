"""
South Punjab Development Dashboard
Streamlit Application — Premium UI

An interactive dashboard analyzing socioeconomic disparities
across South Punjab districts using real government data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_data, clean_data, filter_south_punjab, filter_rest_of_punjab,
    get_summary_stats, get_district_profile, get_rankings,
    load_historical_data, load_budget_data, override_2023_anchor,
    SOUTH_PUNJAB_DISTRICTS
)
from src import historical_analyzer
from src.eda import (
    plot_literacy_comparison, plot_poverty_map, plot_gender_gap,
    plot_enrollment_trends, plot_health_indicators,
    plot_correlation_heatmap, plot_south_vs_rest,
    plot_division_comparison, plot_literacy_vs_poverty,
    plot_out_of_school, plot_infrastructure,
    plot_rural_urban_literacy, plot_temporal_comparison
)
from src.ml_model import (
    prepare_features, train_linear, train_ridge, find_best_alpha,
    evaluate_model, plot_predictions, plot_feature_importance, plot_residuals
)
from src.historical_viz import (
    plot_indicator_trends, plot_budget_comparison, plot_disparity_gap
)

# Page Config

st.set_page_config(
    page_title="South Punjab Development Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

   /* ─── Global ──────────────────────────────────────── */
    /* Remove the [class*="st-"] selector to prevent breaking native icons */
    html, body {
        font-family: 'Inter', sans-serif;
    }
    
    /* Safely apply Inter to typical text elements instead of everything */
    p, span, div, label {
        font-family: 'Inter', sans-serif;
    }
    
    /* Explicitly protect Streamlit's material icons */
    .material-symbols-rounded, .material-symbols-outlined {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', sans-serif !important;
    }
    
    .main .block-container {
        padding: 1.5rem 2rem 2rem;
        max-width: 1300px;
    }
    .main { background: #F8FAFC; }

    /* ─── Hide default Streamlit branding ─────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }

    /* ─── Sidebar ─────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        margin-bottom: 2px;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(99, 102, 241, 0.15);
        color: #E0E7FF !important;
    }
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.2) 100%);
        color: #C7D2FE !important;
        font-weight: 600;
    }

    /* ─── Headers ─────────────────────────────────────── */
    h1 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        font-size: 2rem !important;
    }
    h2 {
        color: #1E293B !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* ─── Custom Metric Cards ─────────────────────────── */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        border: 1px solid #E2E8F0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08), 0 4px 8px rgba(0,0,0,0.04);
        border-color: #C7D2FE;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        border-radius: 16px 16px 0 0;
    }
    .metric-card.blue::before { background: linear-gradient(90deg, #6366F1, #818CF8); }
    .metric-card.red::before { background: linear-gradient(90deg, #EF4444, #F87171); }
    .metric-card.green::before { background: linear-gradient(90deg, #10B981, #34D399); }
    .metric-card.amber::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .metric-card.purple::before { background: linear-gradient(90deg, #8B5CF6, #A78BFA); }
    .metric-card.teal::before { background: linear-gradient(90deg, #14B8A6, #2DD4BF); }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        display: inline-block;
    }
    .metric-delta.negative {
        color: #DC2626;
        background: #FEE2E2;
    }
    .metric-delta.positive {
        color: #059669;
        background: #D1FAE5;
    }
    .metric-delta.neutral {
        color: #D97706;
        background: #FEF3C7;
    }

    /* ─── Page Banner ─────────────────────────────────── */
    .page-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 40%, #1E293B 100%);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .page-banner::after {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 60%; height: 200%;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 60%);
        border-radius: 50%;
    }
    .page-banner h1 {
        color: white !important;
        font-size: 1.8rem !important;
        margin-bottom: 0.25rem;
        position: relative;
        z-index: 1;
    }
    .page-banner p {
        color: #94A3B8;
        font-size: 1rem;
        max-width: 700px;
        position: relative;
        z-index: 1;
    }
    .page-banner .badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #C7D2FE;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 6px;
        margin-bottom: 8px;
    }

    /* ─── Section Cards ───────────────────────────────── */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .section-card h3 {
        margin-top: 0 !important;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #F1F5F9;
        margin-bottom: 1rem;
    }

    /* ─── Tabs ────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #F1F5F9;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.9rem;
        background: transparent;
        border: none;
        color: #64748B;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        color: #4F46E5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ─── DataFrames ──────────────────────────────────── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ─── Selectbox, Slider ───────────────────────────── */
    .stSelectbox > div > div,
    .stSlider > div { border-radius: 10px; }

    /* ─── Divider ─────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
        margin: 1.5rem 0;
    }

    /* ─── Insight Box ─────────────────────────────────── */
    .insight-box {
        background: linear-gradient(135deg, #EEF2FF, #F8FAFC);
        border-left: 4px solid #6366F1;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #334155;
    }
    .insight-box strong { color: #4F46E5; }

    /* ─── About page cards ────────────────────────────── */
    .about-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
        transition: all 0.3s ease;
    }
    .about-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }
    .about-card .icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .about-card h4 { color: #1E293B; margin: 0.5rem 0 0.25rem; }
    .about-card p { color: #64748B; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# Helper: Custom Metric Card

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


# Load Data (cached)

@st.cache_data
def load_and_clean():
    try:
        df = load_data()
        df = clean_data(df)
        return df
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

@st.cache_data
def train_models(_df):
    try:
        data = prepare_features(_df)
        lr = train_linear(data["X_train"], data["y_train"])
        best_alpha, alpha_df = find_best_alpha(data["X_train"], data["y_train"])
        ridge = train_ridge(data["X_train"], data["y_train"], alpha=best_alpha)
        return lr, ridge, data, best_alpha, alpha_df
    except Exception as e:
        st.error(f"❌ Error training models: {e}")
        st.stop()

df = load_and_clean()
sp_df = filter_south_punjab(df)
rest_df = filter_rest_of_punjab(df)

# Pre-compute dynamic insights (used across pages)
_worst_lit   = df.loc[df["literacy_rate"].idxmin()]
_best_lit    = df.loc[df["literacy_rate"].idxmax()]
_worst_pov   = df.loc[df["poverty_headcount"].idxmax()]
_best_pov    = df.loc[df["poverty_headcount"].idxmin()]
_worst_imm   = sp_df.loc[sp_df["immunization_coverage"].idxmin()]
_worst_water = sp_df.loc[sp_df["clean_water_access"].idxmin()]

# Count how many of top-10 most impoverished are South Punjab
_top10_pov = get_rankings(df, "poverty_headcount", ascending=False).head(10)
_sp_in_top10 = _top10_pov[_top10_pov["region"] == "South Punjab"].shape[0]

# Enrollment drop-off: avg middle/primary ratio in South Punjab
_sp_enroll_ratio = (sp_df["middle_enrollment_rate"].mean() /
                    sp_df["primary_enrollment_rate"].mean() * 100)

# New data-driven insights
_sp_unemp = sp_df["unemployment_rate"].mean() if "unemployment_rate" in sp_df.columns else None
_rest_unemp = rest_df["unemployment_rate"].mean() if "unemployment_rate" in rest_df.columns else None
_sp_oos = sp_df["out_of_school_rate"].mean() if "out_of_school_rate" in sp_df.columns else None
_rest_oos = rest_df["out_of_school_rate"].mean() if "out_of_school_rate" in rest_df.columns else None
_sp_internet = sp_df["internet_access"].mean() if "internet_access" in sp_df.columns else None
_rest_internet = rest_df["internet_access"].mean() if "internet_access" in rest_df.columns else None


# Sidebar

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size: 2rem; margin-bottom: 0.25rem;">📊</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #E2E8F0 !important; letter-spacing: -0.02em;">
            South Punjab
        </div>
        <div style="font-size: 0.75rem; font-weight: 400; color: #64748B !important; letter-spacing: 0.05em; text-transform: uppercase;">
            Development Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🏘️ District Profiles", "📈 EDA", 
         "📅 Temporal Trends", "💰 Budget Accountability",
         "🤖 ML Predictions", "ℹ️ About"],
        label_visibility="collapsed"
    )

    # Load Historical & Budget Data, then anchor 2023 values to master
    df_hist = load_historical_data()
    if df_hist is not None:
        df_hist = override_2023_anchor(df_hist, df)
    # Load budget (nominal and inflation adjusted)
    df_budget_nom  = load_budget_data(adjust_for_inflation=False)
    df_budget_real = load_budget_data(adjust_for_inflation=True)

    st.markdown("---")

    st.markdown(f"""
    <div style="padding: 0 0.5rem;">
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #475569 !important; margin-bottom: 0.75rem; font-weight: 600;">
            Dataset Summary
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <div style="background: rgba(99,102,241,0.1); border-radius: 10px; padding: 0.6rem 0.75rem;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #C7D2FE !important;">{len(df)}</div>
                <div style="font-size: 0.65rem; color: #64748B !important;">Districts</div>
            </div>
            <div style="background: rgba(239, 68, 68, 0.1); border-radius: 10px; padding: 0.6rem 0.75rem;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #FCA5A5 !important;">{len(sp_df)}</div>
                <div style="font-size: 0.65rem; color: #64748B !important;">South Punjab</div>
            </div>
            <div style="background: rgba(16, 185, 129, 0.1); border-radius: 10px; padding: 0.6rem 0.75rem;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #6EE7B7 !important;">{len(rest_df)}</div>
                <div style="font-size: 0.65rem; color: #64748B !important;">Rest of Punjab</div>
            </div>
            <div style="background: rgba(245, 158, 11, 0.1); border-radius: 10px; padding: 0.6rem 0.75rem;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #FCD34D !important;">{len(df.columns)}</div>
                <div style="font-size: 0.65rem; color: #64748B !important;">Indicators</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 0.5rem; font-size: 0.65rem; color: #475569 !important; line-height: 1.5;">
        📄 PBS Census 2023 | PSLM | PB Budget
    </div>
    """, unsafe_allow_html=True)


# PAGE 1: OVERVIEW

if page == "🏠 Overview":
    st.markdown("""
    <div class="page-banner">
        <div class="badge">PBS CENSUS 2023</div>
        <div class="badge">CENSUS 2017</div>
        <div class="badge">PSLM 2019-20</div>
        <div class="badge">36 DISTRICTS</div>
        <div class="badge">43 INDICATORS</div>
        <h1>📊 South Punjab Development Dashboard</h1>
        <p>Analyzing socioeconomic disparities across 11 South Punjab districts
        compared to the rest of Punjab using real government data from
        PBS Census 2023, PSLM surveys, and provincial budget archives.</p>
    </div>
    """, unsafe_allow_html=True)

    sp_lit   = sp_df["literacy_rate"].mean()
    rest_lit = rest_df["literacy_rate"].mean()
    sp_pov   = sp_df["poverty_headcount"].mean()
    rest_pov = rest_df["poverty_headcount"].mean()
    sp_imm   = sp_df["immunization_coverage"].mean()
    rest_imm = rest_df["immunization_coverage"].mean()
    sp_gap   = sp_df["gender_literacy_gap"].mean()
    rest_gap = rest_df["gender_literacy_gap"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Avg Literacy (South)", f"{sp_lit:.1f}%",
                    f"▼ {abs(sp_lit - rest_lit):.1f}% vs Rest", "negative", "blue")
    with col2:
        metric_card("Avg Poverty (South)", f"{sp_pov:.1f}%",
                    f"▲ +{sp_pov - rest_pov:.1f}% vs Rest", "negative", "red")
    with col3:
        metric_card("Immunization (South)", f"{sp_imm:.1f}%",
                    f"▼ {abs(sp_imm - rest_imm):.1f}% vs Rest", "negative", "teal")
    with col4:
        metric_card("Gender Literacy Gap", f"{sp_gap:.1f}%",
                    f"▲ +{sp_gap - rest_gap:.1f}% vs Rest", "negative", "amber")

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Second row of metrics — new data dimensions
    col5, col6, col7, col8 = st.columns(4)
    if _sp_unemp is not None:
        with col5:
            metric_card("Unemployment (South)", f"{_sp_unemp:.1f}%",
                        f"▲ +{_sp_unemp - _rest_unemp:.1f}% vs Rest", "negative", "purple")
    if _sp_oos is not None:
        with col6:
            metric_card("Out of School (South)", f"{_sp_oos:.1f}%",
                        f"▲ +{_sp_oos - _rest_oos:.1f}% vs Rest", "negative", "red")
    if _sp_internet is not None:
        with col7:
            metric_card("Internet Access (South)", f"{_sp_internet:.1f}%",
                        f"▼ {abs(_sp_internet - _rest_internet):.1f}% vs Rest", "negative", "blue")
    with col8:
        sp_sanit = sp_df["sanitation_access"].mean() if "sanitation_access" in sp_df.columns else 0
        rest_sanit = rest_df["sanitation_access"].mean() if "sanitation_access" in rest_df.columns else 0
        metric_card("Sanitation (South)", f"{sp_sanit:.1f}%",
                    f"▼ {abs(sp_sanit - rest_sanit):.1f}% vs Rest", "negative", "green")

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Dynamic insight
    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Key Insight:</strong> The most impoverished district is
        <strong>{_worst_pov['district']}</strong>
        ({_worst_pov['poverty_headcount']:.1f}% poverty) while
        <strong>{_best_pov['district']}</strong> has the lowest
        ({_best_pov['poverty_headcount']:.1f}%). South Punjab averages
        <strong>{sp_pov - rest_pov:.1f}% higher</strong> poverty,
        <strong>{_sp_unemp - _rest_unemp:.1f}% higher</strong> unemployment, and
        <strong>{abs(_sp_internet - _rest_internet):.1f}% lower</strong> internet access
        than the rest of Punjab.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🔴 Most Impoverished Districts")
        poverty_rank = get_rankings(df, "poverty_headcount", ascending=False)
        st.dataframe(
            poverty_rank.head(10).style.background_gradient(
                subset=["poverty_headcount"], cmap="Reds"),
            width="stretch", hide_index=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🟢 Most Literate Districts")
        lit_rank = get_rankings(df, "literacy_rate", ascending=False)
        st.dataframe(
            lit_rank.head(10).style.background_gradient(
                subset=["literacy_rate"], cmap="Greens"),
            width="stretch", hide_index=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Literacy vs Poverty — All Punjab Districts")
    fig = plot_literacy_vs_poverty(df)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# PAGE 2: DISTRICT PROFILES

elif page == "🏘️ District Profiles":
    st.markdown("""
    <div class="page-banner">
        <h1>🏘️ District Profiles</h1>
        <p>Explore detailed socioeconomic indicators for any Punjab district.</p>
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox("Select a District", sorted(df["district"].unique()))
    profile  = get_district_profile(df, selected)

    is_south   = profile["region"] == "South Punjab"
    region_color = "#EF4444" if is_south else "#6366F1"
    region_bg    = "#FEE2E2" if is_south else "#EEF2FF"

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0;">
        <h2 style="margin: 0;">{profile['district']}</h2>
        <span style="background: {region_bg}; color: {region_color}; padding: 4px 12px;
              border-radius: 20px; font-size: 0.8rem; font-weight: 600;">{profile['region']}</span>
        <span style="color: #64748B; font-size: 0.9rem;">{profile['division']} Division</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Population", f"{profile['population_2023']:,.0f}", color="blue")
    with c2: metric_card("Area", f"{profile['area_sqkm']:,.0f} km²", color="blue")
    with c3: metric_card("Density", f"{profile['density_per_sqkm']:,.0f} /km²", color="blue")

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        lit_vs     = profile['literacy_rate'] - df['literacy_rate'].mean()
        delta_type = "positive" if lit_vs >= 0 else "negative"
        metric_card("Literacy Rate", f"{profile['literacy_rate']:.1f}%",
                    f"{'▲' if lit_vs >= 0 else '▼'} {abs(lit_vs):.1f}% vs avg", delta_type, "green")
    with c5: metric_card("Male Literacy",   f"{profile['male_literacy']:.1f}%",   color="green")
    with c6: metric_card("Female Literacy", f"{profile['female_literacy']:.1f}%", color="green")
    with c7: metric_card("Gender Gap", f"{profile['gender_literacy_gap']:.1f}%",
                         "Higher = worse", "neutral", "amber")

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    c8, c9, c10, c11 = st.columns(4)
    with c8:
        pov_vs     = profile['poverty_headcount'] - df['poverty_headcount'].mean()
        delta_type = "negative" if pov_vs > 0 else "positive"
        metric_card("Poverty", f"{profile['poverty_headcount']:.1f}%",
                    f"{'▲' if pov_vs > 0 else '▼'} {abs(pov_vs):.1f}% vs avg", delta_type, "red")
    with c9:  metric_card("MPI Score",     f"{profile['mpi_score']:.3f}",              color="red")
    with c10: metric_card("Immunization",  f"{profile['immunization_coverage']:.0f}%", color="teal")
    with c11: metric_card("Clean Water",   f"{profile['clean_water_access']:.1f}%",    color="teal")

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # New indicators row
    c12, c13, c14, c15 = st.columns(4)
    if "unemployment_rate" in profile.index:
        with c12:
            unemp_vs = profile['unemployment_rate'] - df['unemployment_rate'].mean()
            delta_type = "negative" if unemp_vs > 0 else "positive"
            metric_card("Unemployment", f"{profile['unemployment_rate']:.1f}%",
                        f"{'▲' if unemp_vs > 0 else '▼'} {abs(unemp_vs):.1f}% vs avg", delta_type, "purple")
    if "sanitation_access" in profile.index:
        with c13: metric_card("Sanitation",  f"{profile['sanitation_access']:.1f}%", color="green")
    if "internet_access" in profile.index:
        with c14: metric_card("Internet",    f"{profile['internet_access']:.1f}%",  color="blue")
    if "out_of_school_rate" in profile.index:
        with c15:
            oos_vs = profile['out_of_school_rate'] - df['out_of_school_rate'].mean()
            delta_type = "negative" if oos_vs > 0 else "positive"
            metric_card("Out of School", f"{profile['out_of_school_rate']:.1f}%",
                        f"{'▲' if oos_vs > 0 else '▼'} {abs(oos_vs):.1f}% vs avg", delta_type, "red")

    st.markdown("---")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Compared to Averages")

    indicators = ["literacy_rate", "poverty_headcount", "unemployment_rate",
                  "immunization_coverage", "clean_water_access", "sanitation_access",
                  "primary_enrollment_rate", "out_of_school_rate",
                  "electricity_access", "internet_access"]
    compare_data = []
    for ind in indicators:
        compare_data.append({
            "Indicator": ind.replace("_", " ").title(),
            f"📍 {selected}": profile[ind],
            "🔴 South Punjab Avg": sp_df[ind].mean(),
            "🟣 All Punjab Avg": df[ind].mean(),
        })

    compare_df = pd.DataFrame(compare_data)
    st.dataframe(compare_df.style.format({
        f"📍 {selected}": "{:.1f}",
        "🔴 South Punjab Avg": "{:.1f}",
        "🟣 All Punjab Avg": "{:.1f}"
    }).background_gradient(subset=[f"📍 {selected}"], cmap="Blues"),
    use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# PAGE 3: EDA

elif page == "📈 EDA":
    st.markdown("""
    <div class="page-banner">
        <h1>📈 Exploratory Data Analysis</h1>
        <p>Dive deep into the socioeconomic indicators with interactive charts
        comparing South Punjab against the rest of Punjab.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📚 Literacy", "💰 Poverty", "🏫 Education", "🏥 Health", "🏗️ Infrastructure", "📅 Temporal", "🔗 Correlations"
    ])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### District-wise Literacy Rates")
        # Dynamic insight
        st.markdown(f"""<div class="insight-box">
            💡 South Punjab districts cluster at the <strong>bottom</strong> of literacy rankings,
            with <strong>{_worst_lit['district']}</strong>
            ({_worst_lit['literacy_rate']:.1f}%) having the lowest literacy in all of Punjab.
        </div>""", unsafe_allow_html=True)
        fig = plot_literacy_comparison(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Gender Literacy Gap — Worst Districts")
        n_districts = st.slider("Number of districts to show", 5, 20, 15, key="gender_n")
        fig = plot_gender_gap(df, top_n=n_districts)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### District-wise Poverty Headcount")
        # Dynamic insight
        st.markdown(f"""<div class="insight-box">
            💡 <strong>{_sp_in_top10} out of the top 10</strong> most impoverished districts
            in Punjab belong to South Punjab, with DG Khan division being the worst affected.
        </div>""", unsafe_allow_html=True)
        fig = plot_poverty_map(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### South Punjab vs Rest of Punjab")
        fig = plot_south_vs_rest(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### School Enrollment — South Punjab")
        # Dynamic insight
        st.markdown(f"""<div class="insight-box">
            💡 On average, only <strong>{_sp_enroll_ratio:.0f} out of 100</strong> primary
            students in South Punjab advance to middle school — a significant dropout gap
            that worsens in DG Khan division.
        </div>""", unsafe_allow_html=True)
        fig = plot_enrollment_trends(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Out-of-School Children (5-16 years)")
        if _sp_oos is not None:
            st.markdown(f"""<div class="insight-box">
                💡 South Punjab has an average out-of-school rate of <strong>{_sp_oos:.1f}%</strong>
                compared to <strong>{_rest_oos:.1f}%</strong> in the rest of Punjab — a gap of
                <strong>{_sp_oos - _rest_oos:.1f} percentage points</strong>.
            </div>""", unsafe_allow_html=True)
        fig = plot_out_of_school(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Division-wise Comparison")
        indicator = st.selectbox(
            "Choose indicator",
            ["primary_enrollment_rate", "middle_enrollment_rate", "literacy_rate",
             "out_of_school_rate", "unemployment_rate"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="div_indicator"
        )
        fig = plot_division_comparison(df, indicator=indicator)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Health Indicators — South Punjab")
        # Dynamic insight
        st.markdown(f"""<div class="insight-box">
            💡 <strong>{_worst_imm['district']}</strong> has the lowest immunization coverage
            ({_worst_imm['immunization_coverage']:.0f}%) and
            <strong>{_worst_water['district']}</strong> has the worst clean water access
            ({_worst_water['clean_water_access']:.1f}%) in South Punjab.
        </div>""", unsafe_allow_html=True)
        fig = plot_health_indicators(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Infrastructure Access — South Punjab")
        _sp_sanit = sp_df["sanitation_access"].mean() if "sanitation_access" in sp_df.columns else 0
        _sp_inet = sp_df["internet_access"].mean() if "internet_access" in sp_df.columns else 0
        st.markdown(f"""<div class="insight-box">
            💡 South Punjab districts average only <strong>{_sp_sanit:.1f}%</strong> sanitation access
            and <strong>{_sp_inet:.1f}%</strong> internet penetration. Districts like
            <strong>Rajanpur</strong> and <strong>DG Khan</strong> are the worst affected.
        </div>""", unsafe_allow_html=True)
        fig = plot_infrastructure(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Rural vs Urban Literacy — South Punjab")
        fig = plot_rural_urban_literacy(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Literacy Rate Change: 2017 → 2023")
        if "literacy_change" in df.columns:
            sp_change = sp_df["literacy_change"].mean()
            rest_change = rest_df["literacy_change"].mean()
            st.markdown(f"""<div class="insight-box">
                💡 South Punjab literacy improved by an average of <strong>{sp_change:.1f} percentage points</strong>
                from 2017 to 2023, compared to <strong>{rest_change:.1f} pp</strong> for the rest of Punjab.
                While both regions improved, the gap remains significant.
            </div>""", unsafe_allow_html=True)
        fig = plot_temporal_comparison(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab7:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Correlation Matrix")
        _corr_val = df["literacy_rate"].corr(df["poverty_headcount"])
        # Dynamic insight
        st.markdown(f"""<div class="insight-box">
            💡 <strong>Literacy rate</strong> and <strong>poverty headcount</strong> show a strong
            negative correlation (r = {_corr_val:.2f}), confirming that education is the
            strongest predictor of poverty across Punjab districts.
        </div>""", unsafe_allow_html=True)
        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Literacy vs Poverty Scatter")
        fig = plot_literacy_vs_poverty(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)


# PAGE 4: TEMPORAL TRENDS

elif page == "📅 Temporal Trends":
    st.markdown("""
    <div class="page-banner">
        <h1>📅 Developmental Trends (2011–2023)</h1>
        <p>Tracking the socioeconomic evolution across Survey Rounds and Census years.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_hist is not None:
        indicator = st.selectbox(
            "Select Indicator to Track",
            ["literacy_rate", "poverty_headcount", "out_of_school_rate"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("### 📍 Highlight Districts")
            selected_dists = st.multiselect(
                "Choose districts to overlay",
                options=sorted(df["district"].unique()),
                default=["Multan", "Lahore"]
            )
            
            # Growth metrics
            growth_stats = historical_analyzer.get_growth_data(df_hist, indicator)
            st.markdown("---")
            st.markdown("**Top Improvers (CAGR %)**")
            st.dataframe(growth_stats[["total_growth", "cagr"]].head(5), use_container_width=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            fig = plot_indicator_trends(df_hist, indicator, selected_dists, SOUTH_PUNJAB_DISTRICTS)
            st.pyplot(fig)
            plt.close(fig)
            
            # Historical Gap analysis — narrow chart below the main trends
            st.markdown("### 📉 Regional Performance Gap Over Time")
            gap_df = historical_analyzer.compare_historical_gaps(df_hist, indicator, SOUTH_PUNJAB_DISTRICTS)
            if gap_df is not None and "gap" in gap_df.columns:
                fig_gap, ax_gap = plt.subplots(figsize=(10, 3))
                ax_gap.plot(gap_df.index, gap_df["gap"], marker='o', color="#94A3B8", linewidth=2)
                ax_gap.axhline(0, color="#EF4444", linestyle="--", alpha=0.4)
                ax_gap.fill_between(gap_df.index, gap_df["gap"], alpha=0.12, color="#94A3B8")
                ax_gap.set_ylabel("Gap (Rest − South, pp)")
                ax_gap.set_xlabel("Year")
                ax_gap.grid(True, linestyle="--", alpha=0.3)
                fig_gap.tight_layout()
                st.pyplot(fig_gap)
                plt.close(fig_gap)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Historical data file not found.")


# PAGE 5: BUDGET ACCOUNTABILITY

elif page == "💰 Budget Accountability":
    st.markdown("""
    <div class="page-banner">
        <h1>💰 Budget & Fiscal Accountability</h1>
        <p>Analyzing the "White Papers": Comparing Promised Allocations vs. Actual Expenditure.</p>
    </div>
    """, unsafe_allow_html=True)

    adj = st.toggle("🚀 Adjust for Inflation (Real Growth)", value=True, 
                    help="Uses CPI (Consumer Price Index) to convert nominal PKR into constant 2015-16 values.")
    
    current_budget = df_budget_real if adj else df_budget_nom
    
    if current_budget is not None:
        val_col = "allocation_real_bn" if adj else "allocation_pkr_bn"
        exp_col = "expenditure_real_bn" if adj else "expenditure_pkr_bn"
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Total Allocation Trends")
            fig1 = plot_disparity_gap(current_budget, val_col)
            st.pyplot(fig1)
            plt.close(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Promised vs Actually Spent")
            fig2 = plot_budget_comparison(current_budget, val_col)
            st.pyplot(fig2)
            plt.close(fig2)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Fiscal Performance Summary")
        summary = historical_analyzer.get_budget_summary(current_budget)
        # Choose column names dynamically based on inflation toggle
        alloc_col = "allocation_real_bn" if adj and "allocation_real_bn" in summary.columns else "allocation_pkr_bn"
        exp_col_fmt = "expenditure_real_bn" if adj and "expenditure_real_bn" in summary.columns else "expenditure_pkr_bn"
        fmt_dict = {alloc_col: "{:,.1f} Bn", exp_col_fmt: "{:,.1f} Bn", "utilization_rate": "{:.1f}%"}
        st.table(summary.style.format(fmt_dict))
        
        st.info("💡 **Revised Estimates**: Represent the actual funds released and spent by the end of the fiscal year, often revealing significant under-utilization in Southern districts compared to Central Punjab.")
    else:
        st.error("Budget data file not found.")


# PAGE 6: ML PREDICTIONS

elif page == "🤖 ML Predictions":
    st.markdown("""
    <div class="page-banner">
        <div class="badge">LINEAR REGRESSION</div>
        <div class="badge">RIDGE REGRESSION</div>
        <div class="badge">LOOCV</div>
        <h1>🤖 Poverty Prediction Models</h1>
        <p>Using literacy and health indicators to predict district-level poverty
        headcount via Linear and Ridge regression with Leave-One-Out cross-validated
        hyperparameters.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training models..."):
        lr_model, ridge_model, data, best_alpha, alpha_df = train_models(df)

    X_test, y_test = data["X_test"], data["y_test"]
    feature_names  = data["feature_names"]

    y_pred_lr    = lr_model.predict(X_test)
    y_pred_ridge = ridge_model.predict(X_test)

    lr_r2    = r2_score(y_test, y_pred_lr)
    lr_mae   = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    ridge_r2   = r2_score(y_test, y_pred_ridge)
    ridge_mae  = mean_absolute_error(y_test, y_pred_ridge)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Linear Regression")
        m1, m2, m3 = st.columns(3)
        with m1: metric_card("R² Score", f"{lr_r2:.3f}",   color="blue")
        with m2: metric_card("MAE",      f"{lr_mae:.2f}%", color="green")
        with m3: metric_card("RMSE",     f"{lr_rmse:.2f}%",color="amber")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f"### Ridge Regression (α={best_alpha})")
        m4, m5, m6 = st.columns(3)
        with m4: metric_card("R² Score", f"{ridge_r2:.3f}",   color="purple")
        with m5: metric_card("MAE",      f"{ridge_mae:.2f}%", color="green")
        with m6: metric_card("RMSE",     f"{ridge_rmse:.2f}%",color="amber")
        st.markdown('</div>', unsafe_allow_html=True)

    # Dynamic insight
    better_model = "Ridge" if ridge_r2 >= lr_r2 else "Linear"
    st.markdown(f"""
    <div class="insight-box">
        💡 Both models achieve <strong>R² > 0.95</strong>. Note: with only {len(df)} districts,
        high R² is expected due to small sample size — not an indicator of overfitting.
        <strong>{better_model} Regression</strong> performs better on the test set.
        Alpha selected via <strong>Leave-One-Out CV</strong> (LOOCV) for maximum reliability.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📉 Feature Importance", "📋 Alpha Tuning"])

    with tab1:
        colA, colB = st.columns(2)
        with colA:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            fig = plot_predictions(y_test, y_pred_lr, "Linear Regression")
            st.pyplot(fig); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        with colB:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            fig = plot_predictions(y_test, y_pred_ridge, f"Ridge (α={best_alpha})")
            st.pyplot(fig); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        colA, colB = st.columns(2)
        with colA:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            fig = plot_feature_importance(lr_model, feature_names, "Linear Regression")
            st.pyplot(fig); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        with colB:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            fig = plot_feature_importance(ridge_model, feature_names, f"Ridge (α={best_alpha})")
            st.pyplot(fig); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Ridge Alpha — LOOCV Results")
        st.caption("Scored using negative MSE (lower MSE = better). Best alpha highlighted.")
        # Dynamically format columns to handle any cached version
        fmt = {}
        for col in alpha_df.columns:
            if col == "alpha":
                fmt[col] = "{:.2f}"
            else:
                fmt[col] = "{:.4f}"
        # Find a numeric column (not alpha) to highlight minimum
        highlight_col = None
        for candidate in ["cv_mse", "cv_rmse", "cv_neg_mse_mean"]:
            if candidate in alpha_df.columns:
                highlight_col = candidate
                break
        styled = alpha_df.style.format(fmt)
        if highlight_col:
            styled = styled.highlight_min(subset=[highlight_col], color="#D1FAE5")
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Residual Analysis")
    fig = plot_residuals(y_test, y_pred_ridge, f"Ridge (α={best_alpha})")
    st.pyplot(fig); plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# PAGE 7: ABOUT

elif page == "ℹ️ About":
    st.markdown("""
    <div class="page-banner">
        <h1>ℹ️ About This Project</h1>
        <p>A portfolio data science project analyzing development disparities
        in South Punjab using real government statistics.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, icon, name, desc in [
        (col1, "🐍", "Python",      "Core Language"),
        (col2, "🐼", "Pandas",      "Data Processing"),
        (col3, "📊", "Matplotlib",  "Visualizations"),
        (col4, "🧠", "Scikit-learn","ML Models"),
        (col5, "🚀", "Streamlit",   "Dashboard"),
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
        st.markdown("### 🎯 Objective")
        st.markdown("""
        - Identify development gaps in South Punjab
        - Visualize literacy, poverty, health & education indicators
        - Track 15-year developmental trends (2011-2023)
        - Analyze budget allocations vs actual regional spending
        - Apply ML to understand poverty determinants
        - Create an interactive, explorable dashboard
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🏛️ South Punjab Districts")
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
        st.markdown("### 📊 Data Sources")
        st.markdown("""
        | Source | Data | Year |
        |--------|------|------|
        | **PBS Census Archive** | Population, literacy, urbanization (1998, 2017, 2023) | 1998-2023 |
        | **PSLM Archive** | District literacy, enrollment, water/sanitation trends | 2010-2020 |
        | **HIES Archive** | Household income, consumption, poverty trends | 2010-2025 |
        | **Punjab P&D (ADP)** | Annual Development Programme budget allocations | 2015-2025 |
        | **Punjab Finance** | White Papers: Revised Estimates vs Promised Budget | 2015-2025 |
        | **UNDP/MPI** | Poverty headcount, MPI score | 2020 |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Limitations")
        st.markdown(f"""
        - Dataset has **{len(df)} districts** × **{len(df.columns)} indicators** — high R² expected with small samples
        - PSLM indicators (unemployment, sanitation, internet) are from 2019-20 district-level survey
        - Poverty and MPI figures are from UNDP estimates
        - Literacy figures verified from **PBS Census 2023** (Table 12, Punjab Districts)
        - Census 2017 data used for temporal comparison — 6-year gap
        - ML results are illustrative — not suitable for causal inference
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 👨‍💻 Author")
        st.markdown("""
        **BS Data Science** — 4th Semester

        Air University, Islamabad

        *Built as a portfolio project for coursework.*
        """)
        st.markdown('</div>', unsafe_allow_html=True)
