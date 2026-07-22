"""Punjab Districts — Development Longitudinal Analysis.

Interactive dashboard for socioeconomic disparities across
Punjab's 36 districts, with MICS household-level microdata.
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
from src.choropleth import plot_choropleth
from src.plotly_charts import (
    plot_radar, plot_bubble, plot_animated_scatter,
    plot_budget_waterfall, plot_waterfall_all_years
)
from src.ml_explainer import compute_shap_values, plot_shap_summary, plot_shap_waterfall
from src import theme
from src.theme import masthead
from src import mics_page

# Page Config

st.set_page_config(
    page_title="Punjab Districts · Development Longitudinal Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design system: CSS + chart templates (single source of truth in src/theme.py)
theme.inject_theme()
theme.register_plotly_template()
theme.apply_matplotlib_theme()

# CSS now lives in src/theme.py (injected above)


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
        st.error(f"Data file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
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
        st.error(f"Error training models: {e}")
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
        ["Overview", "District Profiles", "Indicators",
         "Trends 2011–2023", "Budget Accountability",
         "Poverty Co-Movement", "MICS Longitudinal Study", "About & Sources"],
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
    <div style="padding: 0 0.75rem;">
        <div style="font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.12em; color: #8A8172 !important; margin-bottom: 0.6rem; font-weight: 700;">
            The Data
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: rgba(246,241,231,0.12); border: 1px solid rgba(246,241,231,0.12);">
            <div style="background: #211B12; padding: 0.55rem 0.7rem;">
                <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #F6F1E7 !important;">{len(df)}</div>
                <div style="font-size: 0.62rem; color: #8A8172 !important; text-transform: uppercase; letter-spacing: 0.06em;">Districts</div>
            </div>
            <div style="background: #211B12; padding: 0.55rem 0.7rem;">
                <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #E0784F !important;">{len(sp_df)}</div>
                <div style="font-size: 0.62rem; color: #8A8172 !important; text-transform: uppercase; letter-spacing: 0.06em;">South Punjab</div>
            </div>
            <div style="background: #211B12; padding: 0.55rem 0.7rem;">
                <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #F6F1E7 !important;">{len(rest_df)}</div>
                <div style="font-size: 0.62rem; color: #8A8172 !important; text-transform: uppercase; letter-spacing: 0.06em;">Rest of Punjab</div>
            </div>
            <div style="background: #211B12; padding: 0.55rem 0.7rem;">
                <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #F6F1E7 !important;">{len(df.columns)}</div>
                <div style="font-size: 0.62rem; color: #8A8172 !important; text-transform: uppercase; letter-spacing: 0.06em;">Indicators</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 0.75rem; font-size: 0.64rem; color: #8A8172 !important; line-height: 1.6;">
        Sources: PBS Census 2023 · PSLM · MICS 2011–2018 · Punjab Finance White Papers
    </div>
    """, unsafe_allow_html=True)


# PAGE 1: OVERVIEW

if page == "Overview":
    masthead(
        "Special Report · Punjab, Pakistan",
        "Uneven Progress: Mapping South Punjab's Development Gap",
        "Eleven southern districts are home to a third of Punjab's people but sit at "
        "the bottom of nearly every development ranking. This atlas measures the gap — "
        "and where it is closing — using official census, survey, and budget records.",
        badges=["PBS Census 2023", "Census 2017", "PSLM 2019-20",
                "MICS 2011-2018", "36 Districts", "43 Indicators"],
    )

    # ── Choropleth Map ──────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    map_indicator = st.selectbox(
        "Map Indicator",
        ["poverty_headcount", "literacy_rate", "out_of_school_rate", "internet_access", "immunization_coverage"],
        format_func=lambda x: x.replace("_", " ").title(),
        key="map_indicator"
    )
    choropleth_fig = plot_choropleth(df, map_indicator)
    if choropleth_fig:
        st.plotly_chart(choropleth_fig, use_container_width=True)
    else:
        st.warning("GeoJSON file not found. Run `python data/raw/extract_geojson.py` to generate it.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

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
        <strong>Key Insight:</strong> The most impoverished district is
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
        st.markdown("### Most Impoverished Districts")
        poverty_rank = get_rankings(df, "poverty_headcount", ascending=False)
        st.dataframe(
            poverty_rank.head(10).style.background_gradient(
                subset=["poverty_headcount"], cmap="Oranges"),
            width="stretch", hide_index=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Most Literate Districts")
        lit_rank = get_rankings(df, "literacy_rate", ascending=False)
        st.dataframe(
            lit_rank.head(10).style.background_gradient(
                subset=["literacy_rate"], cmap="BuGn"),
            width="stretch", hide_index=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Literacy vs Poverty — All Punjab Districts")
    fig = plot_literacy_vs_poverty(df)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# PAGE 2: DISTRICT PROFILES

elif page == "District Profiles":
    masthead(
        "Explore the Districts",
        "District Profiles",
        "Every socioeconomic indicator for any of Punjab's 36 districts, "
        "benchmarked against regional and provincial averages.",
    )

    selected = st.selectbox("Select a District", sorted(df["district"].unique()))
    profile  = get_district_profile(df, selected)

    is_south   = profile["region"] == "South Punjab"
    region_cls = "south" if is_south else "rest"

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0;">
        <h2 style="margin: 0;">{profile['district']}</h2>
        <span class="region-tag {region_cls}">{profile['region']}</span>
        <span style="color: #93897A; font-size: 0.9rem;">{profile['division']} Division</span>
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

    # ── Radar Chart ─────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### District Performance Radar")
    radar_fig = plot_radar(df, selected, SOUTH_PUNJAB_DISTRICTS)
    st.plotly_chart(radar_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Compared to Averages")

    indicators = ["literacy_rate", "poverty_headcount", "unemployment_rate",
                  "immunization_coverage", "clean_water_access", "sanitation_access",
                  "primary_enrollment_rate", "out_of_school_rate",
                  "electricity_access", "internet_access"]
    compare_data = []
    for ind in indicators:
        compare_data.append({
            "Indicator": ind.replace("_", " ").title(),
            f"{selected}": profile[ind],
            "South Punjab Avg": sp_df[ind].mean(),
            "All Punjab Avg": df[ind].mean(),
        })

    compare_df = pd.DataFrame(compare_data)
    st.dataframe(compare_df.style.format({
        f"{selected}": "{:.1f}",
        "South Punjab Avg": "{:.1f}",
        "All Punjab Avg": "{:.1f}"
    }).background_gradient(subset=[f"{selected}"], cmap="BuGn"),
    use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# PAGE 3: EDA

elif page == "Indicators":
    masthead(
        "The Evidence",
        "Indicator Deep-Dive",
        "Literacy, poverty, schooling, health, and infrastructure — "
        "South Punjab measured against the rest of the province, chart by chart.",
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Literacy", "Poverty", "Education", "Health", "Infrastructure", "Temporal", "Correlations"
    ])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### District-wise Literacy Rates")
        # Dynamic insight
        st.markdown(f"""<div class="insight-box">
            South Punjab districts cluster at the <strong>bottom</strong> of literacy rankings,
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
            <strong>{_sp_in_top10} out of the top 10</strong> most impoverished districts
            in Punjab belong to South Punjab, with DG Khan division being the worst affected.
        </div>""", unsafe_allow_html=True)
        fig = plot_poverty_map(df)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Bubble Chart ────────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Literacy vs Poverty Bubble Chart")
        st.markdown("""<div class="insight-box">
            Bubble size = district population. Color = administrative division.
            The dashed trendline confirms the strong negative correlation between literacy and poverty.
        </div>""", unsafe_allow_html=True)
        bubble_fig = plot_bubble(df)
        st.plotly_chart(bubble_fig, use_container_width=True)
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
            On average, only <strong>{_sp_enroll_ratio:.0f} out of 100</strong> primary
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
                South Punjab has an average out-of-school rate of <strong>{_sp_oos:.1f}%</strong>
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
            <strong>{_worst_imm['district']}</strong> has the lowest immunization coverage
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
            South Punjab districts average only <strong>{_sp_sanit:.1f}%</strong> sanitation access
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
                South Punjab literacy improved by an average of <strong>{sp_change:.1f} percentage points</strong>
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
            <strong>Literacy rate</strong> and <strong>poverty headcount</strong> show a strong
            negative correlation (r = {_corr_val:.2f}), confirming that education is the
            indicator most tightly associated with poverty across Punjab districts.
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

elif page == "Trends 2011–2023":
    masthead(
        "Twelve Years of Change",
        "Developmental Trends, 2011–2023",
        "Tracking how each district's indicators moved across survey rounds and "
        "census years — and whether the South is catching up.",
    )

    if df_hist is not None:
        indicator = st.selectbox(
            "Select Indicator to Track",
            ["literacy_rate", "poverty_headcount", "out_of_school_rate"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("### Highlight Districts")
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
            st.markdown("### Regional Performance Gap Over Time")
            gap_df = historical_analyzer.compare_historical_gaps(df_hist, indicator, SOUTH_PUNJAB_DISTRICTS)
            if gap_df is not None and "gap" in gap_df.columns:
                fig_gap, ax_gap = plt.subplots(figsize=(10, 3))
                ax_gap.plot(gap_df.index, gap_df["gap"], marker='o', color=theme.TEAL, linewidth=2)
                ax_gap.axhline(0, color=theme.SOUTH, linestyle="--", alpha=0.5)
                ax_gap.fill_between(gap_df.index, gap_df["gap"], alpha=0.12, color=theme.TEAL)
                ax_gap.set_ylabel("Gap (Rest − South, pp)")
                ax_gap.set_xlabel("Year")
                ax_gap.grid(True, linestyle="--", alpha=0.3)
                fig_gap.tight_layout()
                st.pyplot(fig_gap)
                plt.close(fig_gap)

            # ── Animated Scatter ─────────────────────────────────────
            st.markdown("### Animated District Movement")
            st.caption("Press ▶ Play to watch districts evolve across survey years.")
            anim_fig = plot_animated_scatter(df_hist, indicator, SOUTH_PUNJAB_DISTRICTS)
            if anim_fig:
                st.plotly_chart(anim_fig, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Historical data file not found.")


# PAGE 5: BUDGET ACCOUNTABILITY

elif page == "Budget Accountability":
    masthead(
        "Follow the Money",
        "Budget & Fiscal Accountability",
        "Reading the province's White Papers: what was promised to the South, "
        "what was actually spent, and the gap in between.",
        badges=["Punjab Finance White Papers", "ADP 2015-2025", "CPI-adjusted"],
    )

    adj = st.toggle("Adjust for Inflation (Real Growth)", value=True, 
                    help="Uses CPI (Consumer Price Index) to convert nominal PKR into constant 2015-16 values.")
    
    current_budget = df_budget_real if adj else df_budget_nom
    
    if current_budget is not None:
        # ── Waterfall Chart ───────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Budget Utilization Waterfall")
        year_to_view = st.slider("Select Year to view flow:", 
                                 min_value=int(current_budget["year"].min()), 
                                 max_value=int(current_budget["year"].max()), 
                                 value=int(current_budget["year"].max()))
        waterfall_fig = plot_budget_waterfall(current_budget, year_to_view, "South Punjab", use_real=adj)
        if waterfall_fig:
            st.plotly_chart(waterfall_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown("### Fiscal Performance Summary")
        summary = historical_analyzer.get_budget_summary(current_budget)
        # Choose column names dynamically based on inflation toggle
        alloc_col = "allocation_real_bn" if adj and "allocation_real_bn" in summary.columns else "allocation_pkr_bn"
        exp_col_fmt = "expenditure_real_bn" if adj and "expenditure_real_bn" in summary.columns else "expenditure_pkr_bn"
        fmt_dict = {alloc_col: "{:,.1f} Bn", exp_col_fmt: "{:,.1f} Bn", "utilization_rate": "{:.1f}%"}
        st.table(summary.style.format(fmt_dict))
        
        st.info("**Revised Estimates**: Represent the actual funds released and spent by the end of the fiscal year, often revealing significant under-utilization in Southern districts compared to Central Punjab.")
    else:
        st.error("Budget data file not found.")


# PAGE 6: POVERTY CO-MOVEMENT

elif page == "Poverty Co-Movement":
    masthead(
        "Ecological Correlation · 36 Districts",
        "Which Indicators Track Deprivation",
        "Linear and Ridge regressions describe how literacy, health, and "
        "infrastructure indicators co-move with district poverty headcount. "
        "At this level of aggregation the models measure association along one "
        "shared development gradient — not prediction, and not causation.",
        badges=["Linear Regression", "Ridge Regression", "LOOCV", "SHAP"],
    )

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
        <strong>Read the R² ≈ 0.99 with care — it is not evidence of predictive skill.</strong>
        The target is the UNDP multidimensional poverty (MPI) headcount, an index computed
        from deprivation indicators, and ten of the fourteen input features (literacy,
        enrollment, out-of-school, immunization, water, sanitation, electricity) measure
        the very components the MPI is built from — the regression largely reconstructs
        an index from its own ingredients. The remaining features ride the same district
        development gradient (all features correlate with the target at |r| ≥ 0.57, most
        above 0.93; one principal component carries ~81% of feature variance). A small
        sample makes such a fit <em>less</em> trustworthy, not more. Treat coefficients
        and SHAP values as descriptions of which indicators co-move with deprivation
        across districts — not as causal drivers or forecasting power.
        {better_model} regression fits the test split marginally better; alpha selected
        via Leave-One-Out CV. For a deprivation model built on strictly exogenous
        features, see the <strong>MICS Longitudinal Study</strong> page.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Model Fit", "Indicator Weights", "Alpha Tuning", "Interpretability (SHAP)"])

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
            styled = styled.highlight_min(subset=[highlight_col], color="#E6EBDD")
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Residual Analysis")
    fig = plot_residuals(y_test, y_pred_ridge, f"Ridge (α={best_alpha})")
    st.pyplot(fig); plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Model Interpretability (SHAP)")
        st.markdown("""<div class="insight-box">
            SHAP values decompose each district's fitted poverty level into indicator
            contributions — read them as association along the district development
            gradient, not as causal drivers.
        </div>""", unsafe_allow_html=True)
        shap_values = compute_shap_values(lr_model, data["X_train"], X_test, feature_names)
        
        # Summary Plot
        st.markdown("#### Global Summary: Which indicators carry the association?")
        st.pyplot(plot_shap_summary(shap_values))
        
        st.markdown("---")
        
        # Single District Explanation
        st.markdown("#### Explain a Single District")
        districts_in_test = df.loc[y_test.index, "district"].tolist()
        dist_to_explain = st.selectbox("Select District from Test Set:", districts_in_test)
        idx = districts_in_test.index(dist_to_explain)
        
        st.pyplot(plot_shap_waterfall(shap_values, idx, dist_to_explain))
        
        st.markdown('</div>', unsafe_allow_html=True)



# PAGE 7: ABOUT

elif page == "MICS Longitudinal Study":
    mics_page.render()


elif page == "About & Sources":
    masthead(
        "Methodology & Provenance",
        "About This Project",
        "A data science portfolio project analyzing development disparities in "
        "South Punjab, built entirely on official government statistics.",
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, icon, name, desc in [
        (col1, "", "Python",      "Core Language"),
        (col2, "", "Pandas",      "Data Processing"),
        (col3, "", "Matplotlib",  "Visualizations"),
        (col4, "", "Scikit-learn","ML Models"),
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
        - Identify development gaps in South Punjab
        - Visualize literacy, poverty, health & education indicators
        - Track 15-year developmental trends (2011-2023)
        - Analyze budget allocations vs actual regional spending
        - Apply ML to describe which indicators co-move with poverty
        - Create an interactive, explorable dashboard
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### South Punjab Districts")
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
        | **PBS Census Archive** | Population, literacy, urbanization (1998, 2017, 2023) | 1998-2023 |
        | **PSLM Archive** | District literacy, enrollment, water/sanitation trends | 2010-2020 |
        | **HIES Archive** | Household income, consumption, poverty trends | 2010-2025 |
        | **Punjab P&D (ADP)** | Annual Development Programme budget allocations | 2015-2025 |
        | **Punjab Finance** | White Papers: Revised Estimates vs Promised Budget | 2015-2025 |
        | **UNDP/MPI** | Poverty headcount, MPI score | 2020 |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Limitations")
        st.markdown(f"""
        - Poverty-model R² ≈ 0.99 is inflated by construction: the target is the UNDP MPI headcount and most features overlap with MPI components (see the note on the Poverty Models page)
        - PSLM indicators (unemployment, sanitation, internet) are from 2019-20 district-level survey
        - Poverty and MPI figures are from UNDP estimates
        - Literacy figures verified from **PBS Census 2023** (Table 12, Punjab Districts)
        - Census 2017 data used for temporal comparison — 6-year gap
        - ML results are illustrative — not suitable for causal inference
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Author")
        st.markdown("""
        **M Saad Sadaf**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
