"""
Interactive Plotly Charts Module
South Punjab Development Dashboard

Contains: Radar chart, Bubble chart, Animated temporal, Waterfall budget.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# ── PROMPT 3: Radar Chart ────────────────────────────────────────────

RADAR_DIMS = [
    "literacy_rate",
    "poverty_headcount",  # will be inverted
    "immunization_coverage",
    "internet_access",
    "clean_water_access",
    "primary_enrollment_rate",
]

RADAR_LABELS = {
    "literacy_rate": "Literacy",
    "poverty_headcount": "Low Poverty",  # inverted
    "immunization_coverage": "Immunization",
    "internet_access": "Internet",
    "clean_water_access": "Clean Water",
    "primary_enrollment_rate": "Enrollment",
}


def _normalize(val, col_min, col_max, invert=False):
    """Scale a value to 0–100 relative to min/max across all districts."""
    if col_max == col_min:
        return 50
    normed = (val - col_min) / (col_max - col_min) * 100
    return (100 - normed) if invert else normed


def plot_radar(df, district_name, south_districts):
    """
    Radar chart comparing a district vs South Punjab avg vs All Punjab avg.
    """
    profile = df[df["district"] == district_name].iloc[0]
    sp_df = df[df["district"].isin(south_districts)]

    categories = []
    dist_vals = []
    sp_vals = []
    all_vals = []

    for dim in RADAR_DIMS:
        invert = dim == "poverty_headcount"
        col_min = df[dim].min()
        col_max = df[dim].max()

        dist_vals.append(_normalize(profile[dim], col_min, col_max, invert))
        sp_vals.append(_normalize(sp_df[dim].mean(), col_min, col_max, invert))
        all_vals.append(_normalize(df[dim].mean(), col_min, col_max, invert))
        categories.append(RADAR_LABELS.get(dim, dim))

    # Close the polygon
    categories.append(categories[0])
    dist_vals.append(dist_vals[0])
    sp_vals.append(sp_vals[0])
    all_vals.append(all_vals[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=dist_vals, theta=categories, name=district_name,
        fill="toself", fillcolor="rgba(99, 102, 241, 0.15)",
        line=dict(color="#6366F1", width=2.5),
    ))
    fig.add_trace(go.Scatterpolar(
        r=sp_vals, theta=categories, name="South Punjab Avg",
        fill="toself", fillcolor="rgba(239, 68, 68, 0.08)",
        line=dict(color="#EF4444", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=all_vals, theta=categories, name="All Punjab Avg",
        fill="toself", fillcolor="rgba(148, 163, 184, 0.08)",
        line=dict(color="#94A3B8", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=True,
        height=420,
        margin=dict(t=30, b=30, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


# ── PROMPT 4: Bubble Chart ───────────────────────────────────────────

def plot_bubble(df):
    """
    Interactive bubble chart: x=literacy, y=poverty, size=population, color=division.
    """
    plot_df = df.copy()
    # Scale bubble sizes
    plot_df["pop_scaled"] = plot_df["population_2023"] / 100000

    # Trendline via numpy polyfit
    z = np.polyfit(plot_df["literacy_rate"], plot_df["poverty_headcount"], 1)
    x_line = np.linspace(plot_df["literacy_rate"].min(), plot_df["literacy_rate"].max(), 50)
    y_line = np.polyval(z, x_line)

    fig = px.scatter(
        plot_df,
        x="literacy_rate",
        y="poverty_headcount",
        size="pop_scaled",
        color="division",
        text="district",
        hover_data={
            "literacy_rate": ":.1f",
            "poverty_headcount": ":.1f",
            "population_2023": ":,.0f",
            "pop_scaled": False,
        },
        labels={
            "literacy_rate": "Literacy Rate (%)",
            "poverty_headcount": "Poverty Headcount (%)",
            "division": "Division",
        },
        size_max=45,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(textposition="top center", textfont_size=8)

    # Add trendline
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#94A3B8", width=2, dash="dash"),
        name=f"Trend (r={np.corrcoef(plot_df['literacy_rate'], plot_df['poverty_headcount'])[0,1]:.2f})",
        showlegend=True
    ))

    fig.update_layout(
        height=500,
        margin=dict(t=30, b=30),
        xaxis_title="Literacy Rate (%)",
        yaxis_title="Poverty Headcount (%)",
    )
    return fig


# ── PROMPT 5: Animated Temporal Chart ────────────────────────────────

def plot_animated_trends(df_hist, indicator, selected_districts, south_districts):
    """
    Animated scatter/line showing districts moving year by year.
    """
    if df_hist is None:
        return None

    df = df_hist[df_hist["indicator"] == indicator].copy()
    df["region"] = df["district"].map(
        lambda x: "South Punjab" if x in south_districts else "Rest of Punjab"
    )

    # Regional averages per year
    regional = df.groupby(["year", "region"])["value"].mean().reset_index()
    regional["district"] = regional["region"]
    regional["type"] = "Average"

    # Selected individual districts
    if selected_districts:
        indiv = df[df["district"].isin(selected_districts)].copy()
        indiv["type"] = "District"
    else:
        indiv = pd.DataFrame()

    plot_df = pd.concat([regional, indiv], ignore_index=True)

    fig = px.line(
        plot_df,
        x="year",
        y="value",
        color="district",
        line_dash="type",
        markers=True,
        animation_frame="year",
        range_y=[plot_df["value"].min() - 5, plot_df["value"].max() + 5],
        labels={
            "value": indicator.replace("_", " ").title(),
            "year": "Survey Year",
            "district": "District / Region",
        },
        color_discrete_map={
            "South Punjab": "#EF4444",
            "Rest of Punjab": "#6366F1",
        },
    )

    fig.update_layout(
        height=500,
        margin=dict(t=30, b=30),
        xaxis=dict(dtick=1),
    )

    # The animation_frame on a line chart can be tricky.
    # Let's also provide a static cumulative version:
    return fig


def plot_animated_scatter(df_hist, indicator, south_districts):
    """
    Animated scatter: each dot = one district, moving across years.
    Uses animation_frame on 'year'.
    """
    if df_hist is None:
        return None

    df = df_hist[df_hist["indicator"] == indicator].copy()
    df["region"] = df["district"].map(
        lambda x: "South Punjab" if x in south_districts else "Rest of Punjab"
    )

    fig = px.scatter(
        df,
        x="district",
        y="value",
        color="region",
        animation_frame="year",
        range_y=[df["value"].min() - 5, df["value"].max() + 5],
        hover_name="district",
        hover_data={"value": ":.1f", "source": True},
        labels={
            "value": indicator.replace("_", " ").title(),
            "district": "",
        },
        color_discrete_map={
            "South Punjab": "#EF4444",
            "Rest of Punjab": "#6366F1",
        },
    )

    fig.update_layout(
        height=500,
        margin=dict(t=30, b=30),
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
    )
    fig.update_traces(marker=dict(size=10))

    return fig


# ── PROMPT 6: Waterfall Budget Chart ─────────────────────────────────

def plot_budget_waterfall(df_budget, year, region="South Punjab", use_real=True):
    """
    Waterfall chart: Allocation → Expenditure → Gap for a single year/region.
    """
    if df_budget is None:
        return None

    row = df_budget[(df_budget["year"] == year) & (df_budget["region"] == region)]
    if row.empty:
        return None
    row = row.iloc[0]

    alloc_col = "allocation_real_bn" if use_real and "allocation_real_bn" in row.index else "allocation_pkr_bn"
    exp_col = "expenditure_real_bn" if use_real and "expenditure_real_bn" in row.index else "expenditure_pkr_bn"

    allocation = row[alloc_col]
    expenditure = row[exp_col]
    gap = allocation - expenditure

    fig = go.Figure(go.Waterfall(
        name="Budget Flow",
        orientation="v",
        measure=["absolute", "relative", "relative"],
        x=["Allocated (Promised)", "Spent (Actual)", "Utilization Gap"],
        y=[allocation, -expenditure, 0],
        text=[f"{allocation:.1f} Bn", f"{expenditure:.1f} Bn", f"{gap:.1f} Bn"],
        textposition="outside",
        connector={"line": {"color": "#94A3B8", "width": 1.5}},
        decreasing={"marker": {"color": "#6366F1"}},
        increasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#F59E0B"}},
    ))

    unit = "Real PKR Bn (2015 base)" if use_real else "Nominal PKR Bn"
    fig.update_layout(
        title=f"{region} — FY{year} Budget Flow ({unit})",
        yaxis_title=unit,
        height=400,
        margin=dict(t=60, b=30),
        showlegend=False,
    )
    return fig


def plot_waterfall_all_years(df_budget, region="South Punjab", use_real=True):
    """
    Multi-year waterfall: stacked view of allocation, spent, and gap per year.
    """
    if df_budget is None:
        return None

    reg_df = df_budget[df_budget["region"] == region].copy()
    alloc_col = "allocation_real_bn" if use_real and "allocation_real_bn" in reg_df.columns else "allocation_pkr_bn"
    exp_col = "expenditure_real_bn" if use_real and "expenditure_real_bn" in reg_df.columns else "expenditure_pkr_bn"

    reg_df["gap"] = reg_df[alloc_col] - reg_df[exp_col]
    reg_df["utilization_pct"] = (reg_df[exp_col] / reg_df[alloc_col] * 100).round(1)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=reg_df["year"], y=reg_df[exp_col],
        name="Spent", marker_color="#6366F1",
        text=reg_df["utilization_pct"].apply(lambda x: f"{x:.0f}%"),
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=reg_df["year"], y=reg_df["gap"],
        name="Unspent Gap", marker_color="#EF4444",
        opacity=0.7,
    ))

    unit = "Real PKR Bn (2015)" if use_real else "Nominal PKR Bn"
    fig.update_layout(
        barmode="stack",
        yaxis_title=unit,
        height=400,
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig
