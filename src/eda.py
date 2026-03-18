"""
EDA Visualization Module
South Punjab Development Dashboard

Provides publication-quality charts for exploring socioeconomic
disparities across South Punjab districts.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# Style Configuration

# Modern, clean color palette
SOUTH_PUNJAB_COLOR = "#E63946"   # Red for South Punjab
REST_PUNJAB_COLOR = "#457B9D"    # Steel blue for Rest of Punjab
ACCENT_COLOR = "#F4A261"         # Warm orange accent
DARK_BG = "#1D3557"              # Dark navy
PALETTE = ["#E63946", "#457B9D", "#F4A261", "#2A9D8F", "#264653"]

DIVISION_COLORS = {
    "Multan": "#E63946",
    "Bahawalpur": "#F4A261",
    "DG Khan": "#E76F51",
    "Lahore": "#457B9D",
    "Faisalabad": "#2A9D8F",
    "Rawalpindi": "#264653",
    "Gujranwala": "#6A994E",
    "Sargodha": "#BC6C25",
    "Sahiwal": "#606C38",
}


def set_style():
    """Set consistent plot styling."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "grid.color": "#E8E8E8",
        "font.family": "sans-serif",
        "figure.dpi": 100,
    })


# 1. Literacy Rate Comparison

def plot_literacy_comparison(df, save_path=None):
    """
    Horizontal bar chart: literacy rate by district, color-coded by region.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with 'district', 'literacy_rate', 'region' columns
    save_path : str, optional
        Path to save the figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sorted_df = df.sort_values("literacy_rate", ascending=True)
    colors = [SOUTH_PUNJAB_COLOR if r == "South Punjab" else REST_PUNJAB_COLOR 
              for r in sorted_df["region"]]
    
    bars = ax.barh(sorted_df["district"], sorted_df["literacy_rate"], color=colors, 
                   edgecolor="white", linewidth=0.5, height=0.7)
    
    # Add value labels
    for bar, val in zip(bars, sorted_df["literacy_rate"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9, color="#333")
    
    ax.set_xlabel("Literacy Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("District-wise Literacy Rates in Punjab", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 90)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SOUTH_PUNJAB_COLOR, label="South Punjab"),
        Patch(facecolor=REST_PUNJAB_COLOR, label="Rest of Punjab"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# 2. Poverty Headcount Map

def plot_poverty_map(df, save_path=None):
    """
    Horizontal bar chart: poverty headcount by district (highest first).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sorted_df = df.sort_values("poverty_headcount", ascending=True)
    colors = [SOUTH_PUNJAB_COLOR if r == "South Punjab" else REST_PUNJAB_COLOR 
              for r in sorted_df["region"]]
    
    bars = ax.barh(sorted_df["district"], sorted_df["poverty_headcount"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)
    
    for bar, val in zip(bars, sorted_df["poverty_headcount"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9, color="#333")
    
    ax.set_xlabel("Poverty Headcount (%)", fontsize=12, fontweight="bold")
    ax.set_title("District-wise Poverty Headcount in Punjab", fontsize=14, fontweight="bold", pad=15)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SOUTH_PUNJAB_COLOR, label="South Punjab"),
        Patch(facecolor=REST_PUNJAB_COLOR, label="Rest of Punjab"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# 3. Gender Literacy Gap

def plot_gender_gap(df, top_n=15, save_path=None):
    """
    Grouped bar chart: male vs female literacy, sorted by gap size.
    Shows top_n districts with the largest gender literacy gap.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sorted_df = df.sort_values("gender_literacy_gap", ascending=False).head(top_n)
    
    x = np.arange(len(sorted_df))
    width = 0.35
    
    bars_m = ax.bar(x - width/2, sorted_df["male_literacy"], width, 
                    label="Male Literacy", color="#457B9D", edgecolor="white")
    bars_f = ax.bar(x + width/2, sorted_df["female_literacy"], width,
                    label="Female Literacy", color="#E63946", edgecolor="white")
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df["district"], rotation=45, ha="right")
    ax.set_ylabel("Literacy Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Gender Literacy Gap — Top {top_n} Districts", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    
    # Add gap annotation
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        gap = row["gender_literacy_gap"]
        mid = (row["male_literacy"] + row["female_literacy"]) / 2
        ax.annotate(f"Δ{gap:.0f}%", xy=(i, mid), fontsize=8, ha="center",
                    color="#E76F51", fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# 4. School Enrollment Comparison

def plot_enrollment_trends(df, save_path=None):
    """
    Grouped bar chart: Primary vs Middle enrollment by South Punjab districts.
    """
    set_style()
    sp_df = df[df["region"] == "South Punjab"].sort_values("primary_enrollment_rate", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(sp_df))
    width = 0.35
    
    ax.bar(x - width/2, sp_df["primary_enrollment_rate"], width,
           label="Primary Enrollment", color="#2A9D8F", edgecolor="white")
    ax.bar(x + width/2, sp_df["middle_enrollment_rate"], width,
           label="Middle Enrollment", color="#E76F51", edgecolor="white")
    
    ax.set_xticks(x)
    ax.set_xticklabels(sp_df["district"], rotation=45, ha="right")
    ax.set_ylabel("Enrollment Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("School Enrollment Rates — South Punjab Districts",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# 5. Health Indicators

def plot_health_indicators(df, save_path=None):
    """
    Grouped horizontal bars: immunization + clean water access for South Punjab.
    """
    set_style()
    sp_df = df[df["region"] == "South Punjab"].sort_values("immunization_coverage")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    y = np.arange(len(sp_df))
    height = 0.35
    
    ax.barh(y - height/2, sp_df["immunization_coverage"], height,
            label="Immunization Coverage", color="#2A9D8F", edgecolor="white")
    ax.barh(y + height/2, sp_df["clean_water_access"], height,
            label="Clean Water Access", color="#457B9D", edgecolor="white")
    
    ax.set_yticks(y)
    ax.set_yticklabels(sp_df["district"])
    ax.set_xlabel("Coverage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Health Indicators — South Punjab Districts",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# 6. Correlation Heatmap

def plot_correlation_heatmap(df, save_path=None):
    """
    Correlation heatmap of all numeric socioeconomic indicators.
    """
    set_style()
    
    cols = [
        "literacy_rate", "female_literacy", "poverty_headcount", "mpi_score",
        "primary_enrollment_rate", "middle_enrollment_rate",
        "immunization_coverage", "clean_water_access", "electricity_access",
        "hospitals_per_100k", "gender_literacy_gap"
    ]
    available = [c for c in cols if c in df.columns]
    
    corr = df[available].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title("Correlation Matrix — Socioeconomic Indicators",
                 fontsize=14, fontweight="bold", pad=15)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# 7. South Punjab vs Rest of Punjab

def plot_south_vs_rest(df, save_path=None):
    """
    Box plot comparison: key indicators between South Punjab and Rest of Punjab.
    """
    set_style()
    
    indicators = ["literacy_rate", "poverty_headcount", "immunization_coverage",
                  "clean_water_access", "primary_enrollment_rate"]
    labels = ["Literacy\nRate", "Poverty\nHeadcount", "Immunization\nCoverage",
              "Clean Water\nAccess", "Primary\nEnrollment"]
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=False)
    
    for ax, col, label in zip(axes, indicators, labels):
        data = [
            df[df["region"] == "South Punjab"][col],
            df[df["region"] == "Rest of Punjab"][col]
        ]
        
        bp = ax.boxplot(data, labels=["South\nPunjab", "Rest of\nPunjab"],
                       patch_artist=True, widths=0.6,
                       medianprops=dict(color="black", linewidth=2))
        
        bp["boxes"][0].set_facecolor(SOUTH_PUNJAB_COLOR)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(REST_PUNJAB_COLOR)
        bp["boxes"][1].set_alpha(0.7)
        
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("%" if col != "mpi_score" else "Score")
    
    fig.suptitle("South Punjab vs Rest of Punjab — Key Indicators",
                 fontsize=14, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# 8. Division-wise Comparison

def plot_division_comparison(df, indicator="literacy_rate", save_path=None):
    """
    Bar chart comparing divisions on a specific indicator.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    div_means = df.groupby("division")[indicator].mean().sort_values(ascending=False)
    colors = [DIVISION_COLORS.get(d, "#999999") for d in div_means.index]
    
    bars = ax.bar(div_means.index, div_means.values, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.6)
    
    for bar, val in zip(bars, div_means.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    
    ax.set_ylabel(indicator.replace("_", " ").title() + " (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Division-wise Average: {indicator.replace('_', ' ').title()}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=30)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# 9. Scatter: Literacy vs Poverty

def plot_literacy_vs_poverty(df, save_path=None):
    """
    Scatter plot: literacy rate vs poverty headcount, sized by population.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for region, color in [("South Punjab", SOUTH_PUNJAB_COLOR), ("Rest of Punjab", REST_PUNJAB_COLOR)]:
        subset = df[df["region"] == region]
        sizes = subset["population_2023"] / 50000
        ax.scatter(subset["literacy_rate"], subset["poverty_headcount"],
                   s=sizes, c=color, alpha=0.7, edgecolors="white", linewidth=0.5,
                   label=region)
        
        # Label key districts
        for _, row in subset.iterrows():
            if row["poverty_headcount"] > 45 or row["literacy_rate"] > 70 or row["district"] == "Layyah":
                ax.annotate(row["district"],
                           xy=(row["literacy_rate"], row["poverty_headcount"]),
                           xytext=(5, 5), textcoords="offset points",
                           fontsize=8, color="#333")
    
    ax.set_xlabel("Literacy Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Poverty Headcount (%)", fontsize=12, fontweight="bold")
    ax.set_title("Literacy Rate vs Poverty Headcount (bubble = population)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
