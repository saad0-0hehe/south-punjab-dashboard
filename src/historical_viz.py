"""
Historical Visualization Module
South Punjab Development Dashboard

Handles plotting of time-series trends and budget disparity charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_indicator_trends(df_history, indicator="literacy_rate", selected_districts=None, south_districts=None):
    """
    Plot 15-year trends for a specific indicator.
    """
    if df_history is None:
        return None
        
    df = df_history[df_history["indicator"] == indicator].copy()
    
    # Calculate Regional Averages
    df["region"] = df["district"].map(lambda x: "South Punjab" if x in south_districts else "Rest of Punjab")
    regional_avg = df.groupby(["year", "region"])["value"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot South Punjab Average
    sns.lineplot(data=regional_avg[regional_avg["region"]=="South Punjab"], 
                 x="year", y="value", marker='o', label="South Punjab Avg", 
                 linewidth=3, color="#EF4444", ax=ax)
    
    # Plot Rest of Punjab Average
    sns.lineplot(data=regional_avg[regional_avg["region"]=="Rest of Punjab"], 
                 x="year", y="value", marker='o', label="Rest of Punjab Avg", 
                 linewidth=3, color="#6366F1", ax=ax)
    
    # Plot Selected Districts
    if selected_districts:
        for d in selected_districts:
            d_data = df[df["district"] == d]
            sns.lineplot(data=d_data, x="year", y="value", marker='x', 
                         label=f"📍 {d}", alpha=0.5, linestyle='--', ax=ax)
    
    ax.set_title(f"{indicator.replace('_', ' ').title()} Trends (2011-2023)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Survey Year", fontsize=10)
    ax.set_ylabel(indicator.replace('_', ' ').title(), fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    return fig

def plot_budget_comparison(df_budget, metric="allocation_pkr_bn"):
    """
    Plot Promised (Allocation) vs Spent (Expenditure) per year.
    """
    if df_budget is None:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Reshape for grouped bar chart
    df_plot = df_budget.melt(id_vars=["year", "region"], 
                            value_vars=[metric, metric.replace("allocation", "expenditure")],
                            var_name="Type", value_name="Value")
    
    df_plot["Type"] = df_plot["Type"].map(lambda x: "Promised (Allocation)" if "allocation" in x else "Spent (Expenditure)")
    
    sns.barplot(data=df_plot, x="year", y="Value", hue="Type", palette=["#94A3B8", "#6366F1"], ax=ax)
    
    ax.set_title(f"Annual Budget: Promised vs Actually Spent", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Fiscal Year", fontsize=10)
    ax.set_ylabel("PKR Billions", fontsize=10)
    ax.legend(title="", frameon=False)
    ax.grid(axis='y', linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_disparity_gap(df_budget, metric="allocation_pkr_bn"):
    """
    Plot the gap in allocations between South and Rest of Punjab.
    """
    if df_budget is None:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=df_budget, x="year", y=metric, hue="region", 
                 marker='s', linewidth=2.5, palette={"South Punjab": "#EF4444", "Rest of Punjab": "#6366F1"}, ax=ax)
    
    ax.set_title(f"Budget Allocation Disparity: South vs Rest", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Fiscal Year", fontsize=10)
    ax.set_ylabel("PKR Billions", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    return fig
