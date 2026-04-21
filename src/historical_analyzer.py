"""
Historical Analysis Module
South Punjab Development Dashboard

Handles processing of longitudinal data, CAGR calculation, 
and budget disparity analysis.
"""

import pandas as pd
import numpy as np

def get_growth_data(df_history, indicator="literacy_rate"):
    """
    Calculate growth and CAGR for a specific indicator across districts.
    """
    if df_history is None:
        return None
        
    mask = df_history["indicator"] == indicator
    df = df_history[mask].pivot(index="district", columns="year", values="value")
    
    # Calculate Total Growth and CAGR (2011 to 2023)
    if 2011 in df.columns and 2023 in df.columns:
        df["total_growth"] = df[2023] - df[2011]
        # CAGR = [(Final/Initial)^(1/n) - 1] * 100
        n_years = 2023 - 2011
        df["cagr"] = ((df[2023] / df[2011])**(1/n_years) - 1) * 100
        
    return df.sort_values("total_growth", ascending=False)

def get_budget_summary(df_budget):
    """
    Summarize budget metrics: Utilization rate and South-North disparity.
    """
    if df_budget is None:
        return None
        
    # Calculate utilization rate
    df = df_budget.copy()
    df["utilization_rate"] = (df["expenditure_pkr_bn"] / df["allocation_pkr_bn"]) * 100
    
    summary = df.groupby("region").agg({
        "allocation_pkr_bn": "sum",
        "expenditure_pkr_bn": "sum",
        "utilization_rate": "mean"
    }).reset_index()
    
    return summary

def compare_historical_gaps(df_history, indicator="literacy_rate", region_map=None):
    """
    Compare average indicator values over time for South vs Rest of Punjab.
    """
    if df_history is None or region_map is None:
        return None
        
    df = df_history[df_history["indicator"] == indicator].copy()
    df["region"] = df["district"].map(lambda x: "South Punjab" if x in region_map else "Rest of Punjab")
    
    trend = df.groupby(["year", "region"])["value"].mean().unstack()
    trend["gap"] = trend.get("Rest of Punjab", 0) - trend.get("South Punjab", 0)
    
    return trend
