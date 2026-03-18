"""
01_eda.py — Exploratory Data Analysis
South Punjab Development Dashboard

This script walks through the complete EDA pipeline:
1. Load and clean district-level data
2. Generate summary statistics
3. Produce publication-quality visualizations
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, clean_data, filter_south_punjab, get_summary_stats
from src.eda import (
    plot_literacy_comparison,
    plot_poverty_map,
    plot_gender_gap,
    plot_enrollment_trends,
    plot_health_indicators,
    plot_correlation_heatmap,
    plot_south_vs_rest,
    plot_division_comparison,
    plot_literacy_vs_poverty,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script

# Step 1: Load & Clean Data

print("STEP 1: Loading and Cleaning Data")

df = load_data()
df = clean_data(df)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nRegion distribution:")
print(df["region"].value_counts())

# Step 2: Summary Statistics

print("\n" + "=" * 60)
print("STEP 2: Summary Statistics — South Punjab vs Rest of Punjab")

summary = get_summary_stats(df)
print(summary.to_string())

sp = filter_south_punjab(df)
print(f"\nSouth Punjab Average Literacy:  {sp['literacy_rate'].mean():.1f}%")
print(f"Rest of Punjab Average Literacy: {df[df['region'] == 'Rest of Punjab']['literacy_rate'].mean():.1f}%")
print(f"\nSouth Punjab Average Poverty:   {sp['poverty_headcount'].mean():.1f}%")
print(f"Rest of Punjab Average Poverty:  {df[df['region'] == 'Rest of Punjab']['poverty_headcount'].mean():.1f}%")

# Step 3: Generate Visualizations

print("\n" + "=" * 60)
print("STEP 3: Generating Visualizations")

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(output_dir, exist_ok=True)

charts = [
    ("Literacy Comparison", plot_literacy_comparison, "01_literacy_comparison.png"),
    ("Poverty Map", plot_poverty_map, "02_poverty_map.png"),
    ("Gender Gap", plot_gender_gap, "03_gender_gap.png"),
    ("Enrollment Trends", plot_enrollment_trends, "04_enrollment_trends.png"),
    ("Health Indicators", plot_health_indicators, "05_health_indicators.png"),
    ("Correlation Heatmap", plot_correlation_heatmap, "06_correlation_heatmap.png"),
    ("South vs Rest", plot_south_vs_rest, "07_south_vs_rest.png"),
    ("Division Comparison", plot_division_comparison, "08_division_comparison.png"),
    ("Literacy vs Poverty", plot_literacy_vs_poverty, "09_literacy_vs_poverty.png"),
]

for name, func, filename in charts:
    save_path = os.path.join(output_dir, filename)
    try:
        func(df, save_path=save_path)
        print(f"  ✅ {name} → {filename}")
    except Exception as e:
        print(f"  ❌ {name} failed: {e}")

import matplotlib.pyplot as plt
plt.close("all")

print(f"\n✅ All charts saved to: {output_dir}")
print("EDA COMPLETE!")
