"""
Data Loading and Cleaning Module
South Punjab Development Dashboard

Handles loading, cleaning, and filtering district-level socioeconomic
data for South Punjab analysis.
"""

import pandas as pd
import numpy as np
import os

# Constants

SOUTH_PUNJAB_DISTRICTS = [
    "Multan", "Lodhran", "Khanewal", "Vehari",
    "Bahawalpur", "Bahawalnagar", "Rahim Yar Khan",
    "DG Khan", "Muzaffargarh", "Layyah", "Rajanpur"
]

SOUTH_PUNJAB_DIVISIONS = ["Multan", "Bahawalpur", "DG Khan"]

NUMERIC_COLUMNS = [
    # Demographics
    "population_2023", "population_2017", "population_growth_rate",
    "area_sqkm", "density_per_sqkm",
    # Literacy
    "literacy_rate", "male_literacy", "female_literacy",
    "rural_literacy", "urban_literacy", "rural_urban_literacy_gap",
    "literacy_rate_2017", "literacy_change",
    # Poverty & Employment
    "poverty_headcount", "mpi_score", "unemployment_rate",
    # Education (rates)
    "primary_enrollment_rate", "middle_enrollment_rate",
    "out_of_school_rate", "dropout_rate", "never_attended_rate",
    "higher_education_rate",
    # Education (raw counts)
    "population_5plus", "population_10plus", "ever_attended",
    "enrolment_primary", "enrolment_middle", "enrolment_matric",
    "enrolment_intermediate", "enrolment_graduation_above",
    "out_of_school_5_16", "dropout_5_16", "never_attended_school",
    # Health & Infrastructure
    "immunization_coverage", "clean_water_access",
    "sanitation_access", "electricity_access",
    "internet_access", "hospitals_per_100k",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# Data Loading

def load_data(filepath=None):
    """
    Load the district-level socioeconomic CSV file.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to data/district_socioeconomic.csv
    
    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from CSV
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "district_socioeconomic.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Make sure district_socioeconomic.csv is in the data/ directory."
        )
    
    df = pd.read_csv(filepath)
    print(f"[OK] Loaded {len(df)} rows x {len(df.columns)} columns from {os.path.basename(filepath)}")
    return df


def load_historical_data(filepath=None):
    """Load longitudinal district data (2011-2023)."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "historical", "district_history.csv")
    
    if not os.path.exists(filepath):
        return None
        
    df = pd.read_csv(filepath)

    # Normalize district name to match master dataset
    df["district"] = df["district"].replace("Dera Ghazi Khan", "DG Khan")

    print(f"[OK] Loaded {len(df)} historical records from {os.path.basename(filepath)}")
    return df


def override_2023_anchor(df_history, df_master):
    """
    Replace the 2023 historical values with verified master dataset values.
    This ensures the temporal trend endpoint matches the authoritative 2023 snapshot.

    Parameters
    ----------
    df_history : pd.DataFrame
        Long-form historical data
    df_master : pd.DataFrame
        Cleaned master dataset (district_socioeconomic.csv)

    Returns
    -------
    pd.DataFrame
        Historical data with corrected 2023 anchor values
    """
    indicator_map = {
        "literacy_rate": "literacy_rate",
        "poverty_headcount": "poverty_headcount",
        "out_of_school_rate": "out_of_school_rate",
    }

    rows_to_update = df_history["year"] == 2023
    for hist_col, master_col in indicator_map.items():
        if master_col not in df_master.columns:
            continue
        mask = rows_to_update & (df_history["indicator"] == hist_col)
        # Build a lookup from master
        lookup = df_master.set_index("district")[master_col]
        df_history.loc[mask, "value"] = df_history.loc[mask, "district"].map(lookup)
        df_history.loc[mask, "source"] = "Census"  # Mark as authoritative

    return df_history



def load_budget_data(filepath=None, adjust_for_inflation=False):
    """
    Load annual budget data and optionally adjust for inflation.
    
    Formula: Real PKR = (Nominal PKR / CPI_index) * 100
    Base Year: 2015-16 (CPI = 100)
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "finance", "punjab_budget_analysis.csv")
        
    if not os.path.exists(filepath):
        return None
        
    df = pd.read_csv(filepath)
    
    if adjust_for_inflation:
        # Create 'Real' columns based on CPI index
        df["allocation_real_bn"] = (df["allocation_pkr_bn"] / df["cpi_index"]) * 100
        df["expenditure_real_bn"] = (df["expenditure_pkr_bn"] / df["cpi_index"]) * 100
        print("[INFO] Applied CPI inflation adjustment (Base: 2015)")
        
    print(f"[OK] Loaded {len(df)} budget records from {os.path.basename(filepath)}")
    return df


# Data Cleaning

def clean_data(df):
    """
    Clean and prepare the district-level data.
    
    Steps:
    1. Strip whitespace from string columns
    2. Standardize district names
    3. Convert numeric columns to proper dtypes
    4. Handle missing values
    5. Add derived columns (gender literacy gap, etc.)
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data()
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for analysis
    """
    df = df.copy()
    
    # 1. Strip whitespace from all string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    
    # 2. Standardize district names (title case, fix known variants)
    name_mapping = {
        "D.G. Khan": "DG Khan",
        "D.G.Khan": "DG Khan",
        "Dera Ghazi Khan": "DG Khan",
        "R.Y. Khan": "Rahim Yar Khan",
        "R.Y.Khan": "Rahim Yar Khan",
        "RY Khan": "Rahim Yar Khan",
        "T.T. Singh": "Toba Tek Singh",
        "T.T.Singh": "Toba Tek Singh",
        "M.B.Din": "Mandi Bahauddin",
        "M.B. Din": "Mandi Bahauddin",
    }
    df["district"] = df["district"].replace(name_mapping)
    
    # 3. Ensure numeric columns have correct dtypes
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 4. Handle missing values — fill numeric NaNs with column median
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"[WARN] Filled {n_missing} missing values in '{col}' with median ({median_val:.2f})")
    
    # 5. Add derived columns
    df["gender_literacy_gap"] = df["male_literacy"] - df["female_literacy"]
    df["enrollment_ratio"] = (df["middle_enrollment_rate"] / df["primary_enrollment_rate"] * 100).round(1)
    
    # Ensure rural_urban_literacy_gap is calculated if raw columns exist
    if "urban_literacy" in df.columns and "rural_literacy" in df.columns:
        if "rural_urban_literacy_gap" not in df.columns:
            df["rural_urban_literacy_gap"] = (df["urban_literacy"] - df["rural_literacy"]).round(1)
    
    # Ensure literacy_change is calculated if 2017 data exists
    if "literacy_rate_2017" in df.columns:
        if "literacy_change" not in df.columns:
            df["literacy_change"] = (df["literacy_rate"] - df["literacy_rate_2017"]).round(1)
    
    print(f"✅ Cleaning complete: {len(df)} districts, {len(df.columns)} features")
    return df


# Filtering

def filter_south_punjab(df):
    """
    Filter dataframe to only South Punjab districts (11 districts).
    
    Parameters
    ----------
    df : pd.DataFrame
        Full Punjab dataframe
    
    Returns
    -------
    pd.DataFrame
        Filtered to South Punjab districts only
    """
    mask = df["district"].isin(SOUTH_PUNJAB_DISTRICTS)
    sp_df = df[mask].copy()
    
    if len(sp_df) == 0:
        print("⚠️  Warning: No South Punjab districts found! Check district name spelling.")
    else:
        print(f"✅ Filtered to {len(sp_df)} South Punjab districts")
    
    return sp_df


def filter_rest_of_punjab(df):
    """
    Filter dataframe to non-South Punjab districts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full Punjab dataframe
    
    Returns
    -------
    pd.DataFrame
        Filtered to Rest of Punjab districts only
    """
    mask = ~df["district"].isin(SOUTH_PUNJAB_DISTRICTS)
    rest_df = df[mask].copy()
    print(f"✅ Filtered to {len(rest_df)} Rest of Punjab districts")
    return rest_df


# Summary Statistics

def get_summary_stats(df, group_col="region"):
    """
    Generate summary statistics grouped by region.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe
    group_col : str
        Column to group by (default: 'region')
    
    Returns
    -------
    pd.DataFrame
        Summary statistics table
    """
    key_indicators = [
        "literacy_rate", "female_literacy", "poverty_headcount",
        "mpi_score", "unemployment_rate",
        "primary_enrollment_rate", "out_of_school_rate",
        "immunization_coverage", "clean_water_access",
        "sanitation_access", "internet_access",
        "gender_literacy_gap"
    ]
    
    # Filter to only columns that exist in the dataframe
    available = [col for col in key_indicators if col in df.columns]
    
    summary = df.groupby(group_col)[available].agg(["mean", "min", "max", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    
    return summary.round(2)


def get_district_profile(df, district_name):
    """
    Get a detailed profile for a specific district.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe
    district_name : str
        Name of the district
    
    Returns
    -------
    pd.Series
        All indicators for the specified district
    """
    match = df[df["district"].str.lower() == district_name.lower()]
    
    if len(match) == 0:
        available = ", ".join(df["district"].unique())
        raise ValueError(
            f"District '{district_name}' not found. Available: {available}"
        )
    
    return match.iloc[0]


def get_rankings(df, indicator, ascending=True):
    """
    Rank districts by a specific indicator.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe
    indicator : str
        Column name to rank by
    ascending : bool
        True = lowest first (good for poverty), False = highest first (good for literacy)
    
    Returns
    -------
    pd.DataFrame
        Ranked dataframe with rank column
    """
    ranked = df[["district", "division", "region", indicator]].copy()
    ranked = ranked.sort_values(indicator, ascending=ascending).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "rank"
    return ranked


# Main (for testing)

if __name__ == "__main__":
    print("South Punjab Development Dashboard — Data Loader Test")
        
    # Load & clean
    df = load_data()
    df = clean_data(df)
    
    # Filter
    sp = filter_south_punjab(df)
    rest = filter_rest_of_punjab(df)
    
    # Summary
    print("\n📊 Regional Summary Statistics:")
    print("-" * 40)
    summary = get_summary_stats(df)
    print(summary.to_string())
    
    # Rankings
    print("\n📉 Top 5 Most Impoverished Districts:")
    print("-" * 40)
    poverty_rank = get_rankings(df, "poverty_headcount", ascending=False)
    print(poverty_rank.head().to_string())
    
    print("\n📚 Top 5 Most Literate Districts:")
    print("-" * 40)
    literacy_rank = get_rankings(df, "literacy_rate", ascending=False)
    print(literacy_rank.head().to_string())
    
    print("\n✅ Data loader test passed!")
