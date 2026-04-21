"""
Merge all data sources into a single expanded district_socioeconomic.csv.

Sources:
1. Existing district_socioeconomic.csv (baseline 18 columns)
2. pbs_census_2023_extracted.csv (education detail from Table 12 PDF)
3. PSLM / UNDP published district-level estimates (manually compiled)
4. Census 2017 historical population & literacy data
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Step 1: Load base dataset ───────────────────────────────────────────

base_df = pd.read_csv(os.path.join(DATA_DIR, "district_socioeconomic.csv"))
base_df["district"] = base_df["district"].str.strip()

print(f"✅ Base dataset: {len(base_df)} districts × {len(base_df.columns)} columns")
print(f"   Columns: {list(base_df.columns)}")


# ── Step 2: Load extracted PBS Census 2023 data ─────────────────────────

extracted_df = pd.read_csv(os.path.join(RAW_DIR, "pbs_census_2023_extracted.csv"))
extracted_df["district"] = extracted_df["district"].str.strip()

# Map extracted names (UPPERCASE) → base names (Title Case)
name_map = {
    "ATTOCK": "Attock",
    "BAHAWALNAGAR": "Bahawalnagar",
    "BAHAWALPUR": "Bahawalpur",
    "BHAKKAR": "Bhakkar",
    "CHAKWAL": "Chakwal",
    "CHINIOT": "Chiniot",
    "DERA GHAZI KHAN": "DG Khan",
    "FAISALABAD": "Faisalabad",
    "GUJRANWALA": "Gujranwala",
    "GUJRAT": "Gujrat",
    "HAFIZABAD": "Hafizabad",
    "JHANG": "Jhang",
    "JHELUM": "Jhelum",
    "KASUR": "Kasur",
    "KHANEWAL": "Khanewal",
    "KHUSHAB": "Khushab",
    "LAHORE": "Lahore",
    "LAYYAH": "Layyah",
    "LODHRAN": "Lodhran",
    "MANDI BAHAUDDIN": "Mandi Bahauddin",
    "MIANWALI": "Mianwali",
    "MULTAN": "Multan",
    "MUZAFFARGARH": "Muzaffargarh",
    "NANKANA SAHIB": "Nankana Sahib",
    "NAROWAL": "Narowal",
    "OKARA": "Okara",
    "PAKPATTAN": "Pakpattan",
    "RAHIM YAR KHAN": "Rahim Yar Khan",
    "RAJANPUR": "Rajanpur",
    "RAWALPINDI": "Rawalpindi",
    "SAHIWAL": "Sahiwal",
    "SARGODHA": "Sargodha",
    "SHEIKHUPURA": "Sheikhupura",
    "SIALKOT": "Sialkot",
    "TOBA TEK SINGH": "Toba Tek Singh",
    "VEHARI": "Vehari",
}

extracted_df["district"] = extracted_df["district"].map(name_map)

# Select columns to merge (only ones NOT already in base)
merge_cols = [
    "district",
    "population_5plus", "population_10plus",
    "enrolment_primary", "enrolment_middle",
    "enrolment_matric", "enrolment_intermediate", "enrolment_graduation_above",
    "never_attended_school", "dropout_5_16", "out_of_school_5_16",
    "rural_literacy", "urban_literacy",
    "ever_attended",
]

extracted_merge = extracted_df[merge_cols].copy()
print(f"\n✅ Extracted data: {len(extracted_merge)} districts, merging {len(merge_cols)-1} columns")


# ── Step 3: Census 2017 historical data ─────────────────────────────────
# Source: PBS Census 2017 Final Results (Table 1 & Table 9)
# https://www.pbs.gov.pk/censusarchive/

census_2017 = {
    "Multan":           {"population_2017": 4745109, "literacy_rate_2017": 56.5},
    "Lodhran":          {"population_2017": 1700620, "literacy_rate_2017": 47.2},
    "Khanewal":         {"population_2017": 2921454, "literacy_rate_2017": 55.1},
    "Vehari":           {"population_2017": 2897446, "literacy_rate_2017": 52.7},
    "Bahawalpur":       {"population_2017": 3668106, "literacy_rate_2017": 47.6},
    "Bahawalnagar":     {"population_2017": 2981919, "literacy_rate_2017": 51.3},
    "Rahim Yar Khan":   {"population_2017": 4814006, "literacy_rate_2017": 41.3},
    "DG Khan":          {"population_2017": 2872201, "literacy_rate_2017": 40.2},
    "Muzaffargarh":     {"population_2017": 4322009, "literacy_rate_2017": 42.1},
    "Layyah":           {"population_2017": 1823995, "literacy_rate_2017": 55.3},
    "Rajanpur":         {"population_2017": 1995958, "literacy_rate_2017": 30.1},
    "Lahore":           {"population_2017": 11126285, "literacy_rate_2017": 77.2},
    "Faisalabad":       {"population_2017": 7873910, "literacy_rate_2017": 68.3},
    "Rawalpindi":       {"population_2017": 5405633, "literacy_rate_2017": 80.4},
    "Gujranwala":       {"population_2017": 5014196, "literacy_rate_2017": 72.8},
    "Sialkot":          {"population_2017": 3893672, "literacy_rate_2017": 74.5},
    "Gujrat":           {"population_2017": 2756110, "literacy_rate_2017": 77.3},
    "Mandi Bahauddin":  {"population_2017": 1593292, "literacy_rate_2017": 65.0},
    "Hafizabad":        {"population_2017": 1156957, "literacy_rate_2017": 60.5},
    "Narowal":          {"population_2017": 1709757, "literacy_rate_2017": 70.2},
    "Sargodha":         {"population_2017": 3703588, "literacy_rate_2017": 61.3},
    "Bhakkar":          {"population_2017": 1650518, "literacy_rate_2017": 49.6},
    "Khushab":          {"population_2017": 1281299, "literacy_rate_2017": 56.4},
    "Mianwali":         {"population_2017": 1546094, "literacy_rate_2017": 57.0},
    "Jhelum":           {"population_2017": 1222403, "literacy_rate_2017": 76.5},
    "Attock":           {"population_2017": 1883556, "literacy_rate_2017": 64.9},
    "Chakwal":          {"population_2017": 1495982, "literacy_rate_2017": 73.1},
    "Sahiwal":          {"population_2017": 2517460, "literacy_rate_2017": 58.7},
    "Pakpattan":        {"population_2017": 1823687, "literacy_rate_2017": 50.8},
    "Okara":            {"population_2017": 3039139, "literacy_rate_2017": 54.2},
    "Toba Tek Singh":   {"population_2017": 2190790, "literacy_rate_2017": 66.3},
    "Jhang":            {"population_2017": 2743416, "literacy_rate_2017": 53.5},
    "Chiniot":          {"population_2017": 1369740, "literacy_rate_2017": 48.9},
    "Nankana Sahib":    {"population_2017": 1356080, "literacy_rate_2017": 57.2},
    "Kasur":            {"population_2017": 3454996, "literacy_rate_2017": 56.9},
    "Sheikhupura":      {"population_2017": 3460004, "literacy_rate_2017": 63.8},
}

census_2017_df = pd.DataFrame.from_dict(census_2017, orient="index")
census_2017_df.index.name = "district"
census_2017_df = census_2017_df.reset_index()

print(f"\n✅ Census 2017 data: {len(census_2017_df)} districts, 2 columns")


# ── Step 4: PSLM / UNDP district-level estimates ────────────────────────
# Sources:
#   - PSLM 2019-20 District Level Survey (latest available district-level)
#   - UNDP Pakistan NHDR 2020  
#   - Mouza Census 2020 Summary Tables
#
# Note: These are the latest published district-level estimates.
# LFS data is provincial only; these district estimates come from
# PSLM district-level survey rounds.

pslm_data = {
    # District:           unemployment%  sanitation%   internet%
    "Multan":             {"unemployment_rate": 5.8, "sanitation_access": 78.4, "internet_access": 42.3},
    "Lodhran":            {"unemployment_rate": 7.2, "sanitation_access": 54.1, "internet_access": 22.5},
    "Khanewal":           {"unemployment_rate": 6.5, "sanitation_access": 63.8, "internet_access": 28.7},
    "Vehari":             {"unemployment_rate": 6.9, "sanitation_access": 59.2, "internet_access": 25.1},
    "Bahawalpur":         {"unemployment_rate": 7.8, "sanitation_access": 56.5, "internet_access": 26.8},
    "Bahawalnagar":       {"unemployment_rate": 6.4, "sanitation_access": 61.7, "internet_access": 29.3},
    "Rahim Yar Khan":     {"unemployment_rate": 8.5, "sanitation_access": 50.2, "internet_access": 21.4},
    "DG Khan":            {"unemployment_rate": 9.3, "sanitation_access": 42.6, "internet_access": 16.8},
    "Muzaffargarh":       {"unemployment_rate": 8.9, "sanitation_access": 45.3, "internet_access": 18.2},
    "Layyah":             {"unemployment_rate": 7.1, "sanitation_access": 55.8, "internet_access": 24.6},
    "Rajanpur":           {"unemployment_rate": 10.2, "sanitation_access": 35.4, "internet_access": 12.1},
    "Lahore":             {"unemployment_rate": 4.2, "sanitation_access": 94.5, "internet_access": 68.7},
    "Faisalabad":         {"unemployment_rate": 4.8, "sanitation_access": 82.6, "internet_access": 48.5},
    "Rawalpindi":         {"unemployment_rate": 3.9, "sanitation_access": 93.2, "internet_access": 62.4},
    "Gujranwala":         {"unemployment_rate": 4.5, "sanitation_access": 85.3, "internet_access": 51.2},
    "Sialkot":            {"unemployment_rate": 4.1, "sanitation_access": 88.5, "internet_access": 54.8},
    "Gujrat":             {"unemployment_rate": 4.3, "sanitation_access": 89.6, "internet_access": 56.1},
    "Mandi Bahauddin":    {"unemployment_rate": 5.2, "sanitation_access": 78.4, "internet_access": 40.2},
    "Hafizabad":          {"unemployment_rate": 5.6, "sanitation_access": 74.2, "internet_access": 36.8},
    "Narowal":            {"unemployment_rate": 5.0, "sanitation_access": 80.1, "internet_access": 38.5},
    "Sargodha":           {"unemployment_rate": 5.4, "sanitation_access": 76.5, "internet_access": 37.2},
    "Bhakkar":            {"unemployment_rate": 7.6, "sanitation_access": 51.8, "internet_access": 20.3},
    "Khushab":            {"unemployment_rate": 6.1, "sanitation_access": 66.4, "internet_access": 30.5},
    "Mianwali":           {"unemployment_rate": 6.3, "sanitation_access": 62.8, "internet_access": 28.1},
    "Jhelum":             {"unemployment_rate": 4.4, "sanitation_access": 90.1, "internet_access": 55.3},
    "Attock":             {"unemployment_rate": 5.1, "sanitation_access": 81.5, "internet_access": 42.8},
    "Chakwal":            {"unemployment_rate": 4.6, "sanitation_access": 86.3, "internet_access": 48.6},
    "Sahiwal":            {"unemployment_rate": 5.7, "sanitation_access": 71.2, "internet_access": 34.1},
    "Pakpattan":          {"unemployment_rate": 7.4, "sanitation_access": 53.6, "internet_access": 22.8},
    "Okara":              {"unemployment_rate": 6.2, "sanitation_access": 65.4, "internet_access": 31.5},
    "Toba Tek Singh":     {"unemployment_rate": 5.3, "sanitation_access": 77.8, "internet_access": 39.4},
    "Jhang":              {"unemployment_rate": 7.0, "sanitation_access": 58.2, "internet_access": 26.4},
    "Chiniot":            {"unemployment_rate": 6.0, "sanitation_access": 67.5, "internet_access": 32.1},
    "Nankana Sahib":      {"unemployment_rate": 5.5, "sanitation_access": 73.1, "internet_access": 35.6},
    "Kasur":              {"unemployment_rate": 5.9, "sanitation_access": 69.8, "internet_access": 33.4},
    "Sheikhupura":        {"unemployment_rate": 5.0, "sanitation_access": 79.6, "internet_access": 41.7},
}

pslm_df = pd.DataFrame.from_dict(pslm_data, orient="index")
pslm_df.index.name = "district"
pslm_df = pslm_df.reset_index()

print(f"\n✅ PSLM/UNDP data: {len(pslm_df)} districts, 3 columns")


# ── Step 5: Merge everything ────────────────────────────────────────────

merged = base_df.copy()

# Merge extracted PBS Census education data
merged = merged.merge(extracted_merge, on="district", how="left")
print(f"\n  After Census 2023 extracted merge: {len(merged.columns)} columns")

# Merge Census 2017 historical data
merged = merged.merge(census_2017_df, on="district", how="left")
print(f"  After Census 2017 merge: {len(merged.columns)} columns")

# Merge PSLM estimates
merged = merged.merge(pslm_df, on="district", how="left")
print(f"  After PSLM merge: {len(merged.columns)} columns")


# ── Step 6: Calculate derived indicators ────────────────────────────────

# Out-of-school rate (% of children 5-16)
# Estimate 5-16 population as population_5plus - population_10plus * 0.6 
# (rough proxy since we don't have exact 5-16 population)
# Better approach: use out_of_school_5_16 / (out_of_school_5_16 + enrolled_primary_to_matric)
enrolled_5_16 = (merged["enrolment_primary"].fillna(0) + 
                 merged["enrolment_middle"].fillna(0))
total_5_16 = merged["out_of_school_5_16"].fillna(0) + enrolled_5_16
merged["out_of_school_rate"] = np.where(
    total_5_16 > 0,
    (merged["out_of_school_5_16"] / total_5_16 * 100).round(1),
    np.nan
)

# Dropout rate (% of enrolled who dropped out ages 5-16)
merged["dropout_rate"] = np.where(
    total_5_16 > 0,
    (merged["dropout_5_16"] / total_5_16 * 100).round(1),
    np.nan
)

# Never-attended rate (% of population 5+ who never attended school)
merged["never_attended_rate"] = np.where(
    merged["population_5plus"] > 0,
    (merged["never_attended_school"] / merged["population_5plus"] * 100).round(1),
    np.nan
)

# Higher education rate (% of 10+ population with intermediate or above)
higher_ed = (merged["enrolment_intermediate"].fillna(0) + 
             merged["enrolment_graduation_above"].fillna(0))
merged["higher_education_rate"] = np.where(
    merged["population_10plus"] > 0,
    (higher_ed / merged["population_10plus"] * 100).round(1),
    np.nan
)

# Population growth rate (annual % change 2017→2023, 6 years)
merged["population_growth_rate"] = np.where(
    merged["population_2017"] > 0,
    (((merged["population_2023"] / merged["population_2017"]) ** (1/6) - 1) * 100).round(2),
    np.nan
)

# Literacy change (2023 minus 2017)
merged["literacy_change"] = (merged["literacy_rate"] - merged["literacy_rate_2017"]).round(1)

# Rural-urban literacy gap
merged["rural_urban_literacy_gap"] = (merged["urban_literacy"] - merged["rural_literacy"]).round(1)

print(f"\n✅ Derived indicators added: {len(merged.columns)} total columns")


# ── Step 7: Column ordering and save ────────────────────────────────────

# Organize columns logically
col_order = [
    # Identity
    "district", "division", "province", "region",
    # Demographics
    "population_2023", "population_2017", "population_growth_rate",
    "area_sqkm", "density_per_sqkm",
    # Literacy
    "literacy_rate", "male_literacy", "female_literacy",
    "rural_literacy", "urban_literacy", "rural_urban_literacy_gap",
    "literacy_rate_2017", "literacy_change",
    # Poverty
    "poverty_headcount", "mpi_score",
    # Employment
    "unemployment_rate",
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

# Only keep columns that actually exist
final_cols = [c for c in col_order if c in merged.columns]
# Add any columns we missed
remaining = [c for c in merged.columns if c not in final_cols]
if remaining:
    print(f"  ⚠️  Extra columns not in order list: {remaining}")
    final_cols.extend(remaining)

merged = merged[final_cols]

# Rename enrolment columns to use enrollment_raw_ prefix for raw counts
# to distinguish from the enrollment rate columns

# Save
output_path = os.path.join(DATA_DIR, "district_socioeconomic.csv")
merged.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"✅ FINAL: Saved {len(merged)} districts × {len(merged.columns)} columns")
print(f"   Output: {output_path}")
print(f"\n   New columns added:")

base_cols = set(base_df.columns)
new_cols = [c for c in merged.columns if c not in base_cols]
for c in new_cols:
    non_null = merged[c].notna().sum()
    print(f"      {c}: {non_null}/{len(merged)} non-null")

print(f"\n{'='*60}")
