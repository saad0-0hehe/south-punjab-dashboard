"""Quick test to verify the expanded dataset loads and all modules work."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, clean_data, filter_south_punjab, filter_rest_of_punjab
from src.eda import (plot_out_of_school, plot_infrastructure, 
                     plot_rural_urban_literacy, plot_temporal_comparison)
from src.ml_model import prepare_features, FEATURE_COLUMNS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("INTEGRATION TEST")
print("=" * 60)

# 1. Data loading
df = load_data()
df = clean_data(df)
sp = filter_south_punjab(df)
rest = filter_rest_of_punjab(df)

print(f"\nDataset shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# 2. Check new columns
new_cols = [
    "unemployment_rate", "out_of_school_rate", "sanitation_access",
    "internet_access", "rural_literacy", "urban_literacy",
    "literacy_change", "population_2017", "population_growth_rate",
    "dropout_rate", "never_attended_rate", "higher_education_rate",
]
print(f"\nNew columns check:")
all_ok = True
for c in new_cols:
    present = c in df.columns
    non_null = df[c].notna().sum() if present else 0
    status = "OK" if present and non_null > 0 else "MISSING"
    print(f"  {status} {c}: {non_null}/36 non-null")
    if not present:
        all_ok = False

# 3. Test new EDA charts
print(f"\nNew EDA charts:")
for name, func in [
    ("plot_out_of_school", plot_out_of_school),
    ("plot_infrastructure", plot_infrastructure),
    ("plot_rural_urban_literacy", plot_rural_urban_literacy),
    ("plot_temporal_comparison", plot_temporal_comparison),
]:
    try:
        fig = func(df)
        plt.close(fig)
        print(f"  OK {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        all_ok = False

# 4. Test ML model with expanded features
print(f"\nML Model:")
print(f"  Feature columns: {len(FEATURE_COLUMNS)}")
try:
    data = prepare_features(df)
    print(f"  OK Features prepared: {len(data['feature_names'])} features")
    print(f"  Features used: {data['feature_names']}")
except Exception as e:
    print(f"  FAIL prepare_features: {e}")
    all_ok = False

print(f"\n{'=' * 60}")
if all_ok:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED - check above")
print(f"{'=' * 60}")
