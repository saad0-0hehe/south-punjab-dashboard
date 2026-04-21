import json, pandas as pd

with open("data/pak_admin2.geojson", "r", encoding="utf-8") as f:
    geo = json.load(f)

# Filter Punjab using correct key
punjab = [feat for feat in geo["features"] if feat["properties"].get("adm1_name") == "Punjab"]
print(f"Punjab districts in GeoJSON: {len(punjab)}")

# List all GeoJSON names
geo_names = sorted([f["properties"]["adm2_name"] for f in punjab])
print("\nGeoJSON district names:")
for n in geo_names:
    print(f"  {n}")

# CSV names
df = pd.read_csv("data/district_socioeconomic.csv")
csv_names = sorted(df["district"].unique())
print(f"\nCSV districts: {len(csv_names)}")
for n in csv_names:
    print(f"  {n}")

# Auto-match
geo_set = set(geo_names)
csv_set = set(csv_names)
print(f"\nMatched: {len(geo_set & csv_set)}")
print(f"In GeoJSON but not CSV: {geo_set - csv_set}")
print(f"In CSV but not GeoJSON: {csv_set - geo_set}")

# Save Punjab-only GeoJSON
punjab_geo = {"type": "FeatureCollection", "features": punjab}
with open("data/punjab_districts.geojson", "w", encoding="utf-8") as f:
    json.dump(punjab_geo, f)
print(f"\nSaved: data/punjab_districts.geojson ({len(punjab)} features)")
