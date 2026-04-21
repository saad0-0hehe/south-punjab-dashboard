import json

with open("data/pak_admin2.geojson", "r", encoding="utf-8") as f:
    geo = json.load(f)

# Show all property keys from first feature
feat0 = geo["features"][0]
print("Property keys:", list(feat0["properties"].keys()))
print()

# Show all unique province names
provinces = set()
for feat in geo["features"]:
    props = feat["properties"]
    # Try common keys
    for key in props:
        if "adm1" in key.lower() or "province" in key.lower() or "state" in key.lower():
            provinces.add(f"{key}={props[key]}")
    
print("Province entries found:")
for p in sorted(provinces):
    print(f"  {p}")

# Show first feature full properties
print()
print("First feature properties:")
for k, v in feat0["properties"].items():
    print(f"  {k}: {v}")
