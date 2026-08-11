# Archive: PBS Census 2023 — Dashboard Pages

> **Status:** Archived — not loaded by the live app.  
> **Reason:** The dashboard was refactored (August 2025) to a MICS-only view.  
> **Revival:** These modules are complete and can be dropped back into `src/` to restore the six census-based pages.

---

## What Is Here

| File | Powers | Notes |
|------|--------|-------|
| `src/choropleth.py` | **Overview** page — district choropleth map | Requires `data/punjab_districts.geojson` |
| `src/eda.py` | **Indicators** page — all 7 chart tabs | Matplotlib/Seaborn plots |
| `src/historical_analyzer.py` | **Trends 2011–2023** + **Budget** pages | Growth-rate and budget-summary helpers |
| `src/historical_viz.py` | **Trends 2011–2023** + **Budget** pages | Matplotlib trend/disparity charts |
| `src/ml_explainer.py` | **Poverty Co-Movement** page — SHAP tab | Wraps SHAP library |
| `src/ml_model.py` | **Poverty Co-Movement** page — Linear/Ridge | sklearn training & evaluation helpers |
| `src/plotly_charts.py` | **Overview**, **District Profiles**, **Trends** | Radar, bubble, animated scatter, waterfall |

---

## To Revive the Census Pages

1. Copy all files in `archive_census2023/src/` back into `src/`.
2. In `app.py`, restore the full import block and the six `if/elif` page blocks from git history:
   ```
   git show main:app.py > app_original.py
   ```
3. The data files they depend on are still present in the repo:
   - `data/district_socioeconomic.csv`
   - `data/historical/district_history.csv`
   - `data/finance/punjab_budget_analysis.csv`
   - `data/punjab_districts.geojson`

---

## Pages That Were Archived

| Page | Description |
|------|-------------|
| **Overview** | Headline metric cards, choropleth map, South Punjab vs Rest comparison, top-10 rankings |
| **District Profiles** | Full indicator explorer for any of 36 districts, radar chart, comparison table |
| **Indicators** | Seven EDA tabs: literacy, poverty, education, health, infrastructure, temporal, correlations |
| **Trends 2011–2023** | 15-year trend lines, district overlays, CAGR tables, regional gap tracking, animated scatter |
| **Budget Accountability** | Nominal vs CPI-adjusted ADP allocations, promised vs spent waterfall, fiscal summary |
| **Poverty Co-Movement** | Linear & Ridge regression, feature importance, LOOCV alpha tuning, SHAP interpretability |
