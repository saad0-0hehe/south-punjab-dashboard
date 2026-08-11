# Punjab MICS — Longitudinal Development Study

An interactive dashboard built on UNICEF MICS Punjab household microdata (2011, 2014, 2017-18), tracking human development across Punjab's 36 districts at the household level.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)

---

## What This Does

- Constructs a household-level composite HDI from three MICS Punjab rounds covering **185,303 households** across 36 districts
- Validates the index against the independently published Naveed & Gordon (2024) district rankings (Spearman ρ = 0.969)
- Tests for **unconditional β-convergence**: initially poorer districts grew faster (β = −0.054, p < 0.001)
- Tracks **within-district and between-district inequality** from 2011 to 2017-18
- Identifies deprivation drivers via SHAP on a corrected, exogenous-features-only XGBoost model (honest R² = 0.42–0.48)

### South Punjab Districts (11)

| Multan Division | Bahawalpur Division | DG Khan Division |
|-----------------|---------------------|------------------|
| Multan          | Bahawalpur          | DG Khan          |
| Lodhran         | Bahawalnagar        | Muzaffargarh     |
| Khanewal        | Rahim Yar Khan      | Layyah           |
| Vehari          |                     | Rajanpur         |

---

## Data Sources

| Source | Indicators | Year |
|--------|-----------|------|
| UNICEF MICS Punjab 2011 | Household assets, education, child health | 2011 |
| UNICEF MICS Punjab 2014 | Household assets, education, child health | 2014 |
| UNICEF MICS Punjab 2017-18 | Household assets, education, child health | 2017-18 |
| Naveed & Gordon (2024) | Published district HDI (external validation) | 2024 |
| PDHS 2017-18 | Immunization coverage cross-check | 2018 |

Full citations in [`data/source_references.md`](data/source_references.md).

---

## Running Locally

```bash
git clone https://github.com/saad0-0hehe/Punjab-Longitudinal-Development-analysis-.git
cd Punjab-Longitudinal-Development-analysis-
pip install -r requirements.txt
streamlit run app.py
```

---

## Dashboard Pages

**MICS Longitudinal Study** *(default landing page)* — Five tabs:
- **The Map** — District HDI choropleth across three survey rounds; district trajectory chart
- **Convergence** — β-convergence scatter, convergence speed (λ), gap half-life
- **Inequality** — Within-district Gini dumbbell chart (2011 vs 2017-18); between-district Gini trend
- **Deprivation Drivers** — SHAP feature importance for Punjab, Southern Punjab, and Rest of Punjab
- **Validation & Rigor** — External validation metrics, robustness checks, methodology notes

**About & Sources** — Methodology, data source table, author.

---

## Selected Findings

- **Unconditional convergence confirmed**: β = −0.054 (p < 0.001); a district's 2011 HDI alone explains 55% of its subsequent growth rate
- **The South led the catching-up**: Rajanpur, DG Khan, and Muzaffargarh grew 4.5–5.6%/yr vs ~1%/yr for Lahore and Rawalpindi
- **Between-district inequality fell by ~32%** (spatial Gini, 2011→2017-18); within-district inequality fell in 31 of 36 districts
- **Top deprivation drivers** (SHAP, exogenous model): improved flooring, bank account ownership, open defecation, solid cooking fuel, agricultural land ownership
- **Electricity access** matters significantly more in Southern Punjab than in the rest of the province

---

## Archived Census Pages

The six PBS Census 2023-based pages (Overview, District Profiles, Indicators, Trends 2011–2023, Budget Accountability, Poverty Co-Movement) have been archived to [`archive_census2023/`](archive_census2023/) and are not loaded by the live app. They remain intact for potential revival as a separate project. See [`archive_census2023/README_ARCHIVE.md`](archive_census2023/README_ARCHIVE.md) for revival instructions.

---

## Author

**M Saad Sadaf**
