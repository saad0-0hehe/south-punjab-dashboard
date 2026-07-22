# Punjab Districts — Development Longitudinal Analysis

An interactive dashboard tracking socioeconomic disparities across Punjab's 36 districts, with a focus on the South Punjab development gap. Spans 15 years of government data (2011–2025) and includes a household-level longitudinal study from three rounds of MICS Punjab microdata.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)

---

## What This Does

- Compares 43 socioeconomic indicators across all 36 Punjab districts — literacy, poverty, health, infrastructure, budget
- Tracks 15-year developmental trends (2011–2023) using PSLM and Census anchor points
- Analyzes provincial budget disparity: ADP allocations vs. actual expenditure, nominal and inflation-adjusted
- Presents a household-level MICS longitudinal study (2011, 2014, 2017-18): composite HDI, convergence econometrics, within-district inequality, and SHAP-based deprivation drivers
- Predicts district-level poverty using regression models with feature importance analysis

### South Punjab Districts (11)

| Multan Division | Bahawalpur Division | DG Khan Division |
|-----------------|---------------------|------------------|
| Multan          | Bahawalpur          | DG Khan          |
| Lodhran         | Bahawalnagar        | Muzaffargarh     |
| Khanewal        | Rahim Yar Khan      | Layyah           |
| Vehari          |                     | Rajanpur         |

---

## Data Sources

All data is from official government and UN publications:

| Source | Indicators | Year |
|--------|-----------|------|
| PBS Census 2023 (Table 12) | Literacy, enrollment, out-of-school, dropout | 2023 |
| PBS Census Archive | Historical population and literacy | 2017, 1998 |
| PSLM District Surveys | Unemployment, sanitation, internet, education trends | 2010–2020 |
| HIES Archive | Household income, poverty, consumption trends | 2010–2025 |
| Punjab P&D / ADP | Annual Development Programme allocations | 2015–2025 |
| Punjab Finance Dept | White Papers: Revised vs Budget Estimates | 2015–2025 |
| UNDP / MPI | Poverty headcount, Multidimensional Poverty Index | 2020 |
| UNICEF MICS Punjab | Household microdata: assets, education, child health | 2011, 2014, 2017-18 |

Full citations in [`data/source_references.md`](data/source_references.md).

---

## Running Locally

```bash
git clone https://github.com/saad0-0hehe/punjab-development-analysis.git
cd punjab-development-analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## Dashboard Pages

**Overview** — Headline metric cards, South Punjab vs Rest comparison, top-10 rankings, literacy-poverty scatter.

**District Profiles** — Full indicator explorer for any district, with comparison against regional and provincial averages.

**Indicators** — Seven tabs covering literacy, poverty, education, health, infrastructure, temporal change, and correlations.

**Trends 2011–2023** — 15-year trend lines for any indicator, district overlays, CAGR tables, regional gap tracking.

**Budget Accountability** — Nominal vs inflation-adjusted ADP allocations, promised vs actually spent, fiscal performance.

**Which Indicators Track Deprivation** — Linear and Ridge regression models, feature importance, LOOCV alpha tuning, residual analysis.

**MICS Longitudinal Study** — Household-level HDI across three survey rounds, validated against Naveed & Gordon (2024). Beta-convergence, within-district Gini trajectories, SHAP feature importance from corrected exogenous-features-only ML analysis.

**About & Sources** — Methodology, data source table, limitations.

---

## Selected Findings

- 9 of the top 10 most impoverished Punjab districts are in South Punjab
- Rajanpur has the lowest literacy (36.1%) and highest poverty (58.4%) in all of Punjab
- The out-of-school rate in South Punjab is nearly double that of central/northern Punjab
- In real (inflation-adjusted) terms, South Punjab's ADP growth has been largely eroded by inflation
- MICS household-level analysis confirms strong unconditional convergence: initially poorer districts grew systematically faster (β = −0.054, p < 0.001), with health converging fastest

---

## Author

**M Saad Sadaf**
