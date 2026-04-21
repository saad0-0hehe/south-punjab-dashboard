# 📊 South Punjab Development Dashboard

An interactive, multi-source socioeconomic analytics dashboard analyzing **development disparities across 11 South Punjab districts** vs. the rest of Punjab, Pakistan — spanning 15 years of data (2011–2025).

> Built as a portfolio project for BS Data Science (4th Semester) — Air University, Islamabad

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Data](https://img.shields.io/badge/Data-PBS%20%7C%20PSLM%20%7C%20Budget-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Overview

South Punjab (Multan, Bahawalpur, DG Khan divisions) has historically faced significant development gaps compared to northern Punjab. This dashboard uses **real government data** from multiple official sources to:

- **Visualize** 43 socioeconomic indicators across 36 Punjab districts
- **Compare** South Punjab vs Rest of Punjab across literacy, poverty, health, infrastructure, and budget
- **Track** 15-year developmental trends (2011–2023) using PSLM + Census anchor points
- **Analyze** provincial budget disparity: Promised ADP allocations vs. Actual Expenditure
- **Predict** district-level poverty headcount using Linear and Ridge regression models
- **Explore** any district interactively through a premium Streamlit dashboard

### South Punjab Districts Covered (11)

| Multan Division | Bahawalpur Division | DG Khan Division |
|----------------|--------------------|--------------------|
| Multan         | Bahawalpur         | DG Khan            |
| Lodhran        | Bahawalnagar       | Muzaffargarh       |
| Khanewal       | Rahim Yar Khan     | Layyah             |
| Vehari         |                    | Rajanpur           |

---

## 📁 Project Structure

```
south-punjab-dashboard/
├── app.py                              # Streamlit dashboard (7 pages)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── test_integration.py                 # Automated integration tests
│
├── data/
│   ├── district_socioeconomic.csv      # Master dataset (36 districts × 43 indicators)
│   ├── source_references.md            # Data source citations
│   ├── historical/
│   │   └── district_history.csv        # Longitudinal data (2011–2023, 5 anchor years)
│   ├── finance/
│   │   └── punjab_budget_analysis.csv  # ADP allocations + CPI indices (2015–2024)
│   └── raw/
│       ├── pbs_census_2023_extracted.csv
│       ├── table_12_punjab_districts.pdf
│       ├── merge_datasets.py           # Data pipeline script
│       └── generate_historical_data.py # Historical data compilation script
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Load, clean, filter + historical/budget loaders
│   ├── eda.py                          # 13 EDA visualization functions
│   ├── ml_model.py                     # Linear + Ridge regression models
│   ├── historical_analyzer.py          # CAGR and budget utilization analysis
│   └── historical_viz.py              # Time-series + budget chart functions
│
├── notebooks/
│   └── 01_eda.py                       # EDA walkthrough script
│
└── outputs/                            # Generated charts
```

---

## 📊 Data Sources

All data is compiled from **official government and UN publications**:

| # | Source | Indicators | Year |
|---|--------|-----------|------|
| 1 | **PBS Census 2023** (Table 12) | Literacy (total/male/female/rural/urban), enrollment (primary→graduation), out-of-school, dropout | 2023 |
| 2 | **PBS Census Archive** | Historical population + literacy rates | 2017, 1998 |
| 3 | **PSLM District Surveys** | Unemployment, sanitation, internet access, district education trends | 2010–2020 |
| 4 | **HIES Archive** | Household income, poverty, consumption trends | 2010–2025 |
| 5 | **Punjab P&D / ADP** | Annual Development Programme district/regional allocations | 2015–2025 |
| 6 | **Punjab Finance Dept** | White Papers: Revised Estimates vs Budget Estimates (actual spend) | 2015–2025 |
| 7 | **UNDP / MPI** | Poverty headcount, Multidimensional Poverty Index scores | 2020 |

Full citations: [`data/source_references.md`](data/source_references.md)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/saad0-0hehe/south-punjab-dashboard.git
cd south-punjab-dashboard

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.12** | Core language |
| **Pandas / NumPy** | Data cleaning and manipulation |
| **Matplotlib / Seaborn** | Publication-quality visualizations |
| **Scikit-learn** | Linear and Ridge regression models |
| **Streamlit** | Interactive web dashboard |
| **Requests / pdfplumber** | Data pipeline and PDF extraction |

---

## 📈 Dashboard Pages (7)

### 🏠 Overview
- 8 headline metric cards (literacy, poverty, unemployment, out-of-school, internet, sanitation, immunization, gender gap)
- South Punjab vs Rest of Punjab at a glance
- Top 10 most impoverished + most literate districts rankings
- Literacy vs Poverty scatter plot

### 🏘️ District Profiles
- Full indicator explorer for any of 36 Punjab districts
- 16 metric cards per district with vs-average deltas
- Comparison table: District vs South Punjab Avg vs All Punjab Avg

### 📈 EDA (7 Tabs)
1. **📚 Literacy** — District literacy bars + gender gap chart
2. **💰 Poverty** — Poverty rankings + South vs Rest radar
3. **🏫 Education** — Enrollment trends + out-of-school chart + division comparison
4. **🏥 Health** — Immunization + clean water heatmap
5. **🏗️ Infrastructure** — Sanitation/internet/electricity + rural-urban literacy gap
6. **📅 Temporal** — 2017→2023 literacy change chart
7. **🔗 Correlations** — Heatmap + literacy-poverty scatter

### 📅 Temporal Trends *(New)*
- Select any indicator (literacy, poverty, out-of-school)
- 15-year trend lines for South vs Rest of Punjab (2011–2023 across 5 anchor years)
- District overlay selector to compare specific districts
- CAGR (Compound Annual Growth Rate) table — who improved fastest?
- Regional performance gap chart over time

### 💰 Budget Accountability *(New)*
- **Nominal vs Inflation-Adjusted** toggle (CPI-indexed to 2015 baseline)
- Annual Development Programme allocations: South vs Rest of Punjab
- Promised vs Actually Spent comparison (utilization rate)
- Fiscal performance summary table

### 🤖 ML Predictions
- Linear + Ridge Regression predicting poverty from 14 socioeconomic features
- R², MAE, RMSE metrics
- Feature importance charts (both models)
- Leave-One-Out Cross-Validation for Ridge alpha tuning
- Residual analysis

### ℹ️ About
- Project objective, methodology, tech stack
- Complete data source table with 7 sources
- Limitations and caveats

---

## 🔬 Key Findings

- **9 of the top 10** most impoverished Punjab districts are in South Punjab
- South Punjab has **~18% higher poverty** and **~15% lower literacy** than the rest of Punjab on average
- **Rajanpur** has the lowest literacy (36.1%) and highest poverty (58.4%) in all of Punjab
- The **out-of-school rate** in South Punjab is nearly double that of central/northern Punjab
- In **real (inflation-adjusted) terms**, South Punjab's ADP allocation growth has been largely eroded by Pakistan's high inflation (2022–2023)
- **Sanitation access** in DG Khan division averages only ~42% vs. 94% in Lahore

---

## 👨‍💻 Author

**BS Data Science Student** — Air University, Islamabad — 4th Semester
