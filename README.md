# 📊 South Punjab Development Dashboard

An interactive data analytics dashboard analyzing socioeconomic disparities across **11 South Punjab districts** compared to the rest of Punjab, Pakistan.

> Built as a portfolio project for BS Data Science (4th Semester) — Air University, Islamabad 

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Overview

South Punjab (Multan, Bahawalpur, DG Khan divisions) has historically faced development gaps compared to northern Punjab. This dashboard uses **real government data** to:

- **Visualize** literacy, poverty, health, and education indicators across 36 Punjab districts
- **Compare** South Punjab vs Rest of Punjab on key socioeconomic metrics
- **Predict** district-level poverty headcount using Linear and Ridge regression models
- **Explore** district profiles interactively via a Streamlit web app

### South Punjab Districts Covered (11)
| Multan Division | Bahawalpur Division | DG Khan Division |
|----------------|--------------------|--------------------|
| Multan | Bahawalpur | DG Khan |
| Lodhran | Bahawalnagar | Muzaffargarh |
| Khanewal | Rahim Yar Khan | Layyah |
| Vehari | | Rajanpur |

---

## 📁 Project Structure

```
south-punjab-dashboard/
├── app.py                          # Streamlit dashboard (main entry)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── district_socioeconomic.csv  # Curated district-level data (36 districts)
│   ├── source_references.md        # Data source citations
│   └── raw/                        # Downloaded HDX CSVs
│       ├── poverty_pak.csv
│       └── education_pak.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading, cleaning, filtering
│   ├── eda.py                      # EDA visualization functions
│   └── ml_model.py                 # ML regression models
│
├── notebooks/
│   ├── 01_eda.py                   # EDA walkthrough script
│   └── 02_ml_modeling.py           # ML modeling script
│
└── outputs/                        # Generated charts (created by scripts)
```

---

## 📊 Data Sources

All district-level data is compiled from **official government and UN publications**:

| Source | Indicators | Year |
|--------|-----------|------|
| **PBS Census 2023** | Population, literacy rates | 2023 |
| **PSLM 2019-20** | Poverty headcount, enrollment, health access | 2019-20 |
| **UNDP SDG Reports** | MPI scores | 2020 |
| **DHS 2017-18** | Immunization coverage | 2017-18 |
| **HDX** | National-level education & poverty trends | Various |

Full citations: [`data/source_references.md`](data/source_references.md)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd south-punjab-dashboard

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Run Analysis Scripts

```bash
# EDA — generates charts in outputs/
python notebooks/01_eda.py

# ML Modeling — trains models and evaluates
python notebooks/02_ml_modeling.py
```

---

## 🛠️ Tech Stack

- **Python 3.12** — Core language
- **Pandas** — Data cleaning and manipulation
- **Matplotlib / Seaborn** — Publication-quality visualizations
- **Scikit-learn** — Linear and Ridge regression models
- **Streamlit** — Interactive web dashboard

---

## 📈 Key Features

### Dashboard Pages
1. **🏠 Overview** — Key metric cards, top/bottom district rankings, literacy vs poverty scatter
2. **🏘️ District Profiles** — Dropdown explorer for any district with comparison to averages
3. **📈 EDA** — Tabbed visualizations (literacy, poverty, education, health, correlations)
4. **🤖 ML Predictions** — Model training, evaluation metrics, feature importance, residuals
5. **ℹ️ About** — Project methodology and data source documentation

### Analysis Highlights
- **9 EDA visualizations** including gender literacy gap, division comparisons, correlation heatmap
- **Linear & Ridge regression** predicting poverty from literacy + health indicators
- **Cross-validation** for Ridge alpha parameter tuning
- **36 districts** with 18 socioeconomic indicators each

---

## 👨‍💻 Author

**BS Data Science Student** — Air University, Islamabad  — 4th Semester
