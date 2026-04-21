# Data Source References

## South Punjab Development Dashboard

### Primary Sources

| # | Source | Data Extracted | Year | URL |
|---|--------|---------------|------|-----|
| 1 | **PBS Census 2023** — Table 12 (Punjab Districts) | Literacy rates (total/male/female/rural/urban), enrollment (primary→graduation), out-of-school children, dropout counts, never-attended school | 2023 | https://www.pbs.gov.pk/census/ |
| 2 | **PBS Census Archive** | Population, literacy, urban/rural (1998, 2017, 2023) | 1998-2023 | https://www.pbs.gov.pk/censusarchive/ |
| 3 | **PSLM District Archive** | District literacy, enrollment, water/sanitation trends | 2010-2020 | https://www.pbs.gov.pk/pslm-publications |
| 4 | **HIES Archive** | Household income, consumption, poverty trends (2010-2025) | 2010-2025 | https://www.pbs.gov.pk/hies-publications |
| 5 | **Punjab P&D (ADP)** | Annual Development Programme district/regional allocations | 2015-2025 | https://pnd.punjab.gov.pk/adp |
| 6 | **Punjab Finance (Budget)** | White Papers: Revised Estimates (Spent) vs Budget Estimates (Promised) | 2015-2025 | https://finance.punjab.gov.pk/budgets |
| 7 | **UNDP Pakistan MPI** | Historical Multidimensional Poverty estimates | 2010-2020 | https://www.undp.org/pakistan |

### Data Files

| File | Description | Rows × Cols |
|------|-------------|-------------|
| `district_socioeconomic.csv` | Master dataset with all merged indicators (2023 snapshot) | 36 × 43 |
| `historical/district_history.csv` | Longitudinal data for literacy, poverty, OOS (2011-2023) | 540 × 5 |
| `finance/punjab_budget_analysis.csv` | Annual ADP allocations and expenditures (Nominal & CPI-indexed) | 20 × 5 |
| `raw/pbs_census_2023_extracted.csv` | Raw extraction from PBS Census Table 12 PDF | 36 × 18 |

### Derived Indicators (Calculated)

| Indicator | Formula | Source Columns |
|-----------|---------|---------------|
| `out_of_school_rate` | out_of_school_5_16 / (out_of_school + enrolled) × 100 | PBS Census 2023 |
| `dropout_rate` | dropout_5_16 / total_5_16_cohort × 100 | PBS Census 2023 |
| `never_attended_rate` | never_attended_school / population_5plus × 100 | PBS Census 2023 |
| `higher_education_rate` | (intermediate + graduation) / population_10plus × 100 | PBS Census 2023 |
| `population_growth_rate` | ((pop_2023 / pop_2017)^(1/6) - 1) × 100 | Census 2017 + 2023 |
| `literacy_change` | literacy_rate_2023 - literacy_rate_2017 | Census 2017 + 2023 |
| `rural_urban_literacy_gap` | urban_literacy - rural_literacy | PBS Census 2023 |
| `gender_literacy_gap` | male_literacy - female_literacy | PBS Census 2023 |
| `enrollment_ratio` | middle_enrollment / primary_enrollment × 100 | PBS Census 2023 |

### Notes

- **PSLM data** is from the 2019-20 district-level survey round (latest available with district disaggregation).
- **Labour Force Survey (LFS)** provides provincial-level employment data only — not district-level. District unemployment estimates are derived from PSLM.
- **Mouza Census 2020** data portal (mc2020.pbos.gov.pk) is an interactive JS dashboard without downloadable CSV exports.
- **Census 2017 Punjab Report PDF** available at: https://www.pbs.gov.pk/wp-content/uploads/2020/07/pcr_punjab.pdf
- **PSLM 2024-25 Report PDF** available at: https://www.pbs.gov.pk/wp-content/uploads/2020/07/PSLM_Report_2024-25-Social-1.pdf
