# Data Source References

All district-level data in `district_socioeconomic.csv` is compiled from the following published government and UN sources:

## Primary Sources

### Pakistan Bureau of Statistics (PBS) — Census 2023
- **Website**: https://www.pbs.gov.pk
- **Publication**: 7th Population and Housing Census 2023 — Punjab Tables
- **Indicators sourced**: Population, area, population density, literacy rate (overall, male, female)
- **Reference year**: 2023

### Pakistan Social and Living Standards Measurement (PSLM) Survey 2019-20
- **Website**: https://www.pbs.gov.pk/content/pslm-district-level-survey-2019-20
- **Conducted by**: Pakistan Bureau of Statistics
- **Indicators sourced**: Poverty headcount ratio, primary/middle enrollment rates, immunization coverage, clean water access, electricity access
- **Reference year**: 2019-20
- **Granularity**: District-level published indicator tables

### UNDP South Punjab Regional SDGs Indicators Report
- **Published by**: UNDP Pakistan & Planning and Development Department, South Punjab
- **Indicators sourced**: Multidimensional Poverty Index (MPI) scores, supplementary health/education data
- **Key finding cited**: 31% of South Punjab population lives below the national poverty line

### Pakistan Demographic and Health Survey (DHS) 2017-18
- **Conducted by**: National Institute of Population Studies (NIPS)
- **Indicators sourced**: Supplementary immunization and health facility data
- **Reference year**: 2017-18

## HDX Raw Data (National-Level)
The following CSV files in `data/raw/` are downloaded directly from the Humanitarian Data Exchange:

- `poverty_pak.csv` — National poverty indicators (HDX/World Bank)
  - URL: https://data.humdata.org/dataset/c97d6e9c-895d-4018-b435-aafcde1d8fda
- `education_pak.csv` — National education indicators (HDX/UNESCO)
  - URL: https://data.humdata.org/dataset/25f2a580-b2bb-4260-b2f7-0a52c29a32a6

## Notes
- Literacy rates align with PBS Census 2023 published district tables
- Poverty headcount and enrollment rates align with PSLM 2019-20 district-level published indicators
- MPI scores are derived from OPHI/UNDP Global MPI methodology, applied at subnational level
- Hospital density figures are derived from Punjab Health Department annual reports
