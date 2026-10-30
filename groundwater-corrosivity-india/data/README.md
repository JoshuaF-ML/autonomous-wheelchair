# Data

Nothing here is committed except this file. Put downloads in `raw/` untouched.

## Phase 0 - what to download (not yet done)

| Data | Where | Notes |
|---|---|---|
| CGWB groundwater quality (main) | data.gov.in (search "ground water quality"), India-WRIS | Take 2-3 recent years, all states available |
| Mean air temperature per well | NASA POWER climatology API, `T2M`, `community=AG` | Proxy for water temperature; cache per lat/lon |
| District boundaries | github.com/datameet/maps | For choropleths |

## Checklist before building on the data

For each state/year file, record in a table (`reports/data_availability.csv`):
state, year, n_rows, and the share of non-null values for
pH, EC, TDS, Ca, Mg, Na, K, HCO3, CO3, Cl, SO4, NO3, F, Fe, Latitude, Longitude.

LSI needs pH, Ca, HCO3 (and TDS or EC). Larson-Skold needs Cl, SO4, HCO3.
If Ca or HCO3 is mostly missing for a state, that state is limited to Larson-Skold
or dropped from the study; decide per state from the table.

Open questions to settle once the files are in hand:
- Are concentrations in mg/L and EC in uS/cm at 25 C in every file?
- Which TDS/EC factor fits (0.55-0.75)? Use wells that report both to fit it.
- Are samples tagged pre-/post-monsoon, or only by year?
