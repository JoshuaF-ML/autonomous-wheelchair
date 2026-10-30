# Results: groundwater corrosivity in Telangana, 2018-2020

All numbers come from the files in `reports/`. Regenerate with:

    python -m src.pipeline && python -m src.analysis && python -m src.modeling && python -m src.train_final

## Scope change from the original plan

The plan was all-India. The national CGWB portals (data.gov.in, India-WRIS) could not be reached from the
build environment, and the only national file found (Kaggle "India's Ground Water Quality Statewise
2012-2021") has just pH, conductivity and temperature, which is not enough for the Langelier index.
The full-chemistry data found is the **Telangana State groundwater survey** (pH, EC, TDS, CO3, HCO3, Cl, F, NO3,
SO4, Na, K, Ca, Mg, hardness, lat/lon), pre- and post-monsoon, 2018-2020, redistributed on Kaggle:

- `sivapriyagarladinne/telangana-post-monsoon-ground-water-quality-data` (post-monsoon 2018-2020)
- `sadiyazubair/telangana-ground-water-classification` (pre-monsoon 2018-2020, xlsx)

So this is a **single-state case study**: 33 districts, about 370 wells, 2,196 samples. Original source
and licence not verified; confirm before publishing. TDS in the files is exactly 0.64 x EC (derived, not measured).

## Data quality

2,196 samples loaded; 1 dropped for a missing input; 55 dropped for charge-balance error above 10%; 2,140 kept.
The 2019 post-monsoon file leaves carbonate blank where other files write 0, treated as 0 (affects 161 rows,
a small term in alkalinity). One pH typo ("8..05") was recovered. Iron was not in the data, so the planned
iron cross-check was not possible.

## Findings

**Class mix (2,140 samples):** Scaling 21.1%, Balanced 35.7%, Moderate 42.2%, High 0.9%.

**Telangana water is mostly alkaline and hard.** Median LSI is +0.58, so most water forms carbonate scale rather
than dissolving pipe. Only 49 samples (2.3%) have LSI below -0.5.

**The "at-risk" label is almost entirely a Larson-Skold result.** Of 923 Moderate/High samples, 874 (94.7%)
are flagged by Larson-Skold alone (chloride + sulphate high relative to alkalinity), 29 by LSI alone, 20 by
both. 529 samples are simultaneously scale-forming (LSI > 0.5) and chloride/sulphate-aggressive (LS > 1.2); the
two indices disagree for them, and scale can protect steel, so Moderate probably overstates real risk there.

**Hotspots (share of samples Moderate/High):** Peddapalli 76%, Jogulamba (Gadwal) 74%, Mulugu 73%, Sircilla 72%,
Warangal (Urban) 69%. Lowest: Kumuram Bheem 9%, Karimnagar 13%, Hyderabad 15%. Caution: several districts have
only 4-6 wells (Mulugu 15 samples), so their percentages are noisy. See `reports/district_summary.csv`.

**Season:** dry-season (pre-monsoon) water is more often at risk: 48.8% vs 42.2% (2018), 49.0% vs 42.1% (2019),
39.1% vs 37.4% (2020). For wells sampled in both seasons of a year (1,025 pairs), median LSI is 0.60 pre vs 0.54
post, and post-monsoon LSI is higher at only 46% of wells.

**Label robustness:** assuming water is 0-3 C warmer than air changes 1.0-1.8% of labels; assuming a fixed 20 C
changes 6.5%; fixed 30 C changes 2.0%.

## Machine learning

Target: At-risk (Moderate+High) / Balanced / Scaling. High alone has only 20 samples. Models never see Ca, HCO3,
CO3, Cl, SO4 or the indices. Majority-class baseline macro-F1 is 0.20.

| Inputs | Validation | Best model | Macro-F1 | At-risk recall |
|---|---|---|---|---|
| pH + EC | unseen districts | Hist. gradient boosting | 0.585 | 0.65 |
| pH + EC + location + season + climate | unseen districts | Random forest | 0.610 | 0.70 |
| pH + EC + location + season + climate | train 2018-19, test 2020 | Hist. gradient boosting | 0.633 | 0.75 |
| location + season + climate only | unseen districts | Hist. gradient boosting | 0.387 | 0.46 |

- A random split inflates macro-F1 only slightly (0.65 vs 0.61 for the full feature set), because the models
  rely on chemistry, not on memorising neighbouring wells.
- **Location alone does not predict corrosivity** (0.36-0.39 on unseen districts, chance for three classes is 0.33),
  so the spatial-interpolation idea (Task B) did not work and should not be claimed.
- **Screening from pH and EC is only moderately accurate**: about 59-62% accuracy; about 30% of truly at-risk
  samples are missed (recall 0.70). SHAP shows EC (salinity) dominates, then pH; location and season matter little.
  It is a triage aid, not a replacement for a lab analysis.

## Limitations

- Labels are computed from water chemistry, not measured pipe corrosion; no pipe-failure data was used.
- One state only; results do not transfer to other states. The app says so.
- Air temperature (NASA POWER, 0.5-degree cells) plus 1 C stands in for water temperature.
- The risk rules leave 246 samples with LSI > 0.5 and Larson-Skold 0.8-1.2 in "Balanced" (see README).
- Dataset provenance and licence unverified (Kaggle redistribution).
- No district polygons: Telangana's 33 districts are not in the 2011 boundary files, so maps are well-point plots.
