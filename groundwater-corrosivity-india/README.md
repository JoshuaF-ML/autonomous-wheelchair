# Groundwater corrosivity across India

Compute standard corrosivity indices (Langelier, Ryznar, Larson-Skold) for CGWB
groundwater samples, map hotspots by district, and train a model that predicts
corrosion risk from cheap field measurements (pH, EC, location) so a full lab
analysis is not needed. Goal: guide pipe-material choice (GI/steel vs uPVC/HDPE)
for rural water supply.

## Status

| Phase | State |
|---|---|
| Scaffold, `src/indices.py`, `src/cleaning.py`, tests | done (22 tests pass) |
| 0. Data acquisition and column check | **blocked in the dev sandbox** (see below); run where data.gov.in is reachable |
| 1. Cleaning and QC on real data | not started |
| 2-6. Features, EDA/maps, ML, validation, Streamlit app, report | not started |

The labels are computed from water chemistry, not measured pipe corrosion. State this
as a limitation in the report.

## Run the tests

    pip install pandas numpy pytest
    python -m pytest

## Layout

    src/indices.py    LSI, RSI, Larson-Skold, charge balance, risk_class
    src/cleaning.py   BDL parsing, QC filters with a per-rule removal report
    tests/            index checks (hand-worked example), parsing, QC
    data/             raw -> interim -> processed (git-ignored, see data/README.md)
    notebooks/        01 acquisition, 02 cleaning, 03 labels, 04 EDA/maps, 05 modelling, 06 interpretation
    app/              Streamlit "Check your water" (later)

## Risk class rules

High: LSI < -2, or (LSI < -0.5 and LS > 1.2). Moderate: exactly one of LSI < -0.5 / LS > 1.2.
Scaling: LSI > 0.5 and LS < 0.8. Everything else is Balanced. (LSI > 0.5 with LS in
[0.8, 1.2] lands in Balanced; revisit once real data shows how common that is.)

## ML framing (to avoid leakage)

Models must not see the inputs the label is computed from. Task A uses pH, EC, lat/lon,
season and climate; Task B uses location and climate only. Validate with GroupKFold by
district (or leave-one-state-out) and a temporal holdout, never a random split.
