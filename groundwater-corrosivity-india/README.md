# Groundwater corrosivity across India

Compute standard corrosivity indices (Langelier, Ryznar, Larson-Skold) for CGWB
groundwater samples, map hotspots by district, and train a model that predicts
corrosion risk from cheap field measurements (pH, EC, location) so a full lab
analysis is not needed. Goal: guide pipe-material choice (GI/steel vs uPVC/HDPE)
for rural water supply.

## Status

Scope is now a **Telangana case study** (2018-2020, pre/post-monsoon), because it was the only full-chemistry
data reachable; see `RESULTS.md` for data sources, findings, ML results and limitations.

| Part | State |
|---|---|
| Indices, QC, label, 50 unit tests | done |
| Data loading, NASA POWER temperature, pipeline | done |
| Tables, figures (district bars, well maps, LSI vs LS, confusion matrix, SHAP) | done |
| ML with four validation designs | done (modest accuracy, see RESULTS.md) |
| Streamlit app (`app/streamlit_app.py`), model in `models/` | done, checked headlessly with Streamlit's test harness |
| Notebooks | not made; the pipeline is scripts in `src/` |
| All-India extension | not done (national portals unreachable from the build environment) |

The labels are computed from water chemistry, not measured pipe corrosion. State this as a limitation.

## Reproduce

    pip install -r requirements.txt
    # put the Telangana files in data/raw/telangana/ (sources in RESULTS.md), then:
    python -m src.pipeline && python -m src.analysis && python -m src.modeling && python -m src.train_final
    streamlit run app/streamlit_app.py
    python -m pytest

## Layout

    src/indices.py    LSI, RSI, Larson-Skold, charge balance, risk_class
    src/cleaning.py   BDL parsing, QC filters with a per-rule removal report
    src/data_loading.py, climate.py, pipeline.py, analysis.py, modeling.py, train_final.py
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
