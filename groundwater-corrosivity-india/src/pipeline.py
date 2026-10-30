"""Raw Telangana files -> QC'd table with corrosivity indices and risk labels."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import indices as ix
from .cleaning import qc_filter
from .climate import add_temperature
from .data_loading import load_telangana

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
GW_TEMP_OFFSET_C = 1.0   # groundwater is typically ~1-2 C above mean annual air temperature
REQUIRED = ["pH", "EC", "Ca", "HCO3", "Cl", "SO4", "Mg", "Na", "K"]


def build(temp_offset=GW_TEMP_OFFSET_C):
    """Returns (clean_df, qc_report). Rows missing any required input are dropped first."""
    df = load_telangana()
    n0 = len(df)
    df["CO3"] = df["CO3"].fillna(0.0)          # blank carbonate in the 2019 file means none reported
    df = add_temperature(df)
    df["CBE"] = ix.charge_balance_error(df)
    have = df.dropna(subset=REQUIRED + ["Latitude", "Longitude", "T_air_C"]).copy()
    clean, report = qc_filter(have)
    report = pd.concat([pd.DataFrame([
        {"rule": "start", "rows_removed": 0},
        {"rule": "missing required input", "rows_removed": n0 - len(have)}]), report], ignore_index=True)
    clean = add_indices(clean, temp_offset)
    return clean.reset_index(drop=True), report


def add_indices(df, temp_offset=GW_TEMP_OFFSET_C):
    out = df.copy()
    tds = out["TDS"].fillna(out["EC"] * ix.EC_TO_TDS)
    t = out["T_air_C"] + temp_offset
    args = (tds, t, out["Ca"], out["HCO3"], out["CO3"])
    out["LSI"] = ix.lsi(out["pH"], *args)
    out["RSI"] = ix.rsi(out["pH"], *args)
    out["LS"] = ix.larson_skold(out["Cl"], out["SO4"], out["HCO3"], out["CO3"])
    out["Risk"] = ix.risk_class(out["LSI"].to_numpy(), out["LS"].to_numpy())
    return out.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    clean, report = build()
    clean.to_csv(PROCESSED / "telangana_labelled.csv", index=False)
    report.to_csv(PROCESSED / "qc_report.csv", index=False)
    print(report.to_string(index=False))
    print(len(clean), "rows ->", PROCESSED / "telangana_labelled.csv")
