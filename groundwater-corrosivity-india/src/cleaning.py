"""Cleaning and QC helpers for CGWB groundwater-quality tables."""
import re
import numpy as np
import pandas as pd

# India bounding box (incl. islands), a loose sanity check on coordinates
LAT_RANGE = (6.0, 37.5)
LON_RANGE = (68.0, 97.5)
PH_RANGE = (4.0, 11.0)

_BDL = re.compile(r"^\s*(bdl|nd|nil|<\s*[\d.]+)\s*$", re.I)
_LT = re.compile(r"^\s*<\s*([\d.]+)\s*$")


def parse_concentration(value, bdl_fraction=0.5):
    """Parse a reported value. 'BDL'/'ND'/'Nil' -> 0; '<0.1' -> bdl_fraction * 0.1.
    Plain numbers pass through; anything unparseable -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip()
    m = _LT.match(s)
    if m:
        return bdl_fraction * float(m.group(1))
    if _BDL.match(s):
        return 0.0
    try:
        return float(re.sub(r"\.{2,}", ".", s.replace(",", "")))
    except ValueError:
        return np.nan


def parse_columns(df, cols, bdl_fraction=0.5):
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda v: parse_concentration(v, bdl_fraction))
    return out


def qc_filter(df, cbe_limit=10.0):
    """Apply the validity checks from the project plan and return
    (clean_df, report_df) where report counts rows removed per rule.
    Expects numeric columns pH, Latitude, Longitude and the ion columns,
    plus a precomputed 'CBE' column."""
    steps = [
        ("pH outside 4-11", df["pH"].between(*PH_RANGE)),
        ("coordinates outside India", df["Latitude"].between(*LAT_RANGE) & df["Longitude"].between(*LON_RANGE)),
        ("negative concentration", ~(df[[c for c in ("Ca", "Mg", "Na", "K", "HCO3", "CO3", "Cl", "SO4") if c in df]] < 0).any(axis=1)),
        (f"|charge balance error| > {cbe_limit}%", df["CBE"].abs() <= cbe_limit),
    ]
    rows, keep = [], pd.Series(True, index=df.index)
    for name, ok in steps:
        removed = int((keep & ~ok).sum())
        rows.append({"rule": name, "rows_removed": removed})
        keep &= ok
    return df[keep].copy(), pd.DataFrame(rows)
