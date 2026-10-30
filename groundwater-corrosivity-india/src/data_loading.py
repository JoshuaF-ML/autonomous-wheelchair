"""Load and harmonise the Telangana groundwater-quality files into one table."""
import glob
import re
from pathlib import Path

import pandas as pd

from .cleaning import parse_columns

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "telangana"

RENAME = {
    "district": "District", "mandal": "Mandal", "village": "Village",
    "lat_gis": "Latitude", "long_gis": "Longitude", "gwl": "GWL_m",
}

# Header spellings differ between files ('E.C'/'EC', 'CO_-2 ', 'Ca+2', 'Na+', ...).
_HEADER_PATTERNS = [
    (r"^e\.?c\.?$", "EC"), (r"^t\.?d\.?s$", "TDS"), (r"^t\.?h\.?$", "TH"),
    (r"^co_?3?[\s_+-]*\d*$", "CO3"), (r"^hco[\w\s_+-]*$", "HCO3"), (r"^cl[\s+-]*$", "Cl"),
    (r"^f[\s+-]*$", "F"), (r"^no3[\s+-]*$", "NO3"), (r"^so4[\s+\d-]*$", "SO4"),
    (r"^na[\s+\d]*$", "Na"), (r"^k[\s+]*$", "K"), (r"^ca[\s+\d]*$", "Ca"),
    (r"^mg[\s+\d]*$", "Mg"),
]


def canonical_header(col):
    c = str(col).strip()
    low = c.lower()
    for pat, name in _HEADER_PATTERNS:
        if re.match(pat, low):
            return name
    return RENAME.get(c, c)
ION_COLS = ["pH", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3", "SO4", "Na", "K", "Ca", "Mg", "TH"]
KEEP = ["District", "Mandal", "Village", "Latitude", "Longitude", "GWL_m", "Year", "Season"] + ION_COLS


def parse_season(label):
    """'post monsoon 2019' / 'Pre-monsoon 2020' / 'postmonsoon 2018' -> ('Post', 2019)."""
    s = str(label).lower()
    season = "Post" if "post" in s else "Pre"
    return season, int(re.search(r"(20\d\d)", s).group(1))


def load_telangana(raw_dir=RAW):
    frames = []
    for f in sorted(glob.glob(str(Path(raw_dir) / "*.csv"))):
        frames.append(pd.read_csv(f))
    for f in sorted(glob.glob(str(Path(raw_dir) / "*.xlsx"))):
        frames.append(pd.read_excel(f))
    frames = [f.rename(columns=canonical_header) for f in frames]
    df = pd.concat(frames, ignore_index=True)
    df[["Season", "Year"]] = df["season"].apply(lambda v: pd.Series(parse_season(v)))
    df = parse_columns(df, [c for c in ION_COLS if c in df])
    df["District"] = df["District"].astype(str).str.strip().str.upper()
    df["Mandal"] = df["Mandal"].astype(str).str.strip().str.title()
    df["Village"] = df["Village"].astype(str).str.strip().str.title()
    return df[KEEP].reset_index(drop=True)
