"""Mean annual air temperature per location from NASA POWER (climatology, T2M).

Used as a proxy for groundwater temperature in the Langelier index. Requests are
made per 0.5-degree cell (NASA POWER's native resolution) and cached to CSV.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
CACHE = Path(__file__).resolve().parents[1] / "data" / "interim" / "nasa_t2m_cells.csv"


def cell(lat, lon, size=0.5):
    """Snap to the centre of a size-degree cell."""
    snap = lambda v: np.floor(np.asarray(v, dtype=float) / size) * size + size / 2
    return snap(lat), snap(lon)


def parse_annual_t2m(payload):
    return float(payload["properties"]["parameter"]["T2M"]["ANN"])


def fetch_annual_t2m(lat, lon, timeout=60):
    r = requests.get(URL, params={"parameters": "T2M", "community": "AG", "latitude": lat,
                                  "longitude": lon, "format": "JSON"}, timeout=timeout)
    r.raise_for_status()
    return parse_annual_t2m(r.json())


def add_temperature(df, cache=CACHE):
    """Add 'T_air_C' to df (needs Latitude, Longitude). Fetches only uncached cells."""
    out = df.copy()
    out["cell_lat"], out["cell_lon"] = cell(out["Latitude"], out["Longitude"])
    cells = out[["cell_lat", "cell_lon"]].drop_duplicates()
    cached = pd.read_csv(cache) if Path(cache).exists() else pd.DataFrame(columns=["cell_lat", "cell_lon", "T_air_C"])
    todo = cells.merge(cached, how="left", on=["cell_lat", "cell_lon"])
    todo = todo[todo["T_air_C"].isna()][["cell_lat", "cell_lon"]]
    rows = [dict(cell_lat=la, cell_lon=lo, T_air_C=fetch_annual_t2m(la, lo)) for la, lo in todo.itertuples(index=False)]
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache, index=False)
    cached["T_air_C"] = cached["T_air_C"].astype(float)
    return out.merge(cached, how="left", on=["cell_lat", "cell_lon"]).drop(columns=["cell_lat", "cell_lon"])
