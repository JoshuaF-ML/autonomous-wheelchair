"""Fit the final screening models on all data and save them for the app."""
from pathlib import Path

import joblib

from .modeling import FEATURES, load, models

OUT = Path(__file__).resolve().parents[1] / "models" / "risk_models.joblib"

if __name__ == "__main__":
    d = load()
    bundle = {"classes": None, "trained_on_rows": len(d), "years": sorted(d.Year.unique().tolist()), "models": {}}
    for key in ("A0 pH+EC", "A  pH+EC+location+season+climate"):
        m = models()["Random forest"].set_params(n_estimators=200, min_samples_leaf=5).fit(d[FEATURES[key]], d["y"])
        bundle["models"][key] = {"model": m, "features": FEATURES[key]}
        bundle["classes"] = list(m.classes_)
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, OUT, compress=3)
    print("saved", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
