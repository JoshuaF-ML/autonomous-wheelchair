"""Leakage-free ML: predict corrosion-risk class from cheap measurements and location.

Target (3 classes): 'At-risk' (Moderate + High; High alone has too few samples),
'Balanced', 'Scaling'. The model never sees Ca, HCO3, CO3, Cl, SO4 or the indices.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import plotstyle as ps
from .pipeline import PROCESSED

ROOT = Path(__file__).resolve().parents[1]
REPORTS, FIG = ROOT / "reports", ROOT / "reports" / "figures"
CLASSES = ["Scaling", "Balanced", "At-risk"]
SEED = 42

FEATURES = {
    "A0 pH+EC": ["pH", "EC"],
    "A  pH+EC+location+season+climate": ["pH", "EC", "Latitude", "Longitude", "SeasonPre", "T_air_C"],
    "B  location+season+climate only": ["Latitude", "Longitude", "SeasonPre", "T_air_C"],
}


def load():
    d = pd.read_csv(PROCESSED / "telangana_labelled.csv")
    d["y"] = d["Risk"].replace({"Moderate": "At-risk", "High": "At-risk"})
    d["SeasonPre"] = (d["Season"] == "Pre").astype(int)
    d["well"] = d["Latitude"].round(5).astype(str) + "_" + d["Longitude"].round(5).astype(str)
    return d


def models():
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED),
        "Hist. gradient boosting": HistGradientBoostingClassifier(class_weight="balanced", max_depth=4, learning_rate=0.05,
                                                                  max_iter=200, random_state=SEED),
    }


def splits(d, scheme):
    y = d["y"]
    if scheme == "random (leaky)":
        return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(d, y))
    if scheme == "unseen wells":
        return list(GroupKFold(5).split(d, y, d["well"]))
    if scheme == "unseen districts":
        return list(GroupKFold(5).split(d, y, d["District"]))
    if scheme == "train 2018-19, test 2020":
        return [(np.where(d.Year < 2020)[0], np.where(d.Year == 2020)[0])]
    raise ValueError(scheme)


def evaluate(model, X, y, folds):
    pred = pd.Series(index=y.index, dtype=object)
    for tr, te in folds:
        m = model.__class__(**model.get_params()) if not hasattr(model, "steps") else __import__("sklearn.base", fromlist=["clone"]).clone(model)
        m.fit(X.iloc[tr], y.iloc[tr]); pred.iloc[te] = m.predict(X.iloc[te])
    idx = pred.dropna().index
    yt, yp = y.loc[idx], pred.loc[idx]
    return {"macro_F1": f1_score(yt, yp, average="macro"), "balanced_acc": balanced_accuracy_score(yt, yp),
            "recall_at_risk": recall_score(yt, yp, labels=["At-risk"], average=None)[0],
            "accuracy": float((yt == yp).mean()), "n_test": len(idx)}, yp


def run():
    d = load(); y = d["y"]
    rows, oof = [], {}
    for fname, cols in FEATURES.items():
        for scheme in ["random (leaky)", "unseen wells", "unseen districts", "train 2018-19, test 2020"]:
            folds = splits(d, scheme)
            for mname, model in models().items():
                if fname.startswith("B") and mname == "Logistic regression":
                    continue
                res, yp = evaluate(model, d[cols], y, folds)
                rows.append({"features": fname, "validation": scheme, "model": mname, **res})
                oof[(fname, scheme, mname)] = (yp.index, yp)
    out = pd.DataFrame(rows)
    out.round(3).to_csv(REPORTS / "model_results.csv", index=False)
    return d, out, oof


def fig_confusion(d, oof, key, path, title):
    ps.apply(); import matplotlib.pyplot as plt
    idx, yp = oof[key]
    cm = confusion_matrix(d.loc[idx, "y"], yp, labels=CLASSES)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100); ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pct[i, j]:.0f}%\n(n={cm[i, j]})", ha="center", va="center",
                    color="#ffffff" if pct[i, j] > 55 else ps.INK, fontsize=9)
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (row-normalised)"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / path, dpi=150); plt.close(fig)


def shap_plots(d):
    import shap
    import matplotlib.pyplot as plt
    cols = FEATURES["A  pH+EC+location+season+climate"]
    rf = models()["Random forest"].fit(d[cols], d["y"])
    sv = shap.TreeExplainer(rf).shap_values(d[cols])
    k = list(rf.classes_).index("At-risk")
    vals = sv[:, :, k] if getattr(sv, "ndim", 0) == 3 else sv[k]
    ps.apply()
    shap.summary_plot(vals, d[cols], show=False, plot_size=(7.5, 4.2))
    plt.title("What pushes a sample toward 'At-risk' (SHAP, random forest)", loc="left", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG / "shap_at_risk.png", dpi=150); plt.close()
    imp = pd.Series(np.abs(vals).mean(axis=0), index=cols).sort_values(ascending=False)
    imp.round(4).to_csv(REPORTS / "shap_importance_at_risk.csv", header=["mean_abs_shap"])
    return imp


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    d, res, oof = run()
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    fig_confusion(d, oof, ("A  pH+EC+location+season+climate", "unseen districts", "Random forest"),
                  "confusion_A_unseen_districts.png", "Task A, random forest, unseen districts")
    print(shap_plots(d))
