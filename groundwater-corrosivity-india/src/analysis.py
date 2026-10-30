"""Descriptive statistics, tables and figures from the labelled dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotstyle as ps
from .pipeline import PROCESSED, add_indices

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"


def load():
    d = pd.read_csv(PROCESSED / "telangana_labelled.csv")
    d["Risk"] = pd.Categorical(d["Risk"], ps.RISK_ORDER)
    d["AtRisk"] = d["Risk"].isin(["Moderate", "High"])
    # what drives an at-risk label: low LSI, high Larson-Skold, or both
    d["Driver"] = np.select([(d.LSI < -0.5) & (d.LS > 1.2), d.LSI < -0.5, d.LS > 1.2],
                            ["both", "LSI only", "LS only"], "none")
    return d


def tables(d):
    REPORTS.mkdir(exist_ok=True)
    pct = lambda s: (s.value_counts(normalize=True).reindex(ps.RISK_ORDER).fillna(0) * 100).round(1)
    overall = pd.DataFrame({"n": d.Risk.value_counts().reindex(ps.RISK_ORDER), "pct": pct(d.Risk)})
    overall.to_csv(REPORTS / "risk_overall.csv")

    dist = d.groupby("District").agg(n_samples=("Risk", "size"), n_wells=("Village", "nunique"),
                                     median_LSI=("LSI", "median"), median_LS=("LS", "median"),
                                     median_EC=("EC", "median"), pct_at_risk=("AtRisk", lambda s: 100 * s.mean()),
                                     pct_scaling=("Risk", lambda s: 100 * (s == "Scaling").mean()))
    dist.round(2).sort_values("pct_at_risk", ascending=False).to_csv(REPORTS / "district_summary.csv")

    ss = d.groupby(["Year", "Season"]).agg(n=("Risk", "size"), pct_at_risk=("AtRisk", lambda s: 100 * s.mean()),
                                           median_LSI=("LSI", "median"), median_LS=("LS", "median")).round(2)
    ss.to_csv(REPORTS / "season_year_summary.csv")

    drivers = d[d.AtRisk].Driver.value_counts().rename("n").to_frame()
    drivers["pct"] = (100 * drivers.n / drivers.n.sum()).round(1)
    drivers.to_csv(REPORTS / "at_risk_drivers.csv")

    # paired pre -> post change for wells sampled in both seasons of the same year
    w = d.pivot_table(index=["Latitude", "Longitude", "Year"], columns="Season", values="LSI").dropna()
    paired = pd.Series({"n_pairs": len(w), "median_LSI_pre": w["Pre"].median(), "median_LSI_post": w["Post"].median(),
                        "median_post_minus_pre": (w["Post"] - w["Pre"]).median(),
                        "share_post_higher": (w["Post"] > w["Pre"]).mean()}).round(3)
    paired.to_csv(REPORTS / "paired_season_lsi.csv", header=["value"])

    base = d["Risk"].astype(str)
    rows = []
    for name, tair, off in [("air T +0 C", None, 0.0), ("air T +1 C (baseline)", None, 1.0), ("air T +3 C", None, 3.0),
                            ("fixed 20 C", 19.0, 1.0), ("fixed 30 C", 29.0, 1.0)]:
        x = d.copy()
        if tair is not None:
            x["T_air_C"] = tair
        r = add_indices(x, off)["Risk"].astype(str)
        rows.append({"scenario": name, "labels_changed": int((r != base).sum()), "pct": round(100 * (r != base).mean(), 1)})
    pd.DataFrame(rows).to_csv(REPORTS / "temperature_sensitivity.csv", index=False)
    return overall, dist, ss, drivers, paired


def fig_district(d):
    ps.apply(); import matplotlib.pyplot as plt
    tab = pd.crosstab(d.District, d.Risk, normalize="index").reindex(columns=ps.RISK_ORDER) * 100
    tab = tab.loc[(tab["Moderate"] + tab["High"]).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 8.5)); left = np.zeros(len(tab))
    for k in ps.RISK_ORDER:
        ax.barh(tab.index, tab[k], left=left, color=ps.RISK_COLORS[k], edgecolor=ps.SURFACE, linewidth=1.5, label=k, height=0.78)
        left += tab[k].to_numpy()
    ax.set_xlim(0, 100); ax.set_xlabel("% of samples"); ax.grid(axis="y", visible=False)
    ax.set_title("Corrosion risk class by district, Telangana 2018-2020")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(); fig.savefig(FIG / "risk_by_district.png", dpi=150); plt.close(fig)


def fig_map(d):
    ps.apply(); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, season in zip(axes, ["Pre", "Post"]):
        s = d[d.Season == season]
        for k in ps.RISK_ORDER:
            m = s[s.Risk == k]
            ax.scatter(m.Longitude, m.Latitude, s=14, color=ps.RISK_COLORS[k], edgecolor=ps.SURFACE, linewidth=0.6, label=f"{k} ({len(m)})")
        ax.set_title(f"{season}-monsoon samples"); ax.set_xlabel("Longitude"); ax.set_aspect("equal")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    axes[0].set_ylabel("Latitude")
    fig.tight_layout(); fig.savefig(FIG / "map_wells_by_season.png", dpi=150); plt.close(fig)


def fig_scatter(d):
    ps.apply(); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for k in ps.RISK_ORDER:
        m = d[d.Risk == k]
        ax.scatter(m.LSI, m.LS, s=12, color=ps.RISK_COLORS[k], edgecolor=ps.SURFACE, linewidth=0.4, alpha=0.9, label=f"{k} ({len(m)})")
    ax.set_yscale("log"); ax.set_ylim(0.04, 60)
    for x in (-2, -0.5, 0.5): ax.axvline(x, color=ps.MUTED, lw=0.8, ls="--")
    for y in (0.8, 1.2): ax.axhline(y, color=ps.MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Langelier Saturation Index (LSI)"); ax.set_ylabel("Larson-Skold index (log scale)")
    ax.set_title("Where the two indices agree and disagree"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "lsi_vs_larson_skold.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    d = load()
    for t in tables(d): print(t.to_string(), "\n")
    fig_district(d); fig_map(d); fig_scatter(d)
    print("figures ->", FIG)
