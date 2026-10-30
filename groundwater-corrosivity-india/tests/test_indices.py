import math
import numpy as np
import pandas as pd
import pytest

from src import indices as ix
from src.cleaning import parse_concentration, qc_filter


def test_alkalinity_conversion():
    # 61.02 mg/L HCO3 is 1 meq/L = 50.04 mg/L as CaCO3
    assert ix.alkalinity_caco3(61.02) == pytest.approx(50.04, rel=1e-3)
    assert ix.alkalinity_caco3(0, 30.0) == pytest.approx(50.04, rel=1e-3)


def test_lsi_hand_worked_example():
    # Inputs chosen so Ca as CaCO3 = 150 and alkalinity = 120 mg/L as CaCO3.
    # Hand calculation of the Langelier equation:
    #   A = (log10(320)-1)/10 = 0.1505 ; B = -13.12*log10(298)+34.55 = 2.088
    #   C = log10(150)-0.4 = 1.7761   ; D = log10(120) = 2.0792
    #   pHs = 9.3+0.1505+2.088-1.7761-2.0792 = 7.683 ; LSI(pH 7.5) = -0.183
    ca = 150 / ix.CA_TO_CACO3
    hco3 = 120 / ix.HCO3_TO_CACO3
    assert ix.ph_saturation(320, 25, ca, hco3) == pytest.approx(7.683, abs=0.005)
    assert ix.lsi(7.5, 320, 25, ca, hco3) == pytest.approx(-0.183, abs=0.005)
    assert ix.rsi(7.5, 320, 25, ca, hco3) == pytest.approx(2 * 7.683 - 7.5, abs=0.01)


def test_lsi_sign_behaviour():
    # Raising pH makes water less corrosive; raising temperature makes it more scale-forming.
    args = dict(tds=500, temp_c=25, ca=60, hco3=250)
    assert ix.lsi(8.5, **args) > ix.lsi(7.0, **args)
    assert ix.lsi(7.5, 500, 35, 60, 250) > ix.lsi(7.5, 500, 15, 60, 250)


def test_larson_skold_known_value():
    # Cl 35.45 -> 1 meq, SO4 48.03 -> 1 meq, HCO3 61.02 -> 1 meq  => LS = 2
    assert ix.larson_skold(35.45, 48.03, 61.02) == pytest.approx(2.0)


def test_charge_balance_perfect_and_unbalanced():
    bal = {"Ca": 20.04, "HCO3": 61.02}          # 1 meq each side
    assert ix.charge_balance_error(bal) == pytest.approx(0.0, abs=1e-9)
    off = {"Ca": 40.08, "HCO3": 61.02}          # 2 vs 1 meq
    assert ix.charge_balance_error(off) == pytest.approx(33.33, abs=0.01)


@pytest.mark.parametrize("lsi_v,ls_v,expected", [
    (-2.5, 0.1, "High"),
    (-1.0, 1.5, "High"),
    (-1.0, 0.5, "Moderate"),
    (0.0, 1.5, "Moderate"),
    (0.0, 0.5, "Balanced"),
    (1.0, 0.5, "Scaling"),
    (1.0, 1.0, "Balanced"),   # documented gap in the rules
])
def test_risk_class(lsi_v, ls_v, expected):
    assert ix.risk_class(lsi_v, ls_v) == expected


def test_risk_class_vectorised_and_nan():
    out = ix.risk_class(np.array([-3.0, np.nan]), np.array([0.1, 0.1]))
    assert out[0] == "High" and pd.isna(out[1])


@pytest.mark.parametrize("raw,expected", [
    ("BDL", 0.0), ("nil", 0.0), ("<0.1", 0.05), ("< 2", 1.0),
    ("1,234.5", 1234.5), (7.2, 7.2), ("abc", None), (None, None),
])
def test_parse_concentration(raw, expected):
    got = parse_concentration(raw)
    if expected is None:
        assert math.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_qc_filter_reports_each_rule():
    df = pd.DataFrame({
        "pH": [7.0, 12.0, 7.0, 7.0, 7.0],
        "Latitude": [13.0, 13.0, 50.0, 13.0, 13.0],
        "Longitude": [80.0, 80.0, 80.0, 80.0, 80.0],
        "Ca": [10, 10, 10, -1, 10], "Mg": 5, "Na": 5, "K": 1,
        "HCO3": 50, "CO3": 0, "Cl": 10, "SO4": 5,
        "CBE": [1.0, 1.0, 1.0, 1.0, 25.0],
    })
    clean, report = qc_filter(df)
    assert len(clean) == 1
    assert report["rows_removed"].tolist() == [1, 1, 1, 1]
