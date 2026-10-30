import pandas as pd
import pytest

from src.climate import cell, parse_annual_t2m
from src.cleaning import parse_concentration
from src.data_loading import canonical_header, parse_season


@pytest.mark.parametrize("raw,expected", [
    ("E.C", "EC"), ("EC", "EC"), ("CO3", "CO3"), ("CO_-2 ", "CO3"), ("HCO3", "HCO3"), ("HCO_ - ", "HCO3"),
    ("Cl -", "Cl"), ("F -", "F"), ("NO3 ", "NO3"), ("NO3- ", "NO3"), ("SO4-2", "SO4"),
    ("Na+", "Na"), ("K+", "K"), ("Ca+2", "Ca"), ("Mg+2", "Mg"), ("T.H", "TH"),
    ("lat_gis", "Latitude"), ("district", "District"),
])
def test_canonical_header(raw, expected):
    assert canonical_header(raw) == expected


def test_classification_columns_not_mistaken_for_chloride():
    assert canonical_header("Classification") == "Classification"
    assert canonical_header("Classification.1") == "Classification.1"


@pytest.mark.parametrize("label,expected", [
    ("postmonsoon 2018", ("Post", 2018)), ("post monsoon 2019", ("Post", 2019)),
    ("Post-monsoon 2020", ("Post", 2020)), ("premonsoon 2018", ("Pre", 2018)),
    ("pre monsoon 2019", ("Pre", 2019)), ("Pre-monsoon 2020", ("Pre", 2020)),
])
def test_parse_season(label, expected):
    assert parse_season(label) == expected


def test_typo_ph_value_is_recovered():
    assert parse_concentration("8..05") == pytest.approx(8.05)


def test_cell_snaps_to_half_degree_centres():
    lat, lon = cell(19.668, 78.5247)
    assert (float(lat), float(lon)) == (19.75, 78.75)


def test_parse_annual_t2m():
    payload = {"properties": {"parameter": {"T2M": {"JAN": 20.1, "ANN": 26.99}}}}
    assert parse_annual_t2m(payload) == 26.99
