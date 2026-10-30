"""Water-corrosivity indices and the project's risk-class label.

All concentrations are in mg/L unless noted; EC is in uS/cm at 25 C.
Every function is vectorised: pass floats or pandas/numpy arrays.
"""
import numpy as np

# Equivalent weights (g/eq) used to convert mg/L -> meq/L
EQ_WT = {
    "Ca": 20.04, "Mg": 12.15, "Na": 22.99, "K": 39.10,
    "HCO3": 61.02, "CO3": 30.00, "Cl": 35.45, "SO4": 48.03,
    "NO3": 62.00, "F": 19.00,
}
CA_TO_CACO3 = 50.04 / 20.04      # 2.497
HCO3_TO_CACO3 = 50.04 / 61.02    # 0.820
CO3_TO_CACO3 = 50.04 / 30.00     # 1.668
EC_TO_TDS = 0.64                 # typical; Indian groundwater ranges ~0.55-0.75


def meq(conc_mgl, ion):
    """mg/L -> meq/L for a named ion."""
    return np.asarray(conc_mgl, dtype=float) / EQ_WT[ion]


def tds_from_ec(ec_uscm, factor=EC_TO_TDS):
    return np.asarray(ec_uscm, dtype=float) * factor


def alkalinity_caco3(hco3, co3=0.0):
    """Total alkalinity as mg/L CaCO3 from bicarbonate and carbonate."""
    return np.asarray(hco3, dtype=float) * HCO3_TO_CACO3 + np.asarray(co3, dtype=float) * CO3_TO_CACO3


def ph_saturation(tds, temp_c, ca, hco3, co3=0.0):
    """pHs: the pH at which water is in equilibrium with CaCO3 (Langelier)."""
    tds = np.asarray(tds, dtype=float)
    a = (np.log10(tds) - 1.0) / 10.0
    b = -13.12 * np.log10(np.asarray(temp_c, dtype=float) + 273.0) + 34.55
    c = np.log10(np.asarray(ca, dtype=float) * CA_TO_CACO3) - 0.4
    d = np.log10(alkalinity_caco3(hco3, co3))
    return (9.3 + a + b) - (c + d)


def lsi(ph, tds, temp_c, ca, hco3, co3=0.0):
    """Langelier Saturation Index = pH - pHs. <0 corrosive, >0 scale-forming."""
    ph = np.asarray(ph, dtype=float)
    return ph - ph_saturation(tds, temp_c, ca, hco3, co3)


def rsi(ph, tds, temp_c, ca, hco3, co3=0.0):
    """Ryznar Stability Index = 2*pHs - pH. <6.2 scaling, 6.2-6.8 neutral, >6.8 corrosive."""
    ph = np.asarray(ph, dtype=float)
    return 2.0 * ph_saturation(tds, temp_c, ca, hco3, co3) - ph


def larson_skold(cl, so4, hco3, co3=0.0):
    """Larson-Skold index: (Cl + SO4) / (HCO3 + CO3), all in meq/L.
    <0.8 low risk to mild steel, 0.8-1.2 moderate, >1.2 high."""
    num = meq(cl, "Cl") + meq(so4, "SO4")
    den = meq(hco3, "HCO3") + meq(co3, "CO3")
    return num / den


def charge_balance_error(df_like):
    """Charge-balance error (%) from a mapping/DataFrame with mg/L columns
    Ca, Mg, Na, K, HCO3, CO3, Cl, SO4, NO3, F. Missing minor ions count as 0."""
    def col(name):
        try:
            return np.nan_to_num(np.asarray(df_like[name], dtype=float))
        except KeyError:
            return 0.0

    cations = sum(meq(col(i), i) for i in ("Ca", "Mg", "Na", "K"))
    anions = sum(meq(col(i), i) for i in ("HCO3", "CO3", "Cl", "SO4", "NO3", "F"))
    return (cations - anions) / (cations + anions) * 100.0


def risk_class(lsi_val, ls_val):
    """Project label combining LSI and Larson-Skold. Rules, in priority order:
      High      : LSI < -2, or (LSI < -0.5 and LS > 1.2)
      Moderate  : LSI < -0.5, or LS > 1.2 (one condition only)
      Scaling   : LSI > 0.5 and LS < 0.8
      Balanced  : everything else
    Note: LSI > 0.5 with LS in [0.8, 1.2] falls through to Balanced by design.
    Returns an object array; NaN inputs give NaN.
    """
    l = np.asarray(lsi_val, dtype=float)
    s = np.asarray(ls_val, dtype=float)
    out = np.full(l.shape, "Balanced", dtype=object)
    out[(l > 0.5) & (s < 0.8)] = "Scaling"
    out[(l < -0.5) | (s > 1.2)] = "Moderate"
    out[(l < -2.0) | ((l < -0.5) & (s > 1.2))] = "High"
    out[np.isnan(l) | np.isnan(s)] = np.nan
    return out if out.ndim else out.item()
