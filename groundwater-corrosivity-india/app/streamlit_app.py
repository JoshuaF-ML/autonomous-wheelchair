"""Check your water: corrosivity indices from a lab report, or ML screening from pH + EC."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import indices as ix  # noqa: E402

ADVICE = {
    "High": "Avoid GI and mild steel. Use uPVC, HDPE or CPVC, or lined pipe. Inspect hand pumps more often.",
    "Moderate": "Chloride/sulphate attack on steel is likely. GI is acceptable only with a shorter replacement cycle; prefer plastic where practical.",
    "Balanced": "Any standard material is fine.",
    "Scaling": "Expect carbonate scale and clogging. Plan descaling and avoid very small bore pipe.",
    "At-risk": "Water is aggressive toward steel (Moderate or High). Prefer uPVC/HDPE; if GI is used, plan early replacement.",
}

st.set_page_config(page_title="Check your water", page_icon="💧")
st.title("Check your water")
st.caption("Trained on Telangana state groundwater data (2018-2020). Outside Telangana, treat ML results as indicative only.")


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "risk_models.joblib")


tab1, tab2 = st.tabs(["Full lab report", "Quick screen (pH + EC)"])

with tab1:
    st.write("Enter concentrations in mg/L (EC in µS/cm). Gives the exact indices.")
    c1, c2 = st.columns(2)
    ph = c1.number_input("pH", 4.0, 11.0, 7.8, 0.1)
    ec = c2.number_input("EC (µS/cm)", 1.0, 20000.0, 1200.0, 10.0)
    temp = c1.number_input("Water temperature (°C)", 5.0, 45.0, 28.0, 0.5)
    ca = c2.number_input("Calcium (mg/L)", 0.1, 1000.0, 60.0)
    hco3 = c1.number_input("Bicarbonate HCO3 (mg/L)", 1.0, 3000.0, 250.0)
    co3 = c2.number_input("Carbonate CO3 (mg/L)", 0.0, 500.0, 0.0)
    cl = c1.number_input("Chloride (mg/L)", 0.0, 5000.0, 150.0)
    so4 = c2.number_input("Sulphate (mg/L)", 0.0, 3000.0, 40.0)
    tds = ec * ix.EC_TO_TDS
    lsi = float(ix.lsi(ph, tds, temp, ca, hco3, co3))
    rsi = float(ix.rsi(ph, tds, temp, ca, hco3, co3))
    ls = float(ix.larson_skold(cl, so4, hco3, co3))
    risk = ix.risk_class(lsi, ls)
    m1, m2, m3 = st.columns(3)
    m1.metric("LSI", f"{lsi:.2f}")
    m2.metric("RSI", f"{rsi:.2f}")
    m3.metric("Larson-Skold", f"{ls:.2f}")
    st.subheader(f"Risk class: {risk}")
    st.write(ADVICE[risk])
    st.caption(f"TDS estimated as {ix.EC_TO_TDS} x EC = {tds:.0f} mg/L. LSI > 0.5 means scale-forming, < -0.5 corrosive. "
               "Larson-Skold > 1.2 means chloride/sulphate are aggressive toward steel.")

with tab2:
    st.write("Only pH and EC needed (a cheap field meter). Add location and season for a better estimate.")
    bundle = load_bundle()
    ph2 = st.number_input("pH ", 4.0, 11.0, 7.8, 0.1, key="ph2")
    ec2 = st.number_input("EC (µS/cm) ", 1.0, 20000.0, 1200.0, 10.0, key="ec2")
    use_loc = st.checkbox("I know the location (Telangana)")
    if use_loc:
        lat = st.number_input("Latitude", 15.8, 19.95, 17.4, 0.01)
        lon = st.number_input("Longitude", 77.2, 81.4, 78.5, 0.01)
        season = st.radio("Season", ["Pre-monsoon", "Post-monsoon"], horizontal=True)
        key = "A  pH+EC+location+season+climate"
        row = {"pH": ph2, "EC": ec2, "Latitude": lat, "Longitude": lon, "SeasonPre": int(season == "Pre-monsoon"),
               "T_air_C": 26.9}   # state-wide mean annual air temperature in the training data
    else:
        key = "A0 pH+EC"
        row = {"pH": ph2, "EC": ec2}
    spec = bundle["models"][key]
    proba = spec["model"].predict_proba(pd.DataFrame([row])[spec["features"]])[0]
    best = int(np.argmax(proba)); cls = spec["model"].classes_[best]
    st.subheader(f"Most likely: {cls} ({proba[best]:.0%})")
    st.bar_chart(pd.Series(proba, index=spec["model"].classes_, name="probability"))
    st.write(ADVICE[cls])
    st.caption("A screening estimate, not a measurement. Confirm with a lab analysis before specifying pipe material for a scheme.")
