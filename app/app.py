# app.py — Flip Estimator (refactor) + 401k-style theming

import os
import re
import json
import requests
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Flip Estimator + CA Model", layout="centered")

ATTOM_URL = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/property/detail"


st.markdown(f"<style>{Path('app/flip_estimator_styles.css').read_text()}</style>", unsafe_allow_html=True)

def parse_zip(s: str | None) -> str | None:
    if not s:
        return None
    m = re.search(r"\b(\d{5})\b", str(s))
    return m.group(1) if m else None

def to_float(x):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        return float(x)
    except Exception:
        return None

def _get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

def predict_price(model, bed, bath, sqft, acre_lot, zip_code: str) -> float | None:
    try:
        row = pd.DataFrame([{
            "bed": float(bed) if bed is not None else None,
            "bath": float(bath) if bath is not None else None,
            "house_size": float(sqft) if sqft is not None else None,
            "acre_lot": float(acre_lot) if acre_lot is not None else 0.0,
            "zip_code": str(zip_code) if zip_code else None,
            "zip3": str(zip_code)[:3] if zip_code else None,  # important for your pipeline
        }])
        return float(model.predict(row)[0])
    except Exception as e:
        st.warning(f"Model prediction failed: {e}")
        return None

DEFAULTS = {
    "api_key": _get_secret("ATTOM_API_KEY", os.getenv("ATTOM_API_KEY", "")),
    "addr1": "1141 Langton Dr",
    "addr2": "San Ramon, CA 94582",
    "model_path": "ml/model_ca_zip_hgbr.joblib",

    # property facts (editable)
    "one_line": "",
    "zip_code": "",
    "beds": 3,
    "baths": 2.0,
    "sqft": 2000,
    "acre_lot": 0.10,
    "year": None,

    # model/cache
    "use_model_exit": False,
    "model_loaded": False,
    "last_attom_json": {},
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---------------------------
# Sidebar (kept functional; gets styled by CSS)
# ---------------------------
st.sidebar.header("Setup")
st.session_state.api_key = st.sidebar.text_input("ATTOM API key", value=st.session_state.api_key, type="password")
st.session_state.addr1 = st.sidebar.text_input("Street (address1)", st.session_state.addr1)
st.session_state.addr2 = st.sidebar.text_input("City, State Zip (address2)", st.session_state.addr2)

st.sidebar.markdown("---")
st.session_state.model_path = st.sidebar.text_input("ML model path", value=st.session_state.model_path)
model = None
if os.path.exists(st.session_state.model_path):
    try:
        model = load_model(st.session_state.model_path)
        st.session_state.model_loaded = True
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.session_state.model_loaded = False
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.session_state.model_loaded = False
    st.sidebar.info("Provide a trained model path (e.g., ml/model_ca_zip_hgbr.joblib)")

# ===========================
# Main content shell + cards
# ===========================
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

# ---- Header card
st.title("Simple Flip Estimator (MVP) + CA ZIP Model")
st.caption("Optionally fetch facts via ATTOM to prefill fields, then estimate value and run renovation what-ifs. All fields remain editable.")
st.markdown('</div>', unsafe_allow_html=True)

# ---- ATTOM fetch card
st.divider() 
st.subheader("Prefill from ATTOM (optional)")
c_fetch1, c_fetch2 = st.columns([1, 2])
with c_fetch1:
    if st.button("Fetch Property from ATTOM"):
        if not st.session_state.api_key:
            st.error("Add your ATTOM API key in the sidebar.")
        else:
            with st.spinner("Calling ATTOM…"):
                data = {}
                try:
                    r = requests.get(
                        ATTOM_URL,
                        headers={"accept": "application/json", "apikey": st.session_state.api_key},
                        params={"address1": st.session_state.addr1.strip(), "address2": st.session_state.addr2.strip()},
                        timeout=20,
                    )
                    if r.headers.get("content-type", "").startswith("application/json"):
                        data = r.json()
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    data = {}

                # Parse safely and PREFILL session_state
                try:
                    p = (data.get("property") or [])[0]
                    a = p.get("address", {}) or {}
                    b = p.get("building", {}) or {}
                    size = b.get("size", {}) or {}
                    rooms = b.get("rooms", {}) or {}

                    st.session_state.one_line = a.get("oneLine") or f"{st.session_state.addr1}, {st.session_state.addr2}"
                    st.session_state.zip_code = a.get("postal1") or parse_zip(st.session_state.addr2) or st.session_state.zip_code

                    beds = to_float(rooms.get("beds"))
                    baths = to_float(rooms.get("bathsfull") or rooms.get("bathstotal"))
                    sqft = to_float(size.get("universalsize") or size.get("livingsize") or size.get("bldgsize"))
                    year = (p.get("summary", {}) or {}).get("yearbuilt") or (b.get("summary", {}) or {}).get("yearbuilt")

                    if beds is not None:
                        st.session_state.beds = int(beds)
                    if baths is not None:
                        st.session_state.baths = float(baths)
                    if sqft is not None:
                        st.session_state.sqft = int(sqft)
                    if year:
                        st.session_state.year = int(year)

                    st.session_state.last_attom_json = data
                    st.success("ATTOM data fetched. Fields updated — you can still edit them below.")
                except Exception:
                    st.info("No usable ATTOM payload; keeping your current values.")
with c_fetch2:
    st.text_input("Resolved Address", value=st.session_state.one_line or f"{st.session_state.addr1}, {st.session_state.addr2}", disabled=True)

# ---- Property Snapshot card
st.divider() 
st.subheader("Property Snapshot (editable)")
st.markdown('<div class="section-hint">Correct anything ATTOM guessed wrong — these drive both model & math.</div>', unsafe_allow_html=True)

p1, p2, p3, p4, p5 = st.columns(5)

p1.number_input("Beds", min_value=0, step=1, key="beds")
p2.number_input("Baths", min_value=0.0, step=0.5, format="%.1f", key="baths")
p3.number_input("Sqft", min_value=200, step=50, key="sqft")
p4.number_input("Year", min_value=1800, max_value=2100, step=1,
                value=(st.session_state.year or 1990), key="year")
p5.text_input("ZIP code", key="zip_code")

l1, l2 = st.columns(2)
l1.number_input("Lot size (acres)", min_value=0.0, step=0.01, format="%.2f", key="acre_lot")
with l2:
    st.caption("Rooms like offices may not count as bedrooms; feel free to correct the numbers.")
st.markdown('</div>', unsafe_allow_html=True)

st.divider() 
st.subheader("Renovation Planner (CA costs)")
st.markdown('<div class="section-hint">California ballpark pricing — pick a level and tweak counts/sf below.</div>', unsafe_allow_html=True)

COST_PRESETS = {
    "Basic": {
        "kitchen": 30000, "full_bath": 12000, "half_bath": 6000,
        "bedroom_cosmetic": 3000, "living_cosmetic": 4000,
        "flooring_per_sqft": 6, "paint_interior_per_sqft": 2.5,
        "window_each": 700, "roof_replace": 14000,
        "electrical_panel": 2500, "repipe_plumbing": 6000,
        "permits_allowance": 2000,
    },
    "Mid": {
        "kitchen": 45000, "full_bath": 22000, "half_bath": 12000,
        "bedroom_cosmetic": 6000, "living_cosmetic": 8000,
        "flooring_per_sqft": 9, "paint_interior_per_sqft": 3.5,
        "window_each": 1100, "roof_replace": 20000,
        "electrical_panel": 4000, "repipe_plumbing": 10000,
        "permits_allowance": 4000,
    },
    "High": {
        "kitchen": 70000, "full_bath": 35000, "half_bath": 18000,
        "bedroom_cosmetic": 9000, "living_cosmetic": 12000,
        "flooring_per_sqft": 14, "paint_interior_per_sqft": 5,
        "window_each": 1600, "roof_replace": 30000,
        "electrical_panel": 6500, "repipe_plumbing": 16000,
        "permits_allowance": 8000,
    },
}

cc1, cc2 = st.columns([1,1])
with cc1:
    preset = st.selectbox("Cost level", list(COST_PRESETS.keys()), index=1, help="California ballpark pricing — adjust counts/sf below.")
with cc2:
    contingency_pct = st.number_input("Contingency %", min_value=0, max_value=50, value=10, step=1)

colA, colB, colC = st.columns(3)
with colA:
    n_kitchens = st.number_input("Kitchens (gut/replace)", min_value=0, value=1, step=1)
    n_full_baths = st.number_input("Full Baths", min_value=0, value=2, step=1)
    n_half_baths = st.number_input("Half Baths", min_value=0, value=0, step=1)
    n_bedrooms = st.number_input("Bedrooms (cosmetic)", min_value=0, value=max(0, int(st.session_state.beds)), step=1)
with colB:
    n_living = st.number_input("Living/Great/Other rooms (cosmetic)", min_value=0, value=2, step=1)
    flooring_sf = st.number_input("Flooring to replace (sqft)", min_value=0, value=int(st.session_state.sqft), step=50)
    paint_sf = st.number_input("Interior paint (sqft)", min_value=0, value=int(st.session_state.sqft), step=100)
    n_windows = st.number_input("Windows to replace (count)", min_value=0, value=0, step=1)
with colC:
    roof_replace = st.checkbox("Roof replacement")
    need_panel = st.checkbox("Electrical panel upgrade")
    need_repipe = st.checkbox("Whole-home repipe")
    permits_extra = st.number_input("Permits/Design allowance ($)", min_value=0, value=COST_PRESETS[preset]["permits_allowance"], step=500)

P = COST_PRESETS[preset]
subtotal = (
    n_kitchens * P["kitchen"]
    + n_full_baths * P["full_bath"]
    + n_half_baths * P["half_bath"]
    + n_bedrooms * P["bedroom_cosmetic"]
    + n_living * P["living_cosmetic"]
    + flooring_sf * P["flooring_per_sqft"]
    + paint_sf * P["paint_interior_per_sqft"]
    + n_windows * P["window_each"]
    + (P["roof_replace"] if roof_replace else 0)
    + (P["electrical_panel"] if need_panel else 0)
    + (P["repipe_plumbing"] if need_repipe else 0)
    + permits_extra
)
contingency = subtotal * (contingency_pct / 100.0)
computed_reno_cost = int(round(subtotal + contingency))
st.session_state.reno_cost = computed_reno_cost

st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
for label, val in [
    ("Reno Subtotal", f"${subtotal:,.0f}"),
    ("Contingency", f"${contingency:,.0f}"),
    ("Total Reno Cost", f"${computed_reno_cost:,.0f}"),
    ("Preset", preset),
]:
    st.markdown(f'''
    <div class="metric">
      <div class="label">{label}</div>
      <div class="value">{val}</div>
    </div>
    ''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Adjustments (optional)"):
    manual_adjust = st.number_input("Manual tweak (+/− $)", value=0, step=500)
    if manual_adjust != 0:
        st.session_state.reno_cost = int(max(0, computed_reno_cost + manual_adjust))
        st.caption(f"Applied adjustment: ${manual_adjust:,.0f}")
st.markdown('</div>', unsafe_allow_html=True)

# ---- Deal Math card

# ---- What-If card
st.divider() 
st.subheader("Renovation What-If (sqft add-on)")

w1, w2, w3 = st.columns(3)
add_sqft = w1.number_input("Add Sqft", min_value=0, value=300, step=50)
cost_per_sqft = w2.number_input("Cost per Sqft", min_value=0, value=350, step=25)
include_in_total = w3.checkbox("Include in Total Reno Cost", value=False,
    help="If checked, the add-on CapEx is included in the Total Reno Cost metric above.")

capex = int(add_sqft * cost_per_sqft)

# If you want the add-on to optionally change the computed total:
total_with_addon = st.session_state.reno_cost + (capex if include_in_total else 0)

st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
for label, val in [
    ("Add-on CapEx", f"${capex:,.0f}"),
    ("Total Reno Cost", f"${st.session_state.reno_cost:,.0f}"),
    ("Reno Cost (with add-on)", f"${total_with_addon:,.0f}" if include_in_total else "—"),
]:
    st.markdown(f'''
    <div class="metric">
      <div class="label">{label}</div>
      <div class="value">{val}</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close What-If card

# ---- Debug JSON card
st.markdown('<div class="card">', unsafe_allow_html=True)
with st.expander("See raw ATTOM JSON (debug)"):
    st.json(st.session_state.last_attom_json if isinstance(st.session_state.last_attom_json, dict) else {"note": "no JSON"})
st.markdown('</div>', unsafe_allow_html=True)

# Close shell
st.markdown('</div>', unsafe_allow_html=True)
