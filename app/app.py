# app.py — Flip Estimator (refactor)
# Goals:
# - Let users freely EDIT all financial fields (purchase price, reno cost, etc.) at any time
# - Make ATTOM fetch optional; when used, it just pre-fills fields (never locks them)
# - Remove the auto-forced "Current Value". Instead, let users set Exit Value (ARV) manually
#   with an optional toggle to apply the model estimate
# - Stabilize state with st.session_state so inputs persist across reruns
# - Keep the code simple and readable

import os
import re
import json
import requests
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flip Estimator + CA Model", page_icon="🏠", layout="centered")

ATTOM_URL = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/property/detail"

# ---------------------------
# Helpers
# ---------------------------

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


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def predict_price(model, bed, bath, sqft, acre_lot, zip_code: str) -> float | None:
    try:
        row = pd.DataFrame([
            {
                "bed": float(bed) if bed is not None else None,
                "bath": float(bath) if bath is not None else None,
                "house_size": float(sqft) if sqft is not None else None,
                "acre_lot": float(acre_lot) if acre_lot is not None else 0.0,
                "zip_code": str(zip_code) if zip_code else None,
                "zip3": str(zip_code)[:3] if zip_code else None,  # important for your pipeline
            }
        ])
        return float(model.predict(row)[0])
    except Exception as e:
        st.warning(f"Model prediction failed: {e}")
        return None


# ---------------------------
# Initialize session_state with stable keys
# ---------------------------
DEFAULTS = {
    "api_key": os.getenv("ATTOM_API_KEY", ""),
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

    # finance inputs (editable)
    "purchase_price": 1_100_000,
    "reno_cost": 50_000,
    "monthly_carry": 3_000,
    "months": 6,
    "exit_value": 1_200_000,  # manual ARV; NOT auto-forced by model

    # model/cache
    "use_model_exit": False,
    "model_loaded": False,
    "last_attom_json": {},
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("🔧 Setup")
st.session_state.api_key = st.sidebar.text_input(
    "ATTOM API key", value=st.session_state.api_key, type="password"
)
st.session_state.addr1 = st.sidebar.text_input("Street (address1)", st.session_state.addr1)
st.session_state.addr2 = st.sidebar.text_input("City, State Zip (address2)", st.session_state.addr2)

st.sidebar.markdown("---")
st.session_state.model_path = st.sidebar.text_input("ML model path", value=st.session_state.model_path)
model = None
if os.path.exists(st.session_state.model_path):
    try:
        model = load_model(st.session_state.model_path)
        st.session_state.model_loaded = True
        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.session_state.model_loaded = False
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.session_state.model_loaded = False
    st.sidebar.info("Provide a trained model path (e.g., ml/model_ca_zip_hgbr.joblib)")

# ---------------------------
# Main
# ---------------------------
st.title("🏠 Simple Flip Estimator (MVP) + CA ZIP Model")
st.caption("Optionally fetch facts via ATTOM to prefill fields, then estimate value and run renovation what‑ifs. All fields remain editable.")

# ---- ATTOM fetch just PREFILLS; never locks editing
c_fetch1, c_fetch2 = st.columns([1, 2])
with c_fetch1:
    if st.button("🔎 Fetch Property from ATTOM"):
        if not st.session_state.api_key:
            st.error("Add your ATTOM API key in the sidebar.")
        else:
            with st.spinner("Calling ATTOM…"):
                data = {}
                try:
                    r = requests.get(
                        ATTOM_URL,
                        headers={"accept": "application/json", "apikey": st.session_state.api_key},
                        params={
                            "address1": st.session_state.addr1.strip(),
                            "address2": st.session_state.addr2.strip(),
                        },
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

st.markdown("---")

# ---- Editable Property Snapshot & Inputs (always editable)
st.subheader("📄 Property Snapshot (editable)")
p1, p2, p3, p4, p5 = st.columns(5)
st.session_state.beds = p1.number_input("Beds", min_value=0, step=1, value=int(st.session_state.beds))
st.session_state.baths = p2.number_input("Baths", min_value=0.0, step=0.5, value=float(st.session_state.baths))
st.session_state.sqft = p3.number_input("Sqft", min_value=200, step=50, value=int(st.session_state.sqft))
st.session_state.year = p4.number_input("Year", min_value=1800, max_value=2100, step=1, value=int(st.session_state.year) if st.session_state.year else 1990)
st.session_state.zip_code = p5.text_input("ZIP code", value=st.session_state.zip_code)

l1, l2 = st.columns(2)
st.session_state.acre_lot = l1.number_input("Lot size (acres)", min_value=0.0, step=0.01, format="%.2f", value=float(st.session_state.acre_lot))
with l2:
    st.caption("Rooms like offices may not count as bedrooms; feel free to correct the numbers.")

st.markdown("---")

# ---- Renovation Planner (CA cost presets)
st.subheader("🛠️ Renovation Planner (CA costs)")

# Cost presets typical for CA (rough 2025 ballparks)
COST_PRESETS = {
    "Basic": {
        "kitchen": 30000,
        "full_bath": 12000,
        "half_bath": 6000,
        "bedroom_cosmetic": 3000,
        "living_cosmetic": 4000,
        "flooring_per_sqft": 6,
        "paint_interior_per_sqft": 2.5,
        "window_each": 700,
        "roof_replace": 14000,
        "electrical_panel": 2500,
        "repipe_plumbing": 6000,
        "permits_allowance": 2000,
    },
    "Mid": {
        "kitchen": 45000,
        "full_bath": 22000,
        "half_bath": 12000,
        "bedroom_cosmetic": 6000,
        "living_cosmetic": 8000,
        "flooring_per_sqft": 9,
        "paint_interior_per_sqft": 3.5,
        "window_each": 1100,
        "roof_replace": 20000,
        "electrical_panel": 4000,
        "repipe_plumbing": 10000,
        "permits_allowance": 4000,
    },
    "High": {
        "kitchen": 70000,
        "full_bath": 35000,
        "half_bath": 18000,
        "bedroom_cosmetic": 9000,
        "living_cosmetic": 12000,
        "flooring_per_sqft": 14,
        "paint_interior_per_sqft": 5,
        "window_each": 1600,
        "roof_replace": 30000,
        "electrical_panel": 6500,
        "repipe_plumbing": 16000,
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

# Reflect into session state's reno_cost (so math below uses it) but show transparently
st.session_state.reno_cost = computed_reno_cost

rc1, rc2, rc3 = st.columns(3)
rc1.metric("Reno Subtotal", f"${subtotal:,.0f}")
rc2.metric("Contingency", f"${contingency:,.0f}")
rc3.metric("Total Reno Cost", f"${computed_reno_cost:,.0f}")

# Optional manual tweak without losing transparency
with st.expander("Adjustments (optional)"):
    manual_adjust = st.number_input("Manual tweak (+/− $)", value=0, step=500)
    if manual_adjust != 0:
        st.session_state.reno_cost = int(max(0, computed_reno_cost + manual_adjust))
        st.caption(f"Applied adjustment: ${manual_adjust:,.0f}")

st.markdown("---")

# ---- Deal Math (moved to bottom; uses computed reno cost)
st.subheader("📊 Deal Math")

# Optional model estimate (FYI only)
est_val = None
if st.session_state.model_loaded and st.session_state.zip_code:
    est_val = predict_price(
        model,
        st.session_state.beds,
        st.session_state.baths,
        st.session_state.sqft,
        st.session_state.acre_lot,
        st.session_state.zip_code,
    )

d1, d2 = st.columns([1,1])
with d1:
    st.session_state.purchase_price = st.number_input(
        "Purchase Price",
        min_value=0,
        value=int(st.session_state.purchase_price),
        step=10_000,
    )
with d2:
    st.session_state.monthly_carry = st.number_input(
        "Monthly Carry (tax/HOA/ins/interest)",
        min_value=0,
        value=int(st.session_state.monthly_carry),
        step=500,
    )
    st.session_state.months = st.number_input(
        "Holding Months",
        min_value=1,
        value=int(st.session_state.months),
        step=1,
    )

# Exit value controls
carv1, carv2 = st.columns([2,1])
with carv1:
    st.session_state.exit_value = st.number_input(
        "Exit Value (ARV) — manual",
        min_value=0,
        value=int(st.session_state.exit_value),
        step=10_000,
    )
with carv2:
    st.session_state.use_model_exit = st.checkbox(
        "Use model estimate",
        value=bool(st.session_state.use_model_exit and est_val is not None),
        disabled=est_val is None,
        help="If checked, ARV will be set to the model's estimate for this ZIP and inputs.",
    )
    if est_val is not None:
        st.caption(f"Model estimate (FYI): **${est_val:,.0f}**")

if st.session_state.use_model_exit and est_val is not None:
    computed_exit_value = est_val
else:
    computed_exit_value = float(st.session_state.exit_value)

# Totals
purchase_price = float(st.session_state.purchase_price)
reno_cost = float(st.session_state.reno_cost)
monthly_carry = float(st.session_state.monthly_carry)
months = int(st.session_state.months)

total_costs = purchase_price + reno_cost + (monthly_carry * months)
profit = computed_exit_value - total_costs

pm1, pm2, pm3, pm4 = st.columns(4)
pm1.metric("Reno Cost (calc)", f"${reno_cost:,.0f}")
pm2.metric("Total Costs", f"${total_costs:,.0f}")
pm3.metric("Exit Value (ARV)", f"${computed_exit_value:,.0f}")
pm4.metric("Estimated Profit", f"${profit:,.0f}")

st.markdown("---")

# ---- Renovation What-If (sqft add-on)
st.subheader("🔧 Renovation What-If (sqft add‑on)")
w1, w2, w3 = st.columns(3)
add_sqft = w1.number_input("Add Sqft", min_value=0, value=300, step=50)
cost_per_sqft = w2.number_input("Cost per Sqft", min_value=0, value=350, step=25)
use_model_after = w3.checkbox(
    "Use model for after-repair value",
    value=True if st.session_state.model_loaded else False,
    disabled=not st.session_state.model_loaded,
)

if use_model_after and st.session_state.model_loaded and st.session_state.zip_code:
    after_val = predict_price(
        model,
        st.session_state.beds,
        st.session_state.baths,
        st.session_state.sqft + add_sqft,
        st.session_state.acre_lot,
        st.session_state.zip_code,
    )
else:
    # simple linear proxy if no model
    after_val = computed_exit_value + (add_sqft * 200)

capex = add_sqft * cost_per_sqft
uplift = (after_val - computed_exit_value) if (after_val and computed_exit_value is not None) else None
roi = ((uplift - capex) / capex) if uplift is not None and capex > 0 else None

cA, cB, cC, cD = st.columns(4)
cA.metric("After Value", f"${after_val:,.0f}" if after_val else "—")
cB.metric("Value Uplift", f"${uplift:,.0f}" if uplift is not None else "—")
cC.metric("CapEx", f"${capex:,.0f}")
cD.metric("ROI", f"{roi*100:,.1f}%" if roi is not None else "—")

# ---- Debug JSON (optional)
with st.expander("See raw ATTOM JSON (debug)"):
    st.json(st.session_state.last_attom_json if isinstance(st.session_state.last_attom_json, dict) else {"note": "no JSON"})
