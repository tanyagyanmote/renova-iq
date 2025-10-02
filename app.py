import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CA ZIP Price Predictor", layout="centered")
st.title("California ZIP House Price Predictor (MVP)")

# Load model (must exist at ml/model_ca_zip_ridge.joblib)
@st.cache_resource
def load_model():
    return joblib.load("ml/model_ca_zip_ridge.joblib")

try:
    model = load_model()
    feature_cols = list(model.feature_names_in_)
except Exception as e:
    st.error("Model failed to load. Make sure ml/model_ca_zip_ridge.joblib exists.")
    st.exception(e)
    st.stop()

col1, col2 = st.columns(2)
with col1:
    bed = st.slider("Bedrooms", 1, 8, 3)
    bath = st.slider("Bathrooms", 1, 5, 2)
with col2:
    sqft = st.number_input("House size (sqft)", 300, 6000, 1450, step=50)
    acre_lot = st.number_input("Lot size (acres)", 0.0, 5.0, 0.10, step=0.01)

zip_code = st.text_input("ZIP code (CA)", "94582")

def build_row(bed, bath, sqft, acre_lot, zip_code):
    # Base numeric features
    row = pd.DataFrame([{
        "bed": bed, "bath": bath, "house_size": sqft, "acre_lot": acre_lot
    }])
    # Align to training columns
    row = row.reindex(columns=feature_cols, fill_value=0)
    # Turn on ZIP one-hot if present in the model
    col = f"zip_code_{zip_code}"
    if col in row.columns:
        row[col] = 1
    return row

if st.button("Predict price"):
    X = build_row(bed, bath, sqft, acre_lot, zip_code)
    price = float(model.predict(X)[0])
    st.success(f"Estimated price: ${price:,.0f}")

    # Simple renovation scenario (+400 sqft @ $350/sqft capex)
    add_sqft = 400
    capex_per_sqft = 350
    X_after = build_row(bed, bath, sqft + add_sqft, acre_lot, zip_code)
    price_after = float(model.predict(X_after)[0])
    uplift = price_after - price
    capex = add_sqft * capex_per_sqft
    roi_pct = (uplift - capex) / capex if capex else 0.0

    st.subheader("Renovation Scenario (+400 sqft)")
    st.write(f"New estimated price: ${price_after:,.0f}")
    st.write(f"Value uplift: ${uplift:,.0f}")
    st.write(f"Capex (est.): ${capex:,.0f}")
    st.write(f"ROI: {roi_pct*100:.1f}%")
