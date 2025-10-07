Renova-IQ — Flip Estimator (CA ZIP Model)

A simple Streamlit app to estimate renovation costs and flip math. Optionally train and use a California home-price model (zipcode-aware) to suggest ARV. ATTOM API can be used to prefill property facts, but it’s optional.

Features

Editable inputs for beds, baths, sqft, lot size, purchase price, carry costs, months, and ARV.

Renovation cost presets (Basic/Mid/High) with contingency and manual adjustments.

Optional ML model for ARV suggestion (HistGradientBoostingRegressor with ZIP3 one-hot).

Optional ATTOM fetch to prefill fields (never locks editing).

Requirements

Python 3.10+ is recommended.

Install dependencies:

pip install -r requirements.txt


requirements.txt:

streamlit
pandas
numpy
scikit-learn
joblib
requests

Data for Training

To train the model, you need a CSV with at least these columns:

price (numeric)

bed (numeric)

bath (numeric)

house_size (numeric, sqft)

acre_lot (numeric, acres; missing allowed)

zip_code (string or numeric; 5-digit ZIPs)

state (string; includes “CA” rows)

The training script filters to California (state == "CA"), normalizes ZIPs to 5-digit and derives zip3, drops rows with missing required fields, and clips extreme outliers.

Train the Model

Place your dataset at data/realtor-data.csv (or adjust the path), then run:

python3 ml/train_ca_zip_model.py --csv data/realtor-data.csv --out ml/model_ca_zip_hgbr.joblib


This prints metrics (MAE, R²) and saves a pipeline (preprocessing + model) to ml/model_ca_zip_hgbr.joblib.


In the sidebar:

Set the ML model path to ml/model_ca_zip_hgbr.joblib (or wherever you saved it).

You can edit all fields freely at any time.

Optional: ATTOM Prefill

The app can call ATTOM to prefill address, beds, baths, sqft, and year. This is optional; you can use the app without any secrets.

To enable ATTOM prefill, add your key to .streamlit/secrets.toml:

ATTOM_API_KEY = "YOUR_ATTOM_KEY_HERE"


Alternatively, set an environment variable:

export ATTOM_API_KEY="YOUR_ATTOM_KEY_HERE"


Then click “Fetch Property from ATTOM” in the app. The fetched values will prefill but remain editable.

Project Structure 
.
├─ app/
│  └─ app.py
├─ ml/
│  ├─ train_ca_zip_model.py
│  └─ model_ca_zip_hgbr.joblib           # created by training step
├─ data/
│  └─ realtor-data.csv                   # your training data (not committed)
├─ .streamlit/
│  └─ secrets.toml                       # optional; for ATTOM_API_KEY
├─ requirements.txt
└─ README.md

