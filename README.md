# Renova-IQ: Flip Estimator + CA ZIP Model

A Streamlit-based tool for estimating renovation costs and flip profitability, with optional property data prefill from ATTOM and a trained California home price model.

## Features

* Fully editable property and financial inputs
* Optional ATTOM API fetch for property facts (prefill only)
* Renovation planner with California cost presets
* Deal math: purchase, renovation, carry, ARV, profit
* Optional ML-based ARV prediction using ZIP3 one-hot encoded model
* "What-if" analysis for adding square footage

## Project Structure

```
.
├── app/
│   └── app.py
├── ml/
│   ├── train_ca_zip_model.py
│   └── model_ca_zip_hgbr.joblib   # generated after training
├── data/
│   └── realtor-data.csv           # your training dataset
├── requirements.txt
└── .streamlit/                    # optional secrets folder (not required to run)
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Create a Virtual Environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)

If you want to generate your own model, provide a CSV containing at least these columns:
`price, bed, bath, house_size, acre_lot, zip_code, state`

Run:

```bash
python3 ml/train_ca_zip_model.py --csv data/realtor-data.csv --out ml/model_ca_zip_hgbr.joblib
```

This creates `ml/model_ca_zip_hgbr.joblib`, which the app will automatically load.

### 5. Run the App

```bash
streamlit run app/app.py
```

The app opens in your browser at:

```
http://localhost:8501
```

### Notes

* You do **not** need to create `.streamlit/secrets.toml` if cloning the repo.
* ATTOM API usage is optional. The app will still work manually without it.
* To use ATTOM, add your key in the sidebar or store it in `.streamlit/secrets.toml`.

### Example `.streamlit/secrets.toml` (optional)

```
ATTOM_API_KEY = "your_attom_key_here"
```

## Summary

* Train your model (optional)
* Launch the Streamlit app
* Enter property details manually or prefill via ATTOM
* Adjust renovation scope, costs, and assumptions
* Review deal math, ARV, and profit estimates
