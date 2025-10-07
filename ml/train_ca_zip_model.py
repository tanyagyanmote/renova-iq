import argparse, os, joblib, re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def norm_state(x):
    if pd.isna(x): return x
    s = str(x).strip().lower()
    if s in {"ca","california"}:
        return "CA"
    return s.upper()

def norm_zip5(z):
    if pd.isna(z): return np.nan
    s = re.sub(r"\D", "", str(z))
    return s[:5] if len(s) >= 5 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to realtor-data.csv (Kaggle-style)")
    ap.add_argument("--out", default="ml/model_ca_zip_hgbr.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)

    need = ["price", "bed", "bath", "house_size", "acre_lot", "zip_code", "state"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = df[need].copy()

    for c in ["price","bed","bath","house_size","acre_lot"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["state"] = df["state"].map(norm_state)
    df = df[df["state"] == "CA"].copy()

    df["zip5"] = df["zip_code"].map(norm_zip5)
    df["zip3"] = df["zip5"].str[:3]

    # minimal cleaning
    df = df.dropna(subset=["price","house_size","bed","bath","zip3"])
    df["acre_lot"] = df["acre_lot"].fillna(0)

    # remove crazy outliers
    df = df[(df["house_size"] >= 200) & (df["house_size"] <= 10000)]
    df = df[(df["price"] >= 50000) & (df["price"] <= 10_000_000)]

    # features/target
    num = ["bed","bath","house_size","acre_lot"]
    cat = ["zip3"]
    X = df[num + cat]
    y = df["price"]


    pre = ColumnTransformer(
        [
            ("num", "passthrough", num),
            ("zip", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=None,
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    mae = mean_absolute_error(yte, pred)
    r2  = r2_score(yte, pred)
    rel = mae / np.median(yte)

    print(f"✅ CA+ZIP model | MAE: ${mae:,.0f} | R²: {r2:.3f} | Rel MAE: {rel:.1%} | n={len(df):,}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"💾 Saved -> {args.out}")

if __name__ == "__main__":
    main()
