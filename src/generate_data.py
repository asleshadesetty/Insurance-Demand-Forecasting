"""
generate_data.py
----------------
Downloads and preprocesses the Pharma Sales Data from Kaggle:
https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data

The dataset contains weekly sales of pharmaceutical products across
drug categories. This script:
  1. Loads the raw CSV (salesdaily.csv or salesweekly.csv)
  2. Maps drug categories to insurance-relevant labels
  3. Aggregates to weekly frequency
  4. Saves cleaned data to data/insurance_demand.csv

SETUP:
  1. Download the dataset from Kaggle:
     https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data
  2. Place salesweekly.csv inside the data/ folder
  3. Run: python src/generate_data.py

If you do not have Kaggle access, run with --synthetic flag to generate
a representative synthetic dataset instead:
     python src/generate_data.py --synthetic
"""

import argparse
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

DATA_DIR  = "data"
OUT_PATH  = f"{DATA_DIR}/insurance_demand.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# Mapping from Kaggle pharma category codes to descriptive labels
# M01AB, M01AE  -> Musculoskeletal (anti-inflammatories)
# N02BA, N02BE  -> Cardiovascular  (analgesics/antipyretics)
# N05B, N05C   -> Mental Health    (anxiolytics/hypnotics)
# R03           -> Respiratory     (bronchodilators)
# R06           -> Diabetes proxy  (antihistamines — used as stand-in)
CATEGORY_MAP = {
    "M01AB": "Musculoskeletal",
    "M01AE": "Musculoskeletal",
    "N02BA": "Cardiovascular",
    "N02BE": "Cardiovascular",
    "N05B":  "Mental Health",
    "N05C":  "Mental Health",
    "R03":   "Respiratory",
    "R06":   "Diabetes",
}


# ── Real data path ─────────────────────────────────────────────────────────────
def load_kaggle(path="data/salesweekly.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] '{path}' not found.\n"
            "Download salesweekly.csv from:\n"
            "https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data\n"
            "and place it in the data/ folder.\n"
            "Or run with --synthetic to use generated data instead."
        )

    df_raw = pd.read_csv(path, parse_dates=["datum"])
    df_raw = df_raw.rename(columns={"datum": "date"})

    records = []
    for code, label in CATEGORY_MAP.items():
        if code not in df_raw.columns:
            continue
        sub = df_raw[["date", code]].copy()
        sub = sub.rename(columns={code: "demand"})
        sub["category"] = label
        records.append(sub)

    df = pd.concat(records, ignore_index=True)
    df = (
        df.groupby(["date", "category"], as_index=False)["demand"]
        .sum()
        .sort_values(["category", "date"])
        .reset_index(drop=True)
    )
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"]  = df["date"].dt.isocalendar().week.astype(int)
    return df


# ── Synthetic fallback ─────────────────────────────────────────────────────────
CATEGORIES = {
    "Cardiovascular":   {"base": 420, "trend": 0.0008, "noise": 18},
    "Diabetes":         {"base": 380, "trend": 0.0010, "noise": 15},
    "Respiratory":      {"base": 290, "trend": 0.0005, "noise": 22},
    "Mental Health":    {"base": 210, "trend": 0.0012, "noise": 12},
    "Musculoskeletal":  {"base": 175, "trend": 0.0004, "noise": 20},
}

def generate_synthetic(start="2020-01-01", end="2023-12-31"):
    print("[INFO] Generating synthetic dataset (Kaggle data not provided).")
    dates   = pd.date_range(start=start, end=end, freq="W-MON")
    records = []
    for i, date in enumerate(dates):
        annual = np.sin(2 * np.pi * (date.dayofyear - 30) / 365)
        for cat, params in CATEGORIES.items():
            trend   = params["base"] * (1 + params["trend"] * i)
            demand  = max(0, trend + trend * 0.12 * annual
                          + np.random.normal(0, params["noise"]))
            records.append({"date": date, "category": cat, "demand": round(demand, 1)})
    df = pd.DataFrame(records)
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"]  = df["date"].dt.isocalendar().week.astype(int)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of Kaggle dataset")
    args = parser.parse_args()

    if args.synthetic:
        df = generate_synthetic()
    else:
        try:
            df = load_kaggle()
            print("[INFO] Loaded Kaggle pharma sales dataset.")
        except FileNotFoundError as e:
            print(e)
            exit(1)

    df.to_csv(OUT_PATH, index=False)
    print(f"[INFO] Dataset saved to {OUT_PATH}  |  Shape: {df.shape}")
    print(df.head(10).to_string(index=False))
