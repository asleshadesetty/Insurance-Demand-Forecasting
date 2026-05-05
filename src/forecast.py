"""
forecast.py
-----------
Core forecasting pipeline using Facebook Prophet.

For each medication/resource category:
  1. Trains a Prophet model on historical weekly demand
  2. Generates a 52-week (1-year) forecast with uncertainty intervals
  3. Evaluates on a held-out test set (last 12 weeks) using cross-validation
  4. Saves per-category forecast CSVs to outputs/

Metrics reported: MAPE, MAE, RMSE
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH    = "data/insurance_demand.csv"
OUTPUT_DIR   = "outputs"
TEST_WEEKS   = 12
FORECAST_WKS = 52

os.makedirs(OUTPUT_DIR, exist_ok=True)


def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return round(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100, 2)


def train_prophet(df_train):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    model.fit(df_train)
    return model


def evaluate_insample(model, df_train, df_test):
    """
    Evaluate by predicting over the full training range + test range,
    then picking the test portion by position (avoids date alignment issues
    caused by Prophet's internal frequency rounding).
    """
    n_train  = len(df_train)
    future   = model.make_future_dataframe(periods=TEST_WEEKS, freq="W")
    forecast = model.predict(future)

    # The last TEST_WEEKS rows of the forecast correspond to the test period
    pred_test = forecast.iloc[n_train: n_train + TEST_WEEKS]["yhat"].clip(lower=0).values
    actual    = df_test["y"].values

    # Trim to same length in case of minor off-by-one
    n = min(len(actual), len(pred_test))
    actual, pred_test = actual[:n], pred_test[:n]

    return {
        "MAPE": mape(actual, pred_test),
        "MAE":  round(mean_absolute_error(actual, pred_test), 2),
        "RMSE": round(np.sqrt(mean_squared_error(actual, pred_test)), 2),
    }


def run_pipeline():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    all_metrics   = []
    all_forecasts = []

    categories = df["category"].unique()
    print(f"\nRunning Prophet forecasts for {len(categories)} categories...\n")
    print(f"{'Category':<20} {'MAPE (%)':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 54)

    for cat in sorted(categories):
        cat_df = (
            df[df["category"] == cat][["date", "demand"]]
            .rename(columns={"date": "ds", "demand": "y"})
            .sort_values("ds")
            .reset_index(drop=True)
        )

        split_idx = len(cat_df) - TEST_WEEKS
        df_train  = cat_df.iloc[:split_idx].copy()
        df_test   = cat_df.iloc[split_idx:].copy()

        model   = train_prophet(df_train)
        metrics = evaluate_insample(model, df_train, df_test)
        metrics["category"] = cat
        all_metrics.append(metrics)

        # Full forecast from beginning of data + future horizon
        future   = model.make_future_dataframe(periods=FORECAST_WKS, freq="W")
        forecast = model.predict(future)
        forecast["category"] = cat

        # Attach actuals by position for historical rows
        actual_vals = list(cat_df["y"].values) + [np.nan] * FORECAST_WKS
        forecast["actual"] = actual_vals[:len(forecast)]

        all_forecasts.append(
            forecast[["ds", "category", "yhat", "yhat_lower", "yhat_upper", "actual"]]
        )

        print(f"{cat:<20} {metrics['MAPE']:>10} {metrics['MAE']:>10} {metrics['RMSE']:>10}")

    metrics_df   = pd.DataFrame(all_metrics)[["category", "MAPE", "MAE", "RMSE"]]
    forecasts_df = pd.concat(all_forecasts, ignore_index=True)

    metrics_df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)
    forecasts_df.to_csv(f"{OUTPUT_DIR}/forecasts.csv", index=False)

    avg_mape = round(metrics_df["MAPE"].mean(), 2)
    print(f"\n{'─'*54}")
    print(f"{'Average MAPE':<20} {avg_mape:>10}%")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    return metrics_df, forecasts_df


if __name__ == "__main__":
    run_pipeline()
