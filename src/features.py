"""Data loading, cleaning and feature engineering for the freight-rate model.

Design rules
------------
* The SAME cleaning is applied to train, validation and the December chart rows.
  No row is ever dropped from a prediction set.
* Only label-free information is pooled across datasets (date-level averages of
  ``market_index`` / ``quote_signal``); ``posted_rate`` never leaks into features.
* Features must be computable from the December chart input columns (pickup,
  delivery, distance, equipment, weight, date) plus date-level market signals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

WEIGHT_CAP = 47_500.0
MIN_DISTANCE = 70.0
EQUIPMENT_CODES = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
TARGET = "posted_rate"

# --- feature sets used in the experiments / final model --------------------
BASE = [
    "log_dist", "dist", "equip", "weight", "weight_neg", "weight_cap", "weight_missing",
    "dist_floor", "dow",
]
MARKET_DATE = ["mi_date", "qs_date"]          # date-level averages (smooth signals)
MARKET_ROW = ["mi_row", "qs_row"]             # row-level values as supplied
COORDS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
CITY_IDS = ["pickup_id", "delivery_id"]
CALENDAR_TREND = ["doy"]                      # kept only to demonstrate it hurts


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    return train, val, dec


def city_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) row per city, pooled over every frame that has coordinates."""
    parts = []
    for f in frames:
        if "pickup_lat" not in f.columns:
            continue
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    table = pd.concat(parts).drop_duplicates("city").set_index("city").sort_index()
    return table


def date_signals(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date mean of market_index and quote_signal (label-free, pooled)."""
    pooled = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames if "market_index" in f.columns])
    return pooled.groupby("date").agg(mi_date=("market_index", "mean"), qs_date=("quote_signal", "mean"))


def fit_fill_values(train: pd.DataFrame) -> dict:
    """Imputation constants learned from labeled training rows only."""
    w = train["weight"].abs()
    return {"weight_median": float(w.median()), "mi_median": float(train["market_index"].median()),
            "qs_median": float(train["quote_signal"].median())}


def build_features(df: pd.DataFrame, signals: pd.DataFrame, fill: dict, cities: pd.DataFrame) -> pd.DataFrame:
    """Clean ``df`` and return a numeric feature frame aligned to ``df.index``."""
    X = pd.DataFrame(index=df.index)
    dist = df["distance"].astype(float)
    X["dist"] = dist
    X["log_dist"] = np.log(dist)
    X["dist_floor"] = (dist <= MIN_DISTANCE).astype(int)          # distances are floored at 70 mi
    X["equip"] = df["equipment"].map(EQUIPMENT_CODES).astype(int)

    # weight: sign errors (abs), clipped at 47,500 (flag), missing (median + flag)
    w = df["weight"].astype(float)
    X["weight_neg"] = (w < 0).astype(int)
    X["weight_missing"] = w.isna().astype(int)
    w = w.abs()
    X["weight_cap"] = (w >= WEIGHT_CAP).astype(int)
    X["weight"] = w.fillna(fill["weight_median"])

    X["dow"] = df["date"].dt.dayofweek
    X["doy"] = df["date"].dt.dayofyear

    # market signals: date-level average (smooth) and the row value (missing -> date average)
    sig = signals.reindex(df["date"]).reset_index(drop=True)
    sig.index = df.index
    X["mi_date"] = sig["mi_date"].fillna(fill["mi_median"])
    X["qs_date"] = sig["qs_date"].fillna(fill["qs_median"])
    if "market_index" in df.columns:
        X["mi_row"] = df["market_index"].fillna(X["mi_date"])
        X["qs_row"] = df["quote_signal"].fillna(X["qs_date"])
    else:  # December chart rows carry no row-level signal
        X["mi_row"] = X["mi_date"]
        X["qs_row"] = X["qs_date"]

    # coordinates (coarse geography only - see EDA: haversine is unreliable)
    for side in ("pickup", "delivery"):
        if f"{side}_lat" in df.columns:
            X[f"{side}_lat"] = df[f"{side}_lat"]
            X[f"{side}_lon"] = df[f"{side}_lon"]
        else:
            X[f"{side}_lat"] = df[side].map(cities["lat"])
            X[f"{side}_lon"] = df[side].map(cities["lon"])
    codes = {c: i for i, c in enumerate(cities.index)}
    X["pickup_id"] = df["pickup"].map(codes)
    X["delivery_id"] = df["delivery"].map(codes)
    return X


def target(train: pd.DataFrame) -> pd.Series:
    return np.log(train[TARGET])
