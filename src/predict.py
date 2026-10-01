"""Generate predictions with the trained model.

Usage:
    python -m src.predict --target validation   -> outputs/validation_predictions.csv
    python -m src.predict --target december     -> outputs/december_predictions.csv
    python -m src.predict --target both         -> both (default)
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from src import features as F


def _load_bundle():
    with open(F.ROOT / "outputs" / "model" / "two_stage.pkl", "rb") as fh:
        return pickle.load(fh)


def predict_frame(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    X = F.build_features(df, bundle["signals"], bundle["fill"], bundle["cities"])
    return bundle["model"].predict(X, df["date"], bundle["signals"])


def run_validation(bundle: dict) -> None:
    _, val, _ = F.load_raw()
    pred = predict_frame(val, bundle)
    out = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": np.round(pred, 2)})
    assert len(out) == 12_000 and out["load_id"].is_unique and (out["predicted_rate"] > 0).all()
    out.to_csv(F.ROOT / "outputs" / "validation_predictions.csv", index=False)
    print(f"wrote outputs/validation_predictions.csv  ({len(out):,} rows)")
    print(out["predicted_rate"].describe().round(2).to_string())


def run_december(bundle: dict) -> None:
    _, _, dec = F.load_raw()
    pred = predict_frame(dec, bundle)
    out = dec.copy()
    out["predicted_rate"] = np.round(pred, 2)
    assert list(out.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    out.to_csv(F.ROOT / "outputs" / "december_predictions.csv", index=False)
    print(f"wrote outputs/december_predictions.csv  ({len(out)} rows)")
    print(out[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["validation", "december", "both"], default="both")
    args = ap.parse_args()
    bundle = _load_bundle()
    if args.target in ("validation", "both"):
        run_validation(bundle)
    if args.target in ("december", "both"):
        run_december(bundle)
