"""Exploratory analysis: figures for the report + data-quality statistics.

Usage:  python -m src.eda      (writes figures/*.png and outputs/eda_stats.json)
"""
from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import features as F

TEAL, ORANGE, GREY, RED = "#064A56", "#E07A1F", "#8A9BA0", "#C0392B"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
FIG = F.ROOT / "figures"


def main() -> None:
    FIG.mkdir(exist_ok=True)
    (F.ROOT / "outputs").mkdir(exist_ok=True)
    tr, va, _ = F.load_raw()
    stats: dict = {}

    # ------------------------------------------------------------- data quality
    seen = set(tr.pickup) | set(tr.delivery)
    val_cities = set(va.pickup) | set(va.delivery)
    new_cities = sorted(val_cities - seen)
    unseen_rows = int((~va.pickup.isin(seen) | ~va.delivery.isin(seen)).sum())
    route_med = pd.concat([tr, va]).assign(route=lambda d: d.pickup + ">" + d.delivery)
    route_med["ratio"] = route_med.distance / route_med.groupby("route").distance.transform("median")
    stats["quality"] = {
        "train_rows": len(tr), "val_rows": len(va),
        "date_range_train": [str(tr.date.min().date()), str(tr.date.max().date())],
        "date_range_val": [str(va.date.min().date()), str(va.date.max().date())],
        "negative_weight_train": int((tr.weight < 0).sum()), "negative_weight_val": int((va.weight < 0).sum()),
        "weight_at_cap_train": int((tr.weight.abs() == F.WEIGHT_CAP).sum()),
        "missing_weight_train": int(tr.weight.isna().sum()), "missing_weight_val": int(va.weight.isna().sum()),
        "missing_market_index_train": int(tr.market_index.isna().sum()),
        "missing_market_index_val": int(va.market_index.isna().sum()),
        "distance_at_floor_70_train": int((tr.distance <= F.MIN_DISTANCE).sum()),
        "distance_vs_route_median_within_15pct_%": float(100 * ((route_med.ratio - 1).abs() <= 0.15).mean()),
        "new_cities_in_val": new_cities, "val_rows_with_new_city": unseen_rows,
        "duplicate_load_ids": int(tr.load_id.duplicated().sum() + va.load_id.duplicated().sum()),
    }

    # ---------------------------------------------- outlier structure of the target
    rpm = tr.posted_rate / tr.distance
    X = F.build_features(tr, F.date_signals(tr, va), F.fit_fill_values(tr), F.city_table(tr, va))
    y = F.target(tr)
    base = ["log_dist", "equip", "weight"]
    m = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=8,
                          min_child_samples=100, verbose=-1, n_jobs=1).fit(X[base], y)
    resid = y - m.predict(X[base])
    mad_sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    wild = np.abs(resid) > 0.5
    stats["target_noise"] = {
        "rate_per_mile_median": float(rpm.median()), "rate_per_mile_p01": float(rpm.quantile(.01)),
        "rate_per_mile_p99": float(rpm.quantile(.99)), "rate_per_mile_max": float(rpm.max()),
        "core_sigma_log_(MAD)": float(mad_sigma), "overall_std_log": float(resid.std()),
        "wild_outlier_share_%": float(100 * wild.mean()),
        "wild_high_%": float(100 * (resid > 0.5).mean()), "wild_low_%": float(100 * (resid < -0.5).mean()),
    }

    # ---------------------------------------------------------------- figures
    # 1. rate vs distance by equipment
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for eq, c in zip(["Dry Van", "Flatbed", "Reefer"], [TEAL, ORANGE, GREY]):
        s = tr[tr.equipment == eq].sample(3000, random_state=0)
        ax.scatter(s.distance, s.posted_rate, s=5, alpha=.35, color=c, label=eq)
    ax.set_xlabel("Distance (miles)"); ax.set_ylabel("Posted rate ($)")
    ax.set_title("Rate is ~linear in distance; equipment shifts the slope")
    ax.legend(markerscale=3, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "01_rate_vs_distance.png", dpi=170); plt.close(fig)

    # 2. residual distribution (wild outliers)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(np.clip(resid, -1.5, 1.5), bins=150, color=TEAL)
    ax.set_yscale("log"); ax.set_xlabel("log(actual / simple-model prediction), clipped at +/-1.5")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title(f"~{100*wild.mean():.1f}% wild outliers; clean-row noise is only ~{100*mad_sigma:.1f}%")
    fig.tight_layout(); fig.savefig(FIG / "02_target_outliers.png", dpi=170); plt.close(fig)

    # 3. market_index over time (train vs validation) and daily rate effect
    sig = F.date_signals(tr, va)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(sig.index, sig.mi_date, color=TEAL, lw=1.4)
    ax.axvline(pd.Timestamp("2025-11-01"), color=RED, ls="--", lw=1)
    ax.text(pd.Timestamp("2025-11-03"), sig.mi_date.max() * .98, "validation\nperiod", color=RED, fontsize=9, va="top")
    ax.set_ylabel("Daily mean market_index"); ax.set_title("Market level drifts; validation sits after all training data")
    fig.tight_layout(); fig.savefig(FIG / "03_market_index_time.png", dpi=170); plt.close(fig)

    # 4. cities: train vs new-in-validation
    cities = F.city_table(tr, va)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    is_new = cities.index.isin(new_cities)
    ax.scatter(cities.lon[~is_new], cities.lat[~is_new], color=TEAL, s=22, label="in training")
    ax.scatter(cities.lon[is_new], cities.lat[is_new], color=RED, s=42, marker="D", label="only in validation")
    for c in new_cities:
        ax.annotate(c, (cities.lon[c], cities.lat[c]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Longitude (as supplied)"); ax.set_ylabel("Latitude (as supplied)")
    ax.set_title(f"{len(new_cities)} cities are new in validation ({unseen_rows:,} rows)")
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.08, 0.01, "Note: supplied coordinates are distorted (not true geography) but are still usable, "
             "consistent per-city features", fontsize=8.5, color="#455A60")
    fig.savefig(FIG / "04_new_cities.png", dpi=170); plt.close(fig)

    # 5. random vs time split optimism is produced in report script (needs models)
    (F.ROOT / "outputs" / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
