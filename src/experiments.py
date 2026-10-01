"""Run the model / feature comparison under time-CV and city-holdout CV.

Usage:  python -m src.experiments   (~10 min on 1 CPU; resumable - finished specs are cached in
        outputs/cv_cache/; writes outputs/cv_results.csv and outputs/cv_by_fold.csv)
"""
from __future__ import annotations

import re
import sys
import time

import numpy as np
import pandas as pd

from src import features as F
from src import validation as V

STRUCT = F.BASE + F.COORDS      # distance, equipment, weight(+flags), day-of-week, city coordinates


def two_stage(name, features=STRUCT, **kw) -> V.Spec:
    cfg = dict(kind="twostage", objective="l2", trim=0.15, date_mode="zero")
    cfg.update(kw)
    return V.Spec(name, features, **cfg)


SPECS = [
    # --- baselines ---------------------------------------------------------
    V.Spec("A naive: median $/mile by equipment x distance", F.BASE, kind="naive"),
    V.Spec("B ridge log-linear (no geography)", F.BASE, kind="ridge"),
    # --- first attempt: single GBM, raw target, market signals -------------
    V.Spec("C GBM L2, raw target, + market signals", F.BASE + F.MARKET_DATE, objective="l2"),
    V.Spec("D GBM Huber + trim, + coordinates", STRUCT, objective="huber", trim=0.25),
    V.Spec("E  = D + day-of-year (time proxy)", STRUCT + F.CALENDAR_TREND, objective="huber", trim=0.25),
    # --- two-stage family ---------------------------------------------------
    two_stage("F two-stage, no outlier trim", trim=None),
    two_stage("G two-stage, no coordinates", F.BASE),
    two_stage("H two-stage + city ids", STRUCT + F.CITY_IDS),
    two_stage("I two-stage + quote_signal (row)", STRUCT + ["qs_row"]),
    two_stage("J two-stage + market_index (row)", STRUCT + ["mi_row"]),
    two_stage("K two-stage, Huber loss", objective="huber"),
    two_stage("L two-stage, forecast = last-28-day level", date_mode="carry28"),
    two_stage("M two-stage, forecast = market_index anchored", date_mode="mi_anchored"),
    two_stage("N two-stage FINAL (L2, trim 0.15, long-run level, 5 seeds)", n_seeds=5),
]


def run(specs=SPECS, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val, _ = F.load_raw()
    signals = F.date_signals(train, val)
    cities = F.city_table(train, val)
    fill = F.fit_fill_values(train)
    X = F.build_features(train, signals, fill, cities)
    y = F.target(train)
    actual = train[F.TARGET].to_numpy()

    schemes = {
        "time": list(V.time_folds(train["date"])),
        "time2": list(V.time_folds_long(train["date"])),
        "city": list(V.city_folds(train["pickup"], train["delivery"])),
    }
    cache_dir = F.ROOT / "outputs" / "cv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        t0 = time.time()
        cache = cache_dir / (re.sub(r"[^A-Za-z0-9]+", "_", spec.name).strip("_") + ".csv")
        if cache.exists():                       # resumable: reuse finished specs
            rows.extend(pd.read_csv(cache).to_dict("records"))
            continue
        spec_rows = []
        for scheme, folds in schemes.items():
            for fold, tr_idx, te_idx in folds:
                pred = V.fit_predict(spec, X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[te_idx],
                                     train["date"].iloc[tr_idx], train["date"].iloc[te_idx], signals)
                m = V.metrics(actual[te_idx], pred)
                spec_rows.append({"model": spec.name, "scheme": scheme, "fold": fold, "n_test": len(te_idx), **m})
        pd.DataFrame(spec_rows).to_csv(cache, index=False)
        rows.extend(spec_rows)
        if verbose:
            print(f"  done {spec.name!r} in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    by_fold = pd.DataFrame(rows)
    metric_cols = ["MAE", "RMSE", "MAPE_%", "MdAPE_%", "coreMAPE_%"]
    summary = (by_fold.groupby(["model", "scheme"], sort=False)[metric_cols].mean()
               .unstack("scheme").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False))
    return summary, by_fold


if __name__ == "__main__":
    F.ROOT.joinpath("outputs").mkdir(exist_ok=True)
    summary, by_fold = run()
    by_fold.to_csv(F.ROOT / "outputs" / "cv_by_fold.csv", index=False)
    summary.to_csv(F.ROOT / "outputs" / "cv_results.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summary.round(3).to_string())
