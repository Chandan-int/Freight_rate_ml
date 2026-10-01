"""Train the final model on ALL of train_test.csv and save it + the fitted
preprocessing artifacts (city table, date signals, fill values) to outputs/model/.

This is the model selected in src/experiments.py: spec "O" - a two-stage
two-stage model (see src/validation.py:TwoStage) using distance, equipment,
weight (+flags), day-of-week, city coordinates and quote_signal, LightGBM with
an L2 objective on the de-trended target, training rows with |residual| > 0.15
excluded from the second stage, and a 5-seed ensemble. Date effects for
prediction dates outside the training range fall back to the long-run
("zero"-mode -> the structure model's own intercept) daily-effect level; see
TwoStage.forecast_D.

Usage:  python -m src.train
"""
from __future__ import annotations

import json
import pickle

from src import features as F
from src import validation as V

FINAL_SPEC = V.Spec(
    "O two-stage FINAL + quote_signal row, 5 seeds",
    F.BASE + F.COORDS + ["qs_row"],
    kind="twostage", objective="l2", trim=0.15, date_mode="zero", n_seeds=5,
)


def main() -> None:
    train, val, dec = F.load_raw()
    signals = F.date_signals(train, val, dec)   # date-level averages: label-free, pool every date we'll ever predict
    cities = F.city_table(train, val)            # December rows have no coordinates -> looked up from this table
    fill = F.fit_fill_values(train)

    X = F.build_features(train, signals, fill, cities)
    y = F.target(train)

    model = V.TwoStage(FINAL_SPEC).fit(X, y, train["date"], signals)

    out = F.ROOT / "outputs" / "model"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "two_stage.pkl", "wb") as fh:
        pickle.dump({"model": model, "signals": signals, "cities": cities, "fill": fill, "spec": FINAL_SPEC}, fh)

    report = {
        "n_train_rows": len(train),
        "daily_effect_std": float(model.daily_.std()),
        "daily_effect_mean": float(model.daily_.mean()),
        "long_run_level_used_for_nov_dec": float(model.daily_.iloc[-28:].mean()),
        "mi_slope": model.slope_, "mi_anchor": model.anchor_,
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
