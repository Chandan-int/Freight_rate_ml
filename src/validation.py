"""Validation splits, metrics and model factories.

Two complementary validation schemes are used, because the real test differs from
the training data in two ways:

1. **Time** - validation.csv is Nov-Dec 2025, strictly *after* all training data
   (Jan-Oct). => expanding-window, forward-chaining time CV (never a random split).
2. **New cities** - 8 cities (12% of validation rows) never appear in training.
   => city-holdout CV where the test rows touch held-out cities and NO training row
   does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# --------------------------------------------------------------------------- splits
def time_folds(dates: pd.Series, test_months=(7, 8, 9, 10)) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Forward chaining: train on every month before the test month."""
    m = dates.dt.month.to_numpy()
    for tm in test_months:
        yield f"time_m{tm:02d}", np.where(m < tm)[0], np.where(m == tm)[0]


def time_folds_long(dates: pd.Series) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Two-month horizons (Nov-Dec is a two-month forecast): test Jul-Aug and Sep-Oct."""
    m = dates.dt.month.to_numpy()
    for lo, hi in ((7, 8), (9, 10)):
        yield f"time2_m{lo:02d}-{hi:02d}", np.where(m < lo)[0], np.where((m >= lo) & (m <= hi))[0]


def city_folds(pickup: pd.Series, delivery: pd.Series, n_splits: int = 4, seed: int = 0):
    """Hold out groups of cities. Train rows touch none of the held-out cities."""
    cities = np.array(sorted(set(pickup) | set(delivery)))
    rng = np.random.default_rng(seed)
    rng.shuffle(cities)
    for i, held in enumerate(np.array_split(cities, n_splits)):
        held = set(held)
        touches = pickup.isin(held).to_numpy() | delivery.isin(held).to_numpy()
        yield f"city_g{i}", np.where(~touches)[0], np.where(touches)[0]


# ------------------------------------------------------------------------- metrics
def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    ape = np.abs(actual - pred) / actual
    keep = ape <= np.quantile(ape, 0.985)          # ignore the 1.5% wildest rows
    return {
        "MAE": float(np.mean(np.abs(actual - pred))),
        "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "MAPE_%": float(100 * ape.mean()),
        "MdAPE_%": float(100 * np.median(ape)),
        "coreMAPE_%": float(100 * ape[keep].mean()),
    }


# --------------------------------------------------------------------------- models
LGB_PARAMS = dict(
    n_estimators=700, learning_rate=0.03, num_leaves=15, min_child_samples=60,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=2.0,
    n_jobs=1, verbose=-1, random_state=0,
)


@dataclass
class Spec:
    """A model configuration: which features, which loss, and how to treat outliers."""
    name: str
    features: list[str]
    kind: str = "lgbm"                  # lgbm | ridge | naive | twostage
    objective: str = "l2"               # l2 | l1 | huber (on log target)
    trim: float | None = None           # drop training rows with |log residual| > trim
    params: dict | None = None
    date_mode: str | None = None        # two-stage only: how the daily market effect is forecast
    n_seeds: int = 1                    # two-stage only: average this many LightGBM seeds


def _fit_lgbm(X, y, objective, params):
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    if objective == "huber":
        p.update(objective="huber", alpha=0.15)
    elif objective == "l1":
        p.update(objective="l1")
    else:
        p.update(objective="regression")
    return lgb.LGBMRegressor(**p).fit(X, y)


def fit_predict(spec: Spec, Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame,
                dates_tr: pd.Series | None = None, dates_te: pd.Series | None = None,
                signals: pd.DataFrame | None = None) -> np.ndarray:
    """Fit on the training fold and return predictions in dollars for ``Xte``."""
    cols = spec.features
    if spec.kind == "twostage":
        return TwoStage(spec).fit(Xtr, ytr, dates_tr, signals).predict(Xte, dates_te, signals)
    if spec.kind == "naive":
        return _naive(Xtr, ytr, Xte)

    if spec.kind == "ridge":
        def design(X):
            d = pd.DataFrame(index=X.index)
            d["ld"], d["ld2"] = X["log_dist"], X["log_dist"] ** 2
            for e in (1, 2):
                d[f"e{e}"] = (X["equip"] == e).astype(float)
                d[f"e{e}_ld"] = d[f"e{e}"] * X["log_dist"]
            d["w"] = X["weight"] / 1e4
            for k in range(1, 7):
                d[f"dow{k}"] = (X["dow"] == k).astype(float)
            for c in cols:
                if c in ("mi_date", "qs_date", "mi_row", "qs_row"):
                    d[c] = X[c]
            return d
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        keep = _trim_mask(Xtr, ytr, spec)
        mdl.fit(design(Xtr[keep]), ytr[keep])
        return np.exp(mdl.predict(design(Xte)))

    keep = _trim_mask(Xtr, ytr, spec)
    mdl = _fit_lgbm(Xtr.loc[keep, cols], ytr[keep], spec.objective, spec.params)
    return np.exp(mdl.predict(Xte[cols]))


def _trim_mask(Xtr, ytr, spec: Spec) -> np.ndarray:
    """Optionally drop suspect training targets using a robust first-pass fit."""
    if spec.trim is None:
        return np.ones(len(Xtr), dtype=bool)
    first = _fit_lgbm(Xtr[["log_dist", "equip", "weight"]], ytr, "l1",
                      dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=100))
    resid = ytr.to_numpy() - first.predict(Xtr[["log_dist", "equip", "weight"]])
    return np.abs(resid) <= spec.trim


def _naive(Xtr, ytr, Xte):
    """Median rate-per-mile by equipment x distance bucket, times distance."""
    edges = np.quantile(Xtr["dist"], np.linspace(0, 1, 11)[1:-1])
    tr_key = pd.DataFrame({"e": Xtr["equip"], "b": np.digitize(Xtr["dist"], edges)})
    rpm = np.exp(ytr) / Xtr["dist"]
    table = rpm.groupby([tr_key["e"], tr_key["b"]]).median()
    te_key = pd.DataFrame({"e": Xte["equip"], "b": np.digitize(Xte["dist"], edges)})
    est = pd.Series(list(zip(te_key["e"], te_key["b"])), index=Xte.index).map(table.to_dict())
    return (est.fillna(rpm.median()) * Xte["dist"]).to_numpy()


# ------------------------------------------------------------------ two-stage model
class TwoStage:
    """log(rate) = structure(distance, equipment, weight, geography, dow) + D(date) + noise.

    Stage 1 estimates a robust *daily market effect* D(date) (median residual per day,
    iterated with the structure model) and removes it from the target, so the structure
    model learns lane/equipment/weight/geography effects from a much cleaner signal.
    Stage 2 supplies D for the prediction dates:
      * dates seen in training -> the estimated D (interpolation);
      * future dates           -> a forecast chosen by ``spec.date_mode``.
    """

    def __init__(self, spec: Spec, n_iter: int = 3, outlier_band: float = 0.25):
        self.spec, self.n_iter, self.band = spec, n_iter, outlier_band

    # -- helpers ---------------------------------------------------------------
    def _structure(self, X, y, objective=None):
        return _fit_lgbm(X[self.spec.features], y, objective or self.spec.objective, self.spec.params)

    def fit(self, X, y, dates, signals):
        y = y.to_numpy()
        dates = pd.to_datetime(pd.Series(dates.to_numpy(), index=X.index))
        D = np.zeros(len(X))
        for _ in range(self.n_iter):
            m = _fit_lgbm(X[self.spec.features], pd.Series(y - D, index=X.index), "l1",
                          dict(n_estimators=200, learning_rate=0.08))          # fast pass
            resid = y - m.predict(X[self.spec.features])
            core = np.abs(resid - D) < self.band
            daily = pd.Series(np.where(core, resid, np.nan), index=X.index).groupby(dates.to_numpy()).median()
            D = dates.map(daily).to_numpy()
        self.daily_ = daily.dropna()
        self.daily_.index = pd.to_datetime(self.daily_.index)
        y_adj = pd.Series(y - D, index=X.index)
        keep = _trim_mask(X, y_adj, self.spec)
        self.models_ = []
        for seed in range(self.spec.n_seeds):
            params = dict(self.spec.params or {}, random_state=seed)
            self.models_.append(_fit_lgbm(X.loc[keep, self.spec.features], y_adj[keep], self.spec.objective, params))
        # anchoring statistics for the forecast
        sig = signals["mi_date"].reindex(self.daily_.index)
        d7, m7 = self.daily_.diff(7).dropna(), sig.diff(7).dropna()
        self.slope_ = float((d7 * m7).sum() / (m7 ** 2).sum())
        self.anchor_ = float((self.daily_ - self.slope_ * sig).iloc[-28:].mean())
        self.carry28_ = float(self.daily_.iloc[-28:].mean())
        self.carry56_ = float(self.daily_.iloc[-56:].mean())
        self.mean_ = float(self.daily_.mean())
        return self

    def forecast_D(self, dates, signals) -> np.ndarray:
        dates = pd.to_datetime(pd.Series(dates.to_numpy()))
        mode = self.spec.date_mode or "zero"
        mi = signals["mi_date"].reindex(dates).to_numpy()
        fc = {"zero": np.zeros(len(dates)),
              "carry28": np.full(len(dates), self.carry28_),
              "carry56": np.full(len(dates), self.carry56_),
              "half_carry28": np.full(len(dates), 0.5 * self.carry28_),
              "mi_anchored": self.slope_ * mi + self.anchor_}[mode]
        known = dates.map(self.daily_).to_numpy()            # NaN for unseen (future) dates
        return np.where(np.isnan(known), fc, known)

    def predict(self, X, dates, signals):
        log_pred = np.mean([m.predict(X[self.spec.features]) for m in self.models_], axis=0)
        return np.exp(log_pred + self.forecast_D(dates, signals))
