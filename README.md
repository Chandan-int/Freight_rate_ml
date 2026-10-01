# Freight Rate Prediction — Chandan's Submission

Predicts `posted_rate` for 12,000 Nov–Dec 2025 loads (`data/validation.csv`),
trained on 48,000 labeled Jan–Oct 2025 loads (`data/train_test.csv`).

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.10+; tested on 3.12. All four provided CSVs already sit in `data/`.

## Run everything (reproducible, one seed)

```bash
python -m src.eda            # figures/*.png + outputs/eda_stats.json  (data-quality report)
python -m src.experiments    # outputs/cv_results.csv + cv_by_fold.csv (model comparison, ~9 min on 1 CPU;
                              #   resumable — finished models are cached under outputs/cv_cache/)
python -m src.train           # outputs/model/two_stage.pkl            (fits the FINAL model on all 48k rows)
python -m src.predict         # outputs/validation_predictions.csv + outputs/december_predictions.csv
python score.py --predictions outputs/validation_predictions.csv \
                 --december-predictions outputs/december_predictions.csv \
                 --output-dir outputs/scorer_results
```

The last command is the provided scorer; it validates both files and writes
`outputs/scorer_results/candidate_december.png`.

`validation_predictions.csv` (the file to submit) is also copied to the repo
root for convenience.

## Repository layout

```
data/            the 4 provided CSVs
src/
  features.py    cleaning + feature engineering (shared by train/val/December)
  validation.py  time-based & city-holdout CV splits, metrics, the TwoStage model
  experiments.py model/feature comparison harness (writes outputs/cv_*.csv)
  eda.py         data-quality checks + report figures
  train.py       fits the final model on all of train_test.csv
  predict.py     scores validation.csv and december_chart_inputs.csv
figures/         PNGs used in the report
outputs/         everything the scripts produce (git-ignored except predictions)
report/          report.pdf — approach, EDA, validation results, December chart
score.py         provided scorer (unmodified)
```

## Approach in one paragraph

`validation.csv` is a forecast into the future (Nov–Dec, strictly after all
training data) and includes 8 cities never seen in training, so the model is
selected using forward-chaining time-based cross-validation (not a random
split) and cross-checked with a city-holdout scheme. The target is modeled as
`log(rate) = structure(distance, equipment, weight, geography, day-of-week) +
daily market effect`, fit in two stages with LightGBM: stage 1 robustly
estimates and removes a per-day market effect (median residual, iterated),
stage 2 fits the structural relationship on the cleaner de-trended target,
with wild training outliers (~1.4% of rows, evenly split high/low) trimmed
out. For dates beyond the training range (the December chart and,
implicitly, Nov–Dec of the validation set) the daily effect is not
extrapolated — training shows recent levels and market-index-based forecasts
of that effect are *less* accurate going forward than simply using its
long-run average, so that's what's used. See `report/report.pdf` for full
detail, the data-quality writeup, and the model-comparison table.

## Notes on data quality (see `report/report.pdf` §2 for detail)

- Negative weights (sign errors) — used as `abs(weight)`, with a flag.
- Weight clipped at 47,500 lb — flagged.
- Missing `weight` / `market_index` (both sides) — filled from training
  medians / date-matched averages, with a missing-flag where relevant.
- Distance floored at 70 miles.
- ~1.4% of training rows have a wildly inconsistent rate for their lane —
  excluded from the second training stage.
- Supplied lat/lon are distorted (not true geography) but still work as
  consistent per-city features.
