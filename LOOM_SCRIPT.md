# Loom script (target ~2:45)

Record your screen with the repo and `report/report.pdf` open. Rough timing below —
practice once; you don't need to hit these to the second.

---

## 0:00–0:30 — Key findings (talk over report.pdf page 1–2)

> "The prediction set is Nov–Dec 2025 — strictly after all the training data, which
> runs Jan–Oct. So this is a real forecasting problem, not just an interpolation
> problem, and I validated it that way instead of using a random split.
> Two data issues mattered most: about 1.4% of rows have a rate that's wildly
> inconsistent with everything else about the load — I treat those as label noise,
> not signal. And 8 cities in the prediction set never show up in training at all,
> so the model has to generalize from coordinates, not just memorize lanes."

## 0:30–1:05 — Data-quality fixes (page 2–4, figs 1–4)

> "Weight had sign errors — some negative values — so I use absolute value plus a
> flag. It's also capped at exactly 47,500 pounds for over a thousand rows, which
> looks like a hard ceiling in the source system, so that's flagged too. Market
> index and weight both have a modest amount of missing data, spread evenly across
> months, so I impute from training medians and date-matched averages. And the
> supplied coordinates are geographically distorted — some cities land in the
> ocean — but they're consistent per city, so I still use them as features; I just
> don't trust the haversine distance computed from them."

## 1:05–1:40 — Model and why (page 5, fig 5)

> "The model is a two-stage LightGBM setup. Stage one pulls out a daily market
> effect — basically, how much rates are running above or below normal on a given
> day — using a robust median-residual estimate so the handful of wild outliers
> don't contaminate it. Stage two fits the structural relationship — distance,
> equipment, weight, geography, day of week — on the de-trended target, with the
> wildest training rows excluded from that second fit.
> The biggest single jump in accuracy came from trimming those outliers and adding
> coordinates — that took median error from about 3.2% down to 1.8%. The two-stage
> structure and a 5-seed ensemble shave off a bit more on top of that."

## 1:40–2:05 — Validation approach (page 2, fig 6)

> "I used two validation schemes. Forward-chaining time CV — train on everything
> before a month, test on that month, repeated across July through October — which
> mirrors the actual forecast gap. And a city-holdout scheme, where I hide groups
> of cities entirely from training and test on rows that touch them, to mirror the
>12% of prediction rows with unseen cities. The final model is the one that did
> well on both, not just whichever looked best on one."

## 2:05–2:30 — One honest limitation + the December chart (page 7, fig 7)

> "One thing I want to be upfront about: for forecasting more than a few weeks out,
> I tested carrying forward recent market levels and anchoring to market_index, but
> both were less accurate than just using the long-run average — the daily effect
> is basically mean-reverting at this horizon. So that's what the model falls back
> to for the December chart and the forward part of the predictions. If the real
> Nov–Dec market moves a lot from the training-period average, that's the main risk."

## 2:30–2:45 — Code walkthrough (switch to editor: src/features.py, src/validation.py)

> "Quickly on the code — features.py has all the cleaning and feature engineering,
> shared identically across training, validation and the December inputs. validation.py
> has the two CV schemes and the TwoStage model class. train.py fits the final model
> on all 48k rows, predict.py scores both prediction files, and everything's
> reproducible with the commands in the README."

---

**Don't forget to say out loud, once, near the start or end:** your name, and that
this is the Freight Rate Prediction assessment — makes the video easy to file.
