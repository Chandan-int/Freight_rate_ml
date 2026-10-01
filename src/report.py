"""Build report/report.pdf from the figures, cv results and eda stats.

Usage: python -m src.report
"""
from __future__ import annotations

import json

from PIL import Image as PILImage
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from src import features as F

TEAL = colors.HexColor("#064A56")
GREY = colors.HexColor("#455A60")
LIGHT = colors.HexColor("#EEF3F4")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], textColor=TEAL, spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], textColor=TEAL, fontSize=12.5, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.8, leading=14, spaceAfter=6))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.3, textColor=GREY, leading=11, spaceAfter=10))
styles.add(ParagraphStyle("MyBullet", parent=styles["Body"], leftIndent=14, bulletIndent=4))
styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], textColor=TEAL))

stats = json.loads((F.ROOT / "outputs" / "eda_stats.json").read_text())
q, noise = stats["quality"], stats["target_noise"]
train_report = json.loads((F.ROOT / "outputs" / "model" / "train_report.json").read_text())
cv = pd.read_csv(F.ROOT / "outputs" / "cv_results.csv", header=[0, 1], index_col=0)


def mdape(model, scheme):
    return cv.loc[model, (scheme, "MdAPE_%")]


def mae(model, scheme):
    return cv.loc[model, (scheme, "MAE")]


story = []


def fig(path, width=6.4, caption=None):
    w_px, h_px = PILImage.open(F.ROOT / path).size
    story.append(Image(str(F.ROOT / path), width=width * inch, height=width * inch * (h_px / w_px)))
    if caption:
        story.append(Paragraph(caption, styles["Caption"]))
    story.append(Spacer(1, 4))


def bullets(items):
    for it in items:
        story.append(Paragraph(f"&bull;&nbsp;&nbsp;{it}", styles["MyBullet"]))
    story.append(Spacer(1, 4))


# ----------------------------------------------------------------- Title page
story.append(Spacer(1, 1.6 * inch))
story.append(Paragraph("Freight Rate Prediction — Assessment Report", styles["TitleBig"]))
story.append(Spacer(1, 10))
story.append(Paragraph("Chandan", styles["Normal"]))
story.append(Paragraph("Predicting posted load rates for Nov&ndash;Dec 2025 freight; "
                       "model, validation design and results.", styles["Normal"]))
story.append(Spacer(1, 24))
summary_tbl = Table([
    ["Training data", f"{q['train_rows']:,} loads, {q['date_range_train'][0]} to {q['date_range_train'][1]}"],
    ["Prediction set", f"{q['val_rows']:,} loads, {q['date_range_val'][0]} to {q['date_range_val'][1]}"],
    ["Final model", "Two-stage LightGBM (daily market effect + structural model), 5-seed ensemble"],
    ["Validation design", "Forward-chaining time CV + city-holdout CV (never a random split)"],
    ["Headline forward accuracy", f"~{mdape('N two-stage FINAL (L2, trim 0.15, long-run level, 5 seeds)', 'time'):.2f}% "
                                    "median absolute % error (time CV), vs. "
                                    f"{mdape('A naive: median $/mile by equipment x distance', 'time'):.2f}% for a naive baseline"],
], colWidths=[1.7 * inch, 4.7 * inch])
summary_tbl.setStyle(TableStyle([
    ("FONTSIZE", (0, 0), (-1, -1), 9.3), ("TEXTCOLOR", (0, 0), (0, -1), TEAL),
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LIGHT),
]))
story.append(summary_tbl)
story.append(PageBreak())

# ----------------------------------------------------------------- 1. Problem & split
story.append(Paragraph("1. Problem framing and validation design", styles["H1"]))
story.append(Paragraph(
    "The prediction set (validation.csv) is strictly in the future relative to the training data: "
    f"training covers {q['date_range_train'][0]} through {q['date_range_train'][1]}, while the "
    f"predictions cover {q['date_range_val'][0]} through {q['date_range_val'][1]}. The December "
    "chart inputs ask for a further 31-day forecast beyond even that. A model validated with a "
    "random train/test split would be scored on data drawn from the same months it trained on, "
    "which does not test its ability to forecast forward and tends to look better than it will "
    "perform in practice.", styles["Body"]))
story.append(Paragraph(
    f"Separately, {len(q['new_cities_in_val'])} cities in the prediction set "
    f"({', '.join(q['new_cities_in_val'])}) never appear in training at all, affecting "
    f"{q['val_rows_with_new_city']:,} of the 12,000 rows (12%). A model that only learns lane-specific "
    "effects would fail on these.", styles["Body"]))
story.append(Paragraph("Two complementary validation schemes were used instead of a single random split:", styles["Body"]))
bullets([
    "<b>Forward-chaining time CV</b> &mdash; train on all months before a held-out month, test on that "
    "month, repeated for July, August, September and October. This mirrors the actual Nov&ndash;Dec forecast.",
    "<b>City-holdout CV</b> &mdash; cities are split into 4 groups; for each group, every row touching "
    "those cities is held out and training uses only rows that touch none of them. This mirrors the "
    "12% of prediction rows with unseen cities.",
])
story.append(Paragraph("Every model below is scored on both schemes; the final model was chosen for consistently "
                       "low error on both, not just the better-looking one.", styles["Body"]))

# ----------------------------------------------------------------- 2. Data quality
story.append(Paragraph("2. Data-quality findings", styles["H1"]))
fig("figures/01_rate_vs_distance.png", caption=f"Fig. 1 &mdash; posted_rate is close to linear in distance "
    f"(median ${noise['rate_per_mile_median']:.2f}/mile); equipment shifts the slope slightly. The scattered "
    "points far from the trend line are the target outliers discussed below.")
story.append(Paragraph(
    "The following issues were found in both training and the prediction set and handled identically "
    "in both (the prediction set is never filtered &mdash; all rows must still be predicted):", styles["Body"]))
bullets([
    f"<b>Negative weights</b> ({q['negative_weight_train']} train / {q['negative_weight_val']} prediction rows)"
    " &mdash; clear sign errors. Used as <font face='Courier'>abs(weight)</font>, with a flag feature.",
    f"<b>Weight capped at 47,500 lb</b> ({q['weight_at_cap_train']} train rows sit exactly at this value) "
    "&mdash; looks like a hard ceiling in the source system. Flagged rather than treated as a real value.",
    f"<b>Missing values</b> &mdash; weight missing in {q['missing_weight_train']} train / {q['missing_weight_val']} "
    f"prediction rows; market_index missing in {q['missing_market_index_train']} train / "
    f"{q['missing_market_index_val']} prediction rows. Spread evenly across months (looks random, not "
    "systematic). Imputed from training medians / date-matched averages, each with a missing-flag.",
    f"<b>Distance</b> is reliable &mdash; {q['distance_vs_route_median_within_15pct_%']:.1f}% of rows fall "
    f"within 15% of their lane's median distance &mdash; but floored at 70 miles "
    f"({q['distance_at_floor_70_train']} rows sit exactly at the floor).",
    "<b>Supplied coordinates are distorted</b> (not true geography &mdash; e.g. some cities land in the "
    "ocean) but are consistent per city across both files, so they still work as per-city features; "
    "haversine distance computed from them was <i>not</i> reliable enough to use.",
    f"<b>Target (rate) outliers</b> &mdash; after removing the distance/equipment/weight trend, "
    f"{noise['wild_outlier_share_%']:.1f}% of rows have a residual beyond what the rest of the data would "
    f"predict (roughly evenly split high vs. low: {noise['wild_high_%']:.1f}% high, {noise['wild_low_%']:.1f}% "
    f"low). Core noise is modest (&asymp;{noise['core_sigma_log_(MAD)']*100:.1f}% in log terms) but these "
    "outliers alone roughly triple the apparent noise and dominate RMSE if left in. Training rows "
    "identified as outliers by a first-pass robust fit are excluded before the final structural model is fit "
    "(see &sect;3); no prediction row is ever dropped.",
])
fig("figures/02_target_outliers.png", caption="Fig. 2 &mdash; residual distribution from a simple "
    "distance+equipment+weight model (log scale). The bulk of loads sit in a tight band; a small but "
    "non-negligible tail on both sides is noise rather than signal.")
fig("figures/03_market_index_time.png", caption="Fig. 3 &mdash; daily mean market_index over time. The "
    "prediction period (right of the dashed line) sits entirely after the training window, with a lower "
    "average level than training's mid-year peak.")
fig("figures/04_new_cities.png", width=5.6, caption="Fig. 4 &mdash; the 8 cities that only appear in the "
    "prediction set (red diamonds). Coordinates are as supplied (distorted, not true lat/lon).")
story.append(PageBreak())

# ----------------------------------------------------------------- 3. Model
story.append(Paragraph("3. Model", styles["H1"]))
story.append(Paragraph(
    "The target is modeled as <font face='Courier'>log(rate) = structure(distance, equipment, weight, "
    "geography, day-of-week) + daily market effect</font>, fit in two stages with LightGBM gradient boosting:",
    styles["Body"]))
bullets([
    "<b>Stage 1 &mdash; daily market effect.</b> A robust first-pass model (L1 loss) estimates the "
    "structural relationship; the median residual for each calendar date is taken as that day's market "
    "effect, iterated 3 times so the structure model isn't confused by day-to-day rate swings that have "
    "nothing to do with distance, equipment or weight.",
    "<b>Stage 2 &mdash; structural model.</b> With the daily effect removed, a second LightGBM model "
    "(L2 loss, 5-seed ensemble averaged for stability) fits the remaining lane/equipment/weight/geography "
    "structure. Training rows whose first-pass residual exceeds 0.15 in log terms (the outliers from "
    "&sect;2) are excluded from this stage only.",
    "<b>Forecasting the daily effect forward.</b> For December-chart dates and the forward part of "
    "validation that extends past training, the daily effect can't be read off directly. Three forecasts "
    "were tested: carrying forward the last 28/56 days' average, a market_index-anchored linear forecast, "
    "and simply using the long-run average (effectively zero, since the effect is mean-reverting with "
    f"daily std. only {train_report['daily_effect_std']*100:.1f}% in log terms). The long-run average "
    "won on every forward time-CV fold &mdash; recent levels and market_index are both too noisy to "
    "extrapolate reliably two months out.",
])
story.append(Paragraph("Final features: log(distance), distance-floor flag, equipment, abs(weight) with "
    "negative/capped/missing flags, day-of-week, pickup &amp; delivery coordinates, and quote_signal.",
    styles["Body"]))

story.append(Paragraph("3.1 Model comparison", styles["H2"]))
fig("figures/06_model_comparison.png", caption="Fig. 5 &mdash; median absolute % error on forward "
    "time-CV for each stage of model development. Outlier trimming plus coordinates is the single "
    "biggest jump (grey &rarr; the Huber+trim+coords bar); the two-stage daily-effect model and final "
    "ensembling add a further, smaller improvement (orange/teal).")
cmp_rows = [["Model", "Time CV MdAPE", "Time CV MAE", "City-holdout MdAPE"]]
for label, name in [
    ("A. Naive (median $/mi by equipment & distance bucket)", "A naive: median $/mile by equipment x distance"),
    ("B. Ridge, log-linear", "B ridge log-linear (no geography)"),
    ("C. Single GBM, raw target + market signals", "C GBM L2, raw target, + market signals"),
    ("D. Single GBM, Huber loss + outlier trim + coordinates", "D GBM Huber + trim, + coordinates"),
    ("Two-stage, no coordinates", "G two-stage, no coordinates"),
    ("Two-stage, + coordinates", "H two-stage + city ids"),
    ("FINAL: two-stage + quote_signal, 5-seed ensemble", "N two-stage FINAL (L2, trim 0.15, long-run level, 5 seeds)"),
]:
    cmp_rows.append([label, f"{mdape(name,'time'):.2f}%", f"${mae(name,'time'):.0f}", f"{mdape(name,'city'):.2f}%"])
t = Table(cmp_rows, colWidths=[3.55 * inch, 1.0 * inch, 0.9 * inch, 1.15 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.4), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ("TEXTCOLOR", (0, -1), (-1, -1), TEAL), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5),
]))
story.append(t)
story.append(Spacer(1, 8))
story.append(Paragraph("Notable ablations (not shown above): adding row-level market_index actively hurt "
    "forward accuracy (it doesn't generalize past the training window the way the date-level average + "
    "trend does); a day-of-year feature looked best on a 1-month-ahead fold but was worse 2 months ahead "
    "&mdash; a sign of overfitting to the training period's seasonal shape rather than learning something "
    "that forecasts. Both were left out of the final model.", styles["Body"]))
story.append(Paragraph("3.2 Error by forecast horizon", styles["H2"]))
fig("figures/05_error_by_month.png", caption="Fig. 6 &mdash; error stays low and stable from one month "
    "ahead out to four months ahead for the final model, a reasonable proxy for the real forecast gap "
    "to Nov&ndash;Dec.")
story.append(PageBreak())

# ----------------------------------------------------------------- 4. December chart
story.append(Paragraph("4. December 2025 chart", styles["H1"]))
story.append(Paragraph(
    "The December chart inputs hold pickup, delivery, distance, equipment and weight fixed "
    "(Lexington&rarr;Fort Wayne, 360 mi, Dry Van, 32,000 lb) and vary only the date, with no "
    "market_index, quote_signal or coordinates supplied. These were reconstructed without using any "
    "label information:", styles["Body"]))
bullets([
    "<b>Coordinates</b> &mdash; looked up from the training/validation city table (both cities appear there).",
    "<b>market_index / quote_signal</b> &mdash; not used by the final model for forecast dates (see "
    "&sect;3); the daily effect falls back to its long-run average, exactly as it does for the "
    "forward-looking part of the main predictions.",
])
_dec_w, _dec_h = PILImage.open(F.ROOT / "outputs/scorer_results/candidate_december.png").size
story.append(Image(str(F.ROOT / "outputs/scorer_results/candidate_december.png"), width=6.3 * inch, height=6.3 * inch * (_dec_h / _dec_w)))
story.append(Paragraph("Fig. 7 &mdash; scorer-generated chart from outputs/december_predictions.csv. The "
    "weekly pattern (weekday vs. weekend) comes entirely from the day-of-week feature learned from "
    "training data; the absolute level (~$820&ndash;831, &asymp;$2.30/mile) matches the Dry Van rate-per-mile "
    "for a 360-mile lane seen in training.", styles["Caption"]))

story.append(Paragraph("5. Limitations & risks", styles["H1"]))
bullets([
    "The ~1.4% wild-outlier rate in the target is the practical floor on RMSE; no feature or model choice "
    "removes genuine label noise.",
    "If the real Nov&ndash;Dec market level moves meaningfully from the training-period average (beyond "
    f"the typical &plusmn;{train_report['daily_effect_std']*100:.1f}%/day swings seen within training), the "
    "long-run-average forecast for the daily effect will be systematically off in one direction.",
    "The 8 unseen cities rely entirely on coordinate-based generalization; city-holdout CV shows this "
    "works about as well as predicting known cities, but it's untested against these specific cities.",
])

doc = SimpleDocTemplate(str(F.ROOT / "report" / "report.pdf"), pagesize=letter,
                        topMargin=0.7*inch, bottomMargin=0.7*inch, leftMargin=0.8*inch, rightMargin=0.8*inch,
                        title="Freight Rate Prediction - Assessment Report")
doc.build(story)
print("wrote report/report.pdf")
