# SSU Data Dash: predict Toronto home prices

## Results at a glance

Walk-forward results use December 2025-May 2026, with each month predicted
using only earlier sales. Lower MdAPE is better; MAPE breaks ties. The current
default uses community-comparable XGBoost, with a list-price-ratio target on
the list-aware track.

| Model / experiment | Off-market MdAPE | Off-market MAPE | List-aware MdAPE | List-aware MAPE | Notes |
|---|---:|---:|---:|---:|---|
| LightGBM, safe nearby comps | 9.26% | 12.22% | 4.42% | 5.54% | [report](outputs/experiments/lightgbm-nearby-comps-safe-v1/reports/validation_report.md) |
| XGBoost + ARIMA time trend | 9.29% | 12.01% | 4.34% | 5.51% | Tied after rounding; [report](outputs/experiments/xgboost-arima-v1/reports/validation_report.md) |
| XGBoost, community comps + list-price ratio | 9.31% | 12.01% | 4.27% | 5.34% | Current default; [report](outputs/experiments/xgboost-list-ratio-v1/reports/validation_report.md) |
| XGBoost, community comps (sale-price target) | 9.31% | 12.01% | 4.37% | 5.51% | [report](outputs/experiments/xgboost-community-comps-v1/reports/validation_report.md) |
| XGBoost, Optuna tuned | 9.31% | 12.01% | 4.37% | 5.51% | [report](outputs/experiments/xgb-optuna-v1/reports/validation_report.md) |
| XGBoost + 262k hashed text features (Ridge) | 9.34% | 12.03% | 4.83% | 6.01% | Text branch did not improve overall; [report](outputs/experiments/xgboost-hashed-text-v2/reports/validation_report.md) |
| XGBoost, nearby comps | 9.43% | 12.51% | 4.43% | 5.51% | [report](outputs/experiments/xgboost-nearby-comps/reports/validation_report.md) |
| XGBoost, rotated coordinates | 9.48% | 12.55% | 4.40% | 5.54% | [report](outputs/experiments/xgboost-rotated-v1/reports/validation_report.md) |
| XGBoost, safe nearby comps | 9.53% | 12.59% | 4.42% | 5.53% | [report](outputs/experiments/xgboost-nearby-comps-safe-v2/reports/validation_report.md) |
| LightGBM, community comps | 9.66% | 12.32% | 4.32% | 5.47% | [report](outputs/experiments/lightgbm-community-comps-v1/reports/validation_report.md) |
| XGBoost + Matérn GP (all-data residual field) | 10.13% | 13.44% | 6.42% | 8.02% | Worse than default; [report](outputs/experiments/xgboost-matern-grf-v2/reports/validation_report.md) |
| XGBoost + Matérn GP (held-out residuals) | 10.44% | 13.29% | 4.46% | 5.64% | Worse than default; [report](outputs/experiments/xgboost-matern-grf-v1/reports/validation_report.md) |
| XGBoost, word features | 10.84% | 13.73% | 4.48% | 5.56% | [report](outputs/experiments/xgboost-wordfeatures/reports/validation_report.md) |
| XGBoost, basic | 10.84% | 13.77% | 4.47% | 5.59% | [report](outputs/experiments/xgboost/reports/validation_report.md) |
| Spatiotemporal GAM | 12.88% | 16.45% | 12.88% | 16.45% | [report](outputs/experiments/spatiotemporal-gam/reports/validation_report.md) |
| Feature model | 14.87% | 19.23% | 4.61% | 5.82% | [report](outputs/experiments/feature-model/reports/validation_report.md) |
| ConvLSTM, full data | 15.12% | 18.11% | 12.87% | 16.16% | [report](outputs/experiments/convlstm-spatial-full-v1/reports/validation_report.md) |
| ConvLSTM, 60k rows | 19.61% | 25.24% | 18.62% | 26.32% | [report](outputs/experiments/convlstm-spatial-60k-v1/reports/validation_report.md) |
| ConvLSTM, initial run | 43.50% | 70.56% | 36.45% | 125.35% | [report](outputs/experiments/convlstm-spatial-v1/reports/validation_report.md) |
| XGBoost + hashed text (initial SGD) | 81.14% | 158.65% | 64.74% | 72.12% | Unstable first prototype; [report](outputs/experiments/xgboost-hashed-text-v1/reports/validation_report.md) |

Exploratory six-model ensemble scores below use the Aug 2025-May 2026 blend
window, which was also used to choose blend weights; they are not unbiased
walk-forward comparisons.

| Ensemble experiment | Off-market MdAPE | Off-market MAPE | List-aware MdAPE | List-aware MAPE |
|---|---:|---:|---:|---:|
| TF-IDF remarks | 9.27% | 11.88% | 4.34% | 5.39% |
| Engineered features | 9.51% | 12.47% | 4.36% | 5.43% |
| Comparable features | 9.61% | 12.52% | 4.46% | 5.53% |
| Market trends | 9.70% | 12.49% | 4.44% | 5.49% |

Start with the case handout, [`handout.md`](handout.md), and the judging rubric, [`rubric.md`](rubric.md).

Two tracks:
- **Off-market:** value a home with no list price.
- **List-aware:** predict a listed home's sale price, knowing its list price.

Train on 150,000 sales (January 2021 to May 2026). Predict homes that sold June to September 2026.

**Data:** [huggingface.co/datasets/Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash).
All of it is synthetic.

## Setup

Python 3.10 or newer (3.11+ on macOS 27):

```
pip install -r requirements.txt
```

The data downloads itself on first use. No Python? Use the
[Colab notebook](https://colab.research.google.com/github/erwinzhang7/SSU-Data-Dash/blob/main/notebooks/quickstart.ipynb).
Prefer R? See `r/baseline.R`.

## Commands

```
python run.py validate --experiment baseline-v1   # test on Dec 2025 to May 2026
python run.py report --experiment baseline-v1     # results table and a figure
python run.py ensemble --experiment six-model-v1  # tune six regressors and blend them
python run.py predict    # write both submission files
python run.py check      # check them
```

Each validation experiment is saved under
`outputs/experiments/<name>/` with its own predictions, report, and figure.
The `ensemble` command is a separate experiment: it tunes Lasso, Ridge, SVR,
Gradient Boosting, LightGBM, and XGBoost on a chronological tuning window, then
optimizes nonnegative blend weights on a later window. It writes each model's
predictions, the ensemble predictions, selected parameters, weights, and MAPE
scores to `outputs/experiments/<name>/`. The blend-window scores are used to
choose its weights, so use the existing walk-forward `validate` workflow for
an unbiased comparison before changing the default model.
Both comparable feature sets are built from each fold's training sales only:
nearby same-type sales within 3 km and same-community, same-type sales, with
same-month sales excluded.
The comparable-feature run is saved in
`outputs/experiments/six-model-comps-v1/`. On its Aug 2025-May 2026 blend
window, MAPE was 12.52% off-market and 5.53% list-aware. Blend weights were
optimized on that same window, so these scores are exploratory rather than an
unbiased estimate of future performance.
The follow-up `six-model-features-v1` adds rectangular lot area (frontage x
depth), lot perimeter and aspect ratio, floor area relative to lot area,
log-scaled home/lot area, room and parking ratios, rotated coordinates, and
price-per-square-foot/list-to-comparable features. On the same blend window,
MAPE was 12.47% off-market and 5.43% list-aware (versus 12.52% and 5.53%
before these additions). Blend weights were fit on this window, so use
walk-forward validation before concluding the improvement generalizes.
The `six-model-tfidf-v1` experiment also vectorizes remarks with up to 1,200
training-fold TF-IDF unigram/bigram terms; Gradient Boosting uses a
16-component SVD reduction of those text features. The text vocabulary is
learned only from each model's training rows. To keep the RBF SVR runtime
manageable, it fits on a deterministic 1,200-row training sample. On the same
blend window, this run had 11.88% off-market and 5.39% list-aware MAPE versus
12.47% and 5.43% without TF-IDF. These blend-window results are exploratory
because the blending weights were selected on that period; use walk-forward
validation to assess whether the change generalizes.
The `six-model-trends-v1` run adds exponential recency-weighted price-per-square-foot
features over the prior 3 months, a 3-to-12-month comparison window, their
relative trend, and recent sale counts, grouped by community/property type and
property type overall. All windows end at the start of the listing month. On
the same blend window, adding these features changed MAPE from 11.88% to 12.49%
off-market and 5.39% to 5.49% list-aware. They did not improve this experiment's
blend score, so treat them as an experimental candidate rather than a default.
Experiment names must be unique; validation and ensemble runs will stop rather
than overwrite an existing experiment. Omit `--experiment NAME` to create a
timestamped experiment;
`python run.py report` then reports the most recently validated experiment.

On Windows, use `py` if `python` is not found.

## What to edit

- **`features.py`**: the features.
- **`models.py`**: set `MODEL` to `"baseline"`, `"feature"`, `"spatiotemporal_gam"`, `"xgboost"`, `"xgboost_rotated"`, `"xgboost_community_comps"`, `"xgboost_matern_grf"`, `"xgboost_arima"`, `"xgboost_hashed_text"`, `"lightgbm_community_comps"`, `"lightgbm_nearby_comps_safe"`, or `"convlstm_spatial"` to choose a model.
  The GAM-style model mirrors the supplied `mgcv::bam` formula with spline bases and Ridge
  regularization; it approximates, but does not reproduce, `bam`'s REML smoothness selection.
  XGBoost uses the same property, location, and time inputs. The default list-aware
  model learns the sale-to-list-price ratio, then scales that prediction by list price;
  off-market remains on the original sale-price target. The distance-based variants use same-type comparable sales within
  3 km; the community variant instead uses sales of the same type in the same community.
  Both calculate median price, median price per square foot, and count from sales before
  the home's listing month. The distance-based variant also includes nearest distance.
  The exploratory word associations from `analyze_words.py` are not currently used as
  model features.

To compare XGBoost, set `MODEL = "xgboost"` and run
`python run.py validate --experiment xgboost-v1`, then
`python run.py report --experiment xgboost-v1`.
The experimental `"convlstm_spatial"` model represents the 12 months before each
listing as 1 km spatial grids of same-property-type sale counts, median prices, and
median price per square foot. It processes each home's 7-by-7 km neighborhood with
a ConvLSTM and combines that representation with the structured property features.
Every frame excludes its own and future listing months. Set
`MODEL = "convlstm_spatial"` and validate it with a new experiment name. It is a prototype
and may be slower or less accurate than the tree models. To keep CPU training manageable,
each fold trains for two epochs on all earlier sales.
To try LightGBM with the same community-comparable and rotated-coordinate features,
set `MODEL = "lightgbm_community_comps"` and run validation/report with a new
experiment name. This keeps the XGBoost community-comps model as the default until
the LightGBM validation results have been compared. In the Dec 2025-May 2026
walk-forward validation, LightGBM scored 12.32% off-market MAPE and 5.47%
list-aware MAPE, compared with 12.01% and 5.51% for XGBoost. The results are in
`outputs/experiments/lightgbm-community-comps-v1/`.
To try the safe nearby-comps version instead, set
`MODEL = "lightgbm_nearby_comps_safe"`. It uses same-type sales within 3 km,
strictly before the listing month, and includes the rotated coordinate features.
On the Dec 2025-May 2026 walk-forward validation it scored 12.22% off-market
MAPE and 5.54% list-aware MAPE, compared with 12.59% and 5.53% for the existing
XGBoost safe nearby-comps experiment. Results are in
`outputs/experiments/lightgbm-nearby-comps-safe-v1/`.
The default `"xgboost_community_comps"` model keeps rotated longitude and latitude inputs
but replaces distance-matched comparable sales with same-community, same-property-type
sales; only sales before the listing month are used. On this project's Dec 2025-May 2026
walk-forward validation, it improved MAPE from 12.55% to 12.01% off-market and
from 5.54% to 5.51% list-aware compared with `"xgboost_rotated"`. Its results are in
`outputs/experiments/xgboost-community-comps-v1/`.
To compare the rotated distance-based version, set `MODEL = "xgboost_rotated"` and
validate/report it under a new experiment name.
Both models keep the original longitude and latitude inputs and add local coordinates
rotated 45 degrees (with longitude scaled for Toronto's latitude). This can help
axis-aligned trees learn diagonal geographic boundaries; it is not guaranteed to
improve every dataset. On this project's Dec 2025-May 2026
walk-forward validation, it improved MAPE from 13.77% to 12.55% off-market and
from 5.59% to 5.54% list-aware relative to the non-rotated XGBoost variant. Its results are in
`outputs/experiments/xgboost-rotated-v1/`.

The experimental `"xgboost_matern_grf"` model uses the community-comparable
XGBoost model for property, time, community, and location features, trained on
the entire training fold. It combines those predictions with a Matérn 3/2
Gaussian spatial correction. To keep the GP tractable, all training sales
contribute to residual means in 1 km location cells; those residuals are
relative to the property-type and sale-month median log price. The correction
is added at half strength. Set `MODEL` to this name and run walk-forward
validation with a new experiment name before comparing it with the current
default.

The experimental `"xgboost_arima"` model keeps the community-comparable
XGBoost model for property, location, and comparable-sale features. It adds a
damped ARIMA(1,1,0) forecast of log median sale price per square foot, grouped
by property type, using only sale months before each listing month. The ARIMA
adjustment is fit independently from the booster and added at quarter strength.
The default remains `"xgboost_community_comps"`; compare with walk-forward
validation before switching. On the Dec 2025-May 2026 walk-forward, this
variant scored 12.01% off-market and 5.51% list-aware MAPE, effectively tying
the default at those rounded scores; the unrounded scores moved slightly in
opposite directions. Results are in
`outputs/experiments/xgboost-arima-v1/`.

The experimental `"xgboost_hashed_text"` model blends the community-comparable
XGBoost predictions with a sparse Ridge model using 262,144 hashed remark
unigram/bigram features plus the structured features. On the Dec 2025-May 2026
walk-forward it scored 12.03% off-market and 6.01% list-aware MAPE, so it did
not improve on the default. Results are in
`outputs/experiments/xgboost-hashed-text-v2/`.

To explore which words in listing remarks are associated with higher or lower
prices after controlling for the structured property, location, and time features,
run `python analyze_words.py`. It prints the 20 highest and lowest word
associations and saves all results to `outputs/word_associations.csv`. These are
exploratory associations, not causal effects or proof that adding text will
improve validation accuracy.

## Validation

`validate` predicts each month from the sales before it, as in the real test. A random split lets
the model see the future and makes your score look better than it is. Judges compare your
validation error with your hidden-test error.

## Submission

One PDF report (template: `report/report.tex`, 4 pages max) that links to:
1. your video on YouTube, 5 minutes max (unlisted is fine, not private);
2. your GitHub repository (public, or shared with the organisers), holding your code and two files:

```
submissions/submission_off_market.csv    one row per home in test_off_market
submissions/submission_list_aware.csv    one row per home in test_list_aware

ml_num,predicted_price
SYN-0123456,815000
```

`predict` writes both; run `check` before you hand in. Deadline: see the case email.

## Rules

- Outside data only if it covers May 2026 or earlier. List every source with its URL.
- Neighbourhoods, coordinates and streets are synthetic, so map data will not line up.
- No real listing or sales records.
- Do not try to recover the hidden test prices.
