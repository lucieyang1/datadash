# SSU Data Dash: predict Toronto home prices

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
Experiment names must be unique; validation and ensemble runs will stop rather
than overwrite an existing experiment. Omit `--experiment NAME` to create a
timestamped experiment;
`python run.py report` then reports the most recently validated experiment.

On Windows, use `py` if `python` is not found.

## What to edit

- **`features.py`**: the features.
- **`models.py`**: set `MODEL` to `"baseline"`, `"feature"`, `"spatiotemporal_gam"`, `"xgboost"`, `"xgboost_rotated"`, `"xgboost_community_comps"`, `"lightgbm_community_comps"`, `"lightgbm_nearby_comps_safe"`, or `"convlstm_spatial"` to choose a model.
  The GAM-style model mirrors the supplied `mgcv::bam` formula with spline bases and Ridge
  regularization; it approximates, but does not reproduce, `bam`'s REML smoothness selection.
  XGBoost uses the same property, location, and time inputs and adds list price on the
  list-aware track. The distance-based variants use same-type comparable sales within
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
