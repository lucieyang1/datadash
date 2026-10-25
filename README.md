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
python run.py predict    # write both submission files
python run.py check      # check them
```

Each validation experiment is saved under
`outputs/experiments/<name>/` with its own predictions, report, and figure.
Experiment names must be unique; validation will stop rather than overwrite an
existing experiment. Omit `--experiment NAME` to create a timestamped experiment;
`python run.py report` then reports the most recently validated experiment.

On Windows, use `py` if `python` is not found.

## What to edit

- **`features.py`**: the features.
- **`models.py`**: set `MODEL` to `"baseline"`, `"feature"`, `"spatiotemporal_gam"`, or `"xgboost"` to choose a model.
  The GAM-style model mirrors the supplied `mgcv::bam` formula with spline bases and Ridge
  regularization; it approximates, but does not reproduce, `bam`'s REML smoothness selection.
  XGBoost uses the same property, location, and time inputs and adds list price on the
  list-aware track. Its remarks features include binary presence flags for the
  strongest generic words from `analyze_words.py`; location names are omitted because
  the model already has community and municipality categories.

To compare XGBoost, set `MODEL = "xgboost"` and run
`python run.py validate --experiment xgboost-v1`, then
`python run.py report --experiment xgboost-v1`.

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
