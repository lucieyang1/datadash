"""Tune and blend six tabular regressors using chronological data splits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor

import data
import features

MODEL_NAMES = ("lasso", "ridge", "svr", "gbm", "lgbm", "xgboost")
CATEGORICAL_COLUMNS = (
    "house_type_name",
    "house_style",
    "municipality",
    "community",
)
NUMERIC_COLUMNS = (
    "bedrooms",
    "washrooms",
    "parking_total",
    "garage",
    "house_area",
    "lot_front",
    "lot_depth",
    "longitude",
    "latitude",
    "months",
    "month_sin",
    "month_cos",
)
TRIALS = {
    "lasso": [{"alpha": alpha} for alpha in (0.001, 0.01, 0.1)],
    "ridge": [{"alpha": alpha} for alpha in (1.0, 10.0, 100.0)],
    "svr": [
        {"C": c, "epsilon": epsilon}
        for c in (0.5, 2.0)
        for epsilon in (0.05, 0.1)
    ],
    "gbm": [
        {"n_estimators": trees, "learning_rate": rate, "max_depth": depth}
        for trees, rate, depth in ((100, 0.05, 2), (200, 0.05, 2), (150, 0.05, 3))
    ],
    "lgbm": [
        {"n_estimators": trees, "learning_rate": rate, "num_leaves": leaves}
        for trees, rate, leaves in ((300, 0.05, 15), (500, 0.03, 31))
    ],
    "xgboost": [
        {"n_estimators": trees, "learning_rate": rate, "max_depth": depth}
        for trees, rate, depth in ((300, 0.05, 6), (500, 0.03, 8))
    ],
}
FIT_LIMITS = {"svr": 5_000, "gbm": 30_000}


def make_features(homes, track):
    """Build shared, answer-safe features for both prediction tracks."""
    numeric = pd.DataFrame(index=homes.index)
    for column in NUMERIC_COLUMNS:
        if column == "house_area":
            numeric[column] = features.square_feet(homes[column])
        elif column in homes:
            numeric[column] = pd.to_numeric(homes[column], errors="coerce")

    if track == "list_aware":
        numeric["list_price"] = pd.to_numeric(homes["list_price"], errors="coerce")
    listed = pd.to_datetime(homes["date_listed"], errors="coerce")
    numeric["months"] = (listed - pd.Timestamp("2021-01-01")).dt.days / 30.44
    month_angle = 2 * np.pi * (listed.dt.month - 1) / 12
    numeric["month_sin"] = np.sin(month_angle)
    numeric["month_cos"] = np.cos(month_angle)
    categories = homes[list(CATEGORICAL_COLUMNS)].fillna("(missing)").astype(str)
    return pd.concat([numeric, categories], axis=1)


def _preprocessor(model_name, track):
    numeric_columns = list(NUMERIC_COLUMNS)
    if track == "list_aware":
        numeric_columns.append("list_price")
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    if model_name == "gbm":
        category_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="(missing)")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ])
        return ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", category_pipeline, list(CATEGORICAL_COLUMNS)),
        ])

    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="(missing)")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", category_pipeline, list(CATEGORICAL_COLUMNS)),
    ])


def make_pipeline(model_name, track, params):
    """Create one of the six requested regressors and its preprocessing."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}.")
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"Unknown track {track!r}; choose 'off_market' or 'list_aware'.")

    if model_name == "lasso":
        estimator = Lasso(**params, max_iter=5000, random_state=0)
    elif model_name == "ridge":
        estimator = Ridge(**params, solver="lsqr")
    elif model_name == "svr":
        estimator = SVR(**params, kernel="rbf")
    elif model_name == "gbm":
        estimator = GradientBoostingRegressor(**params, random_state=0)
    elif model_name == "lgbm":
        estimator = LGBMRegressor(
            **params, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=4, random_state=0, verbosity=-1,
        )
    else:
        estimator = XGBRegressor(
            **params, objective="reg:squarederror", eval_metric="rmse",
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", n_jobs=4, random_state=0,
        )

    return Pipeline([
        ("features", _preprocessor(model_name, track)),
        ("regressor", estimator),
    ])


def _sample_training_rows(frame, model_name):
    limit = FIT_LIMITS.get(model_name)
    if limit is None or len(frame) <= limit:
        return frame
    positions = np.linspace(0, len(frame) - 1, num=limit, dtype=int)
    return frame.iloc[positions]


def _predict_prices(model, model_name, homes, track):
    transformed = model.named_steps["features"].transform(make_features(homes, track))
    if model_name == "lgbm":
        prediction = model.named_steps["regressor"].booster_.predict(transformed)
    else:
        prediction = model.named_steps["regressor"].predict(transformed)
    return np.exp(np.clip(prediction, np.log(10_000), np.log(100_000_000)))


def _fit_predict(model_name, params, track, train, homes):
    fit_rows = _sample_training_rows(train, model_name)
    model = make_pipeline(model_name, track, params)
    model.fit(
        make_features(fit_rows, track),
        np.log(pd.to_numeric(fit_rows["sold_price"], errors="coerce").to_numpy()),
    )
    return _predict_prices(model, model_name, homes, track)


def _mape(actual, predicted):
    return float(np.mean(np.abs(predicted - actual) / actual) * 100)


def optimize_weights(predictions, actual):
    """Find nonnegative blend weights summing to one that minimize MAPE."""
    values = np.asarray(predictions, dtype=float)
    truth = np.asarray(actual, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(truth) or values.shape[1] == 0:
        raise ValueError("Predictions must be a non-empty 2D array aligned with actual values.")
    if not np.isfinite(values).all() or not np.isfinite(truth).all() or (truth <= 0).any():
        raise ValueError("Predictions and actual values must be finite; actual values must be positive.")

    result = minimize(
        lambda weights: _mape(truth, values @ weights),
        np.full(values.shape[1], 1.0 / values.shape[1]),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * values.shape[1],
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Ensemble weight optimization failed: {result.message}")
    return result.x


def _monthly_splits(listings):
    months = np.sort(listings["date_sold"].dt.to_period("M").dropna().unique())
    if len(months) < 10:
        raise ValueError("At least 10 distinct sale months are required for chronological tuning.")
    tuning_start = months[int(len(months) * 0.70)].start_time
    blend_start = months[int(len(months) * 0.85)].start_time
    return tuning_start, blend_start


def run(experiment, root):
    """Tune model configurations, optimize a blend, and save predictions/results."""
    output_dir = Path(root) / "outputs" / "experiments" / experiment
    if output_dir.exists():
        raise FileExistsError(f"Experiment '{experiment}' already exists; choose a new name.")

    listings = data.load("listings").sort_values("date_sold").reset_index(drop=True)
    tuning_start, blend_start = _monthly_splits(listings)
    tuning_train = listings[listings["date_sold"] < tuning_start]
    tuning_homes = listings[
        (listings["date_sold"] >= tuning_start) & (listings["date_sold"] < blend_start)
    ]
    blend_train = listings[listings["date_sold"] < blend_start]
    blend_homes = listings[listings["date_sold"] >= blend_start]
    results = {}
    prediction_frames = []

    for track in ("off_market", "list_aware"):
        tuning_answers = tuning_homes["sold_price"].to_numpy(dtype=float)
        blend_answers = blend_homes["sold_price"].to_numpy(dtype=float)
        best_params = {}
        tuning_scores = {}

        for model_name in MODEL_NAMES:
            candidates = []
            for params in TRIALS[model_name]:
                predicted = _fit_predict(
                    model_name, params, track, tuning_train,
                    data.hide_answers(tuning_homes, track),
                )
                candidates.append((_mape(tuning_answers, predicted), params))
            score, params = min(candidates, key=lambda item: item[0])
            best_params[model_name] = params
            tuning_scores[model_name] = round(score, 4)
            print(f"{track} {model_name}: tuning MAPE {score:.3f}% ({params})")

        blend_predictions = {}
        for model_name in MODEL_NAMES:
            blend_predictions[model_name] = _fit_predict(
                model_name, best_params[model_name], track, blend_train,
                data.hide_answers(blend_homes, track),
            )
        prediction_matrix = np.column_stack(
            [blend_predictions[name] for name in MODEL_NAMES]
        )
        weights = optimize_weights(prediction_matrix, blend_answers)
        ensemble_predictions = prediction_matrix @ weights

        results[track] = {
            "tuning_period": [
                str(tuning_homes["date_sold"].min().date()),
                str(tuning_homes["date_sold"].max().date()),
            ],
            "blend_period": [
                str(blend_homes["date_sold"].min().date()),
                str(blend_homes["date_sold"].max().date()),
            ],
            "tuning_mape": tuning_scores,
            "blend_mape": {
                name: round(_mape(blend_answers, blend_predictions[name]), 4)
                for name in MODEL_NAMES
            } | {"ensemble": round(_mape(blend_answers, ensemble_predictions), 4)},
            "best_params": best_params,
            "ensemble_weights": {
                name: round(float(weight), 8)
                for name, weight in zip(MODEL_NAMES, weights)
            },
        }
        prediction_frame = pd.DataFrame({
            "ml_num": blend_homes["ml_num"].to_numpy(),
            "track": track,
            "sold_price": blend_answers,
            **{f"{name}_prediction": blend_predictions[name] for name in MODEL_NAMES},
            "ensemble_prediction": ensemble_predictions,
        })
        prediction_frames.append(prediction_frame)
        print(f"{track} ensemble MAPE: {_mape(blend_answers, ensemble_predictions):.3f}%")

    output_dir.mkdir(parents=True, exist_ok=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    (output_dir / "tuning_results.json").write_text(json.dumps({
        "models": list(MODEL_NAMES),
        "objective": "mean absolute percentage error",
        "split_method": "chronological 70% training, 15% tuning, 15% blend validation",
        "tracks": results,
    }, indent=2))
    print(f"Wrote {output_dir.relative_to(root)}")
