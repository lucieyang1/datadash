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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from xgboost import XGBRegressor

import data
import features
from models import CommunityComparableFeatures, NearbyComparableFeatures

MODEL_NAMES = ("lasso", "ridge", "svr", "gbm", "lgbm", "xgboost")
TEXT_COLUMN = "remarks"
TFIDF_MAX_FEATURES = 1200
TFIDF_MIN_DF = 2
TFIDF_N_COMPONENTS = 16
CATEGORICAL_COLUMNS = (
    "house_type_name",
    "house_style",
    "municipality",
    "community",
)
BASE_NUMERIC_COLUMNS = (
    "bedrooms",
    "washrooms",
    "parking_total",
    "garage",
    "house_area",
    "lot_front",
    "lot_depth",
    "longitude",
    "latitude",
)
DERIVED_NUMERIC_COLUMNS = (
    "months",
    "month_sin",
    "month_cos",
    "rotated_longitude",
    "rotated_latitude",
    "log_house_area",
    "lot_area_sqft",
    "log_lot_area_sqft",
    "lot_perimeter_ft",
    "lot_aspect_ratio",
    "floor_area_to_lot_area",
    "bedrooms_per_1000_sqft",
    "washrooms_per_bedroom",
    "parking_per_bedroom",
    "total_bedrooms_washrooms",
    "nearby_comparable_median_price",
    "nearby_comparable_median_price_per_sqft",
    "nearby_comparable_count",
    "nearby_comparable_nearest_km",
    "nearby_comparable_estimated_price",
    "nearby_to_community_comparable_price_ratio",
    "nearby_to_community_comparable_ppsf_ratio",
    "community_comparable_median_price",
    "community_comparable_median_price_per_sqft",
    "community_comparable_count",
    "community_comparable_estimated_price",
    "community_type_ppsf_recent_weighted_3m",
    "community_type_ppsf_prior_weighted_3_12m",
    "community_type_ppsf_trend_3m_vs_prior",
    "community_type_sales_count_3m",
    "property_type_ppsf_recent_weighted_3m",
    "property_type_ppsf_prior_weighted_3_12m",
    "property_type_ppsf_trend_3m_vs_prior",
    "property_type_sales_count_3m",
    "list_price_per_sqft",
    "list_price_to_nearby_comparable",
    "list_price_to_community_comparable",
)
NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + DERIVED_NUMERIC_COLUMNS
COMPARABLE_COLUMNS = (
    "nearby_comparable_median_price",
    "nearby_comparable_median_price_per_sqft",
    "nearby_comparable_count",
    "nearby_comparable_nearest_km",
    "community_comparable_median_price",
    "community_comparable_median_price_per_sqft",
    "community_comparable_count",
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
FIT_LIMITS = {"svr": 1_200, "gbm": 30_000}


class ComparableFeatureBuilder:
    """Generate leak-safe neighborhood and community comparable summaries."""

    def fit(self, sales):
        self.nearby = NearbyComparableFeatures().fit(sales)
        self.community = CommunityComparableFeatures().fit(sales)
        return self

    def transform(self, homes):
        return pd.concat(
            [self.nearby.transform(homes), self.community.transform(homes)],
            axis=1,
        )


class MarketPriceTrendFeatures:
    """Past-only recency-weighted price-per-square-foot trends."""

    half_life_days = 45.0
    recent_months = 3
    history_months = 12

    def fit(self, sales):
        sale_dates = pd.to_datetime(sales["date_sold"], errors="coerce")
        prices = pd.to_numeric(sales["sold_price"], errors="coerce").to_numpy(
            dtype=float
        )
        sizes = features.square_feet(sales["house_area"]).to_numpy(dtype=float)
        ppsf = np.divide(
            prices,
            sizes,
            out=np.full(len(prices), np.nan),
            where=sizes > 0,
        )
        valid = (
            sale_dates.notna().to_numpy()
            & np.isfinite(ppsf)
            & (prices > 0)
            & (sizes > 0)
        )
        dates = sale_dates.to_numpy(dtype="datetime64[ns]")[valid]
        ppsf = ppsf[valid]
        communities = sales["community"].fillna("").astype(str).to_numpy()[valid]
        property_types = sales["house_type_name"].fillna("").astype(str).to_numpy()[
            valid
        ]
        self.sales_by_group = {"community_type": {}, "property_type": {}}
        groups = {
            "community_type": zip(communities, property_types),
            "property_type": ((property_type,) for property_type in property_types),
        }
        for dimension, keys in groups.items():
            grouped = {}
            for position, key in enumerate(keys):
                if "" not in key:
                    grouped.setdefault(key, []).append(position)
            for key, positions in grouped.items():
                positions = np.asarray(positions)
                order = np.argsort(dates[positions], kind="stable")
                selected = positions[order]
                self.sales_by_group[dimension][key] = (
                    dates[selected],
                    ppsf[selected],
                )
        return self

    @staticmethod
    def _weighted_mean(dates, values, start, end, cutoff):
        left = np.searchsorted(dates, start.to_datetime64(), side="left")
        right = np.searchsorted(dates, end.to_datetime64(), side="left")
        if right <= left:
            return np.nan, 0
        selected_dates = dates[left:right]
        selected_values = values[left:right]
        age_days = (
            cutoff.to_datetime64() - selected_dates
        ) / np.timedelta64(1, "D")
        weights = np.exp2(-age_days / MarketPriceTrendFeatures.half_life_days)
        return float(np.average(selected_values, weights=weights)), int(right - left)

    def transform(self, homes):
        columns = {}
        for dimension in ("community_type", "property_type"):
            prefix = dimension
            columns.update({
                f"{prefix}_ppsf_recent_weighted_3m": np.full(len(homes), np.nan),
                f"{prefix}_ppsf_prior_weighted_3_12m": np.full(len(homes), np.nan),
                f"{prefix}_ppsf_trend_3m_vs_prior": np.full(len(homes), np.nan),
                f"{prefix}_sales_count_3m": np.zeros(len(homes), dtype=float),
            })
        if homes.empty:
            return pd.DataFrame(columns, index=homes.index)

        listing_dates = pd.to_datetime(homes["date_listed"], errors="coerce")
        communities = homes["community"].fillna("").astype(str).to_numpy()
        property_types = homes["house_type_name"].fillna("").astype(str).to_numpy()
        periods = listing_dates.dt.to_period("M")
        period_values = periods.to_numpy()
        valid_homes = listing_dates.notna().to_numpy() & (property_types != "")

        for dimension in ("community_type", "property_type"):
            query_keys = (
                list(zip(communities, property_types))
                if dimension == "community_type"
                else [(property_type,) for property_type in property_types]
            )
            valid_dimension = valid_homes.copy()
            if dimension == "community_type":
                valid_dimension &= communities != ""
            grouped_homes = pd.DataFrame({
                "period": period_values[valid_dimension],
                "key": [query_keys[i] for i in np.flatnonzero(valid_dimension)],
                "position": np.flatnonzero(valid_dimension),
            })
            for (period, key), group in grouped_homes.groupby(
                ["period", "key"], sort=False
            ):
                sales_group = self.sales_by_group[dimension].get(key)
                if sales_group is None:
                    continue
                rows = group["position"].to_numpy()
                cutoff = period.start_time
                recent_start = cutoff - pd.DateOffset(
                    months=self.recent_months
                )
                history_start = cutoff - pd.DateOffset(
                    months=self.history_months
                )
                dates, values = sales_group
                recent_mean, recent_count = self._weighted_mean(
                    dates, values, recent_start, cutoff, cutoff
                )
                prior_mean, _ = self._weighted_mean(
                    dates, values, history_start, recent_start, cutoff
                )
                columns[f"{dimension}_ppsf_recent_weighted_3m"][rows] = recent_mean
                columns[f"{dimension}_ppsf_prior_weighted_3_12m"][rows] = prior_mean
                columns[f"{dimension}_sales_count_3m"][rows] = recent_count
                if np.isfinite(recent_mean) and np.isfinite(prior_mean) and prior_mean > 0:
                    columns[f"{dimension}_ppsf_trend_3m_vs_prior"][rows] = (
                        recent_mean / prior_mean - 1
                    )

        return pd.DataFrame(columns, index=homes.index)


def make_features(homes, track, comparables=None, market_trends=None):
    """Build shared, answer-safe features for both prediction tracks."""
    numeric = pd.DataFrame(index=homes.index)
    for column in BASE_NUMERIC_COLUMNS:
        if column == "house_area":
            numeric[column] = features.square_feet(homes[column])
        else:
            numeric[column] = pd.to_numeric(homes[column], errors="coerce")

    front = numeric["lot_front"]
    depth = numeric["lot_depth"]
    valid_lot = (front > 0) & (depth > 0)
    lot_area = (front * depth).where(valid_lot)
    numeric["lot_area_sqft"] = lot_area
    numeric["log_lot_area_sqft"] = np.log1p(lot_area)
    numeric["lot_perimeter_ft"] = (2 * (front + depth)).where(valid_lot)
    numeric["lot_aspect_ratio"] = (
        pd.concat([front, depth], axis=1).max(axis=1)
        / pd.concat([front, depth], axis=1).min(axis=1).where(valid_lot)
    )
    numeric["floor_area_to_lot_area"] = (
        numeric["house_area"] / lot_area
    ).where(valid_lot)
    numeric["log_house_area"] = np.log1p(numeric["house_area"].where(
        numeric["house_area"] > 0
    ))
    numeric["bedrooms_per_1000_sqft"] = (
        1000 * numeric["bedrooms"] / numeric["house_area"].where(
            numeric["house_area"] > 0
        )
    )
    numeric["washrooms_per_bedroom"] = (
        numeric["washrooms"] / numeric["bedrooms"].where(numeric["bedrooms"] > 0)
    )
    numeric["parking_per_bedroom"] = (
        numeric["parking_total"] / numeric["bedrooms"].where(
            numeric["bedrooms"] > 0
        )
    )
    numeric["total_bedrooms_washrooms"] = (
        numeric["bedrooms"] + numeric["washrooms"]
    )
    numeric["rotated_longitude"] = (
        numeric["longitude"] * 0.72 + numeric["latitude"]
    ) * np.sqrt(0.5)
    numeric["rotated_latitude"] = (
        numeric["latitude"] - numeric["longitude"] * 0.72
    ) * np.sqrt(0.5)

    listed = pd.to_datetime(homes["date_listed"], errors="coerce")
    numeric["months"] = (listed - pd.Timestamp("2021-01-01")).dt.days / 30.44
    month_angle = 2 * np.pi * (listed.dt.month - 1) / 12
    numeric["month_sin"] = np.sin(month_angle)
    numeric["month_cos"] = np.cos(month_angle)
    comparable_values = (
        comparables.transform(homes)
        if comparables is not None
        else pd.DataFrame(np.nan, index=homes.index, columns=COMPARABLE_COLUMNS)
    )
    numeric = numeric.join(comparable_values)
    if market_trends is not None:
        numeric = numeric.join(market_trends.transform(homes))
    else:
        trend_columns = [
            column for column in DERIVED_NUMERIC_COLUMNS
            if column.startswith(("community_type_", "property_type_"))
        ]
        numeric[trend_columns] = np.nan
    for location in ("nearby", "community"):
        ppsf = numeric[f"{location}_comparable_median_price_per_sqft"]
        estimated = ppsf * numeric["house_area"]
        numeric[f"{location}_comparable_estimated_price"] = estimated
    numeric["nearby_to_community_comparable_price_ratio"] = (
        numeric["nearby_comparable_median_price"]
        / numeric["community_comparable_median_price"].where(
            numeric["community_comparable_median_price"] > 0
        )
    )
    numeric["nearby_to_community_comparable_ppsf_ratio"] = (
        numeric["nearby_comparable_median_price_per_sqft"]
        / numeric["community_comparable_median_price_per_sqft"].where(
            numeric["community_comparable_median_price_per_sqft"] > 0
        )
    )
    if track == "list_aware":
        numeric["list_price"] = pd.to_numeric(homes["list_price"], errors="coerce")
        numeric["list_price_per_sqft"] = (
            numeric["list_price"]
            / numeric["house_area"].where(numeric["house_area"] > 0)
        )
        for location in ("nearby", "community"):
            comparable_price = numeric[f"{location}_comparable_median_price"]
            numeric[f"list_price_to_{location}_comparable"] = (
                numeric["list_price"] / comparable_price.where(comparable_price > 0)
            )
    else:
        numeric["list_price_per_sqft"] = np.nan
        numeric["list_price_to_nearby_comparable"] = np.nan
        numeric["list_price_to_community_comparable"] = np.nan
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    categories = homes[list(CATEGORICAL_COLUMNS)].fillna("(missing)").astype(str)
    remarks = homes.get(TEXT_COLUMN, pd.Series("", index=homes.index))
    text = remarks.fillna("").astype(str).rename(TEXT_COLUMN)
    return pd.concat([numeric, categories, text], axis=1)


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
        text_steps = [("tfidf", _tfidf_vectorizer()), (
            "svd", TruncatedSVD(n_components=TFIDF_N_COMPONENTS, random_state=0)
        )]
    else:
        category_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="(missing)")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        text_steps = [("tfidf", _tfidf_vectorizer())]
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", category_pipeline, list(CATEGORICAL_COLUMNS)),
        ("remarks", Pipeline(text_steps), TEXT_COLUMN),
    ])


def _tfidf_vectorizer():
    return TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=TFIDF_MIN_DF,
        max_df=0.98,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )


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


def _predict_prices(model, model_name, homes_features):
    transformed = model.named_steps["features"].transform(homes_features)
    if model_name == "lgbm":
        prediction = model.named_steps["regressor"].booster_.predict(transformed)
    else:
        prediction = model.named_steps["regressor"].predict(transformed)
    return np.exp(np.clip(prediction, np.log(10_000), np.log(100_000_000)))


def _fit_predict(
    model_name, params, track, train, homes, train_features=None, homes_features=None
):
    fit_rows = _sample_training_rows(train, model_name)
    if train_features is None or homes_features is None:
        comparables = ComparableFeatureBuilder().fit(train)
        market_trends = MarketPriceTrendFeatures().fit(train)
        train_features = make_features(train, track, comparables, market_trends)
        homes_features = make_features(
            homes, track, comparables, market_trends
        )
    model = make_pipeline(model_name, track, params)
    model.fit(
        train_features.loc[fit_rows.index],
        np.log(pd.to_numeric(fit_rows["sold_price"], errors="coerce").to_numpy()),
    )
    return _predict_prices(model, model_name, homes_features)


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
        tuning_comparables = ComparableFeatureBuilder().fit(tuning_train)
        tuning_market_trends = MarketPriceTrendFeatures().fit(tuning_train)
        tuning_train_features = make_features(
            tuning_train, track, tuning_comparables, tuning_market_trends
        )
        tuning_homes_features = make_features(
            data.hide_answers(tuning_homes, track),
            track,
            tuning_comparables,
            tuning_market_trends,
        )
        blend_comparables = ComparableFeatureBuilder().fit(blend_train)
        blend_market_trends = MarketPriceTrendFeatures().fit(blend_train)
        blend_train_features = make_features(
            blend_train, track, blend_comparables, blend_market_trends
        )
        blend_homes_features = make_features(
            data.hide_answers(blend_homes, track),
            track,
            blend_comparables,
            blend_market_trends,
        )
        best_params = {}
        tuning_scores = {}

        for model_name in MODEL_NAMES:
            candidates = []
            for params in TRIALS[model_name]:
                predicted = _fit_predict(
                    model_name, params, track, tuning_train,
                    data.hide_answers(tuning_homes, track),
                    tuning_train_features, tuning_homes_features,
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
                blend_train_features, blend_homes_features,
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
        "comparable_features": [
            "past-only nearby same-property-type median price, price per sqft, count, and nearest distance within 3 km",
            "past-only same-community and same-property-type median price, price per sqft, and count",
        ],
        "engineered_features": [
            "rectangular lot area, log lot area, perimeter, aspect ratio, and interior floor-area-to-lot-area ratio",
            "log interior area, bedrooms per 1,000 sqft, washrooms and parking per bedroom, and total bedrooms plus washrooms",
            "45-degree rotated local coordinates",
            "comparable price-per-sqft times subject area, nearby-to-community comparable ratios, and list-price-per-sqft and list-price-to-comparable ratios when list price is available",
        ],
        "text_features": {
            "source": "listing remarks",
            "method": "TF-IDF unigrams and bigrams, English stop words removed, sublinear term frequency",
            "vocabulary": f"up to {TFIDF_MAX_FEATURES} terms, minimum document frequency {TFIDF_MIN_DF}, fit only on each model training fold",
            "gradient_boosting": f"{TFIDF_N_COMPONENTS}-component truncated SVD of the training-fold TF-IDF matrix",
            "svr_fit_sample_limit": FIT_LIMITS["svr"],
        },
        "market_trend_features": {
            "groupings": ["community and property type", "property type"],
            "recent_window": f"weighted average price per sqft in the {MarketPriceTrendFeatures.recent_months} months before the listing month",
            "comparison_window": f"weighted average price per sqft from {MarketPriceTrendFeatures.recent_months} to {MarketPriceTrendFeatures.history_months} months before the listing month",
            "recency_weighting": f"exponential decay with {MarketPriceTrendFeatures.half_life_days:g}-day half-life",
            "trend": "recent weighted average divided by prior weighted average, minus one",
            "cutoff": "start of listing month; same-month and later sales excluded",
        },
        "tracks": results,
    }, indent=2))
    print(f"Wrote {output_dir.relative_to(root)}")
