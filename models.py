"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBRegressor

import features

MODEL = "xgboost" 


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if MODEL == "spatiotemporal_gam":
        return SpatiotemporalGAM()
    if MODEL == "xgboost":
        return XGBoostModel(track)
    if MODEL == "feature":
        if track == "list_aware":
            return FeatureModel(features.list_aware_features)
        return FeatureModel(features.off_market_features)
    if MODEL != "baseline":
        raise ValueError(
            f"Unknown model {MODEL!r}; choose 'baseline', 'feature', "
            "'spatiotemporal_gam', or 'xgboost'."
        )
    if track == "list_aware":
        return ListPriceBaseline()
    return NearestSalesBaseline(k=10)


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


class SpatiotemporalGAM:
    """GAM-style log-price model mirroring the supplied mgcv formula.

    Uses one-hot categorical effects, linear property effects, a 10-by-10
    tensor-product spatial spline, a 20-basis time spline, and a 12-basis
    periodic month spline. Ridge regularization is a practical approximation
    to mgcv's REML smoothness selection, not an identical fitting procedure.
    Longitude/latitude represent x/y, and both time terms come from date_listed.
    House-area ranges are converted to their midpoint.
    """

    categorical_columns = ["house_type_name", "house_style", "municipality", "community"]
    numeric_columns = [
        "bedrooms", "washrooms", "parking_total", "garage",
        "house_area", "lot_front", "lot_depth",
    ]

    def __init__(self, alpha=1.0):
        self.numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.numeric_scaler = StandardScaler()
        self.category_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
        self.coordinate_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.time_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.month_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.longitude_spline = SplineTransformer(
            n_knots=8, degree=3, extrapolation="linear"
        )
        self.latitude_spline = SplineTransformer(
            n_knots=8, degree=3, extrapolation="linear"
        )
        self.time_spline = SplineTransformer(
            n_knots=18, degree=3, extrapolation="linear"
        )
        month_knots = np.linspace(0.5, 12.5, 13).reshape(-1, 1)
        self.month_spline = SplineTransformer(
            knots=month_knots, degree=3, extrapolation="periodic"
        )
        self.regressor = Ridge(alpha=alpha, solver="lsqr")

    def _values(self, homes):
        numeric = homes[self.numeric_columns].copy()
        numeric["house_area"] = features.square_feet(numeric["house_area"])
        numeric = numeric.apply(pd.to_numeric, errors="coerce")

        categories = homes[self.categorical_columns].fillna("(missing)").astype(str)
        coordinates = homes[["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce")
        listed = pd.to_datetime(homes["date_listed"], errors="coerce")
        time = ((listed - pd.Timestamp("2021-01-01")).dt.days / 30.44).to_numpy().reshape(-1, 1)
        month = listed.dt.month.to_numpy(dtype=float).reshape(-1, 1)
        return numeric, categories, coordinates, time, month

    def _design_matrix(self, homes, fit=False):
        numeric, categories, coordinates, time, month = self._values(homes)

        if fit:
            numeric_values = self.numeric_scaler.fit_transform(
                self.numeric_imputer.fit_transform(numeric)
            )
            category_values = self.category_encoder.fit_transform(categories)
            coordinates = self.coordinate_imputer.fit_transform(coordinates)
            time = self.time_imputer.fit_transform(time)
            month = self.month_imputer.fit_transform(month)
            longitude_values = self.longitude_spline.fit_transform(coordinates[:, [0]])
            latitude_values = self.latitude_spline.fit_transform(coordinates[:, [1]])
            time_values = self.time_spline.fit_transform(time)
            month_values = self.month_spline.fit_transform(month)
        else:
            numeric_values = self.numeric_scaler.transform(self.numeric_imputer.transform(numeric))
            category_values = self.category_encoder.transform(categories)
            coordinates = self.coordinate_imputer.transform(coordinates)
            time = self.time_imputer.transform(time)
            month = self.month_imputer.transform(month)
            longitude_values = self.longitude_spline.transform(coordinates[:, [0]])
            latitude_values = self.latitude_spline.transform(coordinates[:, [1]])
            time_values = self.time_spline.transform(time)
            month_values = self.month_spline.transform(month)

        spatial_values = (
            longitude_values[:, :, np.newaxis] * latitude_values[:, np.newaxis, :]
        ).reshape(len(homes), -1)
        return sparse.hstack([
            sparse.csr_matrix(numeric_values),
            category_values,
            sparse.csr_matrix(spatial_values),
            sparse.csr_matrix(time_values),
            sparse.csr_matrix(month_values),
        ], format="csr")

    def fit(self, train):
        self.regressor.fit(self._design_matrix(train, fit=True), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.regressor.predict(self._design_matrix(homes)))


class XGBoostModel:
    """XGBoost regressor for log sale price using property, location, and time data."""

    categorical_columns = ["house_type_name", "house_style", "municipality", "community"]
    numeric_columns = [
        "bedrooms", "washrooms", "parking_total", "garage",
        "house_area", "lot_front", "lot_depth", "longitude", "latitude",
        "months", "month_sin", "month_cos",
    ]

    def __init__(self, track, n_estimators=400, max_depth=8, learning_rate=0.05):
        if track not in {"off_market", "list_aware"}:
            raise ValueError(f"Unknown track {track!r}; choose 'off_market' or 'list_aware'.")
        self.track = track
        self.include_list_price = track == "list_aware"
        self.numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.category_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        self.model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=0,
        )

    def _values(self, homes):
        numeric = pd.DataFrame(index=homes.index)
        for column in [
            "bedrooms", "washrooms", "parking_total", "garage",
            "lot_front", "lot_depth", "longitude", "latitude",
        ]:
            numeric[column] = pd.to_numeric(homes[column], errors="coerce")
        numeric["house_area"] = features.square_feet(homes["house_area"])

        listed = pd.to_datetime(homes["date_listed"], errors="coerce")
        numeric["months"] = (listed - pd.Timestamp("2021-01-01")).dt.days / 30.44
        month_angle = 2 * np.pi * (listed.dt.month - 1) / 12
        numeric["month_sin"] = np.sin(month_angle)
        numeric["month_cos"] = np.cos(month_angle)
        numeric = numeric.join(features.remarks_word_features(homes["remarks"]))
        if self.include_list_price:
            numeric["list_price"] = pd.to_numeric(homes["list_price"], errors="coerce")

        categories = homes[self.categorical_columns].fillna("(missing)").astype(str)
        return numeric, categories

    def _design_matrix(self, homes, fit=False):
        numeric, categories = self._values(homes)
        if fit:
            numeric_values = self.numeric_imputer.fit_transform(numeric)
            category_values = self.category_encoder.fit_transform(categories)
        else:
            numeric_values = self.numeric_imputer.transform(numeric)
            category_values = self.category_encoder.transform(categories)
        return sparse.hstack(
            [sparse.csr_matrix(numeric_values, dtype=np.float32), category_values],
            format="csr",
            dtype=np.float32,
        )

    def fit(self, train):
        self.model.fit(self._design_matrix(train, fit=True), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self._design_matrix(homes)))
