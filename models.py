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
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

import features

MODEL = "xgboost_community_comps"


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if MODEL == "spatiotemporal_gam":
        return SpatiotemporalGAM()
    if MODEL == "xgboost":
        return XGBoostModel(track)
    if MODEL == "xgboost_rotated":
        return RotatedCoordinateXGBoostModel(track)
    if MODEL == "xgboost_community_comps":
        return CommunityComparableXGBoostModel(track)
    if MODEL == "xgboost_matern_grf":
        return MaternSpatialXGBoostModel(track)
    if MODEL == "xgboost_arima":
        return ARIMATimeXGBoostModel(track)
    if MODEL == "lightgbm_community_comps":
        return CommunityComparableLightGBMModel(track)
    if MODEL == "lightgbm_nearby_comps_safe":
        return SafeNearbyComparableLightGBMModel(track)
    if MODEL == "convlstm_spatial":
        from convlstm import ConvLSTMPriceModel

        return ConvLSTMPriceModel(track)
    if MODEL == "feature":
        if track == "list_aware":
            return FeatureModel(features.list_aware_features)
        return FeatureModel(features.off_market_features)
    if MODEL != "baseline":
        raise ValueError(
            f"Unknown model {MODEL!r}; choose 'baseline', 'feature', "
            "'spatiotemporal_gam', 'xgboost', 'xgboost_rotated', or "
            "'xgboost_community_comps', 'xgboost_matern_grf', "
            "'xgboost_arima', 'lightgbm_community_comps', or "
            "'lightgbm_nearby_comps_safe', or 'convlstm_spatial'."
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
        self.comparables = NearbyComparableFeatures()
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
        numeric = numeric.join(self.comparables.transform(homes))
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
        self.comparables.fit(train)
        self.model.fit(self._design_matrix(train, fit=True), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self._design_matrix(homes)))


class RotatedCoordinateXGBoostModel(XGBoostModel):
    """XGBoost model with additional 45-degree local coordinate features."""

    def _values(self, homes):
        numeric, categories = super()._values(homes)
        longitude = pd.to_numeric(homes["longitude"], errors="coerce")
        latitude = pd.to_numeric(homes["latitude"], errors="coerce")
        x = longitude * 0.72
        y = latitude
        scale = np.sqrt(0.5)
        numeric["coordinate_rotated_45_x"] = (x + y) * scale
        numeric["coordinate_rotated_45_y"] = (y - x) * scale
        return numeric, categories


class CommunityComparableXGBoostModel(RotatedCoordinateXGBoostModel):
    """Rotated XGBoost using same-community rather than distance-based comps."""

    def __init__(self, track, **kwargs):
        super().__init__(track, **kwargs)
        self.comparables = CommunityComparableFeatures()


class MaternSpatialXGBoostModel(CommunityComparableXGBoostModel):
    """Full-data community-comparable boosting plus a binned Matérn spatial field."""

    spatial_cell_km = 1.0
    spatial_strength = 0.5
    longitude_scale = 0.72
    km_per_degree = 111.2

    def __init__(self, track, **kwargs):
        super().__init__(track, **kwargs)
        self.spatial_model = None

    def _spatial_coordinates(self, homes):
        coordinates = homes[["longitude", "latitude"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        coordinates[:, 0] *= self.longitude_scale
        return coordinates * self.km_per_degree

    def fit(self, train):
        self.comparables.fit(train)
        self.model.fit(
            self._design_matrix(train, fit=True),
            np.log(train["sold_price"]),
        )
        coordinates = self._spatial_coordinates(train)
        log_prices = np.log(pd.to_numeric(train["sold_price"], errors="coerce"))
        sale_months = pd.to_datetime(train["date_sold"], errors="coerce").dt.to_period("M")
        property_types = train["house_type_name"].fillna("(missing)").astype(str)
        market_baseline = log_prices.groupby(
            [sale_months, property_types], dropna=False
        ).transform("median")
        spatial_residuals = log_prices - market_baseline
        valid = (
            np.isfinite(coordinates).all(axis=1)
            & np.isfinite(spatial_residuals.to_numpy(dtype=float))
        )
        if valid.any():
            valid_coordinates = coordinates[valid]
            grid = np.floor(valid_coordinates / self.spatial_cell_km).astype(int)
            cells = pd.DataFrame({
                "grid_x": grid[:, 0],
                "grid_y": grid[:, 1],
                "x": valid_coordinates[:, 0],
                "y": valid_coordinates[:, 1],
                "residual": spatial_residuals.to_numpy(dtype=float)[valid],
            })
            grouped = cells.groupby(["grid_x", "grid_y"], sort=False).agg(
                x=("x", "mean"),
                y=("y", "mean"),
                residual=("residual", "mean"),
                count=("residual", "size"),
            )
            kernel = (
                ConstantKernel(1.0, (0.01, 100.0))
                * Matern(length_scale=3.0, length_scale_bounds=(0.25, 25.0), nu=1.5)
                + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 0.5))
            )
            self.spatial_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.01 / grouped["count"].to_numpy(dtype=float),
                normalize_y=False,
                n_restarts_optimizer=0,
                random_state=0,
            )
            self.spatial_model.fit(
                grouped[["x", "y"]].to_numpy(),
                grouped["residual"].to_numpy(),
            )
            self.spatial_training_rows_ = int(grouped["count"].sum())
        return self

    def predict(self, homes):
        predictions = self.model.predict(self._design_matrix(homes))
        if self.spatial_model is not None and not homes.empty:
            coordinates = self._spatial_coordinates(homes)
            valid = np.isfinite(coordinates).all(axis=1)
            if valid.any():
                predictions[valid] += (
                    self.spatial_strength
                    * self.spatial_model.predict(coordinates[valid])
                )
        return np.exp(predictions)


class ARIMATimeXGBoostModel(CommunityComparableXGBoostModel):
    """Best boosted model plus a prior-only ARIMA price-per-square-foot trend."""

    arima_order = (1, 1, 0)
    minimum_months = 12
    arima_strength = 0.25

    def __init__(self, track, **kwargs):
        super().__init__(track, **kwargs)
        self.market_series = {}
        self.forecast_cache = {}

    def fit(self, train):
        super().fit(train)
        sale_dates = pd.to_datetime(train["date_sold"], errors="coerce")
        sale_months = sale_dates.dt.to_period("M")
        prices = pd.to_numeric(train["sold_price"], errors="coerce")
        sizes = features.square_feet(train["house_area"])
        valid = (
            sale_months.notna()
            & prices.gt(0)
            & sizes.gt(0)
            & np.isfinite(sizes)
        )
        history = pd.DataFrame({
            "month": sale_months[valid],
            "property_type": train.loc[valid, "house_type_name"].fillna("(missing)").astype(str),
            "log_ppsf": np.log(prices[valid] / sizes[valid]),
        })
        self.market_series = {
            property_type: group.groupby("month")["log_ppsf"].median().sort_index()
            for property_type, group in history.groupby("property_type", sort=False)
        }
        self.market_series["(all)"] = history.groupby("month")["log_ppsf"].median().sort_index()
        self.forecast_cache = {}
        return self

    def _forecast_market_change(self, property_type, target_month):
        cache_key = (property_type, target_month)
        if cache_key in self.forecast_cache:
            return self.forecast_cache[cache_key]

        series = self.market_series.get(property_type, self.market_series["(all)"])
        prior = series[series.index < target_month]
        change = 0.0
        if len(prior) >= self.minimum_months:
            complete_months = pd.period_range(prior.index.min(), prior.index.max(), freq="M")
            observed_positions = prior.index.asi8
            complete_values = np.interp(
                complete_months.asi8,
                observed_positions,
                prior.to_numpy(dtype=float),
            )
            horizon = target_month.ordinal - complete_months[-1].ordinal
            if horizon > 0:
                forecast = ARIMA(
                    complete_values,
                    order=self.arima_order,
                    trend="t",
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit().forecast(steps=horizon)
                change = float(forecast[-1] - complete_values[-1])

        self.forecast_cache[cache_key] = change
        return change

    def predict(self, homes):
        predictions = super().predict(homes)
        listed_dates = pd.to_datetime(homes["date_listed"], errors="coerce")
        property_types = homes["house_type_name"].fillna("(missing)").astype(str)
        changes = np.zeros(len(homes), dtype=float)
        for position, (listed_date, property_type) in enumerate(
            zip(listed_dates, property_types)
        ):
            if pd.notna(listed_date):
                target_month = listed_date.to_period("M")
                changes[position] = self._forecast_market_change(
                    property_type, target_month
                )
        return predictions * np.exp(self.arima_strength * changes)


class CommunityComparableLightGBMModel(RotatedCoordinateXGBoostModel):
    """LightGBM using rotated coordinates and same-community comparable sales."""

    def __init__(self, track):
        super().__init__(track)
        self.comparables = CommunityComparableFeatures()
        self.model = LGBMRegressor(
            objective="regression",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=0,
            verbosity=-1,
        )

    def predict(self, homes):
        return np.exp(self.model.booster_.predict(self._design_matrix(homes)))


class SafeNearbyComparableLightGBMModel(RotatedCoordinateXGBoostModel):
    """LightGBM using past-only nearby comps and rotated coordinates."""

    def __init__(self, track):
        super().__init__(track)
        self.model = LGBMRegressor(
            objective="regression",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=0,
            verbosity=-1,
        )

    def predict(self, homes):
        return np.exp(self.model.booster_.predict(self._design_matrix(homes)))


class CommunityComparableFeatures:
    """Past-only comparable sale summaries for the same community and home type."""

    def fit(self, sales):
        self.sale_dates = pd.to_datetime(sales["date_sold"], errors="coerce").to_numpy()
        self.communities = sales["community"].fillna("").astype(str).to_numpy()
        self.property_types = sales["house_type_name"].fillna("").astype(str).to_numpy()
        self.prices = pd.to_numeric(sales["sold_price"], errors="coerce").to_numpy(dtype=float)
        self.square_feet = features.square_feet(sales["house_area"]).to_numpy(dtype=float)
        self.sales_by_group = {}
        valid = (
            pd.notna(self.sale_dates)
            & (self.communities != "")
            & (self.property_types != "")
            & np.isfinite(self.prices)
            & (self.prices > 0)
        )
        valid_positions = np.flatnonzero(valid)
        grouped_sales = pd.DataFrame({
            "community": self.communities[valid_positions],
            "property_type": self.property_types[valid_positions],
            "position": valid_positions,
        })
        for key, group in grouped_sales.groupby(
            ["community", "property_type"], sort=False
        ):
            positions = group["position"].to_numpy()
            order = np.argsort(self.sale_dates[positions], kind="stable")
            self.sales_by_group[key] = (
                self.sale_dates[positions][order],
                self.prices[positions][order],
                self.square_feet[positions][order],
            )
        return self

    def transform(self, homes):
        columns = [
            "community_comparable_median_price",
            "community_comparable_median_price_per_sqft",
            "community_comparable_count",
        ]
        result = pd.DataFrame(np.nan, index=homes.index, columns=columns, dtype=float)
        result["community_comparable_count"] = 0
        if homes.empty:
            return result

        home_dates = pd.to_datetime(homes["date_listed"], errors="coerce")
        communities = homes["community"].fillna("").astype(str).to_numpy()
        property_types = homes["house_type_name"].fillna("").astype(str).to_numpy()
        periods = home_dates.dt.to_period("M")
        period_values = periods.to_numpy()
        valid_homes = home_dates.notna().to_numpy() & (communities != "") & (
            property_types != ""
        )

        valid_positions = np.flatnonzero(valid_homes)
        query_groups = pd.DataFrame({
            "period": period_values[valid_positions],
            "community": communities[valid_positions],
            "property_type": property_types[valid_positions],
            "position": valid_positions,
        })
        median_prices = np.full(len(homes), np.nan)
        median_ppsf = np.full(len(homes), np.nan)
        comparable_counts = np.zeros(len(homes), dtype=float)
        for (period, community, property_type), group_homes in query_groups.groupby(
            ["period", "community", "property_type"], sort=False
        ):
            rows = group_homes["position"].to_numpy()
            cutoff = period.start_time.to_datetime64()
            sales_group = self.sales_by_group.get((community, property_type))
            if sales_group is None:
                continue
            sale_dates, prices, sizes = sales_group
            earlier_count = np.searchsorted(sale_dates, cutoff, side="left")
            if earlier_count == 0:
                continue

            prices = prices[:earlier_count]
            sizes = sizes[:earlier_count]
            median_prices[rows] = np.median(prices)
            valid_sizes = np.isfinite(sizes) & (sizes > 0)
            if valid_sizes.any():
                median_ppsf[rows] = np.median(prices[valid_sizes] / sizes[valid_sizes])
            comparable_counts[rows] = earlier_count
        result["community_comparable_median_price"] = median_prices
        result["community_comparable_median_price_per_sqft"] = median_ppsf
        result["community_comparable_count"] = comparable_counts
        return result


class NearbyComparableFeatures:
    """Past-only nearby sale summaries for same-type homes.

    Candidate locations come only from sales before the listing month; this
    deliberately excludes same-month sales to keep future locations out too.
    """

    candidate_count = 128
    radius_km = 3.0
    longitude_scale = 0.72
    km_per_degree = 111.2
    tree_cache = {}

    def fit(self, sales):
        self.sales = sales
        self.sale_dates = pd.to_datetime(sales["date_sold"], errors="coerce").to_numpy()
        self.property_types = sales["house_type_name"].fillna("").astype(str).to_numpy()
        self.prices = pd.to_numeric(sales["sold_price"], errors="coerce").to_numpy(dtype=float)
        self.square_feet = features.square_feet(sales["house_area"]).to_numpy(dtype=float)
        self.sale_coords = sales[["longitude", "latitude"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        self.valid_sales = np.isfinite(self.sale_coords).all(axis=1)
        return self

    def transform(self, homes):
        result = pd.DataFrame(
            index=homes.index,
            columns=[
                "nearby_comparable_median_price",
                "nearby_comparable_median_price_per_sqft",
                "nearby_comparable_count",
                "nearby_comparable_nearest_km",
            ],
            dtype=float,
        )
        result["nearby_comparable_count"] = 0
        if homes.empty:
            return result
        home_coords = homes[["longitude", "latitude"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        valid_homes = np.isfinite(home_coords).all(axis=1)
        home_dates = pd.to_datetime(homes["date_listed"], errors="coerce")
        valid_homes &= home_dates.notna().to_numpy()
        if not valid_homes.any():
            return result

        periods = home_dates.dt.to_period("M")
        period_values = periods.to_numpy()
        home_types = homes["house_type_name"].fillna("").astype(str).to_numpy()
        for period in pd.unique(period_values[valid_homes]):
            rows = np.flatnonzero(valid_homes & (period_values == period))
            cutoff = period.start_time.to_datetime64()
            historical_sales = np.flatnonzero(
                self.valid_sales & (self.sale_dates < cutoff)
            )
            if len(historical_sales) == 0:
                continue

            historical_labels = self.sales.index.to_numpy()[historical_sales]
            tree_coords = self.sale_coords[historical_sales].copy()
            tree_coords[:, 0] *= self.longitude_scale
            label_hash = hash(pd.util.hash_pandas_object(
                pd.Index(historical_labels), index=False
            ).values.tobytes())
            coords_hash = hash(tree_coords.tobytes())
            cache_key = (period.ordinal, len(historical_sales), label_hash, coords_hash)
            cached_tree = self.tree_cache.get(cache_key)
            if cached_tree is None:
                tree = cKDTree(tree_coords)
                self.tree_cache[cache_key] = (tree, historical_labels)
            else:
                tree, historical_labels = cached_tree
                historical_sales = self.sales.index.get_indexer(historical_labels)
            query_coords = home_coords[rows].copy()
            query_coords[:, 0] *= self.longitude_scale
            k = min(self.candidate_count, len(historical_sales))
            distances, candidate_positions = tree.query(query_coords, k=k)
            distances = np.asarray(distances).reshape(len(rows), k) * self.km_per_degree
            candidates = historical_sales[
                np.asarray(candidate_positions).reshape(len(rows), k)
            ]
            date_cutoffs = home_dates.iloc[rows].to_numpy()[:, np.newaxis]
            matching = (
                (distances <= self.radius_km)
                & (self.sale_dates[candidates] < date_cutoffs)
                & (self.property_types[candidates] == home_types[rows, np.newaxis])
            )
            candidate_prices = self.prices[candidates]
            candidate_sizes = self.square_feet[candidates]
            matching &= np.isfinite(candidate_prices) & (candidate_prices > 0)
            counts = matching.sum(axis=1)
            has_comparables = counts > 0

            median_prices = np.full(len(rows), np.nan)
            median_prices[has_comparables] = np.nanmedian(
                np.where(matching[has_comparables], candidate_prices[has_comparables], np.nan),
                axis=1,
            )
            valid_size = np.isfinite(candidate_sizes) & (candidate_sizes > 0)
            ppsf_matches = matching & valid_size
            has_ppsf = ppsf_matches.any(axis=1)
            median_ppsf = np.full(len(rows), np.nan)
            median_ppsf[has_ppsf] = np.nanmedian(
                np.where(
                    ppsf_matches[has_ppsf],
                    candidate_prices[has_ppsf] / candidate_sizes[has_ppsf],
                    np.nan,
                ),
                axis=1,
            )
            nearest_distances = np.min(
                np.where(matching, distances, np.inf), axis=1
            )

            result.iloc[rows, result.columns.get_loc("nearby_comparable_median_price")] = median_prices
            result.iloc[
                rows, result.columns.get_loc("nearby_comparable_median_price_per_sqft")
            ] = median_ppsf
            result.iloc[rows, result.columns.get_loc("nearby_comparable_count")] = counts
            result.iloc[
                rows, result.columns.get_loc("nearby_comparable_nearest_km")
            ] = np.where(has_comparables, nearest_distances, np.nan)
        return result
