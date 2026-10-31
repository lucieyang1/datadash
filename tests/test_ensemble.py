import numpy as np
import pandas as pd
import pytest

import data
import ensemble


def make_sales(n=32):
    rng = np.random.default_rng(42)
    sold = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 400, n), unit="D"
    )
    return pd.DataFrame({
        "ml_num": [f"SYN-{i:07d}" for i in range(n)],
        "sold_price": rng.uniform(400_000, 1_500_000, n),
        "list_price": rng.uniform(400_000, 1_500_000, n),
        "date_listed": sold - pd.to_timedelta(10, unit="D"),
        "date_sold": sold,
        "house_type_name": rng.choice(["Condo Apt", "Detached"], n),
        "house_style": "Test",
        "bedrooms": rng.integers(1, 5, n),
        "washrooms": rng.integers(1, 4, n),
        "parking_total": rng.integers(0, 3, n),
        "garage": rng.integers(0, 2, n),
        "house_area": rng.choice(["600-699", "1200"], n),
        "lot_front": rng.uniform(0, 40, n),
        "lot_depth": rng.uniform(0, 100, n),
        "municipality": "Toronto",
        "community": rng.choice(["A", "B", "C"], n),
        "latitude": rng.normal(43.7, 0.02, n),
        "longitude": rng.normal(-79.4, 0.02, n),
        "remarks": rng.choice([
            "renovated family home modern kitchen private garden",
            "bright condo balcony updated kitchen near parks",
            "original finishes needs work large lot and garage",
            "turnkey home new floors ensuite bathroom outdoor space",
        ], n),
    })


def test_six_model_features_respect_track_answer_visibility():
    homes = make_sales(4)
    comparables = ensemble.ComparableFeatureBuilder().fit(homes)

    off_market = ensemble.make_features(
        data.hide_answers(homes, "off_market"), "off_market", comparables
    )
    list_aware = ensemble.make_features(
        data.hide_answers(homes, "list_aware"), "list_aware", comparables
    )

    assert "list_price" not in off_market
    assert "list_price" in list_aware
    assert not {"sold_price", "date_sold"}.intersection(off_market.columns)
    assert not {"sold_price", "date_sold"}.intersection(list_aware.columns)
    assert off_market["house_area"].notna().all()
    assert {
        "nearby_comparable_median_price",
        "nearby_comparable_median_price_per_sqft",
        "nearby_comparable_count",
        "nearby_comparable_nearest_km",
        "community_comparable_median_price",
        "community_comparable_median_price_per_sqft",
        "community_comparable_count",
    }.issubset(off_market.columns)


def test_engineered_lot_size_and_list_price_features():
    homes = make_sales(1).assign(
        house_area="1200",
        lot_front=20.0,
        lot_depth=100.0,
        bedrooms=3,
        washrooms=2,
        parking_total=2,
        list_price=600_000.0,
    )
    engineered = ensemble.make_features(
        data.hide_answers(homes, "list_aware"), "list_aware"
    ).iloc[0]

    assert engineered["lot_area_sqft"] == pytest.approx(2_000)
    assert engineered["log_lot_area_sqft"] == pytest.approx(np.log1p(2_000))
    assert engineered["lot_perimeter_ft"] == pytest.approx(240)
    assert engineered["lot_aspect_ratio"] == pytest.approx(5)
    assert engineered["floor_area_to_lot_area"] == pytest.approx(0.6)
    assert engineered["bedrooms_per_1000_sqft"] == pytest.approx(2.5)
    assert engineered["washrooms_per_bedroom"] == pytest.approx(2 / 3)
    assert engineered["parking_per_bedroom"] == pytest.approx(2 / 3)
    assert engineered["list_price_per_sqft"] == pytest.approx(500)


def test_lot_and_room_ratios_handle_missing_or_zero_denominators():
    homes = make_sales(2).assign(
        lot_front=[0.0, np.nan],
        lot_depth=[100.0, np.nan],
        bedrooms=[0, 0],
        house_area=["1200", "600"],
    )
    engineered = ensemble.make_features(
        data.hide_answers(homes, "off_market"), "off_market"
    )

    ratio_columns = [
        "lot_area_sqft",
        "lot_aspect_ratio",
        "floor_area_to_lot_area",
        "washrooms_per_bedroom",
        "parking_per_bedroom",
    ]
    assert engineered[ratio_columns].isna().all().all()
    assert not np.isinf(engineered.select_dtypes(include="number").to_numpy()).any()


def test_tfidf_vocabulary_is_learned_from_training_remarks_only():
    sales = make_sales(32)
    train = sales.iloc[:-4].copy()
    homes = data.hide_answers(sales.iloc[-4:].copy(), "off_market")
    homes.loc[homes.index[0], "remarks"] += " holdoutunseenvocabulary"
    comparables = ensemble.ComparableFeatureBuilder().fit(train)
    model = ensemble.make_pipeline("ridge", "off_market", {"alpha": 10.0})

    model.fit(
        ensemble.make_features(data.hide_answers(train, "off_market"), "off_market", comparables),
        np.log(train["sold_price"]),
    )

    vocabulary = model.named_steps["features"].named_transformers_["remarks"].named_steps[
        "tfidf"
    ].vocabulary_
    assert "holdoutunseenvocabulary" not in vocabulary
    transformed = model.named_steps["features"].transform(
        ensemble.make_features(homes, "off_market", comparables)
    )
    assert transformed.shape[0] == len(homes)
    assert np.isfinite(transformed.data).all()
    assert transformed.shape[1] > len(ensemble.NUMERIC_COLUMNS)


def test_comparable_features_exclude_same_month_and_future_sales():
    sales = make_sales(3)
    sales["house_type_name"] = "Detached"
    sales["community"] = "Same"
    sales["latitude"] = 43.7
    sales["longitude"] = -79.4
    sales["date_sold"] = pd.to_datetime([
        "2024-12-15", "2025-01-05", "2025-02-05"
    ])
    sales["date_listed"] = pd.to_datetime([
        "2024-12-01", "2025-01-01", "2025-02-01"
    ])
    sales["sold_price"] = [500_000, 900_000, 1_200_000]
    home = sales.iloc[[2]].assign(
        ml_num="SYN-9999999",
        date_listed=pd.Timestamp("2025-01-15"),
        date_sold=pd.Timestamp("2025-03-30"),
        sold_price=1_500_000,
    )

    builder = ensemble.ComparableFeatureBuilder().fit(sales)
    comparables = builder.transform(data.hide_answers(home, "off_market")).iloc[0]

    assert comparables["community_comparable_median_price"] == 500_000
    assert comparables["community_comparable_count"] == 1
    assert comparables["nearby_comparable_median_price"] == 500_000
    assert comparables["nearby_comparable_count"] == 1


def test_price_trends_use_only_sales_before_listing_month():
    sales = make_sales(6)
    sales["house_type_name"] = "Detached"
    sales["community"] = ["A", "A", "A", "A", "A", "B"]
    sales["house_area"] = "1000"
    sales["date_sold"] = pd.to_datetime([
        "2024-01-15",
        "2024-09-15",
        "2024-11-15",
        "2025-01-10",
        "2025-02-10",
        "2024-12-10",
    ])
    sales["sold_price"] = [
        100_000,
        200_000,
        400_000,
        10_000_000,
        20_000_000,
        800_000,
    ]
    home = sales.iloc[[0]].assign(
        ml_num="SYN-9999999",
        community="A",
        date_listed=pd.Timestamp("2025-01-20"),
        date_sold=pd.Timestamp("2025-03-01"),
        sold_price=1_000_000,
    )
    trend_features = ensemble.MarketPriceTrendFeatures().fit(sales).transform(home)
    result = trend_features.iloc[0]

    assert result["community_type_sales_count_3m"] == 1
    assert result["property_type_sales_count_3m"] == 2
    assert result["community_type_ppsf_recent_weighted_3m"] == pytest.approx(400)
    assert 400 < result["property_type_ppsf_recent_weighted_3m"] < 800
    assert np.isfinite(result["community_type_ppsf_trend_3m_vs_prior"])
    assert np.isnan(
        ensemble.MarketPriceTrendFeatures()
        .fit(sales[sales["date_sold"] < "2024-01-01"])
        .transform(home)["community_type_ppsf_recent_weighted_3m"]
        .iloc[0]
    )


@pytest.mark.parametrize("model_name", ensemble.MODEL_NAMES)
@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_each_model_fits_and_predicts(model_name, track):
    sales = make_sales()
    homes = data.hide_answers(sales.iloc[-5:], track)
    comparables = ensemble.ComparableFeatureBuilder().fit(sales)
    market_trends = ensemble.MarketPriceTrendFeatures().fit(sales)
    model = ensemble.make_pipeline(model_name, track, ensemble.TRIALS[model_name][0])

    features_frame = ensemble.make_features(
        data.hide_answers(sales, track), track, comparables, market_trends
    )
    model.fit(features_frame, np.log(sales["sold_price"]))
    homes_features = ensemble.make_features(
        homes, track, comparables, market_trends
    )
    predictions = ensemble._predict_prices(model, model_name, homes_features)

    assert predictions.shape == (len(homes),)
    assert np.isfinite(predictions).all()


def test_ensemble_weights_are_nonnegative_and_sum_to_one():
    actual = np.array([500_000.0, 700_000.0, 900_000.0])
    predictions = np.column_stack([actual, actual * 1.2, actual * 1.3])

    weights = ensemble.optimize_weights(predictions, actual)

    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > 0.99
