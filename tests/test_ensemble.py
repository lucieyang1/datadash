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
    })


def test_six_model_features_respect_track_answer_visibility():
    homes = make_sales(4)

    off_market = ensemble.make_features(
        data.hide_answers(homes, "off_market"), "off_market"
    )
    list_aware = ensemble.make_features(
        data.hide_answers(homes, "list_aware"), "list_aware"
    )

    assert "list_price" not in off_market
    assert "list_price" in list_aware
    assert not {"sold_price", "date_sold"}.intersection(off_market.columns)
    assert not {"sold_price", "date_sold"}.intersection(list_aware.columns)
    assert off_market["house_area"].notna().all()


@pytest.mark.parametrize("model_name", ensemble.MODEL_NAMES)
@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_each_model_fits_and_predicts(model_name, track):
    sales = make_sales()
    homes = data.hide_answers(sales.iloc[-5:], track)
    model = ensemble.make_pipeline(model_name, track, ensemble.TRIALS[model_name][0])

    features_frame = ensemble.make_features(data.hide_answers(sales, track), track)
    model.fit(features_frame, np.log(sales["sold_price"]))
    predictions = ensemble._predict_prices(model, model_name, homes, track)

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
