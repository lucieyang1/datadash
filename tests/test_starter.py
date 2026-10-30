"""Quick checks on tiny made-up data (nothing is downloaded). Run from the repo folder:

    pytest
"""
import numpy as np
import pandas as pd
import pytest

import check_submission
from convlstm import ConvLSTMPriceModel, SpatialSaleHistory
import data
import features
import models
import run
from analyze_words import word_associations


def make_listings(n, seed=0, first_id=0):
    """A tiny made-up listings table with the same columns as the real one."""
    rng = np.random.default_rng(seed)
    sold = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 516, n), unit="D")
    price = rng.uniform(4e5, 2e6, n).round(-2)
    return pd.DataFrame({
        "ml_num": [f"SYN-{first_id + i:07d}" for i in range(n)],
        "sold_price": price,
        "list_price": (price * rng.uniform(0.9, 1.1, n)).round(-2),
        "date_listed": sold - pd.to_timedelta(rng.integers(1, 60, n), unit="D"),
        "date_sold": sold,
        "days_on_market": 10,
        "house_type_name": rng.choice(["Condo Apt", "Detached"], n),
        "house_style": "Apartment",
        "bedrooms": rng.integers(1, 5, n),
        "washrooms": rng.integers(1, 4, n),
        "parking_total": rng.integers(0, 3, n),
        "garage": 0.0,
        "house_area": rng.choice(["600-699", "1200"], n),
        "lot_front": np.nan,
        "lot_depth": np.nan,
        "municipality": "Toronto",
        "community": "Testville",
        "latitude": 43.7 + rng.normal(0, 0.05, n),
        "longitude": -79.4 + rng.normal(0, 0.05, n),
        "external_avm_estimate": price,
        "external_avm_estimate_date": sold,
        "remarks": "A home.",
    })


# ---- the nearest-sales baseline never looks into the future ----

def test_nearest_sales_ignores_later_sales():
    listings = make_listings(400)
    homes = data.hide_answers(make_listings(50, seed=1, first_id=1000), "off_market")
    homes = homes[homes["date_listed"] > pd.Timestamp("2025-03-01")]   # so earlier sales exist
    with_later_sales = models.NearestSalesBaseline(k=10).fit(listings).predict(homes)
    for i, (_, home) in enumerate(homes.iterrows()):
        earlier_only = listings[listings["date_sold"] < home["date_listed"]]
        alone = models.NearestSalesBaseline(k=10).fit(earlier_only).predict(homes.loc[[home.name]])
        assert alone[0] == with_later_sales[i]


def test_a_later_sale_next_door_is_not_used():
    listings = make_listings(400)
    home = listings.iloc[[0]].assign(ml_num="SYN-8888888", latitude=44.0, date_listed=pd.Timestamp("2026-01-15"))
    next_door = home.assign(ml_num="SYN-9999999", date_sold=pd.Timestamp("2026-02-01"), sold_price=1e9)
    model = models.NearestSalesBaseline(k=1).fit(pd.concat([listings, next_door]))
    assert model.predict(data.hide_answers(home, "off_market"))[0] < 1e9


def test_nearest_sales_uses_the_nearest_homes_of_the_same_type():
    def sales(n, kind, latitude, price, first_id):
        rows = make_listings(n, first_id=first_id)
        return rows.assign(house_type_name=kind, latitude=latitude, longitude=-79.4,
                           sold_price=price, date_sold=pd.Timestamp("2025-01-01"))
    listings = pd.concat([sales(10, "Detached", 43.70, 500_000, 0),     # near, same type
                          sales(10, "Condo Apt", 43.70, 2_000_000, 100),  # near, other type
                          sales(10, "Detached", 43.90, 100_000, 200)])    # far, same type
    home = listings.iloc[[0]].assign(date_listed=pd.Timestamp("2026-01-01"))
    model = models.NearestSalesBaseline(k=10).fit(listings)
    assert model.predict(data.hide_answers(home, "off_market"))[0] == 500_000


def test_spatiotemporal_gam_fits_and_predicts_for_both_tracks(monkeypatch):
    listings = make_listings(120)
    homes = data.hide_answers(make_listings(8, seed=1, first_id=1000), "off_market")
    monkeypatch.setattr(models, "MODEL", "spatiotemporal_gam")

    model = models.make_model("off_market").fit(listings)
    predictions = model.predict(homes)

    assert isinstance(model, models.SpatiotemporalGAM)
    assert predictions.shape == (len(homes),)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_spatiotemporal_gam_month_spline_is_cyclic():
    model = models.SpatiotemporalGAM()
    january = model.month_spline.fit_transform(np.arange(1, 13).reshape(-1, 1))[:1]
    january_next_year = model.month_spline.transform([[13.0]])

    assert january.shape == (1, 12)
    np.testing.assert_allclose(january, january_next_year)


@pytest.mark.parametrize(
    ("model_name", "model_type", "rotated"),
    [
        ("xgboost", models.XGBoostModel, False),
        ("xgboost_rotated", models.RotatedCoordinateXGBoostModel, True),
        ("xgboost_community_comps", models.CommunityComparableXGBoostModel, True),
        (
            "lightgbm_community_comps",
            models.CommunityComparableLightGBMModel,
            True,
        ),
        (
            "lightgbm_nearby_comps_safe",
            models.SafeNearbyComparableLightGBMModel,
            True,
        ),
    ],
)
@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_xgboost_fits_and_predicts_for_both_tracks(
    track, model_name, model_type, rotated, monkeypatch
):
    listings = make_listings(80)
    homes = data.hide_answers(make_listings(6, seed=2, first_id=1000), track)
    homes.loc[homes.index[0], "remarks"] = "LANDSCAPED!"
    monkeypatch.setattr(models, "MODEL", model_name)

    model = models.make_model(track).fit(listings)
    predictions = model.predict(homes)

    assert isinstance(model, model_type)
    assert predictions.shape == (len(homes),)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()
    assert ("list_price" in model._values(homes)[0]) == (track == "list_aware")
    assert not any(column.startswith("word_") for column in model._values(homes)[0].columns)
    comparable_features = model._values(homes)[0]
    if model_name in {"xgboost_community_comps", "lightgbm_community_comps"}:
        assert "community_comparable_median_price" in comparable_features
        assert "nearby_comparable_median_price" not in comparable_features
    else:
        assert "nearby_comparable_median_price" in comparable_features
    assert ("coordinate_rotated_45_x" in model._values(homes)[0]) == rotated
    assert ("coordinate_rotated_45_y" in model._values(homes)[0]) == rotated


def test_rotated_xgboost_coordinates_use_local_longitude_scale():
    homes = make_listings(3)
    model = models.RotatedCoordinateXGBoostModel("off_market")
    model.comparables.fit(homes)

    numeric, _ = model._values(homes)

    scale = np.sqrt(0.5)
    x = homes["longitude"] * 0.72
    y = homes["latitude"]
    np.testing.assert_allclose(numeric["coordinate_rotated_45_x"], (x + y) * scale)
    np.testing.assert_allclose(numeric["coordinate_rotated_45_y"], (y - x) * scale)


def test_remarks_word_features_are_case_insensitive_and_match_whole_words():
    remarks = pd.Series([
        "A landscaped courtyard with a chef's kitchen.",
        "A studio, compact layout.",
        "Landscaping nearby, no garden note.",
        None,
    ])

    words = features.remarks_word_features(remarks)

    assert words["word_landscaped"].tolist() == [1, 0, 0, 0]
    assert words["word_courtyard"].tolist() == [1, 0, 0, 0]
    assert words["word_chef"].tolist() == [1, 0, 0, 0]
    assert words["word_studio"].tolist() == [0, 1, 0, 0]
    assert words["word_compact"].tolist() == [0, 1, 0, 0]


def test_nearby_comparable_features_only_use_prior_nearby_same_type_sales():
    sales = make_listings(4)
    sales["house_type_name"] = "Detached"
    sales["latitude"] = 43.70
    sales["longitude"] = -79.40
    sales["date_sold"] = pd.to_datetime([
        "2024-12-05", "2024-12-08", "2025-01-25", "2024-12-08"
    ])
    sales["sold_price"] = [500_000, 700_000, 9_000_000, 2_000_000]
    sales.loc[sales.index[3], "longitude"] = -79.70

    homes = make_listings(1, first_id=2000)
    homes["date_listed"] = pd.Timestamp("2025-01-20")
    homes["latitude"] = 43.70
    homes["longitude"] = -79.40
    homes["house_type_name"] = "Detached"

    result = models.NearbyComparableFeatures().fit(sales).transform(homes).iloc[0]

    assert result["nearby_comparable_median_price"] == 600_000
    assert result["nearby_comparable_count"] == 2
    assert result["nearby_comparable_nearest_km"] >= 0
    assert result["nearby_comparable_median_price_per_sqft"] > 0


def test_nearby_comparable_features_exclude_sales_from_listing_month():
    sales = make_listings(2)
    sales["house_type_name"] = "Detached"
    sales["latitude"] = 43.70
    sales["longitude"] = -79.40
    sales["date_sold"] = pd.to_datetime(["2024-12-15", "2025-01-10"])
    sales["sold_price"] = [500_000, 900_000]
    home = make_listings(1, first_id=3000)
    home["house_type_name"] = "Detached"
    home["date_listed"] = pd.Timestamp("2025-01-20")
    home["latitude"] = 43.70
    home["longitude"] = -79.40

    result = models.NearbyComparableFeatures().fit(sales).transform(home).iloc[0]

    assert result["nearby_comparable_median_price"] == 500_000
    assert result["nearby_comparable_count"] == 1


def test_community_comps_use_only_past_sales_of_same_community_and_type():
    sales = make_listings(5)
    sales["house_type_name"] = "Detached"
    sales["community"] = ["A", "A", "A", "B", "A"]
    sales["date_sold"] = pd.to_datetime([
        "2024-12-05", "2024-12-08", "2025-01-10", "2024-12-08", "2024-12-08"
    ])
    sales["sold_price"] = [500_000, 700_000, 9_000_000, 2_000_000, 1_000_000]
    sales.loc[sales.index[4], "house_type_name"] = "Condo Apt"

    homes = make_listings(2, first_id=2000)
    homes["date_listed"] = pd.Timestamp("2025-01-20")
    homes["community"] = ["A", "B"]
    homes["house_type_name"] = "Detached"

    result = models.CommunityComparableFeatures().fit(sales).transform(homes)

    assert result.iloc[0]["community_comparable_median_price"] == 600_000
    assert result.iloc[0]["community_comparable_count"] == 2
    assert result.iloc[0]["community_comparable_median_price_per_sqft"] > 0
    assert result.iloc[1]["community_comparable_median_price"] == 2_000_000


def test_convlstm_spatial_history_uses_only_months_before_listing():
    sales = make_listings(2)
    sales["house_type_name"] = "Detached"
    sales["latitude"] = 43.70
    sales["longitude"] = -79.40
    sales["date_sold"] = pd.to_datetime(["2024-12-15", "2025-01-10"])
    sales["sold_price"] = [500_000, 9_000_000]
    home = make_listings(1, first_id=3000)
    home["house_type_name"] = "Detached"
    home["date_listed"] = pd.Timestamp("2025-01-20")
    home["latitude"] = 43.70
    home["longitude"] = -79.40

    history = SpatialSaleHistory().fit(sales)
    patch = history.patches(home, np.array([0]))

    assert patch.shape == (1, 12, 3, 7, 7)
    assert patch[0, -1, 0, 3, 3] == pytest.approx(np.log(2) / 4)
    assert patch[0, -1, 1, 3, 3] == pytest.approx(np.log(500_000) - 13.3, abs=1e-5)


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_convlstm_model_fits_and_predicts(track):
    listings = make_listings(40)
    homes = data.hide_answers(make_listings(3, seed=3, first_id=4000), track)
    model = ConvLSTMPriceModel(track)
    model.epochs = 1
    model.batch_size = 16

    predictions = model.fit(listings).predict(homes)

    assert predictions.shape == (len(homes),)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_word_associations_finds_price_correlated_words():
    listings = make_listings(100)
    listings["sold_price"] = np.where(
        np.arange(len(listings)) < 50, 500_000, 1_500_000
    )
    listings["remarks"] = np.where(
        np.arange(len(listings)) < 50, "ordinary home", "premium home"
    )

    associations = word_associations(listings, min_df=5)
    premium = associations.set_index("word").loc["premium"]
    ordinary = associations.set_index("word").loc["ordinary"]

    assert premium["estimated_price_association_pct"] > 0
    assert ordinary["estimated_price_association_pct"] < 0
    assert premium["listings_with_word"] == 50


# ---- check_submission rejects bad files ----

IDS = ["SYN-0000001", "SYN-0000002", "SYN-0000003"]
GOOD = "ml_num,predicted_price\nSYN-0000001,500000\nSYN-0000002,600000\nSYN-0000003,700000\n"
BAD_FILES = {
    "wrong column name": GOOD.replace("predicted_price", "price"),
    "extra column": GOOD.replace("predicted_price\n", "predicted_price,x\n"),
    "a home missing": "ml_num,predicted_price\nSYN-0000001,500000\nSYN-0000002,600000\n",
    "a home twice": GOOD + "SYN-0000003,700000\n",
    "an id from another file": GOOD + "SYN-0000004,700000\n",
    "the other track's homes": "ml_num,predicted_price\nSYN-0000007,500000\nSYN-0000008,600000\n",
    "empty price": GOOD.replace("700000", ""),
    "price is text": GOOD.replace("700000", "seven hundred"),
    "negative price": GOOD.replace("700000", "-700000"),
    "zero price": GOOD.replace("700000", "0"),
    "infinite price": GOOD.replace("700000", "inf"),
    "empty file": "",
}


def test_check_accepts_a_good_file(tmp_path):
    path = tmp_path / "submission.csv"
    path.write_text(GOOD)
    assert check_submission.check_file(path, IDS) == []


@pytest.mark.parametrize("problem", BAD_FILES)
def test_check_rejects_a_bad_file(tmp_path, problem):
    path = tmp_path / "submission.csv"
    path.write_text(BAD_FILES[problem])
    assert check_submission.check_file(path, IDS) != []


# ---- the report computes the metrics correctly ----

def test_metrics_on_known_errors():
    # Errors of -10%, +10%, +20% and 0%.
    predictions = pd.DataFrame({"sold_price": [100.0] * 4, "predicted_price": [90.0, 110.0, 120.0, 100.0]})
    scores = run.metrics(predictions)
    assert scores["n"] == 4
    assert scores["mape"] == 10.0                  # mean of 10, 10, 20, 0
    assert scores["mdape"] == 10.0                 # median of 0, 10, 10, 20
    assert scores["median_signed_error"] == 5.0    # median of -10, 0, 10, 20
    low, high = scores["mape_95ci"]
    assert low <= 10.0 <= high


# ---- every command runs end to end on tiny files ----

def test_whole_pipeline_on_tiny_data(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    make_listings(1500).to_parquet(folder / "listings.parquet")
    later = make_listings(40, seed=2, first_id=5000)
    data.hide_answers(later.iloc[:20], "off_market").to_parquet(folder / "test_off_market.parquet")
    data.hide_answers(later.iloc[20:], "list_aware").to_parquet(folder / "test_list_aware.parquet")
    monkeypatch.setenv("DATA_DIR", str(folder))
    monkeypatch.setattr(run, "ROOT", tmp_path)

    run.validate()
    run.report()
    run.predict()
    assert run.check()
    experiments = list((tmp_path / "outputs" / "experiments").iterdir())
    assert len(experiments) == 1
    assert (experiments[0] / "validation_predictions.csv").exists()
    assert (experiments[0] / "reports" / "validation_report.md").exists()
    assert (experiments[0] / "figures" / "error_by_price_range.png").exists()


def test_experiment_names_do_not_overwrite_existing_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    path = run.experiment_dir("first-run", create=True)
    marker = path / "keep.txt"
    marker.write_text("existing results")

    with pytest.raises(FileExistsError):
        run.experiment_dir("first-run", create=True)

    assert marker.read_text() == "existing results"


@pytest.mark.parametrize("name", ["../outside", "with spaces", ""])
def test_experiment_names_reject_unsafe_or_invalid_values(name):
    with pytest.raises(ValueError):
        run.experiment_dir(name)
