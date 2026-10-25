"""Explore listing words associated with sale prices after controlling for home details."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data
import features

NUMERIC_COLUMNS = [
    "bedrooms", "washrooms", "parking_total", "garage",
    "house_area", "lot_front", "lot_depth", "latitude", "longitude",
    "months", "month_sin", "month_cos",
]
CATEGORICAL_COLUMNS = ["house_type_name", "house_style", "municipality", "community"]


def word_associations(listings, min_df=100, alpha=10.0):
    """Fit a regularized log-price model and return its word coefficients."""
    frame = listings.copy()
    frame["remarks"] = frame["remarks"].fillna("").astype(str)
    frame["house_area"] = features.square_feet(frame["house_area"])
    listed = pd.to_datetime(frame["date_listed"], errors="coerce")
    frame["months"] = (listed - pd.Timestamp("2021-01-01")).dt.days / 30.44
    month_angle = 2 * np.pi * (listed.dt.month - 1) / 12
    frame["month_sin"] = np.sin(month_angle)
    frame["month_cos"] = np.cos(month_angle)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                ]),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="(missing)")),
                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_COLUMNS,
            ),
            (
                "words",
                CountVectorizer(
                    binary=True,
                    min_df=min_df,
                    max_df=0.8,
                    stop_words="english",
                    ngram_range=(1, 1),
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                    dtype=np.float32,
                ),
                "remarks",
            ),
        ],
        sparse_threshold=1.0,
    )
    model = Pipeline([
        ("features", preprocessing),
        ("regressor", Ridge(alpha=alpha, solver="lsqr")),
    ])
    model.fit(frame, np.log(frame["sold_price"]))

    vectorizer = preprocessing.named_transformers_["words"]
    word_coefficients = model.named_steps["regressor"].coef_[-len(vectorizer.vocabulary_):]
    words = vectorizer.get_feature_names_out()
    word_counts = np.asarray(vectorizer.transform(frame["remarks"]).sum(axis=0)).ravel().astype(int)
    return pd.DataFrame({
        "word": words,
        "listings_with_word": word_counts,
        "log_price_coefficient": word_coefficients,
        "estimated_price_association_pct": 100 * np.expm1(word_coefficients),
    }).sort_values("log_price_coefficient", ascending=False, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-df", type=int, default=100,
                        help="minimum listings a word must appear in (default: 100)")
    parser.add_argument("--output", type=Path, default=Path("outputs/word_associations.csv"),
                        help="CSV output path (default: outputs/word_associations.csv)")
    args = parser.parse_args()
    if args.min_df < 2:
        parser.error("--min-df must be at least 2")

    associations = word_associations(data.load("listings"), min_df=args.min_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    associations.to_csv(args.output, index=False)

    columns = ["word", "listings_with_word", "estimated_price_association_pct"]
    print("Top positive adjusted associations:")
    print(associations.head(20)[columns].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nTop negative adjusted associations:")
    print(associations.tail(20).sort_values("log_price_coefficient")[columns].to_string(
        index=False, float_format=lambda x: f"{x:+.2f}"
    ))
    print(f"\nSaved all word associations to {args.output}")


if __name__ == "__main__":
    main()
