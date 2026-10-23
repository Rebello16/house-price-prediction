"""Tests for src/features.py: no NaNs, expected columns, correct maths, no leakage."""
import numpy as np
import pandas as pd
import pytest

from src.data import drop_outliers, get_split, load_raw
from src.features import (NOMINAL_COLS, ORDINAL_COLS, HousingCleaner,
                          build_preprocessor)

NEW_FEATURES = ["TotalSF", "HouseAge", "YearsSinceRemodel", "TotalBathrooms",
                "HasGarage", "HasPool", "HasFireplace", "PorchSF"]


@pytest.fixture(scope="module")
def split():
    return get_split()


@pytest.fixture(scope="module")
def fitted(split):
    X_train, _, _, _ = split
    return build_preprocessor().fit(X_train)


def test_outlier_rule_removes_exactly_two_rows():
    raw = load_raw()
    assert len(raw) - len(drop_outliers(raw)) == 2


def test_no_nans_after_transform_on_train_and_test(fitted, split):
    X_train, X_test, _, _ = split
    for part in (X_train, X_test):
        out = fitted.transform(part)
        assert not out.isna().any().any()
        assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_expected_columns_exist(fitted, split):
    cols = set(fitted.transform(split[1]).columns)
    assert set(NEW_FEATURES) <= cols
    assert set(ORDINAL_COLS) <= cols
    assert "Neighborhood_NAmes" in cols  # one-hot output
    assert "GarageYrBlt" not in cols and "Neighborhood" not in cols


def test_train_and_test_have_identical_columns(fitted, split):
    X_train, X_test, _, _ = split
    assert list(fitted.transform(X_train).columns) == list(fitted.transform(X_test).columns)


def test_every_text_column_is_encoded(split):
    """Guards against a column silently falling into the numeric branch."""
    X_train = split[0]
    text_cols = {c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])}
    assert text_cols <= set(ORDINAL_COLS) | set(NOMINAL_COLS)


def test_engineered_features_are_correct(fitted, split):
    row = split[0].iloc[[0]].copy()
    row[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]] = [800, 1000, 500]
    row[["YrSold", "YearBuilt", "YearRemodAdd"]] = [2008, 1990, 2000]
    row[["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]] = [2, 1, 1, 1]
    row[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]] = [10, 20, 30, 40]
    row[["PoolArea", "Fireplaces", "GarageArea"]] = [0, 2, 0]
    clean = fitted.named_steps["clean"].transform(row).iloc[0]
    assert clean["TotalSF"] == 2300
    assert clean["HouseAge"] == 18 and clean["YearsSinceRemodel"] == 8
    assert clean["TotalBathrooms"] == 4.0
    assert clean["PorchSF"] == 100
    assert (clean["HasPool"], clean["HasFireplace"], clean["HasGarage"]) == (0, 1, 0)


def test_house_sold_before_built_has_non_negative_age(fitted, split):
    row = split[0].iloc[[0]].copy()
    row[["YrSold", "YearBuilt", "YearRemodAdd"]] = [2007, 2008, 2008]
    clean = fitted.named_steps["clean"].transform(row).iloc[0]
    assert clean["HouseAge"] == 0 and clean["YearsSinceRemodel"] == 0


def test_absent_features_become_none_not_nan(fitted, split):
    row = split[0].iloc[[0]].copy()
    row[["GarageType", "PoolQC", "BsmtQual"]] = np.nan
    clean = fitted.named_steps["clean"].transform(row).iloc[0]
    assert (clean[["GarageType", "PoolQC", "BsmtQual"]] == "None").all()


def test_lot_frontage_uses_train_neighborhood_median_only(split):
    """Leakage check: learned stats equal the train medians and ignore test data."""
    X_train, X_test, _, _ = split
    cleaner = HousingCleaner().fit(X_train)
    expected = X_train.groupby("Neighborhood")["LotFrontage"].median()
    pd.testing.assert_series_equal(cleaner.lot_frontage_by_hood_, expected)

    before = cleaner.lot_frontage_by_hood_.copy()
    poisoned = X_test.copy()
    poisoned["LotFrontage"] = 9999.0
    cleaner.transform(poisoned)  # transform must not update what was learned
    pd.testing.assert_series_equal(cleaner.lot_frontage_by_hood_, before)

    row = X_train.iloc[[0]].copy()
    row["LotFrontage"] = np.nan
    filled = cleaner.transform(row)["LotFrontage"].iloc[0]
    assert filled == expected[row["Neighborhood"].iloc[0]]


def test_unseen_category_does_not_crash(fitted, split):
    row = split[1].iloc[[0]].copy()
    row["Neighborhood"] = "Atlantis"
    row["KitchenQual"] = "Platinum"
    out = fitted.transform(row)
    assert not out.isna().any().any()
    assert out["KitchenQual"].iloc[0] == -1


def test_skew_columns_are_chosen_on_train_only(split):
    X_train, X_test, _, _ = split
    pre = build_preprocessor().fit(X_train)
    skew = pre.named_steps["encode"].named_transformers_["num"].named_steps["skew"]
    chosen_before = skew.skewed_mask_.copy()
    pre.transform(X_test)
    assert (skew.skewed_mask_ == chosen_before).all()
    assert chosen_before.any()


def test_scaled_variant_is_standardised(split):
    X_train = split[0]
    out = build_preprocessor(scale=True).fit_transform(X_train)
    assert abs(out["TotalSF"].mean()) < 1e-6
    assert out["TotalSF"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)
