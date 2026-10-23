"""Tests for src/predict.py: positive prices, sensible behaviour, strict validation."""
import numpy as np
import pytest

from src.predict import (InputValidationError, build_row, explain_prediction,
                         load_feature_info, predict_price, predict_range)

TYPICAL = {"OverallQual": 6, "GrLivArea": 1500, "Neighborhood": "NAmes"}


def test_returns_positive_float():
    price = predict_price(TYPICAL)
    assert isinstance(price, float) and price > 0


def test_empty_input_uses_defaults_and_is_positive():
    assert predict_price({}) > 0


def test_better_house_costs_more():
    """A coherent poor vs luxury house (correlated features move together)."""
    poor = {"OverallQual": 3, "ExterQual": "Fa", "KitchenQual": "Fa", "BsmtQual": "Fa",
            "1stFlrSF": 700, "2ndFlrSF": 0, "TotalBsmtSF": 600, "GarageCars": 1, "YearBuilt": 1925}
    luxury = {"OverallQual": 9, "ExterQual": "Ex", "KitchenQual": "Ex", "BsmtQual": "Ex",
              "1stFlrSF": 1900, "2ndFlrSF": 800, "TotalBsmtSF": 1900, "GarageCars": 3, "YearBuilt": 2007}
    assert predict_price(luxury) > 2.5 * predict_price(poor)


def test_bigger_living_area_does_not_lower_price():
    assert predict_price({"GrLivArea": 2400}) > predict_price({"GrLivArea": 900})


def test_range_brackets_the_estimate():
    price = predict_price(TYPICAL)
    low, high = predict_range(price)
    assert 0 < low < price < high


def test_consistency_rules_fill_dependent_fields():
    info = load_feature_info()
    row = build_row({"GarageCars": 0, "TotalBsmtSF": 0, "YearBuilt": 2008}).iloc[0]
    assert row["GarageType"] == "None" and row["GarageArea"] == 0
    assert row["BsmtQual"] == "None"
    assert row["YearRemodAdd"] >= row["YearBuilt"]
    assert build_row({"GarageCars": 3}).iloc[0]["GarageArea"] == pytest.approx(3 * info["garage_area_per_car"])
    floors = build_row({"1stFlrSF": 1000, "2ndFlrSF": 500}).iloc[0]
    assert floors["GrLivArea"] >= 1500


@pytest.mark.parametrize("bad", [
    {"NotAFeature": 1},                 # unknown field
    {"GrLivArea": -5},                  # negative area
    {"GrLivArea": float("nan")},        # not finite
    {"GrLivArea": "big"},               # wrong type
    {"GrLivArea": True},                # bool is not a number here
    {"GrLivArea": 1_000_000},           # absurd value
    {"OverallQual": 11},                # outside 1-10
    {"Neighborhood": "Atlantis"},       # unseen category
    {"KitchenQual": "Platinum"},        # unseen quality level
])
def test_invalid_input_is_rejected(bad):
    with pytest.raises(InputValidationError):
        predict_price(bad)


def test_non_dict_is_rejected():
    with pytest.raises(InputValidationError):
        predict_price([("OverallQual", 5)])


def test_explanation_adds_up_to_the_prediction():
    expl = explain_prediction(TYPICAL, top_n=500)  # every feature, so the sum is exact
    log_pred = np.log1p(expl.attrs["base_price"]) + expl["log_effect"].sum()
    assert np.expm1(log_pred) == pytest.approx(predict_price(TYPICAL), rel=1e-3)
    assert len(explain_prediction(TYPICAL, top_n=8)) == 8
