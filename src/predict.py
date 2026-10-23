"""Predict a house price (and explain it) from a dict of property features.

    from src.predict import predict_price
    predict_price({"OverallQual": 7, "GrLivArea": 1800, "Neighborhood": "CollgCr"})

Only the features you pass are overridden; every other feature falls back to the
training-set median (numeric) or mode (categorical) stored in models/feature_info.json.
Bad input raises ``InputValidationError`` instead of silently producing nonsense.
"""
import json
import math
from functools import lru_cache
from numbers import Real

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.features import NOMINAL_COLS

MODEL_PATH = MODELS_DIR / "house_price_pipeline.joblib"
INFO_PATH = MODELS_DIR / "feature_info.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Numeric fields with a natural domain tighter than "observed range +- margin".
HARD_RANGES = {"OverallQual": (1, 10), "OverallCond": (1, 10), "MoSold": (1, 12)}
RANGE_MARGIN = 0.25  # allow values up to 25% of the observed span beyond train min/max


class InputValidationError(ValueError):
    """Raised when the input dict contains unknown fields or out-of-range values."""


# ---------------------------------------------------------------- loading (cached)
@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train first:  python -m src.train")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_info() -> dict:
    return json.loads(INFO_PATH.read_text())


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


# ---------------------------------------------------------------- validation
def validate_inputs(inputs: dict, info: dict | None = None) -> dict:
    """Check field names, types and ranges. Returns a cleaned copy (MSSubClass -> str)."""
    info = info or load_feature_info()
    if not isinstance(inputs, dict):
        raise InputValidationError(f"inputs must be a dict, got {type(inputs).__name__}")
    unknown = sorted(set(inputs) - set(info["defaults"]))
    if unknown:
        raise InputValidationError(f"Unknown feature(s): {unknown}")

    clean = {}
    for name, value in inputs.items():
        if name in info["categorical"]:
            value = str(int(value)) if isinstance(value, Real) and not isinstance(value, bool) else str(value)
            if value not in info["categorical"][name]:
                raise InputValidationError(f"{name}={value!r} is not valid; choose one of {info['categorical'][name]}")
        else:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise InputValidationError(f"{name} must be a finite number, got {value!r}")
            lo, hi = _allowed_range(name, info)
            if not lo <= value <= hi:
                raise InputValidationError(f"{name}={value} is outside the allowed range [{lo:g}, {hi:g}]")
        clean[name] = value
    return clean


def _allowed_range(name: str, info: dict) -> tuple[float, float]:
    if name in HARD_RANGES:
        return HARD_RANGES[name]
    lo, hi = info["numeric"][name]["min"], info["numeric"][name]["max"]
    margin = RANGE_MARGIN * (hi - lo)
    return (max(0.0, lo - margin) if lo >= 0 else lo - margin), hi + margin


# ---------------------------------------------------------------- row building
def build_row(inputs: dict, info: dict | None = None) -> pd.DataFrame:
    """Validated user inputs + defaults + consistency rules -> one raw-format row.

    Consistency rules only touch fields the caller did NOT set, so a house with 0 garage
    cars does not keep the "typical" garage quality, a bigger house gets a bigger living
    area, and a house cannot be remodelled before it was built.
    """
    info = info or load_feature_info()
    given = validate_inputs(inputs, info)
    row = {**info["defaults"], **given}

    def derive(field: str, value) -> None:
        if field not in given:
            row[field] = value

    if "GrLivArea" not in given and ("1stFlrSF" in given or "2ndFlrSF" in given):
        derive("GrLivArea", row["1stFlrSF"] + row["2ndFlrSF"] + row["LowQualFinSF"])
    if "GarageCars" in given:
        derive("GarageArea", row["GarageCars"] * info["garage_area_per_car"])
    if row["GarageCars"] == 0:
        for col in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
            derive(col, "None")
        derive("GarageArea", 0.0)
    if row["TotalBsmtSF"] == 0:
        for col in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
            derive(col, "None")
        for col in ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"):
            derive(col, 0.0)
    if row["Fireplaces"] == 0:
        derive("FireplaceQu", "None")
    derive("YearRemodAdd", max(row["YearRemodAdd"], row["YearBuilt"]))

    # same column order as the training data
    return pd.DataFrame([row], columns=list(info["defaults"]))


# ---------------------------------------------------------------- prediction
def predict_price(inputs: dict, model=None) -> float:
    """Estimated sale price in USD for the given features (validated)."""
    model = model or load_model()
    price = float(model.predict(build_row(inputs))[0])
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError(f"Model returned an invalid price: {price}")
    return price


def predict_range(price: float, metrics: dict | None = None) -> tuple[float, float]:
    """80% prediction band from out-of-fold residuals (log scale, so it is multiplicative)."""
    band = (metrics or load_metrics())["prediction_interval"]
    return price * float(np.exp(band["lower_log"])), price * float(np.exp(band["upper_log"]))


# ---------------------------------------------------------------- explanation
def feature_group(column: str) -> str:
    """Map a pipeline output column back to the original feature (one-hot -> its parent)."""
    for nominal in sorted(NOMINAL_COLS, key=len, reverse=True):
        if column.startswith(nominal + "_"):
            return nominal
    return column


def group_shap(columns, shap_values: np.ndarray) -> pd.Series:
    """Sum SHAP values of one-hot columns so each original feature gets one number."""
    return pd.Series(shap_values, index=[feature_group(c) for c in columns]).groupby(level=0).sum()


def explain_prediction(inputs: dict, model=None, top_n: int = 10) -> pd.DataFrame:
    """What pushed this prediction up or down? Returns the top_n features by |effect|.

    SHAP values are additive in log-price space, so each is also shown as an
    approximate % change in price (exp(shap) - 1). ``.attrs['base_price']`` is the
    model's average prediction before any feature is considered.
    """
    return explain_row(build_row(inputs), model, top_n)


def explain_row(row: pd.DataFrame, model=None, top_n: int = 10) -> pd.DataFrame:
    """Like explain_prediction but for an already-built one-row raw DataFrame
    (skips user-input validation, so real dataset rows with NaNs work too)."""
    import shap  # lazy: heavy import, only needed for explanations

    model = model or load_model()
    pipe = model.regressor_
    pre, booster = pipe.named_steps["pre"], pipe.named_steps["model"]
    encoded = pre.transform(row)
    explainer = shap.TreeExplainer(booster)
    contrib = group_shap(encoded.columns, explainer.shap_values(encoded)[0])
    shown = pre.named_steps["clean"].transform(row).iloc[0]  # readable values (pre log-transform)

    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    out = pd.DataFrame({
        "feature": top.index,
        "value": [shown.get(f, np.nan) for f in top.index],
        "log_effect": top.values,
    })
    out["pct_effect"] = np.expm1(out["log_effect"]) * 100
    out.attrs["base_price"] = float(np.expm1(np.ravel(explainer.expected_value)[0]))
    return out.head(top_n).reset_index(drop=True)
