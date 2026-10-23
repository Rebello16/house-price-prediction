"""Leak-free feature engineering for the Ames Housing data.

Everything that *learns* from data (neighborhood medians, imputer medians, one-hot
categories, which columns are skewed) happens in ``fit`` and is only ever called on
training data. Wrap the whole thing in one sklearn Pipeline and cross-validation /
the test set can never leak into the transformers.

Usage:
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)   # learns from train only
    X_test_t = pre.transform(X_test)         # reuses what was learned
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import validate_data

# ---------------------------------------------------------------- column groups
QUALITY = ["None", "Po", "Fa", "TA", "Gd", "Ex"]  # worst -> best

# Ordered categoricals: column -> categories from lowest to highest.
ORDINAL_ORDERS = {
    "ExterQual": QUALITY, "ExterCond": QUALITY, "BsmtQual": QUALITY,
    "BsmtCond": QUALITY, "HeatingQC": QUALITY, "KitchenQual": QUALITY,
    "FireplaceQu": QUALITY, "GarageQual": QUALITY, "GarageCond": QUALITY,
    "PoolQC": QUALITY,
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "CentralAir": ["N", "Y"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
}
ORDINAL_COLS = list(ORDINAL_ORDERS)

# Unordered categoricals -> one-hot. MSSubClass is a numeric *code*, so it is cast to str.
NOMINAL_COLS = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "Utilities",
    "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType",
    "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
    "MasVnrType", "Foundation", "Heating", "Electrical", "GarageType", "Fence",
    "MiscFeature", "SaleType", "SaleCondition",
]

# NaN here means "the house has no such feature" (verified in 01_eda), not "unknown".
NONE_IF_MISSING = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
]
ZERO_IF_MISSING = [
    "MasVnrArea", "GarageCars", "GarageArea", "TotalBsmtSF", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath",
]

DROP_COLS = ["GarageYrBlt"]  # r=0.83 with YearBuilt and undefined without a garage


# ---------------------------------------------------------------- transformers
class HousingCleaner(TransformerMixin, BaseEstimator):
    """Structural cleaning + feature creation.

    Learned in ``fit`` (train only): the median LotFrontage per Neighborhood and a
    global median fallback for neighborhoods unseen during training.
    Everything else is a deterministic rule, so it cannot leak.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = self._as_frame(X)
        self.columns_in_ = list(X.columns)
        self.lot_frontage_by_hood_ = X.groupby("Neighborhood")["LotFrontage"].median()
        self.lot_frontage_global_ = float(X["LotFrontage"].median())
        # _transform (not transform): sklearn's pandas-output wrapper needs these names
        self.feature_names_out_ = np.array(self._transform(X).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._as_frame(X)
        missing = set(self.columns_in_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        X = X[self.columns_in_].copy()

        X["MSSubClass"] = X["MSSubClass"].astype(str)
        for col in NONE_IF_MISSING:
            X[col] = X[col].astype(object).fillna("None")  # all-NaN columns arrive as float
        for col in ZERO_IF_MISSING:
            X[col] = X[col].fillna(0)

        # LotFrontage: neighborhood median (learned on train), else global median
        by_hood = X["Neighborhood"].map(self.lot_frontage_by_hood_)
        X["LotFrontage"] = X["LotFrontage"].fillna(by_hood).fillna(self.lot_frontage_global_)

        # ---- new features
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        X["HouseAge"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
        X["YearsSinceRemodel"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
        X["TotalBathrooms"] = (X["FullBath"] + 0.5 * X["HalfBath"]
                               + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"])
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        X["PorchSF"] = (X["OpenPorchSF"] + X["EnclosedPorch"]
                        + X["3SsnPorch"] + X["ScreenPorch"])
        return X.drop(columns=DROP_COLS)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousingCleaner expects a pandas DataFrame with named columns")
        return X


class SkewCorrector(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Apply log1p to numeric columns whose *training* skew exceeds ``threshold``.

    Binary flags (<=2 unique values) and columns with negative values are skipped.
    The column choice is learned in ``fit`` so test data cannot influence it.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True)
        df = pd.DataFrame(X)
        skew = df.skew().to_numpy()
        self.skewed_mask_ = (
            (np.abs(skew) > self.threshold)
            & (df.min().to_numpy() >= 0)
            & (df.nunique().to_numpy() > 2)
        )
        return self

    def transform(self, X):
        X = validate_data(self, X, reset=False).copy()
        X[:, self.skewed_mask_] = np.log1p(np.clip(X[:, self.skewed_mask_], 0, None))
        return X


# ---------------------------------------------------------------- pipeline
def _numeric_cols(X: pd.DataFrame) -> list[str]:
    """Numeric = every column that is not ordinal or nominal (works on cleaned data)."""
    return [c for c in X.columns if c not in ORDINAL_ORDERS and c not in NOMINAL_COLS]


def build_preprocessor(scale: bool = False) -> Pipeline:
    """Return the full, unfitted feature pipeline (raw DataFrame -> numeric DataFrame).

    scale=True adds a StandardScaler at the end; needed for Ridge, pointless for trees.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("skew", SkewCorrector()),
    ])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            categories=[ORDINAL_ORDERS[c] for c in ORDINAL_COLS],
            handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),  # e.g. the 1 missing Electrical
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    encode = ColumnTransformer(
        [("num", numeric, _numeric_cols),
         ("ord", ordinal, ORDINAL_COLS),
         ("nom", nominal, NOMINAL_COLS)],
        remainder="drop", verbose_feature_names_out=False,
    )
    steps = [("clean", HousingCleaner()), ("encode", encode)]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")
