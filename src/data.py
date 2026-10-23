"""Load the Ames Housing data, caching a local copy so the project works offline."""
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.config import (ID_COL, RANDOM_STATE, RAW_DIR, RAW_FILE, TARGET,
                        TEST_SIZE)


def download_raw() -> pd.DataFrame:
    """Download Ames Housing from OpenML (no login) and save it to data/raw/."""
    bunch = fetch_openml(name="house_prices", as_frame=True)
    df = bunch.frame
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)
    return df


def load_raw() -> pd.DataFrame:
    """Return the raw dataset, reading the local CSV if it exists, else downloading.

    keep_default_na=False matters: pandas would otherwise turn the *real* category
    string "None" (e.g. MasVnrType == "None") into NaN. Only empty cells are missing.
    """
    if RAW_FILE.exists():
        return pd.read_csv(RAW_FILE, keep_default_na=False, na_values=[""])
    return download_raw()


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the 2 partial-sale mega-houses found in 01_eda (Ids 524 and 1299).

    Narrow on purpose: big houses that sold for a lot (Ids 692, 1183) are real and stay.
    """
    is_outlier = (df["GrLivArea"] > 4000) & (df[TARGET] < 300_000)
    return df.loc[~is_outlier].reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw frame into features X (without Id) and target y (SalePrice)."""
    return df.drop(columns=[TARGET, ID_COL], errors="ignore"), df[TARGET]


def get_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Deterministic train/test split (outliers removed first): X_train, X_test, y_train, y_test.

    Every module (train, evaluate, notebooks, tests) calls this so they all see the same rows.
    """
    X, y = split_xy(drop_outliers(load_raw()))
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


if __name__ == "__main__":
    data = load_raw()
    print(f"Loaded {data.shape[0]} rows x {data.shape[1]} cols from {RAW_FILE}")
