"""Project-wide constants: paths, seeds and column names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "ames_housing.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

TARGET = "SalePrice"
ID_COL = "Id"
