"""Train, tune, compare and save the house price models.

    python -m src.train               # full run (~10-15 min, mostly Optuna)
    python -m src.train --trials 10   # quick run

Design rules that keep the numbers honest:
  * the target is log1p-transformed *inside* the model (TransformedTargetRegressor), so
    every model takes raw features and returns dollars;
  * the preprocessing pipeline is refit inside every CV fold (no leakage);
  * tuning and model selection use cross-validation on the training split only;
    the held-out test set is touched once, at the very end, for reporting.
"""
import argparse
import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import (MODELS_DIR, N_FOLDS, RANDOM_STATE, REPORTS_DIR,
                        TEST_SIZE)
from src.data import get_split
from src.evaluate import regression_metrics
from src.features import (NOMINAL_COLS, ORDINAL_ORDERS, build_preprocessor)

MODEL_PATH = MODELS_DIR / "house_price_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
INFO_PATH = MODELS_DIR / "feature_info.json"


# ---------------------------------------------------------------- model factory
def make_model(estimator, scale: bool = False) -> TransformedTargetRegressor:
    """preprocessing -> estimator, trained on log1p(price) and predicting dollars."""
    pipe = Pipeline([("pre", build_preprocessor(scale=scale)), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def make_xgb(params: dict | None = None) -> TransformedTargetRegressor:
    return make_model(XGBRegressor(**(params or {}), tree_method="hist",
                                   random_state=RANDOM_STATE, n_jobs=-1))


def make_lgbm(params: dict | None = None) -> TransformedTargetRegressor:
    return make_model(LGBMRegressor(**(params or {}), subsample_freq=1,
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))


def baseline_models() -> dict:
    """Linear baselines, a forest, and the two boosters with *library defaults*."""
    return {
        "Linear Regression": make_model(LinearRegression(), scale=True),
        "Ridge": make_model(RidgeCV(alphas=np.logspace(-1, 2.5, 30)), scale=True),
        "Random Forest": make_model(RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
        "XGBoost (default)": make_xgb(),
        "LightGBM (default)": make_lgbm(),
    }


# ---------------------------------------------------------------- cross-validation
def cross_validate(model, X: pd.DataFrame, y: pd.Series, n_folds: int = N_FOLDS):
    """Manual K-fold CV. Returns (mean fold metrics, out-of-fold dollar predictions).

    clone() gives a fresh, unfitted copy per fold, so the whole preprocessing pipeline
    is refit on each training fold and the validation fold stays unseen.
    """
    kfold = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    folds = []
    for train_idx, val_idx in kfold.split(X):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fitted.predict(X.iloc[val_idx])
        folds.append(regression_metrics(y.iloc[val_idx], oof[val_idx]))
    summary = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    summary["rmsle_std"] = float(np.std([f["rmsle"] for f in folds]))
    return summary, oof


# ---------------------------------------------------------------- Optuna tuning
def _xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def _lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 32),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def tune(name: str, space, factory, X, y, n_trials: int) -> optuna.Study:
    """Minimise 5-fold CV RMSLE with Optuna's TPE sampler (seeded)."""
    def objective(trial):
        summary, _ = cross_validate(factory(space(trial)), X, y)
        return summary["rmsle"]

    def progress(study, trial):
        if (trial.number + 1) % 10 == 0:
            print(f"  [{name}] trial {trial.number + 1}/{n_trials}  best CV RMSLE = {study.best_value:.4f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    return study


# ---------------------------------------------------------------- app metadata
def build_feature_info(X_train: pd.DataFrame, fitted) -> dict:
    """Defaults, ranges and allowed categories, saved so the app needs no raw data.

    Defaults come from the *cleaned* training data, so a feature that is usually
    absent (pool, alley...) defaults to "None" rather than to its rarest real value.
    """
    cleaner = fitted.regressor_.named_steps["pre"].named_steps["clean"]
    cleaned = cleaner.transform(X_train)
    defaults, numeric, categorical = {}, {}, {}
    for col in X_train.columns:
        src = cleaned[col] if col in cleaned else X_train[col]
        if col in ORDINAL_ORDERS or col in NOMINAL_COLS:
            src = src.dropna().astype(str)  # astype(str) keeps NaN in pandas 3 (e.g. Electrical)
            present = set(src)
            order = ORDINAL_ORDERS.get(col) or sorted(present)
            categorical[col] = [v for v in order if v in present]
            defaults[col] = str(src.mode().iloc[0])
        else:
            raw = X_train[col].dropna()
            numeric[col] = {"min": float(raw.min()), "max": float(raw.max())}
            defaults[col] = float(src.median())
    has_garage = X_train["GarageCars"] > 0
    per_car = float((X_train.loc[has_garage, "GarageArea"] / X_train.loc[has_garage, "GarageCars"]).median())
    return {"defaults": defaults, "numeric": numeric, "categorical": categorical,
            "garage_area_per_car": per_car}


# ---------------------------------------------------------------- main
def main(n_trials: int = 50) -> None:
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = get_split()
    print(f"train = {len(X_train)} rows, test = {len(X_test)} rows (test set untouched until the end)\n")

    models = baseline_models()
    cv_results, oof_preds = {}, {}
    for name, model in models.items():
        cv_results[name], oof_preds[name] = cross_validate(model, X_train, y_train)
        print(f"CV  {name:<20} RMSLE = {cv_results[name]['rmsle']:.4f}")

    # ---- Optuna
    best_params = {}
    for label, space, factory in [("XGBoost", _xgb_space, make_xgb), ("LightGBM", _lgbm_space, make_lgbm)]:
        print(f"\nTuning {label}: {n_trials} trials x {N_FOLDS}-fold CV")
        study = tune(label, space, factory, X_train, y_train, n_trials)
        best_params[label] = study.best_params
        name = f"{label} (tuned)"
        models[name] = factory(study.best_params)
        cv_results[name], oof_preds[name] = cross_validate(models[name], X_train, y_train)
        print(f"  best params: {study.best_params}")
        print(f"CV  {name:<20} RMSLE = {cv_results[name]['rmsle']:.4f}")

    # ---- model selection uses CV only
    tuned = [n for n in models if n.endswith("(tuned)")]
    best_name = min(tuned, key=lambda n: cv_results[n]["rmsle"])
    print(f"\nBest tuned model by CV RMSLE: {best_name}")

    # ---- held-out test evaluation of every model (fit on the full training split)
    test_results, fitted_models = {}, {}
    for name, model in models.items():
        fitted_models[name] = clone(model).fit(X_train, y_train)
        test_results[name] = regression_metrics(y_test, fitted_models[name].predict(X_test))
    final = fitted_models[best_name]

    # ---- prediction band from out-of-fold residuals (train only, no test peeking)
    resid = np.log1p(y_train.to_numpy()) - np.log1p(oof_preds[best_name])
    interval = {"level": 0.8, "lower_log": float(np.quantile(resid, 0.10)),
                "upper_log": float(np.quantile(resid, 0.90))}

    table = pd.DataFrame({
        n: {**{f"cv_{k}": v for k, v in cv_results[n].items()},
            **{f"test_{k}": v for k, v in test_results[n].items()}}
        for n in models
    }).T
    table.to_csv(REPORTS_DIR / "model_comparison.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:,.4f}".format):
        print("\n", table[["cv_rmsle", "cv_rmse", "cv_mae", "cv_r2", "test_rmsle", "test_rmse", "test_mae", "test_r2"]])

    metrics = {
        "meta": {"random_state": RANDOM_STATE, "n_train": len(X_train), "n_test": len(X_test),
                 "test_size": TEST_SIZE, "cv_folds": N_FOLDS, "optuna_trials": n_trials,
                 "best_model": best_name, "train_minutes": round((time.time() - t0) / 60, 1)},
        "models": {n: {"cv": cv_results[n], "test": test_results[n]} for n in models},
        "best_params": best_params,
        "prediction_interval": interval,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    INFO_PATH.write_text(json.dumps(build_feature_info(X_train, final), indent=2))
    joblib.dump(final, MODEL_PATH, compress=3)
    print(f"\nSaved {MODEL_PATH.name}, {METRICS_PATH.name}, {INFO_PATH.name}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model")
    main(parser.parse_args().trials)
