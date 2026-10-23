"""Metrics, diagnostic plots and SHAP explainability for the trained model.

    python -m src.evaluate     # writes reports/figures/*.png and adds an "evaluation"
                               # section to models/metrics.json
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

from src.config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
PRICE_BANDS = [0, 100_000, 150_000, 200_000, 300_000, np.inf]
BAND_LABELS = ["< $100k", "$100-150k", "$150-200k", "$200-300k", "> $300k"]
usd_k = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE / MAE in dollars, RMSLE (RMSE of log1p prices) and R2."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmsle": float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRAY, "axes.grid": True, "grid.color": "#e6e5e1",
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titleweight": "bold",
        "axes.titlesize": 11, "font.size": 10,
    })


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def log_residuals(y_true, y_pred) -> np.ndarray:
    """log(actual / predicted): +0.10 means the house sold ~10% above the prediction."""
    return np.log1p(np.asarray(y_true, float)) - np.log1p(np.asarray(y_pred, float))


# ---------------------------------------------------------------- tables
def error_by_price_band(y_true, y_pred) -> pd.DataFrame:
    """Accuracy per price band: mean absolute % error, signed % bias and RMSLE."""
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    df["band"] = pd.cut(df["y"], PRICE_BANDS, labels=BAND_LABELS, right=False)
    df["ape"] = (df["p"] - df["y"]).abs() / df["y"] * 100
    df["bias"] = (df["p"] - df["y"]) / df["y"] * 100
    df["sq_log"] = log_residuals(df["y"], df["p"]) ** 2
    out = df.groupby("band", observed=True).agg(n=("y", "size"), mape=("ape", "mean"),
                                                bias=("bias", "mean"), sq_log=("sq_log", "mean"))
    out["rmsle"] = np.sqrt(out.pop("sq_log"))
    return out


def worst_errors(X: pd.DataFrame, y_true, y_pred, n: int = 10) -> pd.DataFrame:
    """The n predictions with the largest relative error, with context columns."""
    res = log_residuals(y_true, y_pred)
    out = X[["Neighborhood", "OverallQual", "GrLivArea", "YearBuilt", "SaleCondition"]].copy()
    out["actual"], out["predicted"] = np.asarray(y_true, float), np.asarray(y_pred, float)
    out["pct_error"] = (out["predicted"] - out["actual"]) / out["actual"] * 100
    return out.iloc[np.argsort(-np.abs(res))[:n]].round({"actual": 0, "predicted": 0, "pct_error": 1})


# ---------------------------------------------------------------- plots
def plot_pred_vs_actual(y_true, y_pred, rmsle: float):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    big_miss = np.abs(log_residuals(y_true, y_pred)) > 0.25  # >~25% off
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true[~big_miss], y_pred[~big_miss], s=16, alpha=0.5, color=BLUE, edgecolors="none", label="within 25%")
    ax.scatter(y_true[big_miss], y_pred[big_miss], s=40, color=ORANGE, edgecolors="white", linewidths=1, label="off by >25%", zorder=3)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, color=GRAY, lw=1.5, ls="--", label="perfect prediction")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual price", ylabel="Predicted price",
           title=f"Predicted vs actual (test set, RMSLE {rmsle:.3f})")
    ax.xaxis.set_major_formatter(usd_k)
    ax.yaxis.set_major_formatter(usd_k)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "pred_vs_actual")
    return fig


def plot_residuals(y_true, y_pred, rmsle: float):
    res, y_pred = log_residuals(y_true, y_pred), np.asarray(y_pred, float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    far = np.abs(res) > 2 * rmsle
    ax1.scatter(y_pred[~far], res[~far], s=14, alpha=0.5, color=BLUE, edgecolors="none")
    ax1.scatter(y_pred[far], res[far], s=40, color=ORANGE, edgecolors="white", linewidths=1, zorder=3,
                label="beyond 2x RMSLE")
    ax1.axhline(0, color=GRAY, lw=1.5)
    for sign in (-2, 2):
        ax1.axhline(sign * rmsle, color=GRAY, lw=1, ls=":")
    ax1.set(xlabel="Predicted price", ylabel="log(actual / predicted)", title="Residuals vs predicted price")
    ax1.xaxis.set_major_formatter(usd_k)
    ax1.legend(frameon=False)
    ax2.hist(res, bins=30, color=BLUE, edgecolor="white", orientation="horizontal")
    ax2.set(xlabel="houses", title="Residual distribution")
    ax2.axhline(0, color=GRAY, lw=1.5)
    ax2.set_ylim(ax1.get_ylim())
    fig.tight_layout()
    _save(fig, "residuals")
    return fig


def plot_error_by_band(bands: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(bands))
    ax1.bar(x, bands["mape"], color=BLUE, width=0.65)
    for i, (m, n) in enumerate(zip(bands["mape"], bands["n"])):
        ax1.text(i, m + 0.2, f"{m:.1f}%\n(n={n})", ha="center", fontsize=9, color="#52514e")
    ax1.set(xticks=x, xticklabels=bands.index, ylabel="mean absolute % error", title="Error by price range")
    ax1.set_ylim(0, bands["mape"].max() * 1.3)
    ax1.grid(axis="x", visible=False)
    colors = [BLUE if b >= 0 else ORANGE for b in bands["bias"]]
    ax2.bar(x, bands["bias"], color=colors, width=0.65)
    ax2.axhline(0, color=GRAY, lw=1)
    ax2.set(xticks=x, xticklabels=bands.index, ylabel="mean signed % error (pred - actual)",
            title="Bias by price range (blue = over-predicts)")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "error_by_price_band")
    return fig


def plot_importance(importance: pd.Series, top: int = 15):
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.65)
    ax.set(xlabel="mean |SHAP value| (log-price units)",
           title=f"Top {top} features by importance (one-hot columns grouped)")
    ax.grid(axis="y", visible=False)
    _save(fig, "feature_importance")
    return fig


def plot_shap_summary(shap_values: np.ndarray, encoded: pd.DataFrame):
    import shap
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, encoded, max_display=15, show=False)
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP summary: how each feature value moves the prediction", fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save(fig, "shap_summary")
    return fig


def plot_shap_waterfall(expl: pd.DataFrame, base_log: float, actual: float, predicted: float):
    """Waterfall for one house, using original (grouped) features."""
    import shap
    explanation = shap.Explanation(values=expl["log_effect"].to_numpy(), base_values=base_log,
                                   data=expl["value"].to_numpy(dtype=object), feature_names=list(expl["feature"]))
    plt.figure()
    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle(f"One prediction explained: actual \\${actual:,.0f}, predicted \\${predicted:,.0f} (log-price units)",
                 fontweight="bold", fontsize=11, y=1.02)
    _save(fig, "shap_waterfall")
    return fig


# ---------------------------------------------------------------- main
def main() -> None:
    from src.data import get_split
    from src.predict import (explain_row, feature_group, load_model)

    apply_style()
    model = load_model()
    _, X_test, _, y_test = get_split()
    pred = model.predict(X_test)
    metrics = json.loads((REPORTS_DIR.parent / "models" / "metrics.json").read_text())
    rmsle = regression_metrics(y_test, pred)["rmsle"]

    plot_pred_vs_actual(y_test, pred, rmsle)
    plot_residuals(y_test, pred, rmsle)
    bands = error_by_price_band(y_test, pred)
    plot_error_by_band(bands)

    # ---- SHAP on the whole test set
    import shap
    pre, booster = model.regressor_.named_steps["pre"], model.regressor_.named_steps["model"]
    encoded = pre.transform(X_test)
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(encoded)
    grouped = pd.DataFrame(sv, columns=encoded.columns).T.groupby(feature_group).sum().T
    importance = grouped.abs().mean().sort_values(ascending=False)
    plot_importance(importance)
    plot_shap_summary(sv, encoded)

    # ---- waterfall for a typical house: the test house closest to the median price
    i = int((np.abs(y_test.to_numpy() - y_test.median())).argmin())
    expl = explain_row(X_test.iloc[[i]], model, top_n=500)
    plot_shap_waterfall(expl, float(np.log1p(expl.attrs["base_price"])), float(y_test.iloc[i]), float(pred[i]))

    # ---- numbers for the written analysis
    lo, hi = (metrics["prediction_interval"][k] for k in ("lower_log", "upper_log"))
    res = log_residuals(y_test, pred)
    coverage = float(np.mean((res >= lo) & (res <= hi)))
    worst = worst_errors(X_test, y_test, pred)
    by_condition = (pd.DataFrame({"cond": X_test["SaleCondition"].to_numpy(), "sq": res ** 2})
                    .groupby("cond")["sq"].agg(["size", "mean"]).assign(rmsle=lambda d: np.sqrt(d.pop("mean"))))
    metrics["evaluation"] = {
        "test": regression_metrics(y_test, pred),
        "interval_coverage_test": coverage,
        "error_by_price_band": bands.round(4).reset_index().astype({"band": str}).to_dict("records"),
        "rmsle_by_sale_condition": by_condition.round(4).reset_index().to_dict("records"),
        "worst_10": worst.reset_index(drop=True).to_dict("records"),
        "importance_top20": importance.head(20).round(4).to_dict(),
        "share_of_test_houses_within_10pct": float(np.mean(np.abs(pred / y_test.to_numpy() - 1) < 0.10)),
    }
    (REPORTS_DIR.parent / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    worst.to_csv(REPORTS_DIR / "worst_predictions.csv")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nError by price band:\n", bands.round(3))
        print("\nWorst 10 predictions:\n", worst)
        print("\nRMSLE by SaleCondition:\n", by_condition.round(3))
        print("\nTop 15 features (grouped mean |SHAP|):\n", importance.head(15).round(3))
        print(f"\n80% band coverage on test: {coverage:.1%}")
        print(f"within 10% of actual: {metrics['evaluation']['share_of_test_houses_within_10pct']:.1%}")


if __name__ == "__main__":
    main()
