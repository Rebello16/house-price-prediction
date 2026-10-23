"""Streamlit app: estimate an Ames, Iowa house price and see why.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `import src...` works when launched from anywhere

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import predict as P
from src.config import FIGURES_DIR

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"

QUALITY_LABELS = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical / average", "Fa": "Fair", "Po": "Poor", "None": "None"}
NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale", "BrkSide": "Brookside",
    "ClearCr": "Clear Creek", "CollgCr": "College Creek", "Crawfor": "Crawford", "Edwards": "Edwards",
    "Gilbert": "Gilbert", "IDOTRR": "Iowa DOT & Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa", "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames", "OldTown": "Old Town", "SWISU": "South & West of Iowa State", "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook", "Timber": "Timberland", "Veenker": "Veenker",
}

# (feature, label, widget, min, max, step, help). Chosen from the SHAP importance ranking.
SLIDERS = [
    ("OverallQual", "Overall quality (1-10)", 1, 10, 1, "Materials and finish. The single most important feature."),
    ("OverallCond", "Overall condition (1-10)", 1, 10, 1, "Current upkeep of the house."),
    ("1stFlrSF", "1st floor area (sq ft)", 300, 3000, 10, None),
    ("2ndFlrSF", "2nd floor area (sq ft)", 0, 2000, 10, "0 for a single-storey house."),
    ("TotalBsmtSF", "Basement area (sq ft)", 0, 3000, 10, "0 = no basement."),
    ("FullBath", "Full bathrooms above ground", 0, 4, 1, None),
    ("HalfBath", "Half bathrooms above ground", 0, 2, 1, None),
    ("GarageCars", "Garage capacity (cars)", 0, 4, 1, "0 = no garage."),
    ("YearBuilt", "Year built", 1870, 2010, 1, None),
    ("YearRemodAdd", "Year remodeled", 1950, 2010, 1, "Same as year built if never remodeled."),
    ("LotArea", "Lot size (sq ft)", 1300, 30000, 100, None),
]
SELECTS = [
    ("Neighborhood", "Neighborhood", None),
    ("KitchenQual", "Kitchen quality", QUALITY_LABELS),
]

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return P.load_model()


@st.cache_resource
def get_info() -> dict:
    return P.load_feature_info()


@st.cache_resource
def get_metrics() -> dict:
    return P.load_metrics()


def usd(value: float) -> str:
    return f"${value:,.0f}"


# ---------------------------------------------------------------- UI pieces
def sidebar_inputs(info: dict) -> dict:
    """Render the input form; returns {feature: value} for the features the user controls."""
    values = {}
    st.sidebar.header("Describe the house")
    with st.sidebar.form("inputs"):
        for feat, label, lo, hi, step, help_ in SLIDERS:
            default = int(min(max(info["defaults"][feat], lo), hi))
            values[feat] = st.slider(label, lo, hi, default, step, help=help_)
        for feat, label, names in SELECTS:
            options = info["categorical"][feat]
            fmt = (lambda c: f"{NEIGHBORHOODS.get(c, c)} ({c})") if feat == "Neighborhood" \
                else (lambda c, names=names: names.get(c, c))
            values[feat] = st.selectbox(label, options, index=options.index(info["defaults"][feat]), format_func=fmt)
        st.session_state["submitted"] = st.form_submit_button("Predict price", type="primary", use_container_width=True)
    st.sidebar.caption("Every feature not listed here is set to the training-set median (numbers) or "
                       "most common value (categories).")
    return values


def effects_chart(expl: pd.DataFrame):
    """Horizontal bars: % change in price caused by each feature, largest first."""
    data = expl.iloc[::-1]
    labels = [f"{f} = {v:,.0f}" if isinstance(v, (int, float)) and pd.notna(v) else f"{f} = {v}"
              for f, v in zip(data["feature"], data["value"])]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(labels, data["pct_effect"], color=[BLUE if v >= 0 else ORANGE for v in data["pct_effect"]], height=0.65)
    ax.axvline(0, color=GRAY, lw=1)
    span = max(data["pct_effect"].abs().max(), 1)
    for y, v in enumerate(data["pct_effect"]):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), y, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=GRAY)
    ax.set_xlim(-span * 1.3, span * 1.3)
    ax.set_xlabel("effect on price vs. an average house")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=GRAY)
    ax.xaxis.label.set_color(GRAY)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.tight_layout()
    return fig


def prediction_tab(user_inputs: dict, model, metrics: dict) -> None:
    if not st.session_state.get("submitted") and "last" not in st.session_state:
        st.info("Set the house details in the sidebar and press **Predict price**.")
        return
    if st.session_state.get("submitted"):
        st.session_state["last"] = user_inputs
    inputs = st.session_state["last"]

    try:
        price = P.predict_price(inputs, model)
        low, high = P.predict_range(price, metrics)
        expl = P.explain_prediction(inputs, model, top_n=10)
    except P.InputValidationError as exc:
        st.error(f"Invalid input: {exc}")
        return
    except Exception as exc:  # last-resort guard so the app never shows a raw traceback
        st.error(f"Something went wrong while predicting ({type(exc).__name__}: {exc}).")
        return

    c1, c2 = st.columns([1, 1])
    c1.metric("Estimated sale price", usd(round(price, -2)))
    c2.metric(f"Likely range ({metrics['prediction_interval']['level']:.0%} of homes)", f"{usd(round(low, -2))} - {usd(round(high, -2))}")
    st.caption(
        f"The range comes from the model's cross-validated errors (typical error {metrics['meta']['best_model']}: "
        f"RMSLE {metrics['models'][metrics['meta']['best_model']]['test']['rmsle']:.3f} on unseen houses, "
        f"i.e. roughly +/-{metrics['models'][metrics['meta']['best_model']]['test']['rmsle'] * 100:.0f}%). "
        "It is a statistical estimate for Ames, Iowa in 2006-2010, not an appraisal."
    )

    st.subheader("What pushed this estimate up or down?")
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(effects_chart(expl), transparent=True)
    with right:
        st.markdown(
            f"- The model starts from an **average house of about {usd(round(expl.attrs['base_price'], -2))}**.\n"
            "- Blue bars raise the estimate and orange bars lower it, shown as approximate % effects.\n"
            "- Effects come from SHAP values and are computed in log-price space, so they multiply "
            "rather than add; they will not sum exactly to the % gap shown above.\n"
            "- Features are **correlated** (a better kitchen usually means a better house overall), so "
            "changing just one input may move the price less than you expect."
        )


def performance_tab(metrics: dict) -> None:
    meta, best = metrics["meta"], metrics["meta"]["best_model"]
    test = metrics["models"][best]["test"]
    st.subheader(f"Best model: {best}")
    cols = st.columns(4)
    cols[0].metric("Test RMSLE", f"{test['rmsle']:.4f}", help="RMSE of log prices; lower is better. About the relative error.")
    cols[1].metric("Test RMSE", usd(test["rmse"]))
    cols[2].metric("Test MAE", usd(test["mae"]))
    cols[3].metric("Test R²", f"{test['r2']:.3f}")
    st.caption(f"Held-out test set: {meta['n_test']} houses never used for training or tuning. "
               f"Cross-validation: {meta['cv_folds']} folds on {meta['n_train']} training houses. Seed {meta['random_state']}.")

    rows = []
    for name, res in metrics["models"].items():
        rows.append({"Model": name, "CV RMSLE": res["cv"]["rmsle"], "CV RMSE ($)": res["cv"]["rmse"],
                     "CV MAE ($)": res["cv"]["mae"], "CV R²": res["cv"]["r2"],
                     "Test RMSLE": res["test"]["rmsle"], "Test RMSE ($)": res["test"]["rmse"],
                     "Test MAE ($)": res["test"]["mae"], "Test R²": res["test"]["r2"]})
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table.style.format({c: "{:,.0f}" if "$" in c else "{:.4f}" for c in table.columns})
                 .highlight_min(subset=["CV RMSLE", "Test RMSLE"], color="#cfe3fa"), use_container_width=True)
    with st.expander("Tuned hyperparameters (Optuna)"):
        st.json(metrics["best_params"])

    figures = [("pred_vs_actual", "Predicted vs actual"), ("residuals", "Residuals"),
               ("error_by_price_band", "Error by price range"), ("feature_importance", "Feature importance"),
               ("shap_summary", "SHAP summary"), ("shap_waterfall", "One prediction explained")]
    for i in range(0, len(figures), 2):
        for col, (name, caption) in zip(st.columns(2), figures[i:i + 2]):
            path = FIGURES_DIR / f"{name}.png"
            if path.exists():
                col.image(str(path), caption=caption, use_container_width=True)


# ---------------------------------------------------------------- main
def main() -> None:
    st.title("🏠 House Price Predictor")
    st.caption("Gradient-boosted model trained on the Ames, Iowa housing data (2006-2010).")
    try:
        model, info, metrics = get_model(), get_info(), get_metrics()
    except FileNotFoundError as exc:
        st.error(f"Model files are missing. Run `python -m src.train` first.\n\n{exc}")
        st.stop()

    user_inputs = sidebar_inputs(info)
    tab_predict, tab_perf = st.tabs(["Price estimate", "Model performance"])
    with tab_predict:
        prediction_tab(user_inputs, model, metrics)
    with tab_perf:
        performance_tab(metrics)


main()
