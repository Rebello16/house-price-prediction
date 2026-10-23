"""Headless smoke test of the Streamlit app (no browser needed)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Price estimate", "Model performance"]


def test_predict_button_shows_a_dollar_estimate():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.button[0].click().run()
    assert not at.exception and not at.error
    estimate = next(m for m in at.metric if m.label == "Estimated sale price")
    assert estimate.value.startswith("$")
