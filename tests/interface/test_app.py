"""Streamlit interface: main journey, stale-result invalidation, recalibration, clearing."""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

APP = str(Path(__file__).resolve().parents[2] / "src" / "calibcheck" / "app" / "streamlit_app.py")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _button(at: AppTest, label_start: str):
    return next(b for b in at.button if b.label.startswith(label_start))


@pytest.mark.slow
def test_demo_journey_and_stale_invalidation():
    at = _app()
    assert "CalibCheck" in at.title[0].value
    assert any("Configure the analysis" in i.value for i in at.info)
    _button(at, "Run analysis").click().run()
    assert not at.exception, at.exception
    assert any("FINDING_OVERPREDICTION" in w.value for w in at.warning)
    assert any(m.label == "Rows analysed" for m in at.metric)
    # changing a setting hides previous numbers instead of showing stale results
    at.selectbox(key="strategy").set_value("quantile").run()
    assert any("Results are hidden" in w.value for w in at.warning)
    assert not any(m.label == "Rows analysed" for m in at.metric)


@pytest.mark.slow
def test_demo_recalibration_and_clear():
    at = _app()
    _button(at, "Run analysis").click().run()
    _button(at, "Fit on calibration rows").click().run()
    assert not at.exception, at.exception
    texts = " ".join(i.value for i in at.info)
    assert "p_model + sigmoid" in texts or "p_model + isotonic" in texts
    _button(at, "Clear loaded data").click().run()
    assert not at.exception
    assert any("Configure the analysis" in i.value for i in at.info)


@pytest.mark.slow
def test_dependence_declaration_disables_intervals():
    at = _app()
    at.radio(key="dependence").set_value("entity_clustered").run()
    _button(at, "Run analysis").click().run()
    assert not at.exception, at.exception
    frames = [df.value for df in at.dataframe]
    joined = " ".join(str(f.to_dict()) for f in frames)
    assert "UNAVAILABLE_DEPENDENCE" in joined
