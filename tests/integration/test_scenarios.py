"""Synthetic scenarios drive honest findings (expected interpretations, not brittle numbers)."""

from __future__ import annotations

import numpy as np
import pytest

from calibcheck import synthetic
from calibcheck.analysis import analyze_predictions
from calibcheck.config import resolve_config
from calibcheck.data import load_predictions
from calibcheck.findings import BANNED_WORDS, generate_findings
from calibcheck.protocol import evaluate_recalibration


def _analyze(sc, **kw):
    cols = {"label": "y_true", "probs": sc.prob_columns, "id": "row_id",
            "groups": sc.group_columns}
    if sc.timestamp_column:
        cols["timestamp"] = sc.timestamp_column
    return analyze_predictions(sc.frame, columns=cols, **kw)


def codes(res):
    return [f.code for f in res.findings]


def test_headline_cohort_observed_rate_not_forced():
    sc = synthetic.headline()
    flagged = sc.frame[sc.frame.segment == "flagged"]
    rate = flagged.y_true.mean()
    assert rate != 0.65  # sampled, not forced
    assert abs(rate - 0.65) < 0.04
    res = _analyze(sc)
    top = res.models[0].binning.bins[-1]
    assert top.mean_pred == pytest.approx(0.9, abs=0.001)
    assert top.ci_low < 0.65 < top.ci_high or abs(top.observed_rate - 0.65) < 0.05
    assert "FINDING_OVERPREDICTION" in codes(res)


def test_headline_recalibration_judged_on_test_rows():
    sc = synthetic.headline(n=4000)
    df = sc.frame.drop(columns="latent_p_true")
    cfg = resolve_config({"columns": {"label": "y_true", "probs": ["p_model"], "id": "row_id"}})
    cal = load_predictions(df[df.split == "calibration"], cfg)
    test = load_predictions(df[df.split == "test"], cfg)
    res, arts = evaluate_recalibration(cal, test, cfg, model="p_model", methods=["isotonic"])
    q = np.interp(0.9, arts["isotonic"].params["x"], arts["isotonic"].params["y"])
    cal_rate = df[(df.split == "calibration") & (df.p_model == 0.9)].y_true.mean()
    assert abs(q - cal_rate) < 0.03  # learns roughly the calibration-cohort rate
    assert res.comparisons and all(c.n == test.n for c in res.comparisons)


def test_strong_ranking_poor_calibration():
    res = _analyze(synthetic.generate("strong_rank_poor_calib"))
    m = res.models[0]
    assert m.metrics["roc_auc"].value > 0.85
    assert m.metrics["ece"].value > 0.05
    assert "FINDING_RANKING_VS_CALIBRATION" in codes(res)


def test_constant_prevalence_predictor():
    res = _analyze(synthetic.generate("constant_prevalence"))
    m = res.models[0]
    assert m.binning.n_nonempty == 1
    assert m.metrics["roc_auc"].value == pytest.approx(0.5)
    assert "FINDING_CONSTANT_PREDICTOR" in codes(res)


def test_group_cancellation_detected():
    res = _analyze(synthetic.generate("group_cancellation"))
    assert abs(res.models[0].metrics["mean_error"].value) < 0.03
    assert "FINDING_GROUP_CANCELLATION" in codes(res)


def test_tiny_sample_flags_limited_evidence():
    res = _analyze(synthetic.generate("tiny_sample"), uncertainty={"bootstrap": True,
                                                                  "n_resamples": 300})
    assert "UNC_SMALL_SUPPORT" in codes(res)
    iv = res.models[0].metrics["brier"].interval
    assert iv.high - iv.low > 0.05


def test_near_identical_models_inconclusive():
    sc = synthetic.generate("near_calibrated", n=1500)
    df = sc.frame.copy()
    rng = np.random.default_rng(0)
    df["p_alt"] = np.clip(df.p_model + rng.normal(0, 0.002, len(df)), 0, 1)
    res = analyze_predictions(df, columns={"label": "y_true", "probs": ["p_model", "p_alt"],
                                           "id": "row_id"},
                              uncertainty={"bootstrap": True, "n_resamples": 500})
    assert "FINDING_COMPARISON_INCONCLUSIVE" in codes(res)
    assert not any("winner" in f.message and "No overall" not in f.message
                   for f in res.findings)


def test_clustered_declared_dependence_disables_intervals():
    sc = synthetic.generate("clustered")
    res = analyze_predictions(sc.frame, columns={"label": "y_true", "probs": ["p_model"],
                                                 "id": "row_id", "entity": "entity_id"},
                              dependence="entity_clustered",
                              uncertainty={"bootstrap": True, "n_resamples": 200})
    assert res.models[0].metrics["brier"].interval.reason == "UNAVAILABLE_DEPENDENCE"


def test_temporal_windows_descriptive():
    sc = synthetic.generate("temporal")
    res = _analyze(sc, time_window="M")
    tw = res.time_windows
    assert tw is not None and len(tw.rows) == 12
    assert tw.rows[-1].prevalence > tw.rows[0].prevalence
    assert "causal drift" in tw.interpretation


@pytest.mark.parametrize("name", sorted(synthetic.SCENARIOS))
def test_findings_vocabulary_and_determinism(name):
    sc = synthetic.generate(name)
    res = _analyze(sc)
    again = generate_findings(res)
    assert [f.model_dump() for f in again] == [f.model_dump() for f in res.findings]
    for f in res.findings:
        text = (f.message + " " + (f.caveat or "")).lower()
        for w in BANNED_WORDS:
            assert w not in text, (name, w, text)


def test_counterexample_files(tmp_path):
    files = synthetic.write_counterexamples(tmp_path)
    assert set(files) >= {"missing_outcomes.csv", "invalid_probabilities.csv",
                          "single_class.csv", "duplicate_ids.csv", "boundary_and_ties.csv"}
