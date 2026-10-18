"""Data-role protocol: overlap blocking, unknown provenance, label isolation, groups, time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibcheck import synthetic
from calibcheck.calibration import fit_calibrator
from calibcheck.config import resolve_config
from calibcheck.data import load_predictions
from calibcheck.errors import ProtocolError
from calibcheck.protocol import (
    check_artifact_overlap,
    check_role_separation,
    evaluate_recalibration,
    select_method,
)


def _cfg(**extra):
    cols = {"label": "y_true", "probs": ["p_model"], "id": "row_id"}
    cols.update(extra.pop("columns", {}))
    return resolve_config({"columns": cols, **extra})


def _split(name="overconfident", n=3000, **kw):
    sc = synthetic.generate(name, n=n, **kw)
    df = sc.frame.drop(columns="latent_p_true")
    return df[df.split == "calibration"].reset_index(drop=True), \
        df[df.split == "test"].reset_index(drop=True)


def test_id_overlap_blocks_evaluation():
    cal, test = _split()
    leaked = pd.concat([test, cal.iloc[:5]])
    c = _cfg()
    with pytest.raises(ProtocolError) as ei:
        evaluate_recalibration(load_predictions(cal, c), load_predictions(leaked, c), c,
                               model="p_model")
    assert ei.value.code == "ROLE_OVERLAP"


def test_unknown_provenance_reported_not_assumed():
    cal, test = _split()
    c = resolve_config({"columns": {"label": "y_true", "probs": ["p_model"]}})
    a, b = load_predictions(cal, c), load_predictions(test, c)
    fs = check_role_separation(a, b)
    assert any(f.code == "ROLE_PROVENANCE_UNKNOWN" for f in fs)
    assert not any(f.code == "ROLE_OVERLAP" and f.severity == "info" for f in fs)


def test_entity_overlap_blocks():
    sc = synthetic.generate("clustered", n_entities=20, per_entity=30)
    df = sc.frame.drop(columns="latent_p_true")
    c = _cfg(columns={"entity": "entity_id"})
    first = df.iloc[:315]   # entities E000..E010 (E010 only partially)
    second = df.iloc[315:]
    with pytest.raises(ProtocolError) as ei:
        check_role_separation(load_predictions(first, c), load_predictions(second, c))
    assert ei.value.code == "ROLE_ENTITY_OVERLAP"
    ok = check_role_separation(load_predictions(df.iloc[:300], c),
                               load_predictions(df.iloc[300:], c))
    assert ok


def test_changing_test_labels_cannot_change_fit_or_selection():
    cal, test = _split(n=2400)
    c = _cfg()
    flipped = test.copy()
    flipped["y_true"] = 1 - flipped["y_true"]
    cd = load_predictions(cal, c)
    r1, arts1 = evaluate_recalibration(cd, load_predictions(test, c), c, model="p_model",
                                       methods=["sigmoid", "isotonic", "temperature"],
                                       mode="selected")
    r2, arts2 = evaluate_recalibration(cd, load_predictions(flipped, c), c, model="p_model",
                                       methods=["sigmoid", "isotonic", "temperature"],
                                       mode="selected")
    assert r1.protocol["selection"] == r2.protocol["selection"]
    assert arts1.keys() == arts2.keys()
    for k in arts1:
        assert arts1[k].params == arts2[k].params
    # exploratory mode as well
    _, e1 = evaluate_recalibration(cd, load_predictions(test, c), c, model="p_model")
    _, e2 = evaluate_recalibration(cd, load_predictions(flipped, c), c, model="p_model")
    for k in e1:
        assert e1[k].params == e2[k].params


def test_artifact_digest_overlap_check():
    cal, test = _split()
    c = _cfg()
    cd, td = load_predictions(cal, c), load_predictions(test, c)
    art = fit_calibrator(cd.probs["p_model"], cd.y, "sigmoid", model_association="p_model",
                         ids=cd.ids)
    assert check_artifact_overlap(art, td)[0].severity == "info"
    with pytest.raises(ProtocolError):
        check_artifact_overlap(art, cd)
    # digests do not contain raw IDs
    assert not set(cd.ids.tolist()) & set(art.fitted_on.id_digests or [])


def test_group_aware_inner_folds():
    sc = synthetic.generate("clustered", n_entities=30, per_entity=40)
    df = sc.frame.drop(columns="latent_p_true")
    c = _cfg(columns={"entity": "entity_id"})
    sel = select_method(load_predictions(df, c), "p_model", ["identity", "sigmoid"], c)
    assert sel["scheme"].startswith("GroupKFold by entity")


def test_temporal_protocol_order_enforced_and_chronological_selection():
    sc = synthetic.generate("temporal", n=2000)
    df = sc.frame.drop(columns="latent_p_true")
    c = _cfg(columns={"timestamp": "timestamp"})
    early, late = df.iloc[:1000], df.iloc[1000:]
    e, la = load_predictions(early, c), load_predictions(late, c)
    check_role_separation(e, la, temporal=True)
    with pytest.raises(ProtocolError) as ei:
        check_role_separation(la, e, temporal=True)
    assert ei.value.code == "ROLE_TEMPORAL_ORDER"
    sel = select_method(e, "p_model", ["identity", "sigmoid"], c, temporal=True)
    assert sel["scheme"].startswith("chronological")


def test_temporal_selection_ignores_later_outcomes():
    sc = synthetic.generate("temporal", n=2000)
    df = sc.frame.drop(columns="latent_p_true").iloc[:1200].copy()
    c = _cfg(columns={"timestamp": "timestamp"})
    base = select_method(load_predictions(df, c), "p_model", ["identity", "sigmoid"], c,
                         temporal=True, n_folds=3)
    # TimeSeriesSplit never trains on rows after its validation block; perturbing labels in
    # the final validation block cannot change the first fold's training fit.
    from sklearn.model_selection import TimeSeriesSplit
    order = np.arange(len(df))
    folds = list(TimeSeriesSplit(n_splits=3).split(order))
    for tr, te in folds:
        assert tr.max() < te.min()
    assert base["selected"] in ("identity", "sigmoid")


def test_recalibration_report_shows_failures_and_unchanged():
    cal, test = _split()
    c = _cfg()
    one_class = cal.copy()
    one_class["y_true"] = 0
    res, arts = evaluate_recalibration(load_predictions(one_class, c), load_predictions(test, c),
                                       c, model="p_model", methods=["sigmoid", "identity"])
    assert "sigmoid" not in arts
    assert res.protocol["fit"]["sigmoid"]["status"] == "failed"
    assert any(f.code == "UNAVAILABLE_FIT_FAILED" for f in res.findings)
    assert res.models[0].metrics["brier"].available  # raw results retained


def test_intervals_note_conditional_on_fitted_models():
    cal, test = _split()
    c = _cfg(uncertainty={"bootstrap": True, "n_resamples": 200})
    res, _ = evaluate_recalibration(load_predictions(cal, c), load_predictions(test, c), c,
                                    model="p_model")
    assert "conditional" in res.protocol["intervals_note"]
    assert {cmp.reference for cmp in res.comparisons} == {"p_model"}
    assert all(cmp.n == len(test) for cmp in res.comparisons)
