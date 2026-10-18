"""The input contract: strict rejection, explicit drop policy, label semantics, IDs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibcheck.analysis import analyze_predictions, inspect_predictions
from calibcheck.config import resolve_config
from calibcheck.data import align_by_id, load_predictions
from calibcheck.errors import ConfigError, InputValidationError


def cfg(**kw):
    base = {"columns": {"label": "y", "probs": ["p"], "id": "id"}}
    return resolve_config(base, kw)


def codes(exc: InputValidationError) -> set[str]:
    return {f.code for f in exc.findings} | {exc.code}


@pytest.mark.parametrize("text,code", [
    ("id,y,p\n1,1,1.2\n2,0,0.3\n", "PROB_OUT_OF_RANGE"),
    ("id,y,p\n1,1,-0.1\n2,0,0.3\n", "PROB_OUT_OF_RANGE"),
    ("id,y,p\n1,1,inf\n2,0,0.3\n", "PROB_NONFINITE"),
    ("id,y,p\n1,1,abc\n2,0,0.3\n", "PROB_NONNUMERIC"),
    ("id,y,p\n1,1,\n2,0,0.3\n", "PROB_MISSING"),
    ("id,y,p\n1,1,NaN\n2,0,0.3\n", "PROB_MISSING"),
    ("id,y,p\n1,,0.4\n2,0,0.3\n", "LABEL_MISSING"),
    ("id,y,p\n1,1,0.4\n1,0,0.3\n", "ID_DUPLICATE"),
    ("id,y,p\n,1,0.4\n2,0,0.3\n", "ID_MISSING"),
])
def test_invalid_values_rejected(write_csv, text, code):
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("f.csv", text), cfg())
    assert code in codes(ei.value)


def test_structural_errors(write_csv):
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("e.csv", ""), cfg())
    assert ei.value.code == "INPUT_EMPTY"
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("h.csv", "id,y,p\n"), cfg())
    assert ei.value.code == "INPUT_EMPTY"
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("d.csv", "id,y,p,p\n1,1,0.2,0.3\n"), cfg())
    assert ei.value.code == "INPUT_DUPLICATE_HEADER"
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("m.csv", "id,y,q\n1,1,0.2\n"), cfg())
    assert ei.value.code == "INPUT_MISSING_COLUMN"


def test_ambiguous_mapping_rejected(write_csv):
    c = resolve_config({"columns": {"label": "y", "probs": ["p"], "id": "p"}})
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("a.csv", "id,y,p\n1,1,0.2\n"), c)
    assert ei.value.code == "INPUT_AMBIGUOUS_MAPPING"


def test_probability_columns_never_guessed(write_csv):
    c = resolve_config({"columns": {"label": "y", "probs": []}})
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("a.csv", "y,p,q\n1,0.2,0.3\n"), c)
    assert ei.value.code == "INPUT_NO_PROB_COLUMNS"


def test_percentages_not_silently_rescaled(write_csv):
    path = write_csv("pc.csv", "id,y,p\n1,1,80\n2,0,35\n")
    with pytest.raises(InputValidationError) as ei:
        load_predictions(path, cfg())
    msg = " ".join(f.message for f in ei.value.findings)
    assert "percent" in msg
    data = load_predictions(path, cfg(transforms={"p": "percent"}))
    assert np.allclose(data.probs["p"], [0.8, 0.35])
    assert any(f.code == "TRANSFORM_APPLIED" for f in data.findings)


def test_exact_zero_and_one_accepted(write_csv):
    data = load_predictions(write_csv("z.csv", "id,y,p\n1,1,1\n2,0,0\n3,1,0.5\n"), cfg())
    assert data.probs["p"].tolist() == [1.0, 0.0, 0.5]
    assert any(f.code == "PROB_ENDPOINTS" for f in data.findings)


def test_drop_policy_records_exclusions_and_common_population(write_csv):
    text = "id,y,p,q\n1,1,0.9,0.8\n2,0,1.5,0.1\n3,,0.2,0.3\n4,0,0.1,\n5,1,0.7,0.6\n"
    c = resolve_config({"columns": {"label": "y", "probs": ["p", "q"], "id": "id"},
                        "missing_policy": "drop"})
    data = load_predictions(write_csv("dr.csv", text), c)
    assert data.n == 2 and data.ids.tolist() == ["1", "5"]
    reasons = {(e.reason, e.field): e.count for e in data.exclusions}
    assert reasons[("PROB_OUT_OF_RANGE", "p")] == 1
    assert reasons[("LABEL_MISSING", "y")] == 1
    assert reasons[("PROB_MISSING", "q")] == 1
    res = analyze_predictions(data, c)
    assert res.dataset.n_rows_input == 5 and res.dataset.n_rows_excluded == 3
    assert {m.n for m in res.models} == {2}


def test_leading_zero_ids_preserved(write_csv):
    data = load_predictions(write_csv("z.csv", "id,y,p\n007,1,0.5\n0007,0,0.4\n"), cfg())
    assert data.ids.tolist() == ["007", "0007"]


def test_label_rules(write_csv):
    yes_no = write_csv("yn.csv", "id,y,p\n1,yes,0.9\n2,no,0.2\n")
    with pytest.raises(InputValidationError) as ei:
        load_predictions(yes_no, cfg())
    assert ei.value.code == "LABEL_POSITIVE_UNDECLARED"
    d = load_predictions(yes_no, cfg(positive_label="yes"))
    assert d.y.tolist() == [1, 0] and d.negative_label == "no"
    # declaring the *other* class as positive flips y but never p
    d2 = load_predictions(yes_no, cfg(positive_label="no"))
    assert d2.y.tolist() == [0, 1] and d2.probs["p"].tolist() == [0.9, 0.2]
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("mc.csv", "id,y,p\n1,a,0.9\n2,b,0.2\n3,c,0.1\n"),
                         cfg(positive_label="a"))
    assert ei.value.code == "LABEL_NOT_BINARY"
    with pytest.raises(InputValidationError) as ei:
        load_predictions(yes_no, cfg(positive_label="maybe"))
    assert ei.value.code == "LABEL_POSITIVE_NOT_FOUND"


def test_numeric_label_canonicalisation(write_csv):
    d = load_predictions(write_csv("f.csv", "id,y,p\n1,1.0,0.9\n2,0.0,0.2\n"), cfg())
    assert d.y.tolist() == [1, 0] and d.positive_label == "1"


def test_single_class_valid_with_warning(write_csv):
    d = load_predictions(write_csv("s.csv", "id,y,p\n1,0,0.1\n2,0,0.3\n"), cfg())
    assert any(f.code == "LABEL_SINGLE_CLASS" for f in d.findings)
    res = analyze_predictions(d, cfg())
    m = res.models[0]
    assert m.metrics["brier"].available and m.metrics["log_loss"].available
    assert m.metrics["roc_auc"].status == "unavailable"


def test_two_valued_column_warns(write_csv):
    d = load_predictions(write_csv("t.csv", "id,y,p\n1,1,1\n2,0,0\n3,1,1\n"), cfg())
    assert any(f.code == "PROB_TWO_VALUED" for f in d.findings)


def test_positional_ids_flagged(write_csv):
    c = resolve_config({"columns": {"label": "y", "probs": ["p"]}})
    d = load_predictions(write_csv("n.csv", "y,p\n1,0.4\n0,0.3\n"), c)
    assert d.id_kind == "positional"
    assert any(f.code == "ID_POSITIONAL" for f in d.findings)


def test_group_missing_visible_and_high_cardinality(write_csv):
    c = resolve_config({"columns": {"label": "y", "probs": ["p"], "id": "id", "groups": ["g"]}})
    d = load_predictions(write_csv("g.csv", "id,y,p,g\n1,1,0.4,a\n2,0,0.3,\n"), c)
    assert d.groups["g"].tolist() == ["a", "(missing)"]
    rows = "".join(f"{i},1,0.5,v{i}\n" for i in range(60))
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("hc.csv", "id,y,p,g\n" + rows), c)
    assert ei.value.code == "GROUP_HIGH_CARDINALITY"


def test_unknown_config_keys_rejected():
    with pytest.raises(ConfigError):
        resolve_config({"binning": {"nbins": 10}})
    with pytest.raises(ConfigError):
        resolve_config({"unknown": 1})


def test_config_precedence():
    c = resolve_config({"binning": {"n_bins": 20, "strategy": "quantile"}},
                       {"binning": {"n_bins": 5, "strategy": None}})
    assert c.binning.n_bins == 5 and c.binning.strategy == "quantile"


def test_inspect_reports_invalid_without_raising(write_csv):
    r = inspect_predictions(write_csv("x.csv", "id,y,p\n1,1,2\n"), cfg())
    assert not r.valid and any(f.severity == "error" for f in r.findings)
    ok = inspect_predictions(write_csv("o.csv", "id,y,p\n1,1,0.2\n2,0,0.1\n"), cfg())
    assert ok.valid and ok.validation_only


def test_dataframe_adapter_matches_csv(write_csv, tiny_frame):
    path = write_csv("t.csv", tiny_frame.to_csv(index=False))
    a = analyze_predictions(path, cfg())
    b = analyze_predictions(tiny_frame, cfg())
    assert a.models[0].metrics["brier"].value == b.models[0].metrics["brier"].value


def test_alignment_by_id_never_by_position(write_csv):
    c1 = cfg()
    a = load_predictions(write_csv("a.csv", "id,y,p\n1,1,0.9\n2,0,0.1\n3,1,0.6\n"), c1)
    c2 = resolve_config({"columns": {"label": "y", "probs": ["q"], "id": "id"}})
    b = load_predictions(write_csv("b.csv", "id,y,q\n3,1,0.7\n1,1,0.8\n2,0,0.2\n"), c2)
    merged, _ = align_by_id(a, b)
    order = {i: k for k, i in enumerate(merged.ids)}
    assert merged.probs["q"][order["3"]] == 0.7 and merged.probs["q"][order["1"]] == 0.8
    # unmatched rows require explicit intersection
    b2 = load_predictions(write_csv("b2.csv", "id,y,q\n1,1,0.8\n2,0,0.2\n9,0,0.5\n"), c2)
    with pytest.raises(InputValidationError) as ei:
        align_by_id(a, b2)
    assert ei.value.code == "ALIGN_UNMATCHED"
    m2, fs = align_by_id(a, b2, allow_intersection=True)
    assert m2.n == 2 and fs[0].count == 2
    # contradictory outcomes are rejected
    b3 = load_predictions(write_csv("b3.csv", "id,y,q\n1,0,0.8\n2,0,0.2\n3,1,0.5\n"), c2)
    with pytest.raises(InputValidationError) as ei:
        align_by_id(a, b3)
    assert ei.value.code == "ALIGN_LABEL_CONFLICT"


def test_validation_messages_do_not_print_rows(write_csv):
    secret = "patient-SECRET-123"
    with pytest.raises(InputValidationError) as ei:
        load_predictions(write_csv("s.csv", f"id,y,p\n{secret},1,7\n"), cfg())
    assert secret not in str(ei.value)
    assert all(secret not in f.message for f in ei.value.findings)


def test_timestamps_parse_and_failures_disable_views(write_csv):
    c = resolve_config({"columns": {"label": "y", "probs": ["p"], "id": "id",
                                    "timestamp": "t"}, "time_window": "M"})
    ok = load_predictions(write_csv("t.csv", "id,y,p,t\n1,1,0.4,2025-01-03\n"
                                             "2,0,0.3,2025-02-01T10:00:00+02:00\n"), c)
    assert ok.timestamps is not None and str(ok.timestamps.dt.tz) == "UTC"
    bad = load_predictions(write_csv("tb.csv", "id,y,p,t\n1,1,0.4,notadate\n2,0,0.3,2025-01-01\n"), c)
    assert bad.timestamps is None
    assert any(f.code == "TIME_UNPARSEABLE" for f in bad.findings)


def test_dataframe_with_numeric_dtypes():
    df = pd.DataFrame({"id": [1, 2, 3], "y": [True, False, True], "p": [0.1, 0.2, 0.3]})
    d = load_predictions(df, cfg(positive_label="True"))
    assert d.y.tolist() == [1, 0, 1]
