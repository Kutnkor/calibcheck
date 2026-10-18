"""Export integrity: offline HTML, escaping, JSON without NaN, CSV safety, privacy, overwrite."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibcheck import synthetic
from calibcheck.analysis import analyze_predictions
from calibcheck.errors import OutputError
from calibcheck.report import MANIFEST_NAME, MARKER, csv_safe, render_report


def _res(df=None, **kw):
    if df is None:
        df = synthetic.generate("two_models", n=600).frame
    cols = kw.pop("columns", {"label": "y_true", "probs": ["p_model", "p_model_b"],
                              "id": "row_id"})
    return analyze_predictions(df, columns=cols, **kw)


def test_full_export_is_complete_and_offline(tmp_path: Path):
    out = tmp_path / "dir with spaces" / "report"
    manifest = render_report(_res(uncertainty={"bootstrap": True, "n_resamples": 200}), out)
    assert manifest["complete"] is True
    assert not (out / MARKER).exists()
    for rel, digest in manifest["files"].items():
        assert (out / rel).exists(), rel
        assert len(digest) == 64
    html = (out / "report.html").read_text(encoding="utf-8")
    # no network access: no external URLs in src/href/url()/@import
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)
    assert "@import" not in html and "url(http" not in html
    assert "<script" not in html.lower()
    assert "data:image/png;base64," in html
    data = json.loads((out / "report.json").read_text())
    assert data["schema_version"] == "1.0"


def test_json_has_no_nan_tokens_and_null_reasons(tmp_path: Path):
    df = pd.DataFrame({"row_id": ["a", "b", "c"], "y_true": [0, 0, 0], "p": [0.1, 0.2, 0.3]})
    res = _res(df, columns={"label": "y_true", "probs": ["p"], "id": "row_id"})
    render_report(res, tmp_path / "r", figures=False)
    text = (tmp_path / "r" / "report.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    auc = data["models"][0]["metrics"]["roc_auc"]
    assert auc["value"] is None and auc["reason"] == "UNAVAILABLE_SINGLE_CLASS"


def test_hostile_names_are_escaped_and_files_sanitised(tmp_path: Path):
    evil = '<script>alert("x")</script>=HYPERLINK("http://e")'
    df = pd.DataFrame({"row_id": [f"{i}" for i in range(40)],
                       "y_true": [i % 2 for i in range(40)],
                       evil: np.linspace(0.05, 0.95, 40),
                       "grp": ['=cmd|" /C calc"!A0', "<b>x</b>"] * 20})
    res = _res(df, columns={"label": "y_true", "probs": [evil], "id": "row_id",
                            "groups": ["grp"]}, privacy={"mode": "private"})
    m = render_report(res, tmp_path / "r")
    html = (tmp_path / "r" / "report.html").read_text()
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "<b>x</b>" not in html
    for rel in m["files"]:
        assert ".." not in rel and "<" not in rel and '"' not in rel
    groups = pd.read_csv(tmp_path / "r" / "tables" / "groups.csv")
    assert "'=cmd|\" /C calc\"!A0" in set(groups["group"])
    assert not any(str(v).startswith(("=", "+", "@")) for v in groups["group"])
    metrics_csv = pd.read_csv(tmp_path / "r" / "tables" / "metrics.csv")
    assert metrics_csv["model"].iloc[0] == evil  # starts with '<', not a formula trigger


def test_csv_safe_leaves_numbers_untouched():
    df = pd.DataFrame({"a": ["-1+2", "ok", "@x"], "b": [-0.5, 0.1, 2.0]})
    out = csv_safe(df)
    assert out["a"].tolist() == ["'-1+2", "ok", "'@x"]
    assert out["b"].tolist() == [-0.5, 0.1, 2.0]


def test_public_mode_redacts(tmp_path: Path):
    sc = synthetic.generate("group_cancellation", n=2000)
    df = sc.frame.copy()
    df.loc[df.index[:5], "region"] = "tiny-village"
    res = analyze_predictions(df, columns={"label": "y_true", "probs": ["p_model"],
                                           "id": "row_id", "groups": ["region"]},
                              privacy={"mode": "public", "min_cell": 10})
    render_report(res, tmp_path / "pub")
    blob = (tmp_path / "pub" / "report.json").read_text() + \
        (tmp_path / "pub" / "report.html").read_text()
    for secret in ("tiny-village", "north", "south", str(tmp_path)):
        assert secret not in blob
    assert res.manifest is not None
    assert all(i.path is None and i.sha256 is None for i in res.manifest.inputs)
    rows = res.groups[0].rows
    supp = [r for r in rows if r.suppressed]
    assert len(supp) >= 2  # small cell plus complementary suppression
    assert all(r.n_pos is None and r.prevalence is None for r in supp)
    with pytest.raises(OutputError):
        render_report(res, tmp_path / "pub2", retained_ids=np.array(["1"]))


def test_refuses_non_calibcheck_dir_and_requires_overwrite(tmp_path: Path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    with pytest.raises(OutputError):
        render_report(_res(), foreign, figures=False)
    assert (foreign / "notes.txt").read_text() == "keep me"
    out = tmp_path / "r"
    render_report(_res(), out, figures=False)
    with pytest.raises(OutputError):
        render_report(_res(), out, figures=False)
    (out / "user_added.txt").write_text("mine")
    render_report(_res(), out, figures=False, overwrite=True)
    assert (out / "user_added.txt").exists()  # only files listed in the manifest are replaced


def test_partial_export_is_identifiable(tmp_path: Path, monkeypatch):
    import calibcheck.report as rep

    def boom(*a, **k):
        raise RuntimeError("renderer crashed")

    monkeypatch.setattr(rep, "render_html", boom)
    out = tmp_path / "partial"
    with pytest.raises(RuntimeError):
        render_report(_res(), out, figures=False)
    assert (out / MARKER).exists()
    man = json.loads((out / MANIFEST_NAME).read_text())
    assert man["complete"] is False and "renderer crashed" in man["error"]


def test_results_digest_stable_across_runs():
    a = _res(uncertainty={"bootstrap": True, "n_resamples": 200, "seed": 3})
    b = _res(uncertainty={"bootstrap": True, "n_resamples": 200, "seed": 3})
    assert a.manifest and b.manifest
    assert a.manifest.results_digest == b.manifest.results_digest


def test_long_model_names_and_missing_metrics_render(tmp_path: Path):
    long = "p_" + "very_long_model_name_" * 8
    df = pd.DataFrame({"row_id": ["1", "2", "3"], "y_true": [1, 1, 1], long: [0.2, 0.5, 0.9]})
    res = _res(df, columns={"label": "y_true", "probs": [long], "id": "row_id"})
    m = render_report(res, tmp_path / "r")
    assert m["complete"]
    html = (tmp_path / "r" / "report.html").read_text()
    assert "unavailable: UNAVAILABLE_SINGLE_CLASS" in html


def test_manifest_records_paths_as_given_not_resolved(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = synthetic.generate("calibrated", n=200).frame
    df.to_csv("in.csv", index=False)
    res = analyze_predictions("in.csv", columns={"label": "y_true", "probs": ["p_model"],
                                                 "id": "row_id"})
    assert res.manifest is not None
    assert res.manifest.inputs[0].path == "in.csv"
    assert str(tmp_path) not in res.model_dump_json()
