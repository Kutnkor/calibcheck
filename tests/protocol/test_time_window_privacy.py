"""Public-mode small-cell suppression applies to time windows, consistently across outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibcheck.analysis import analyze_predictions
from calibcheck.groups import flag_small_cells
from calibcheck.plotting import time_figure
from calibcheck.report import render_report

SECRET_P = 0.456789  # inside the data range, so only a window leak could expose it


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    # Jan-Apr: 40 rows each with mixed outcomes; May: 12 rows, all positive; Jun: one row.
    for month, n in (("01", 40), ("02", 40), ("03", 40), ("04", 40)):
        for i in range(n):
            rows.append((f"2025-{month}-{(i % 27) + 1:02d}", int(rng.random() < 0.4),
                         round(float(rng.uniform(0.2, 0.8)), 4)))
    rows += [(f"2025-05-{i + 1:02d}", 1, 0.7) for i in range(12)]
    rows.append(("2025-06-15", 1, SECRET_P))
    df = pd.DataFrame(rows, columns=["ts", "y", "p"])
    df.insert(0, "id", [f"r{i:04d}" for i in range(len(df))])
    return df


def _analyze(mode: str):
    return analyze_predictions(
        _frame(), columns={"label": "y", "probs": ["p"], "id": "id", "timestamp": "ts"},
        time_window="M", privacy={"mode": mode, "min_cell": 10})


def _row(res, window):
    return next(r for r in res.time_windows.rows if r.window == window)


def test_flag_small_cells_policy():
    assert flag_small_cells([(40, 10), (40, 12), (3, 1)], 10) == [
        (2, "n < 10"), (0, "complementary suppression")]
    assert flag_small_cells([(40, 10), (12, 12)], 10) == [
        (1, "all outcomes identical"), (0, "complementary suppression")]
    # two hidden cells already protect each other: no extra suppression
    assert [i for i, _ in flag_small_cells([(40, 10), (5, 2), (4, 1)], 10)] == [1, 2]
    assert flag_small_cells([(40, 10), (50, 20)], 10) == []


def test_public_windows_suppressed_with_explicit_nulls():
    res = _analyze("public")
    tw = res.time_windows
    assert tw is not None and tw.suppression_policy
    june, may = _row(res, "2025-06"), _row(res, "2025-05")
    assert june.suppressed and "n < 10" in june.suppression_reason
    assert june.n is None and june.n_pos is None and june.prevalence is None
    assert june.per_model["p"] == {"mean_pred": None, "brier": None, "log_loss": None}
    assert may.suppressed and "all outcomes identical" in may.suppression_reason
    assert may.n == 12 and may.n_pos is None and may.prevalence is None
    assert all(v is None for v in may.per_model["p"].values())
    # two windows hidden -> totals cannot isolate either; others published intact
    shown = [r for r in tw.rows if not r.suppressed]
    assert len(shown) == 4
    assert all(r.n_pos is not None and r.per_model["p"]["mean_pred"] is not None for r in shown)


def test_single_suppressed_window_gets_complementary_partner():
    df = _frame()
    df = df[~df.ts.str.startswith("2025-05")]  # leave the singleton as the only small window
    res = analyze_predictions(df, columns={"label": "y", "probs": ["p"], "id": "id",
                                           "timestamp": "ts"},
                              time_window="M", privacy={"mode": "public", "min_cell": 10})
    hidden = [r for r in res.time_windows.rows if r.suppressed]
    assert len(hidden) == 2
    partner = next(r for r in hidden if r.window != "2025-06")
    assert "complementary" in partner.suppression_reason
    assert partner.n == 40 and partner.n_pos is None  # count kept (>= min_cell), outcomes hidden


def test_private_mode_keeps_full_detail():
    res = _analyze("private")
    assert res.time_windows.suppression_policy is None
    june = _row(res, "2025-06")
    assert not june.suppressed and june.n == 1 and june.n_pos == 1
    assert june.per_model["p"]["mean_pred"] == pytest.approx(SECRET_P)
    assert _row(res, "2025-05").prevalence == 1.0


def test_suppression_consistent_across_json_csv_html(tmp_path: Path):
    res = _analyze("public")
    render_report(res, tmp_path / "pub")
    out = tmp_path / "pub"
    data = json.loads((out / "report.json").read_text())
    june = next(r for r in data["time_windows"]["rows"] if r["window"] == "2025-06")
    assert june["suppressed"] is True
    assert june["n"] is None and june["n_pos"] is None and june["prevalence"] is None
    assert june["per_model"]["p"]["mean_pred"] is None
    csv = pd.read_csv(out / "tables" / "time_windows.csv")
    sup = csv[csv.window == "2025-06"].iloc[0]
    assert bool(sup.suppressed) and pd.isna(sup.n_pos) and pd.isna(sup.mean_pred)
    assert sup.n_pos != 0  # NaN, never a zero stand-in
    html = (out / "report.html").read_text()
    assert "2025-06 <span class=\"note\">(suppressed)</span>" in html
    secret = f"{SECRET_P}"
    for blob in ((out / "report.json").read_text(), (out / "tables" / "time_windows.csv").read_text(),
                 html):
        assert secret not in blob and "0.4568" not in blob


def test_plot_omits_suppressed_points():
    res = _analyze("public")
    fig = time_figure(res.time_windows, res.models)
    ax = fig.axes[0]
    windows = [r.window for r in res.time_windows.rows]
    j = windows.index("2025-06")
    k = windows.index("2025-05")
    for line in ax.get_lines():
        y = line.get_ydata()
        assert math.isnan(y[j]) and math.isnan(y[k])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[j].endswith("(suppressed)")
