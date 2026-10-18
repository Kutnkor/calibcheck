"""Bin boundaries, bin statistics, ECE-type summaries and Wilson intervals."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve

from calibcheck.binning import (
    assign_bins,
    bin_summaries,
    bin_table,
    quantile_edges,
    sensitivity_table,
    uniform_edges,
    wilson_interval,
)


def table(y, p, edges=None, *, strategy="uniform", n_bins=10, min_support=1, level=0.95):
    p = np.asarray(p, dtype=float)
    if edges is None:
        edges = uniform_edges(n_bins) if strategy == "uniform" else quantile_edges(p, n_bins)
    return bin_table(np.asarray(y), p, edges, strategy=strategy, n_bins_requested=n_bins,
                     edge_source="test", min_support=min_support, wilson_level=level)


def test_boundary_semantics_left_closed_last_closed():
    e = uniform_edges(10)
    p = np.array([0.0, 0.1, 0.3, 0.7, 0.9999, 1.0, 0.29999999])
    assert assign_bins(p, e).tolist() == [0, 1, 3, 7, 9, 9, 2]


def test_uniform_edges_exact_decimal_points():
    e = uniform_edges(10)
    assert e[3] == 0.3 and e[7] == 0.7 and e[0] == 0.0 and e[-1] == 1.0


def test_every_observation_assigned_once_and_empty_bins_kept():
    y = [1, 0, 1, 0, 1]
    p = [0.05, 0.05, 0.95, 1.0, 0.5]
    bt = table(y, p)
    assert len(bt.bins) == 10
    assert sum(b.count for b in bt.bins) == 5
    empty = [b for b in bt.bins if b.count == 0]
    assert all(b.mean_pred is None and b.observed_rate is None for b in empty)
    assert all(b.ci_status == "unavailable" for b in empty)
    assert bt.n_nonempty == 3
    last = bt.bins[-1]
    assert last.closed == "both" and last.count == 2 and last.n_pos == 1


def test_ece_hand_value():
    # bin0: p=[0.05,0.05] y=[1,0] -> |0.5-0.05|=0.45, w=2/5
    # bin5: p=[0.5] y=[1] -> 0.5, w=1/5
    # bin9: p=[0.95,1.0] y=[1,0] -> |0.5-0.975|=0.475, w=2/5
    y = [1, 0, 1, 0, 1]
    p = [0.05, 0.05, 0.95, 1.0, 0.5]
    bt = table(y, p)
    s = bin_summaries(bt, 5, "pop")
    expected = 0.4 * 0.45 + 0.2 * 0.5 + 0.4 * 0.475
    assert s["ece"].value == pytest.approx(expected, abs=1e-15)
    assert s["max_abs_gap"].value == pytest.approx(0.5)
    assert s["max_abs_gap"].details["bin_support"] == 1
    rms = np.sqrt(0.4 * 0.45**2 + 0.2 * 0.25 + 0.4 * 0.475**2)
    assert s["rms_gap"].value == pytest.approx(rms)
    assert "uniform" in s["ece"].details["partition"]


def test_signed_gap_direction():
    bt = table([0, 0, 0, 1], [0.9, 0.9, 0.9, 0.9])
    b = bt.bins[9]
    assert b.signed_gap == pytest.approx(0.9 - 0.25)  # positive = overprediction


def test_supported_max_gap_discloses_exclusions():
    y = [0] + [0] * 40
    p = [0.95] + [0.15] * 40  # one tiny bin with a huge gap, one large bin with a small gap
    bt = table(y, p, min_support=30)
    s = bin_summaries(bt, 41, "pop")
    assert s["max_abs_gap"].value == pytest.approx(0.95)
    assert s["max_abs_gap"].details["bin_support"] == 1
    assert s["max_abs_gap_supported"].value == pytest.approx(0.15)
    assert s["max_abs_gap_supported"].details["observations_excluded"] == 1


def test_no_supported_bins_unavailable():
    bt = table([1, 0], [0.2, 0.8], min_support=30)
    s = bin_summaries(bt, 2, "pop")
    assert s["max_abs_gap_supported"].status == "unavailable"


def test_matches_sklearn_calibration_curve_off_boundaries():
    rng = np.random.default_rng(5)
    p = rng.random(2000)
    p = p[np.abs(p * 10 - np.round(p * 10)) > 1e-9]  # no value exactly on an interior edge
    y = (rng.random(p.size) < p).astype(int)
    bt = table(y, p)
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    ours = [(b.observed_rate, b.mean_pred) for b in bt.bins if b.count]
    assert np.allclose([o[0] for o in ours], prob_true)
    assert np.allclose([o[1] for o in ours], prob_pred)


def test_boundary_difference_from_sklearn_is_documented():
    # sklearn puts 0.3 in the bin (0.2, 0.3]; CalibCheck uses [0.3, 0.4).
    y = np.array([1, 0])
    p = np.array([0.3, 0.35])
    bt = table(y, p)
    assert bt.bins[3].count == 2
    _, pred = calibration_curve(y, p, n_bins=10)
    assert len(pred) == 2  # sklearn splits them across two bins


def test_quantile_ties_never_split_and_counts_reported():
    p = np.array([0.2] * 50 + [0.9] * 50)
    y = np.r_[np.zeros(40), np.ones(10), np.ones(30), np.zeros(20)].astype(int)
    e = quantile_edges(p, 10)
    bt = table(y, p, e, strategy="quantile", n_bins=10)
    assert bt.n_bins_requested == 10
    assert bt.n_bins_actual == 2
    assert [b.count for b in bt.bins] == [50, 50]


def test_quantile_constant_predictions_single_bin():
    p = np.full(30, 0.4)
    e = quantile_edges(p, 10)
    assert e.tolist() == [0.0, 1.0]
    bt = table(np.r_[np.ones(12), np.zeros(18)].astype(int), p, e, strategy="quantile")
    assert bt.n_bins_actual == 1 and bt.bins[0].count == 30


def test_quantile_bins_roughly_equal_frequency():
    rng = np.random.default_rng(9)
    p = rng.random(1000)
    e = quantile_edges(p, 10)
    counts = np.bincount(assign_bins(p, e))
    assert len(counts) == 10
    assert counts.min() >= 95 and counts.max() <= 105


@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (3, 10), (1, 1), (17, 412), (0, 1)])
def test_wilson_matches_scipy(k, n):
    lo, hi = wilson_interval(np.array([k]), np.array([n]), 0.95)
    ci = binomtest(k, n).proportion_ci(0.95, method="wilson")
    assert lo[0] == pytest.approx(ci.low, abs=1e-12)
    assert hi[0] == pytest.approx(ci.high, abs=1e-12)


def test_wilson_level_90():
    lo, hi = wilson_interval(np.array([7]), np.array([20]), 0.90)
    ci = binomtest(7, 20).proportion_ci(0.90, method="wilson")
    assert (lo[0], hi[0]) == pytest.approx((ci.low, ci.high), abs=1e-12)


def test_sensitivity_includes_primary_and_actual_counts():
    rng = np.random.default_rng(2)
    p = np.round(rng.random(300), 1)  # heavy ties
    y = (rng.random(300) < p).astype(int)
    rows = sensitivity_table(y, p, bins=[5, 10, 20], strategies=["uniform", "quantile"],
                             primary=("uniform", 10), min_support=30)
    assert sum(r.is_primary for r in rows) == 1
    q20 = next(r for r in rows if r.strategy == "quantile" and r.n_bins_requested == 20)
    assert q20.n_bins_actual < 20  # ties collapse edges; reported honestly
