"""Property-based invariants (Hypothesis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from calibcheck.analysis import analyze_predictions
from calibcheck.binning import assign_bins, bin_table, quantile_edges, uniform_edges
from calibcheck.calibration import apply_calibrator, fit_calibrator
from calibcheck.metrics import core_metrics

probs = arrays(np.float64, st.integers(1, 200),
               elements=st.floats(0, 1, allow_nan=False, allow_infinity=False))


@given(probs, st.integers(1, 40), st.sampled_from(["uniform", "quantile"]))
@settings(max_examples=150, deadline=None)
def test_every_observation_in_exactly_one_bin(p, k, strategy):
    e = uniform_edges(k) if strategy == "uniform" else quantile_edges(p, k)
    idx = assign_bins(p, e)
    assert idx.min() >= 0 and idx.max() < len(e) - 1
    assert np.all(p >= e[idx])
    upper = e[idx + 1]
    last = idx == len(e) - 2
    assert np.all((p < upper) | (last & (p <= upper)))
    y = (p > 0.5).astype(int)
    bt = bin_table(y, p, e, strategy=strategy, n_bins_requested=k, edge_source="",
                   min_support=1, wilson_level=0.95)
    assert sum(b.count for b in bt.bins if b.count) == p.size


@given(probs)
@settings(max_examples=100, deadline=None)
def test_brier_in_range_and_permutation_invariant(p):
    rng = np.random.default_rng(0)
    y = (rng.random(p.size) < 0.5).astype(int)
    m = core_metrics(y, p, eps=1e-15, threshold=0.5, population="x")
    assert 0.0 <= m["brier"].value <= 1.0
    perm = rng.permutation(p.size)
    m2 = core_metrics(y[perm], p[perm], eps=1e-15, threshold=0.5, population="x")
    for k in ("brier", "log_loss", "mean_error"):
        assert np.isclose(m[k].value, m2[k].value, rtol=1e-12, atol=1e-15)
    if m["roc_auc"].available:
        assert np.isclose(m["roc_auc"].value, m2["roc_auc"].value)


@given(arrays(np.float64, st.integers(20, 120), elements=st.floats(0.001, 0.999)),
       st.sampled_from(["sigmoid", "isotonic", "temperature"]))
@settings(max_examples=60, deadline=None)
def test_mappings_produce_valid_probabilities(p, method):
    y = (np.arange(p.size) % 3 == 0).astype(int)
    art = fit_calibrator(p, y, method, model_association="p", store_id_digests=False)
    q, _ = apply_calibrator(art, np.linspace(0, 1, 101))
    assert np.all(np.isfinite(q)) and q.min() >= 0 and q.max() <= 1


@given(st.lists(st.sampled_from(["a", "b", "c", ""]), min_size=5, max_size=80))
@settings(max_examples=60, deadline=None)
def test_group_partitions_preserve_counts(groups):
    n = len(groups)
    rng = np.random.default_rng(n)
    df = pd.DataFrame({"id": [str(i) for i in range(n)], "y": rng.integers(0, 2, n),
                       "p": rng.random(n), "g": groups})
    res = analyze_predictions(df, columns={"label": "y", "probs": ["p"], "id": "id",
                                           "groups": ["g"]})
    assert sum(r.n for r in res.groups[0].rows) == n


@given(probs)
@settings(max_examples=60, deadline=None)
def test_identical_predictions_zero_paired_difference(p):
    n = p.size
    df = pd.DataFrame({"id": [str(i) for i in range(n)], "y": (np.arange(n) % 2),
                       "a": p, "b": p})
    res = analyze_predictions(df, columns={"label": "y", "probs": ["a", "b"], "id": "id"})
    for c in res.comparisons:
        if c.delta is not None:
            assert c.delta == 0.0
