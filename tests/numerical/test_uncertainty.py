"""Bootstrap behaviour, pairing, dependence refusal and comparison direction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import bootstrap as scipy_bootstrap

from calibcheck.analysis import analyze_predictions
from calibcheck.binning import assign_bins, ece_from_assignments, uniform_edges
from calibcheck.config import UncertaintyConfig
from calibcheck.metrics import brier_losses
from calibcheck.uncertainty import (
    bootstrap_means,
    frozen_edge_ece_replicates,
    percentile_interval,
)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(np.int8)
    return y, p


@pytest.mark.slow
def test_percentile_interval_agrees_with_scipy():
    y, p = _data()
    losses = brier_losses(y, p)
    u = UncertaintyConfig(bootstrap=True, n_resamples=20000, seed=1)
    ours = percentile_interval(bootstrap_means(losses, u)[:, 0], 0.95, seed=1, method_note="")
    ref = scipy_bootstrap((losses,), np.mean, n_resamples=20000, method="percentile",
                          confidence_level=0.95, rng=np.random.default_rng(2))
    width = ref.confidence_interval.high - ref.confidence_interval.low
    # Different random streams: agreement up to Monte Carlo error (2% of the width).
    assert ours.low == pytest.approx(ref.confidence_interval.low, abs=0.02 * width)
    assert ours.high == pytest.approx(ref.confidence_interval.high, abs=0.02 * width)


def test_seeded_and_batch_size_invariant():
    y, p = _data(300)
    losses = brier_losses(y, p)
    a = bootstrap_means(losses, UncertaintyConfig(n_resamples=500, seed=5))
    b = bootstrap_means(losses, UncertaintyConfig(n_resamples=500, seed=5))
    c = bootstrap_means(losses, UncertaintyConfig(n_resamples=500, seed=5,
                                                  max_elements_per_batch=10_000))
    assert np.array_equal(a, b)
    assert np.allclose(a, c, rtol=0, atol=1e-15)


def test_pairing_identical_models_zero_difference():
    y, p = _data(150)
    import pandas as pd
    df = pd.DataFrame({"id": [str(i) for i in range(150)], "y": y, "a": p, "b": p.copy()})
    res = analyze_predictions(df, columns={"label": "y", "probs": ["a", "b"], "id": "id"},
                              uncertainty={"bootstrap": True, "n_resamples": 200})
    for c in res.comparisons:
        if c.metric in ("brier", "log_loss"):
            assert c.delta == 0.0
            assert c.interval.low == 0.0 and c.interval.high == 0.0
            assert c.interval.reason == "UNC_BOOTSTRAP_DEGENERATE"
            assert c.favours == "neither"


def test_comparison_direction_and_paired_interval():
    import pandas as pd
    rng = np.random.default_rng(8)
    n = 3000
    p_true = rng.random(n)
    y = (rng.random(n) < p_true).astype(int)
    good = p_true
    bad = np.clip(p_true * 0.5 + 0.4, 0, 1)
    df = pd.DataFrame({"id": [str(i) for i in range(n)], "y": y, "bad": bad, "good": good})
    res = analyze_predictions(df, columns={"label": "y", "probs": ["bad", "good"], "id": "id"},
                              uncertainty={"bootstrap": True, "n_resamples": 500})
    c = next(c for c in res.comparisons if c.metric == "brier")
    assert (c.reference, c.candidate) == ("bad", "good")
    direct = brier_losses(y, good).mean() - brier_losses(y, bad).mean()
    assert c.delta == pytest.approx(direct)
    assert c.delta < 0 and c.favours == "candidate"
    assert c.interval.high < 0
    assert "negative favours the candidate" in c.definition


def test_dependence_declared_refuses_iid_intervals():
    import pandas as pd
    y, p = _data(100)
    df = pd.DataFrame({"id": [str(i) for i in range(100)], "y": y, "p": p})
    for dep in ("entity_clustered", "time_dependent", "unknown"):
        res = analyze_predictions(df, columns={"label": "y", "probs": ["p"], "id": "id"},
                                  uncertainty={"bootstrap": True, "n_resamples": 200},
                                  dependence=dep)
        m = res.models[0]
        assert m.metrics["brier"].interval.status == "unavailable"
        assert m.metrics["brier"].interval.reason == "UNAVAILABLE_DEPENDENCE"
        assert all(b.ci_status != "ok" for b in m.binning.bins)
        assert m.metrics["brier"].value is not None  # descriptive results remain


def test_frozen_edge_ece_bootstrap_matches_direct_on_identity_resample():
    y, p = _data(120)
    e = uniform_edges(10)
    a = assign_bins(p, e)
    direct = ece_from_assignments(y.astype(float), p, a, 10)
    from calibcheck import uncertainty as unc

    class Fixed:
        pass

    # replicate statistic evaluated on the identity index matrix equals the full-sample ECE
    captured = {}

    def fake_bootstrap(n, stat, ucfg, k=1):
        captured["v"] = stat(np.arange(n)[None, :])
        return captured["v"]

    orig = unc.bootstrap_statistic
    unc.bootstrap_statistic = fake_bootstrap
    try:
        frozen_edge_ece_replicates(y, [p], [a], [10], UncertaintyConfig())
    finally:
        unc.bootstrap_statistic = orig
    assert captured["v"][0, 0] == pytest.approx(direct, abs=1e-12)


def test_failed_replicates_counted_not_hidden():
    reps = np.r_[np.random.default_rng(0).random(990), np.full(10, np.nan)]
    iv = percentile_interval(reps, 0.95, seed=0, method_note="")
    assert iv.status == "ok" and iv.n_failed == 10 and iv.n_effective == 990
    assert iv.reason == "UNC_BOOTSTRAP_FAILED_REPLICATES"
    bad = percentile_interval(np.r_[np.ones(50), np.full(50, np.nan)], 0.95, seed=0,
                              method_note="")
    assert bad.status == "failed" and bad.low is None


def test_ece_bootstrap_is_opt_in_and_labelled_exploratory():
    import pandas as pd
    y, p = _data(400)
    df = pd.DataFrame({"id": [str(i) for i in range(400)], "y": y, "p": p})
    res = analyze_predictions(df, columns={"label": "y", "probs": ["p"], "id": "id"},
                              uncertainty={"bootstrap": True, "n_resamples": 300,
                                           "bootstrap_bin_metrics": True})
    iv = res.models[0].metrics["ece"].interval
    assert iv.status == "ok" and "frozen" in iv.interpretation
    res2 = analyze_predictions(df, columns={"label": "y", "probs": ["p"], "id": "id"},
                               uncertainty={"bootstrap": True, "n_resamples": 300})
    assert res2.models[0].metrics["ece"].interval is None
