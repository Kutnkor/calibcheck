"""Hand-checkable metric values and cross-checks against scikit-learn (matching conventions)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from calibcheck.metrics import brier_losses, core_metrics, log_losses

EPS = 1e-15


def metrics(y, p, **kw):
    return core_metrics(np.asarray(y), np.asarray(p, dtype=float), eps=kw.pop("eps", EPS),
                        threshold=kw.pop("threshold", 0.5), population="test", **kw)


def test_brier_hand_value():
    # (0.9-1)^2 + (0.2-0)^2 + (0.6-1)^2 + (0.4-0)^2 = 0.01+0.04+0.16+0.16 = 0.37 ; /4
    m = metrics([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4])
    assert m["brier"].value == pytest.approx(0.0925, abs=1e-15)
    assert m["brier"].status == "ok" and m["brier"].n == 4


def test_log_loss_hand_value_natural_log():
    m = metrics([1, 0], [0.8, 0.3])
    expected = -(math.log(0.8) + math.log(0.7)) / 2
    assert m["log_loss"].value == pytest.approx(expected, rel=1e-14)
    assert m["log_loss"].details["log_base"] == "e"


def test_brier_matches_sklearn_binary_scale():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    # Convention under test: binary Brier on [0, 1] = sklearn scale_by_half=True for binary input.
    ref = brier_score_loss(y, p, scale_by_half=True)
    assert metrics(y, p)["brier"].value == pytest.approx(ref, rel=1e-12)


def test_log_loss_matches_sklearn_off_endpoints():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 400)
    y = (rng.random(400) < p).astype(int)
    # No clipping is active here, so the clipping conventions cannot differ.
    assert metrics(y, p)["log_loss"].value == pytest.approx(log_loss(y, p), rel=1e-12)


def test_log_loss_endpoints_documented_convention():
    # sklearn 1.9 clips at float64 machine epsilon; CalibCheck clips at the declared eps.
    y = np.array([1, 0, 1, 0])
    p = np.array([0.0, 1.0, 1.0, 0.0])
    ours = metrics(y, p, eps=np.finfo(np.float64).eps)["log_loss"].value
    assert ours == pytest.approx(log_loss(y, p), rel=1e-12)
    default = metrics(y, p)["log_loss"]
    # q = clip(p, eps, 1 - eps) evaluated in float64 (1 - 1e-15 is itself rounded).
    hi = 1.0 - EPS
    expected = (-math.log(EPS) - math.log1p(-hi) - math.log(hi) - math.log1p(-EPS)) / 4
    assert default.value == pytest.approx(expected, rel=1e-12)
    assert default.details["n_clipped_low"] == 2 and default.details["n_clipped_high"] == 2


def test_brier_uses_unclipped_endpoints():
    m = metrics([1, 0], [1.0, 0.0])
    assert m["brier"].value == 0.0


def test_label_encoding_symmetry():
    # Swapping which class is positive and complementing p leaves losses unchanged.
    y = np.array([1, 0, 0, 1, 1])
    p = np.array([0.7, 0.1, 0.4, 0.55, 0.9])
    a = metrics(y, p)
    b = metrics(1 - y, 1 - p)
    for k in ("brier", "log_loss"):
        assert a[k].value == pytest.approx(b[k].value, rel=1e-12)
    assert a["roc_auc"].value == pytest.approx(b["roc_auc"].value)


def test_single_class_keeps_losses_marks_ranking_unavailable():
    m = metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert m["brier"].value == pytest.approx((0.01 + 0.04 + 0.09) / 3)
    assert m["log_loss"].available
    for k in ("roc_auc", "average_precision"):
        assert m[k].status == "unavailable"
        assert m[k].value is None
        assert m[k].reason == "UNAVAILABLE_SINGLE_CLASS"
    # Evaluation-prevalence reference has zero Brier loss -> skill undefined, not a number.
    assert m["reference_brier"].value == 0.0
    assert m["brier_skill"].status == "unavailable"
    assert m["brier_skill"].reason == "UNAVAILABLE_ZERO_REFERENCE"


def test_brier_skill_with_fixed_reference():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.6, 0.2, 0.1, 0.3])
    m = metrics(y, p, reference_prevalence=0.5)
    bs = ((0.4) ** 2 + 0.04 + 0.01 + 0.09) / 4
    assert m["reference_brier"].value == pytest.approx(0.25)
    assert m["brier_skill"].value == pytest.approx(1 - bs / 0.25)
    assert "training/development" in m["brier_skill"].details["reference_kind"]


def test_ranking_matches_sklearn():
    rng = np.random.default_rng(3)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    m = metrics(y, p)
    assert m["roc_auc"].value == pytest.approx(roc_auc_score(y, p))
    assert m["average_precision"].value == pytest.approx(average_precision_score(y, p))


def test_constant_predictor_auc_half():
    m = metrics([0, 1, 0, 1], [0.4] * 4)
    assert m["roc_auc"].value == pytest.approx(0.5)
    assert m["roc_auc"].details["constant_predictor"] is True


def test_prevalence_mean_error_accuracy():
    m = metrics([1, 0, 0, 0], [0.5, 0.5, 0.2, 0.7], threshold=0.5)
    assert m["prevalence"].value == 0.25
    assert m["mean_pred"].value == pytest.approx(0.475)
    assert m["mean_error"].value == pytest.approx(0.225)
    # predictions >= 0.5 -> [1, 1, 0, 1]; correct: [1, 0, 1, 0] -> 0.5
    assert m["accuracy"].value == 0.5


def test_per_observation_losses_shape_and_values():
    y = np.array([1, 0])
    p = np.array([0.25, 0.25])
    assert np.allclose(brier_losses(y, p), [0.5625, 0.0625])
    assert np.allclose(log_losses(y, p, EPS), [-math.log(0.25), -math.log(0.75)])


def test_empty_population_is_unavailable_not_nan():
    m = metrics([], [])
    assert all(v.status == "unavailable" and v.value is None for v in m.values())
