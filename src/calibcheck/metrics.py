"""Aggregate metrics for binary probability forecasts.

Conventions (see docs/methodology.md):

* ``y`` is 1 for the declared positive label and 0 otherwise; ``p`` is P(Y = positive).
* Brier loss: mean((p - y)^2) on the [0, 1] binary scale (sklearn ``scale_by_half=True``
  for binary input). Uses the original probabilities, never clipped ones.
* Log loss: natural logarithm, -mean(y log q + (1 - y) log(1 - q)) with
  q = clip(p, eps, 1 - eps). Clipping is a numerical stabilisation, recorded per run.
* Ranking metrics (ROC-AUC, average precision) require both classes; otherwise they are
  marked unavailable with ``UNAVAILABLE_SINGLE_CLASS`` rather than returning a number.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from calibcheck.schemas import ClipInfo, Direction, MetricValue


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    category: str
    scale: str
    direction: Direction
    description: str


REGISTRY: dict[str, MetricSpec] = {s.name: s for s in [
    MetricSpec("brier", "Brier loss", "overall", "[0, 1], squared probability error",
               "lower_is_better",
               "mean((p - y)^2). An overall probabilistic score combining calibration and "
               "discrimination; not a pure calibration measure."),
    MetricSpec("log_loss", "Log loss", "overall", "[0, inf), nats", "lower_is_better",
               "-mean(y log q + (1-y) log(1-q)), q = clip(p, eps, 1-eps). Confident wrong "
               "predictions are penalised heavily."),
    MetricSpec("roc_auc", "ROC-AUC", "discrimination", "[0, 1]", "higher_is_better",
               "Probability that a random positive is ranked above a random negative (ties count "
               "1/2). Describes ranking, not calibration."),
    MetricSpec("average_precision", "Average precision", "discrimination", "[0, 1]",
               "higher_is_better",
               "Step-wise area under the precision-recall curve (scikit-learn definition). "
               "Describes ranking, not calibration; depends on prevalence."),
    MetricSpec("mean_pred", "Mean predicted probability", "descriptive", "[0, 1]", "descriptive",
               "Average of p over the population."),
    MetricSpec("prevalence", "Observed prevalence", "descriptive", "[0, 1]", "descriptive",
               "Fraction of observations with the positive label."),
    MetricSpec("mean_error", "Signed mean prediction error", "calibration", "[-1, 1]",
               "descriptive",
               "mean(p) - mean(y). Positive means the average prediction exceeds the observed "
               "prevalence. Local gaps of opposite sign can cancel here."),
    MetricSpec("ece", "Binned positive-class ECE", "calibration", "[0, 1]", "lower_is_better",
               "sum over nonempty bins of (n_b/n)|mean_y_b - mean_p_b| for the declared "
               "partition. A partition-dependent finite-sample diagnostic, not the exact "
               "population calibration error and not top-label ECE."),
    MetricSpec("max_abs_gap", "Maximum absolute bin gap", "calibration", "[0, 1]",
               "lower_is_better",
               "max over nonempty bins of |mean_y_b - mean_p_b|. Reported with the support of "
               "the bin that attains it."),
    MetricSpec("max_abs_gap_supported", "Maximum absolute gap (supported bins)", "calibration",
               "[0, 1]", "lower_is_better",
               "Same as the maximum gap but only over bins with at least the support threshold; "
               "the excluded population is disclosed."),
    MetricSpec("rms_gap", "RMS bin gap", "calibration", "[0, 1]", "lower_is_better",
               "sqrt(sum over nonempty bins of (n_b/n)(mean_y_b - mean_p_b)^2)."),
    MetricSpec("accuracy", "Threshold accuracy", "threshold", "[0, 1]", "higher_is_better",
               "Fraction correct when predicting positive iff p >= threshold. Secondary view."),
    MetricSpec("reference_brier", "Reference Brier loss", "reference",
               "[0, 1], squared probability error", "lower_is_better",
               "Brier loss of the declared constant reference predictor on the same rows."),
    MetricSpec("brier_skill", "Brier skill score", "skill", "(-inf, 1]", "higher_is_better",
               "1 - BS_model / BS_reference on the same rows; undefined when BS_reference = 0."),
]}


def brier_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-observation squared errors (p - y)^2 on the binary [0, 1] scale."""
    return (np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2


def clip_probs(p: np.ndarray, eps: float) -> tuple[np.ndarray, int, int]:
    p = np.asarray(p, dtype=np.float64)
    low = int((p < eps).sum())
    high = int((p > 1.0 - eps).sum())
    return np.clip(p, eps, 1.0 - eps), low, high


def log_losses(y: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    """Per-observation natural-log losses with the declared clipping rule."""
    q, _, _ = clip_probs(p, eps)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(q) + (1.0 - y) * np.log1p(-q))


def clip_info(p: np.ndarray, eps: float) -> ClipInfo:
    _, low, high = clip_probs(p, eps)
    return ClipInfo(eps=eps, n_clipped_low=low, n_clipped_high=high,
                    applies_to=["log_loss", "logit-based recalibration"])


def _mv(name: str, value: float | None, n: int, population: str, *, status: str = "ok",
        reason: str | None = None, **details: object) -> MetricValue:
    spec = REGISTRY[name]
    if value is not None and not np.isfinite(value):
        status, reason, value = "failed", reason or "NONFINITE_RESULT", None
    return MetricValue(name=name, label=spec.label, category=spec.category,
                       value=None if value is None else float(value),
                       status=status if value is not None or status != "ok" else "unavailable",
                       reason=reason, scale=spec.scale, direction=spec.direction, n=n,
                       population=population, details=dict(details))


def unavailable(name: str, reason: str, n: int, population: str, **details: object) -> MetricValue:
    return _mv(name, None, n, population, status="unavailable", reason=reason, **details)


def _ranking(fn: Callable[[np.ndarray, np.ndarray], float], name: str, y: np.ndarray,
             p: np.ndarray, population: str) -> MetricValue:
    n = int(y.size)
    if n == 0:
        return unavailable(name, "UNAVAILABLE_EMPTY", n, population)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == n:
        return unavailable(name, "UNAVAILABLE_SINGLE_CLASS", n, population)
    tied = bool(np.unique(p).size == 1)
    return _mv(name, float(fn(y, p)), n, population,
               constant_predictor=tied,
               note=("constant predictor: every pair is tied" if tied else ""))


def core_metrics(y: np.ndarray, p: np.ndarray, *, eps: float, threshold: float,
                 population: str, reference_prevalence: float | None = None
                 ) -> dict[str, MetricValue]:
    """Compute the aggregate (non-binned) metrics on one population."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    n = int(y.size)
    out: dict[str, MetricValue] = {}
    if n == 0:
        for name in ("brier", "log_loss", "roc_auc", "average_precision", "mean_pred",
                     "prevalence", "mean_error", "accuracy", "reference_brier", "brier_skill"):
            out[name] = unavailable(name, "UNAVAILABLE_EMPTY", 0, population)
        return out
    bs = float(brier_losses(y, p).mean())
    ll = float(log_losses(y, p, eps).mean())
    prev = float(y.mean())
    out["brier"] = _mv("brier", bs, n, population, convention="binary, unweighted, [0,1] scale")
    out["log_loss"] = _mv("log_loss", ll, n, population, eps=eps, log_base="e",
                          **clip_info(p, eps).model_dump(include={"n_clipped_low",
                                                                  "n_clipped_high"}))
    out["roc_auc"] = _ranking(roc_auc_score, "roc_auc", y, p, population)
    out["average_precision"] = _ranking(average_precision_score, "average_precision", y, p,
                                        population)
    out["mean_pred"] = _mv("mean_pred", float(p.mean()), n, population)
    out["prevalence"] = _mv("prevalence", prev, n, population, n_pos=int(y.sum()))
    out["mean_error"] = _mv("mean_error", float(p.mean()) - prev, n, population)
    acc = float(((p >= threshold).astype(np.int8) == y).mean())
    out["accuracy"] = _mv("accuracy", acc, n, population, threshold=threshold,
                          rule="predict positive iff p >= threshold")
    # --- constant reference predictor ------------------------------------------------------
    if reference_prevalence is not None:
        r = float(reference_prevalence)
        ref_kind = "fixed prevalence supplied from training/development data"
    else:
        r = prev
        ref_kind = ("evaluation-prevalence descriptive reference (uses held-out outcomes; "
                    "not an available deployment predictor)")
    ref_bs = float(((r - y.astype(np.float64)) ** 2).mean())
    out["reference_brier"] = _mv("reference_brier", ref_bs, n, population,
                                 reference_probability=r, reference_kind=ref_kind)
    if ref_bs > 0:
        out["brier_skill"] = _mv("brier_skill", 1.0 - bs / ref_bs, n, population,
                                 reference_probability=r, reference_kind=ref_kind)
    else:
        out["brier_skill"] = unavailable("brier_skill", "UNAVAILABLE_ZERO_REFERENCE", n,
                                         population, reference_probability=r,
                                         reference_kind=ref_kind)
    return out
