"""Data-role protocol for separate-data recalibration.

Roles: the base model was trained elsewhere (CalibCheck cannot verify how); its predictions on
*calibration* rows fit the recalibrator; method selection, when requested, uses inner folds
of the calibration pool only; final evaluation uses untouched *test* rows. Overlap between
calibration and test IDs or entities blocks evaluation. Without trustworthy IDs, separation is
reported as unknown, never as proven.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from calibcheck.analysis import analyze_data
from calibcheck.calibration import (
    METHODS,
    SIMPLICITY,
    CalibratorArtifact,
    apply_calibrator,
    digests_for,
    fit_calibrator,
)
from calibcheck.config import AnalysisConfig
from calibcheck.data import PredictionData
from calibcheck.errors import FittingError, ProtocolError
from calibcheck.metrics import brier_losses, log_losses
from calibcheck.provenance import build_manifest
from calibcheck.schemas import AnalysisResult, Finding

BASE_MODEL_NOTE = ("CalibCheck sees predictions only; it cannot verify that the base model was "
                   "trained without access to the calibration or test rows.")


def check_role_separation(cal: PredictionData, test: PredictionData, *,
                          temporal: bool = False) -> list[Finding]:
    """Raise ProtocolError on forbidden overlap; return findings otherwise."""
    findings = [Finding(code="ROLE_BASE_MODEL_UNVERIFIABLE", severity="info",
                        message=BASE_MODEL_NOTE)]
    if cal.id_kind != "provided" or test.id_kind != "provided":
        findings.append(Finding(
            code="ROLE_PROVENANCE_UNKNOWN", severity="warning",
            message="At least one file has positional IDs, so overlap between calibration and "
                    "test rows cannot be checked. Independence is not established."))
    else:
        overlap = np.intersect1d(cal.ids.astype(str), test.ids.astype(str))
        if overlap.size:
            raise ProtocolError(
                f"{overlap.size} row ID(s) appear in both the calibration and the test data. A "
                "recalibrator must not be evaluated on rows used to fit it.", code="ROLE_OVERLAP",
                findings=[Finding(code="ROLE_OVERLAP", severity="error", count=int(overlap.size),
                                  message=f"{overlap.size} overlapping IDs")])
        findings.append(Finding(code="ROLE_OVERLAP", severity="info", count=0,
                                message="No row IDs are shared between calibration and test "
                                        "data (checked by exact ID match)."))
    if cal.entity is not None and test.entity is not None:
        ov = np.intersect1d(cal.entity.astype(str), test.entity.astype(str))
        if ov.size:
            raise ProtocolError(
                f"{ov.size} entity ID(s) appear in both calibration and test data; with "
                "entity-level observation units the roles must be disjoint by entity.",
                code="ROLE_ENTITY_OVERLAP")
    if temporal:
        if cal.timestamps is None or test.timestamps is None:
            raise ProtocolError("A temporal protocol needs parseable timestamps in both files.",
                                code="ROLE_TEMPORAL_ORDER")
        cmax, tmin = cal.timestamps.max(), test.timestamps.min()
        if not cmax < tmin:
            raise ProtocolError(
                f"Temporal protocol: the latest calibration timestamp ({cmax}) is not earlier "
                f"than the earliest test timestamp ({tmin}).", code="ROLE_TEMPORAL_ORDER")
    return findings


def check_artifact_overlap(art: CalibratorArtifact, test: PredictionData) -> list[Finding]:
    """Use an artifact's salted ID digests to refuse evaluating on its own calibration rows."""
    prov = art.fitted_on
    if prov.id_digests is None or prov.id_digest_salt is None or test.id_kind != "provided":
        return [Finding(code="ROLE_PROVENANCE_UNKNOWN", severity="warning",
                        message="The artifact carries no ID digests (or the evaluation file has "
                                "no IDs), so overlap with its calibration rows cannot be checked.")]
    hits = digests_for(prov.id_digest_salt, test.ids) & set(prov.id_digests)
    if hits:
        raise ProtocolError(
            f"{len(hits)} evaluation row(s) were used to fit calibrator "
            f"'{art.method}' for '{art.model_association}'. Evaluate on untouched rows.",
            code="ROLE_OVERLAP")
    return [Finding(code="ROLE_OVERLAP", severity="info", count=0,
                    message=f"No evaluation rows match the calibration rows of the "
                            f"{art.method} artifact (salted digest check).")]


def _fit_for(data: PredictionData, model: str, method: str, cfg: AnalysisConfig) -> CalibratorArtifact:
    ts = None
    if data.timestamps is not None and data.timestamps.notna().any():
        ts = str(data.timestamps.max())
    return fit_calibrator(
        data.probs[model], data.y, method, eps=cfg.clip_eps, model_association=model,
        positive_label=data.positive_label, label_column=data.label_column, ids=data.ids,
        id_kind=data.id_kind, entities=data.entity, timestamp_max=ts,
        input_sha256=data.input_record.sha256 if data.input_record and
        cfg.privacy.fingerprints_enabled() else None)


def select_method(cal: PredictionData, model: str, methods: list[str], cfg: AnalysisConfig, *,
                  metric: Literal["log_loss", "brier"] = "log_loss", n_folds: int = 5,
                  seed: int = 0, temporal: bool = False) -> dict[str, Any]:
    """Choose a method using inner folds of the calibration data only.

    Folds are grouped by entity when entities exist and chronological (expanding window)
    under a temporal protocol. Out-of-fold predictions are pooled and scored with ``metric``.
    Ties within 1e-12 go to the simpler method.
    """
    n = cal.n
    idx = np.arange(n)
    if temporal and cal.timestamps is not None:
        order = np.argsort(cal.timestamps.to_numpy(), kind="stable")
        splits = [(order[tr], order[te]) for tr, te in TimeSeriesSplit(n_splits=n_folds).split(order)]
        scheme = f"chronological expanding-window, {n_folds} splits"
    elif cal.entity is not None:
        k = min(n_folds, len(np.unique(cal.entity)))
        splits = list(GroupKFold(n_splits=k).split(idx, groups=cal.entity))
        scheme = f"GroupKFold by entity, {k} folds"
    else:
        splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(idx))
        scheme = f"KFold, {n_folds} folds, shuffled with seed {seed}"
    scores: dict[str, float | None] = {}
    failures: dict[str, str] = {}
    for method in methods:
        pooled_y, pooled_q = [], []
        try:
            for tr, te in splits:
                art = fit_calibrator(cal.probs[model][tr], cal.y[tr], method, eps=cfg.clip_eps,
                                     model_association=model, store_id_digests=False)
                q, _ = apply_calibrator(art, cal.probs[model][te])
                pooled_y.append(cal.y[te])
                pooled_q.append(q)
        except FittingError as exc:
            scores[method] = None
            failures[method] = str(exc)
            continue
        yy, qq = np.concatenate(pooled_y), np.concatenate(pooled_q)
        scores[method] = float((log_losses(yy, qq, cfg.clip_eps) if metric == "log_loss"
                                else brier_losses(yy, qq)).mean())
    valid = {m: s for m, s in scores.items() if s is not None}
    if not valid:
        raise FittingError("No candidate method could be fitted in the inner folds.")
    best = min(valid.values())
    chosen = min((m for m, s in valid.items() if s - best <= 1e-12), key=lambda m: SIMPLICITY[m])
    return {"selected": chosen, "metric": metric, "scheme": scheme,
            "inner_scores": scores, "failures": failures,
            "note": "Selection used calibration data only and was frozen before test evaluation."}


def evaluate_recalibration(cal: PredictionData, test: PredictionData, cfg: AnalysisConfig, *,
                           model: str, methods: list[str] | None = None,
                           mode: Literal["exploratory", "selected"] = "exploratory",
                           selection_metric: Literal["log_loss", "brier"] = "log_loss",
                           n_folds: int = 5, temporal: bool = False,
                           title: str | None = None, synthetic: bool = False,
                           data_note: str | None = None
                           ) -> tuple[AnalysisResult, dict[str, CalibratorArtifact]]:
    """Fit on calibration rows, evaluate raw vs recalibrated on the same untouched test rows."""
    methods = list(methods or ["sigmoid", "isotonic"])
    for m in methods:
        if m not in METHODS:
            raise FittingError(f"Unknown method {m!r}.")
    if model not in cal.probs or model not in test.probs:
        raise ProtocolError(f"Probability column {model!r} must exist in both files.")
    if cal.positive_label != test.positive_label:
        raise ProtocolError("Calibration and test files declare different positive labels.")
    role_findings = check_role_separation(cal, test, temporal=temporal)

    selection = None
    if mode == "selected":
        candidates = list(dict.fromkeys(["identity", *methods]))
        selection = select_method(cal, model, candidates, cfg, metric=selection_metric,
                                  n_folds=n_folds, seed=cfg.uncertainty.seed, temporal=temporal)
        to_fit = [selection["selected"]] if selection["selected"] != "identity" else []
    else:
        to_fit = [m for m in methods if m != "identity"]

    artifacts: dict[str, CalibratorArtifact] = {}
    fit_status: dict[str, Any] = {}
    probs = {model: test.probs[model]}
    names = {model: f"{model} (raw)"}
    apply_notes: list[Finding] = []
    for m in to_fit:
        try:
            art = _fit_for(cal, model, m, cfg)
        except FittingError as exc:
            fit_status[m] = {"status": "failed", "reason": exc.code, "message": str(exc)}
            continue
        artifacts[m] = art
        q, notes = apply_calibrator(art, test.probs[model])
        apply_notes += notes
        col = f"{model}__{m}"
        probs[col] = q
        names[col] = f"{model} + {m}"
        fit_status[m] = {"status": "ok", "params": _param_summary(art),
                         "warnings": [w.code for w in art.diagnostics.warnings],
                         "max_abs_change_on_test": float(np.abs(q - test.probs[model]).max())}
    eval_cfg = cfg.model_copy(deep=True)
    eval_cfg.columns.probs = list(probs)
    eval_cfg.reference_model = model
    eval_data = test.with_probs(probs)
    protocol = {
        "roles": {"calibration": {"n": cal.n, "n_pos": int(cal.y.sum())},
                  "test": {"n": test.n, "n_pos": int(test.y.sum())}},
        "mode": mode,
        "mode_note": ("Exploratory: every predeclared method is shown on the test rows. The best "
                      "observed test result is not an unbiased estimate for a method chosen "
                      "because it looked best." if mode == "exploratory" else
                      "Selected: the method was chosen on inner folds of the calibration data "
                      "and frozen before the single test evaluation."),
        "methods_requested": methods, "fit": fit_status, "selection": selection,
        "temporal": temporal,
        "role_checks": [f.model_dump(include={"code", "severity", "message"}) for f in role_findings],
        "intervals_note": ("Intervals resample fixed test predictions; they are conditional on "
                           "the fitted base model and recalibrator and omit training and "
                           "method-selection variability."),
        "base_model_note": BASE_MODEL_NOTE,
    }
    res = analyze_data(eval_data, eval_cfg, kind="recalibration", display_names=names,
                       title=title or "CalibCheck recalibration evaluation", synthetic=synthetic,
                       data_note=data_note, protocol=protocol,
                       extra_inputs=[cal.input_record] if cal.input_record else None)
    extra: list[Finding] = []
    for m, st in fit_status.items():
        if st["status"] == "failed":
            extra.append(Finding(code="UNAVAILABLE_FIT_FAILED", severity="warning",
                                 message=f"{m} could not be fitted ({st['reason']}): "
                                         f"{st['message']} Raw-model results are unaffected."))
        elif st["max_abs_change_on_test"] < 1e-9:
            extra.append(Finding(code="FINDING_RECAL_UNCHANGED", severity="info",
                                 message=f"{m} left the test probabilities unchanged "
                                         "(max change < 1e-9)."))
    if selection and selection["selected"] == "identity":
        extra.append(Finding(code="FINDING_RECAL_UNCHANGED", severity="info",
                             message="Inner-fold selection preferred the identity map (no "
                                     "recalibration) on the calibration data."))
    res.findings = role_findings[1:] + extra + res.findings
    res.validation = res.validation + apply_notes
    inputs = [r for r in (test.input_record, cal.input_record) if r is not None]
    res.manifest = build_manifest(res, inputs, mode="recalibration", protocol=protocol)
    return res, artifacts


def _param_summary(art: CalibratorArtifact) -> dict[str, Any]:
    if art.method == "isotonic":
        return {"n_breakpoints": len(art.params["x"])}
    return dict(art.params)
