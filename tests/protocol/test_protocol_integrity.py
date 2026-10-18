"""Regression checks for held-out evaluation protocol edge cases."""
import pytest

from calibcheck.calibration import fit_calibrator
from calibcheck.data import load_predictions
from calibcheck.errors import ProtocolError
from calibcheck.protocol import check_artifact_overlap


def test_artifact_checks_entity_overlap_even_with_disjoint_row_ids(tiny_frame, cfg_factory):
    cfg = cfg_factory(columns={"entity": "entity"})
    frame = tiny_frame.assign(entity=["E1", "E1", "E2", "E2"])
    cal = load_predictions(frame, cfg)
    art = fit_calibrator(cal.probs["p"], cal.y, "sigmoid", model_association="p",
                         ids=cal.ids, entities=cal.entity)
    test = load_predictions(frame.assign(id=["t1", "t2", "t3", "t4"]), cfg)
    with pytest.raises(ProtocolError) as exc:
        check_artifact_overlap(art, test)
    assert exc.value.code == "ROLE_ENTITY_OVERLAP"


def test_temporal_roles_and_selection_refuse_missing_timestamps(tiny_frame, cfg_factory):
    from calibcheck.protocol import check_role_separation, select_method
    cfg = cfg_factory(columns={"timestamp": "ts"})
    cal = load_predictions(tiny_frame.assign(ts=["2025-01-01", None, "2025-01-03", "2025-01-04"]), cfg)
    test = load_predictions(tiny_frame.assign(id=["t1", "t2", "t3", "t4"], ts="2026-01-01"), cfg)
    with pytest.raises(ProtocolError, match="every row"):
        check_role_separation(cal, test, temporal=True)
    with pytest.raises(ProtocolError, match="every calibration row"):
        select_method(cal, "p", ["identity"], cfg, temporal=True, n_folds=2)


@pytest.mark.parametrize("options", [{"mode": "typo"}, {"selection_metric": "accuracy"}])
def test_evaluation_options_never_silently_fall_back(tiny_frame, cfg_factory, options):
    from calibcheck.protocol import evaluate_recalibration
    cfg = cfg_factory()
    cal = load_predictions(tiny_frame, cfg)
    test = load_predictions(tiny_frame.assign(id=["t1", "t2", "t3", "t4"]), cfg)
    with pytest.raises(ProtocolError):
        evaluate_recalibration(cal, test, cfg, model="p", **options)


@pytest.mark.parametrize("methods,metric", [([], "brier"), (["unknown"], "brier"), (["identity"], "typo")])
def test_selection_rejects_invalid_methods_and_metric(tiny_frame, cfg_factory, methods, metric):
    from calibcheck.errors import FittingError
    from calibcheck.protocol import select_method
    cfg = cfg_factory()
    with pytest.raises(FittingError):
        select_method(load_predictions(tiny_frame, cfg), "p", methods, cfg, metric=metric, n_folds=2)


@pytest.mark.parametrize("folds", [0, 1, True, 2.5, 100])
def test_selection_fold_errors_are_actionable(tiny_frame, cfg_factory, folds):
    from calibcheck.errors import FittingError
    from calibcheck.protocol import select_method
    cfg = cfg_factory()
    with pytest.raises(FittingError):
        select_method(load_predictions(tiny_frame, cfg), "p", ["identity"], cfg, n_folds=folds)


def test_grouped_selection_requires_multiple_entities(tiny_frame, cfg_factory):
    from calibcheck.errors import FittingError
    from calibcheck.protocol import select_method
    cfg = cfg_factory(columns={"entity": "entity"})
    cal = load_predictions(tiny_frame.assign(entity="one"), cfg)
    with pytest.raises(FittingError, match="two distinct entities"):
        select_method(cal, "p", ["identity"], cfg, n_folds=2)
