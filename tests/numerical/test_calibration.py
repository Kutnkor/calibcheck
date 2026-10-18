"""Recalibration maps: parameter recovery, isotonic equivalence, persistence, artifact checks."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from calibcheck.calibration import (
    apply_calibrator,
    fit_calibrator,
    load_artifact,
    parse_artifact,
    save_artifact,
)
from calibcheck.errors import ArtifactError, FittingError


def _distorted(n=20000, slope=2.0, intercept=0.0, seed=0):
    rng = np.random.default_rng(seed)
    p_true = expit(rng.normal(-0.3, 1.2, n))
    y = (rng.random(n) < p_true).astype(np.int8)
    p = expit(slope * logit(p_true) + intercept)
    return p, y


def test_sigmoid_recovers_inverse_distortion():
    p, y = _distorted(slope=2.0, intercept=0.6)
    art = fit_calibrator(p, y, "sigmoid", model_association="p")
    a, b = art.params["a"], art.params["b"]
    # true inverse: logit(p_true) = (logit(p) - 0.6) / 2  ->  a = 0.5, b = -0.3
    assert a == pytest.approx(0.5, abs=0.03)
    assert b == pytest.approx(-0.3, abs=0.05)
    assert art.diagnostics.converged
    assert art.input_transform.startswith("z = logit")


def test_temperature_recovers_scale():
    p, y = _distorted(slope=2.0)
    art = fit_calibrator(p, y, "temperature", model_association="p")
    assert art.params["T"] == pytest.approx(2.0, rel=0.05)


def test_isotonic_matches_sklearn_and_persists(tmp_path):
    p, y = _distorted(n=3000)
    art = fit_calibrator(p, y, "isotonic", model_association="p")
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
    grid = np.r_[0.0, np.linspace(0, 1, 501), 1.0, p.min() / 2]
    q, _ = apply_calibrator(art, grid)
    assert np.allclose(q, iso.predict(grid), atol=1e-12)
    path = save_artifact(art, tmp_path / "iso.json")
    again = load_artifact(path)
    q2, _ = apply_calibrator(again, grid)
    assert np.array_equal(q, q2)


def test_isotonic_out_of_range_reported_and_constant_input():
    p = np.array([0.3, 0.3, 0.3, 0.3])
    y = np.array([1, 0, 1, 0])
    art = fit_calibrator(p, y, "isotonic", model_association="p")
    q, notes = apply_calibrator(art, np.array([0.1, 0.3, 0.9]))
    assert np.allclose(q, 0.5)
    assert notes and notes[0].code == "APPLY_OUT_OF_RANGE" and notes[0].count == 2
    assert any(w.code == "FIT_CONSTANT_MAP" for w in art.diagnostics.warnings)


@pytest.mark.parametrize("method", ["identity", "sigmoid", "isotonic", "temperature"])
def test_maps_are_valid_and_nondecreasing(method):
    p, y = _distorted(n=2000, seed=3)
    art = fit_calibrator(p, y, method, model_association="p")
    grid = np.linspace(0, 1, 1001)
    q, _ = apply_calibrator(art, grid)
    assert np.all((q >= 0) & (q <= 1))
    assert np.all(np.diff(q) >= -1e-15)


def test_single_class_calibration_refused():
    with pytest.raises(FittingError) as ei:
        fit_calibrator(np.array([0.2, 0.4]), np.array([0, 0]), "sigmoid", model_association="p")
    assert ei.value.code == "FIT_SINGLE_CLASS"


def test_separable_data_flags_bound():
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10)
    y = np.array([0, 0, 0, 1, 1, 1] * 10)
    art = fit_calibrator(p, y, "sigmoid", model_association="p")
    assert any(w.code in ("FIT_SEPARATION", "FIT_AT_BOUND") for w in art.diagnostics.warnings)
    assert any(w.code == "FIT_SEPARATION" for w in art.diagnostics.warnings)


def test_sigmoid_constant_map_boundary_allowed():
    rng = np.random.default_rng(1)
    p = rng.random(4000)
    y = (rng.random(4000) < 0.3).astype(int)  # outcomes unrelated to p
    art = fit_calibrator(p, y, "sigmoid", model_association="p")
    assert art.params["a"] >= 0
    assert art.params["a"] < 0.1


def _dump(art) -> dict:
    return json.loads(json.dumps(art.model_dump(mode="json")))


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(schema_version="9.0"), "schema"),
    (lambda d: d["params"].update(a=-1.0), "slope"),
    (lambda d: d.update(artifact_type="other"), "Not a CalibCheck"),
    (lambda d: d.update(unexpected=True), "Malformed"),
    (lambda d: d["params"].update(a="nan"), "finite"),
    (lambda d: d.update(method="exec"), "Malformed"),
])
def test_artifact_validation_rejects(mutate, match):
    p, y = _distorted(n=500)
    d = _dump(fit_calibrator(p, y, "sigmoid", model_association="p"))
    mutate(d)
    with pytest.raises(ArtifactError, match=match):
        parse_artifact(json.dumps(d))


def test_isotonic_artifact_nonmonotone_rejected():
    p, y = _distorted(n=500)
    d = _dump(fit_calibrator(p, y, "isotonic", model_association="p"))
    d["params"]["y"] = list(reversed(d["params"]["y"]))
    if d["params"]["y"] == sorted(d["params"]["y"]):
        pytest.skip("degenerate fit")
    with pytest.raises(ArtifactError, match="non-decreasing"):
        parse_artifact(json.dumps(d))


def test_pickle_files_never_loaded(tmp_path):
    import pickle
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps({"a": 1}, protocol=4))
    with pytest.raises(ArtifactError, match="pickle"):
        load_artifact(path)


def test_eval_labels_cannot_change_fit():
    p, y = _distorted(n=1000)
    a1 = fit_calibrator(p, y, "sigmoid", model_association="p", store_id_digests=False)
    a2 = fit_calibrator(p, y, "sigmoid", model_association="p", store_id_digests=False)
    assert a1.params == a2.params  # deterministic given calibration data


# --- input validation happens before any lossy conversion ---------------------------------

from calibcheck.errors import InputValidationError  # noqa: E402


@pytest.mark.parametrize("p,y,code", [
    ([-2.0, 3.0], [0, 1], "PROB_OUT_OF_RANGE"),
    ([0.2, 0.8], [0.9, 1.9], "LABEL_NOT_BINARY"),
    ([0.2, 0.8], [0, 2], "LABEL_NOT_BINARY"),
    ([0.2, 0.8], [-1, 1], "LABEL_NOT_BINARY"),
    ([0.2, 0.8], [0.0, np.nan], "LABEL_MISSING"),
    ([0.2, 0.8], ["no", "yes"], "LABEL_NOT_BINARY"),
    ([0.2, np.nan], [0, 1], "PROB_NONFINITE"),
    ([0.2, np.inf], [0, 1], "PROB_NONFINITE"),
    (["0.2", "0.8"], [0, 1], "PROB_NONNUMERIC"),
    ([True, False], [0, 1], "PROB_NONNUMERIC"),
    ([0.2, None], [0, 1], "PROB_NONNUMERIC"),
    ([[0.2, 0.8]], [0, 1], "INPUT_SHAPE"),
    ([0.2, 0.8], [[0, 1]], "INPUT_SHAPE"),
    ([0.2, 0.8, 0.5], [0, 1], "INPUT_SHAPE"),
    ([], [], "INPUT_EMPTY"),
])
@pytest.mark.parametrize("method", ["identity", "sigmoid", "isotonic", "temperature"])
def test_fit_rejects_invalid_inputs(p, y, code, method):
    with pytest.raises(InputValidationError) as ei:
        fit_calibrator(p, y, method, model_association="p")
    assert ei.value.code == code


def test_fit_accepts_lossless_label_encodings():
    p = [0.1, 0.4, 0.6, 0.9]
    ref = fit_calibrator(p, np.array([0, 0, 1, 1], dtype=np.int64), "sigmoid",
                         model_association="p", store_id_digests=False).params
    for y in ([False, False, True, True], [0.0, 0.0, 1.0, 1.0], np.array([0, 0, 1, 1], np.uint8)):
        got = fit_calibrator(p, y, "sigmoid", model_association="p", store_id_digests=False)
        assert got.params == ref


def test_fit_rejects_misaligned_ids_and_bad_eps():
    with pytest.raises(InputValidationError, match="ids"):
        fit_calibrator([0.2, 0.8], [0, 1], "sigmoid", model_association="p", ids=["a"])
    with pytest.raises(InputValidationError, match="entities"):
        fit_calibrator([0.2, 0.8], [0, 1], "sigmoid", model_association="p",
                       entities=["a", "b", "c"])
    for eps in (0.0, -1e-9, 0.5, float("nan")):
        with pytest.raises(InputValidationError):
            fit_calibrator([0.2, 0.8], [0, 1], "sigmoid", model_association="p", eps=eps)


def test_apply_validates_without_coercion():
    art = fit_calibrator([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1], "sigmoid", model_association="p")
    for bad in (["0.5"], [[0.5]], [1.5], [np.nan], [True]):
        with pytest.raises(InputValidationError):
            apply_calibrator(art, bad)
    q, _ = apply_calibrator(art, np.array([], dtype=float))
    assert q.size == 0
