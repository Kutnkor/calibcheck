"""Regression checks for safe calibrator persistence and application."""
import json

import numpy as np
import pytest

from calibcheck.calibration import apply_calibrator, fit_calibrator, parse_artifact, validate_artifact
from calibcheck.config import resolve_config
from calibcheck.errors import ArtifactError, ConfigError, InputValidationError


def artifact(method="sigmoid"):
    return fit_calibrator([0.1, 0.3, 0.7, 0.9], [0, 1, 0, 1], method, model_association="p")


def test_tiny_clip_epsilon_rejected_before_endpoint_math():
    with pytest.raises(ConfigError):
        resolve_config({"clip_eps": 1e-100})
    with pytest.raises(InputValidationError):
        fit_calibrator([0, 1], [0, 1], "sigmoid", eps=1e-100, model_association="p")
    art = artifact()
    validate_artifact(parse_artifact(json.dumps(art.model_dump(mode="json"))))
    art.clip_eps = 1e-100
    with pytest.raises(ArtifactError):
        apply_calibrator(art, np.array([0.0, 1.0]))


def test_duplicate_isotonic_coordinates_rejected():
    art = artifact("isotonic")
    art.params = {"x": [0.1, 0.1, 0.9], "y": [0.0, 0.5, 1.0]}
    with pytest.raises(ArtifactError, match="strictly increasing"):
        validate_artifact(art)


def test_nonconverged_artifacts_cannot_be_applied():
    art = artifact()
    art.diagnostics.converged = False
    with pytest.raises(ArtifactError, match="non-converged"):
        apply_calibrator(art, np.array([0.2]))


@pytest.mark.parametrize("update", [
    {"n": 0}, {"n_pos": -1}, {"n_pos": 99}, {"prob_range": [0.2]},
    {"prob_range": [0.8, 0.2]}, {"prob_range": [-1, 1]}, {"prob_range": [0, float("inf")]},
])
def test_invalid_artifact_provenance_rejected(update):
    data = artifact().model_dump(mode="json")
    data["fitted_on"].update(update)
    with pytest.raises(ArtifactError):
        parse_artifact(json.dumps(data))


def test_save_rejects_invalid_artifact_without_touching_destination(tmp_path):
    from calibcheck.calibration import save_artifact
    path = tmp_path / "map.json"
    path.write_text("keep me")
    art = artifact()
    art.params["a"] = -1.0
    with pytest.raises(ArtifactError):
        save_artifact(art, path, overwrite=True)
    assert path.read_text() == "keep me"


def test_atomic_save_preserves_existing_artifact_on_failure(tmp_path, monkeypatch):
    from calibcheck.calibration import save_artifact
    from calibcheck.errors import OutputError
    path = tmp_path / "map.json"
    save_artifact(artifact(), path)
    before = path.read_bytes()

    def fail(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("calibcheck.calibration.os.replace", fail)
    with pytest.raises(OutputError):
        save_artifact(artifact("identity"), path, overwrite=True)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".calibcheck-*.tmp"))


def test_artifact_directory_reports_domain_error(tmp_path):
    from calibcheck.calibration import load_artifact
    with pytest.raises(ArtifactError, match="readable JSON file"):
        load_artifact(tmp_path)


@pytest.mark.parametrize("as_bytes", [True, False])
def test_uploaded_artifact_size_limit_matches_file_limit(monkeypatch, as_bytes):
    raw = json.dumps(artifact().model_dump(mode="json"))
    monkeypatch.setattr("calibcheck.calibration.MAX_ARTIFACT_BYTES", 10)
    with pytest.raises(ArtifactError, match="unreasonably large"):
        parse_artifact(raw.encode() if as_bytes else raw)
