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
