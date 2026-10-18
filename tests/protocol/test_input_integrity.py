"""Regression tests for file and public API input boundaries."""
import pytest

from calibcheck.analysis import compare_predictions
from calibcheck.data import load_predictions
from calibcheck.errors import InputValidationError


def test_comparison_requires_two_models_even_for_validated_data(tiny_frame, cfg_factory):
    data = load_predictions(tiny_frame, cfg_factory())
    with pytest.raises(InputValidationError, match="at least two"):
        compare_predictions(data)
