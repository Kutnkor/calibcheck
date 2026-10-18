"""Regression tests for file and public API input boundaries."""
import pytest

from calibcheck.analysis import compare_predictions
from calibcheck.data import load_predictions
from calibcheck.errors import InputValidationError


def test_comparison_requires_two_models_even_for_validated_data(tiny_frame, cfg_factory):
    data = load_predictions(tiny_frame, cfg_factory())
    with pytest.raises(InputValidationError, match="at least two"):
        compare_predictions(data)


@pytest.mark.parametrize("body", [
    b"y,p\n1,0.8,secret\n0,0.2,secret\n", b"y,p\n1\n0,0.2\n",
    b'y,p\n1,"0.8\n',
])
def test_csv_width_and_quoting_cannot_silently_shift_columns(body, cfg_factory):
    from calibcheck.data import read_csv_bytes
    with pytest.raises(InputValidationError):
        read_csv_bytes(body, cfg_factory())


def test_valid_multiline_csv_and_blank_lines_are_supported(cfg_factory):
    from calibcheck.data import read_csv_bytes
    frame, _ = read_csv_bytes(b'y,p,note\n1,0.8,"line one\nline two"\n\n0,0.2,ok\n', cfg_factory())
    assert len(frame) == 2 and frame.iloc[0]["note"] == "line one\nline two"


@pytest.mark.parametrize("body", [b"y,p,\n1,0.8,note\n", b"y, ,p\n1,x,0.8\n", b"\ny,p\n1,0.8\n"])
def test_blank_headers_are_not_silently_renamed(body, cfg_factory):
    from calibcheck.data import read_csv_bytes
    with pytest.raises(InputValidationError) as exc:
        read_csv_bytes(body, cfg_factory())
    assert exc.value.code == "INPUT_EMPTY_HEADER"
