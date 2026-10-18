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


def test_directory_input_returns_actionable_domain_error(tmp_path, cfg_factory):
    with pytest.raises(InputValidationError) as exc:
        load_predictions(tmp_path, cfg_factory())
    assert exc.value.code == "INPUT_UNREADABLE"


def test_unreadable_csv_returns_domain_error(tmp_path, cfg_factory, monkeypatch):
    from pathlib import Path
    path = tmp_path / "input.csv"
    path.write_text("id,y,p\na,1,.8\n")

    def deny(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", deny)
    with pytest.raises(InputValidationError, match="access permissions"):
        load_predictions(path, cfg_factory())


def test_directory_config_returns_actionable_domain_error(tmp_path):
    from calibcheck.config import load_config_file
    from calibcheck.errors import ConfigError
    with pytest.raises(ConfigError, match="readable TOML file"):
        load_config_file(tmp_path)


def test_misspelled_transform_key_cannot_silently_leave_predictions_unchanged(tiny_frame, cfg_factory):
    cfg = cfg_factory(transforms={"p_typo": "logit"})
    with pytest.raises(InputValidationError) as exc:
        load_predictions(tiny_frame, cfg)
    assert exc.value.code == "INPUT_TRANSFORM_MISMATCH"


def test_csv_probability_decimals_preserve_float64_values(cfg_factory):
    values = ["0.12345678901234567", "0.9999999999999999", "2.2250738585072014e-308", "1e-300"]
    raw = "id,y,p\n" + "\n".join(f"r{i},{i % 2},{p}" for i, p in enumerate(values)) + "\n"
    data = load_predictions(raw.encode(), cfg_factory())
    assert data.probs["p"].tolist() == [float(p) for p in values]
    assert data.probs["p"][1] < 1.0  # must not become a spurious endpoint
