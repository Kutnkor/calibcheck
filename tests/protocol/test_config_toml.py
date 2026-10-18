"""TOML export of AnalysisConfig must reload to an identical configuration."""

from __future__ import annotations

import math
import tomllib
import types
import typing
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from calibcheck.config import AnalysisConfig, config_to_toml, load_config_file, resolve_config

AWKWARD_KEYS = [
    "model.v1",            # dot: must not become nested tables
    "model score",         # space
    "olasılık",            # Unicode
    'say "hi"',            # quotation mark
    "back\\slash",         # backslash
    "tab\tkey",            # tab
    "new\nline",           # newline
    "carriage\rreturn",
    "bell\x07 del\x7f nul\x00",  # other control characters
    "",                    # empty key is legal when quoted
    "[table]",
    "a = b",
    "# not a comment",
    "'single'",
    "trailing\\",
    "emoji 📏",
]


def _roundtrip(cfg: AnalysisConfig, tmp_path: Path) -> AnalysisConfig:
    text = config_to_toml(cfg)
    tomllib.loads(text)  # must be valid TOML on its own
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return resolve_config(load_config_file(path))


@pytest.mark.parametrize("key", AWKWARD_KEYS)
def test_transform_and_column_keys_roundtrip(key: str, tmp_path: Path):
    cfg = resolve_config({"columns": {"label": "y", "probs": [key, "plain"], "groups": [key + "_g"]},
                          "transforms": {key: "percent", "plain": "logit"}})
    parsed = tomllib.loads(config_to_toml(cfg))
    assert parsed["transforms"] == {key: "percent", "plain": "logit"}
    assert parsed["columns"]["probs"] == [key, "plain"]
    assert _roundtrip(cfg, tmp_path) == cfg


def test_default_configuration_roundtrip(tmp_path: Path):
    cfg = AnalysisConfig()
    assert _roundtrip(cfg, tmp_path) == cfg


def test_nested_lists_booleans_numbers_roundtrip(tmp_path: Path):
    cfg = resolve_config({
        "columns": {"label": "outcome", "probs": ["p_a", "p_b"], "id": "row id",
                    "entity": "customer.id", "timestamp": "ts", "split": "role",
                    "groups": ["region", "device type"]},
        "positive_label": "yes \"really\"",
        "missing_policy": "drop",
        "clip_eps": 1e-12,
        "threshold": 0.0,
        "reference_prevalence": 0.1234567890123456,
        "binning": {"strategy": "quantile", "n_bins": 17, "min_support": 1, "sensitivity": False,
                    "sensitivity_bins": [3, 7], "sensitivity_strategies": ["quantile"]},
        "uncertainty": {"level": 0.9, "wilson": False, "bootstrap": True, "n_resamples": 100,
                        "seed": 0, "bootstrap_bin_metrics": True},
        "dependence": "entity_clustered",
        "groups": {"max_groups": 3, "min_group_size": 5, "max_cardinality": 9},
        "time_window": "Q",
        "privacy": {"mode": "public", "min_cell": 4, "reveal_group_names": True,
                    "include_fingerprints": False, "export_retained_rows": False},
        "reference_model": "p_b", "pairwise": True, "primary_metric": "log_loss",
        "max_file_mb": float("inf"), "max_columns": 7,
    })
    again = _roundtrip(cfg, tmp_path)
    assert again == cfg
    assert again.privacy.include_fingerprints is False
    assert math.isinf(again.max_file_mb)
    assert again.reference_prevalence == 0.1234567890123456  # exact float round trip


def _allows_none(annotation: object) -> bool:
    return type(None) in typing.get_args(annotation) or annotation is type(None) or (
        isinstance(annotation, types.UnionType) and type(None) in annotation.__args__)


def _optional_fields(model: type[BaseModel], prefix: str = ""):
    for name, field in model.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            yield from _optional_fields(ann, f"{prefix}{name}.")
        elif _allows_none(ann):
            yield f"{prefix}{name}", field


def test_optional_fields_default_to_none_so_omission_is_lossless():
    # TOML has no null: None values are omitted on export, which is only lossless if every
    # field that accepts None also defaults to None.
    fields = dict(_optional_fields(AnalysisConfig))
    assert "positive_label" in fields and "privacy.include_fingerprints" in fields
    for name, field in fields.items():
        assert field.default is None, name


def test_unrepresentable_values_fail_loudly():
    cfg = AnalysisConfig()
    cfg.columns.probs = ["bad\ud800surrogate"]  # lone surrogate cannot be encoded as UTF-8
    with pytest.raises(ValueError, match="surrogate"):
        config_to_toml(cfg)
    from calibcheck.config import _toml_value
    with pytest.raises(ValueError, match="64-bit"):
        _toml_value(2**64)
    # the schema itself only admits seeds that NumPy and TOML can both represent
    from calibcheck.errors import ConfigError
    for bad in (-1, 2**63):
        with pytest.raises(ConfigError):
            resolve_config({"uncertainty": {"seed": bad}})
    edge = resolve_config({"uncertainty": {"seed": 2**63 - 1}})
    assert resolve_config(tomllib.loads(config_to_toml(edge))) == edge


@given(st.dictionaries(st.text(max_size=12), st.sampled_from(["percent", "logit"]), max_size=4),
       st.lists(st.text(max_size=12), max_size=3))
@settings(max_examples=200, deadline=None)
def test_arbitrary_text_keys_and_values_roundtrip(transforms, probs):
    cfg = resolve_config({"transforms": transforms, "columns": {"probs": probs}})
    text = config_to_toml(cfg)
    again = resolve_config(tomllib.loads(text))
    assert again == cfg
