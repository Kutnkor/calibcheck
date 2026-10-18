"""Typed, validated analysis configuration.

Precedence (highest first): explicit command-line / function arguments, then values from a
TOML configuration file, then the defaults defined here. Unknown keys are rejected so a
misspelled option never silently falls back to a default. Configuration files are parsed as
data only; nothing in them is executed.
"""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from calibcheck.errors import ConfigError

Dependence = Literal["independent", "entity_clustered", "time_dependent", "unknown"]
BinStrategy = Literal["uniform", "quantile"]
_DEFAULT_STRATEGIES: tuple[BinStrategy, ...] = ("uniform", "quantile")
MIN_CLIP_EPS = 2.0**-52  # float64 machine epsilon; keeps 1 - eps below 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ColumnMapping(_Strict):
    """Which CSV columns play which role. Probability columns are never guessed."""

    label: str = "y_true"
    probs: list[str] = Field(default_factory=list)
    id: str | None = None
    entity: str | None = None
    timestamp: str | None = None
    split: str | None = None
    groups: list[str] = Field(default_factory=list)

    def role_columns(self) -> list[tuple[str, str]]:
        """(role, column) pairs for every mapped column."""
        pairs: list[tuple[str, str]] = [("label", self.label)]
        pairs += [("prob", p) for p in self.probs]
        for role in ("id", "entity", "timestamp", "split"):
            col = getattr(self, role)
            if col is not None:
                pairs.append((role, col))
        pairs += [("group", g) for g in self.groups]
        return pairs


class BinningConfig(_Strict):
    strategy: BinStrategy = "uniform"
    n_bins: int = Field(10, ge=1, le=200)
    min_support: int = Field(30, ge=1, description="Practical warning threshold per bin (heuristic).")
    sensitivity: bool = True
    sensitivity_bins: list[int] = Field(default_factory=lambda: [5, 10, 20])
    sensitivity_strategies: list[BinStrategy] = Field(default_factory=lambda: list(_DEFAULT_STRATEGIES))

    @field_validator("sensitivity_bins")
    @classmethod
    def _bins_positive(cls, v: list[int]) -> list[int]:
        if any(b < 1 or b > 200 for b in v):
            raise ValueError("sensitivity bin counts must be between 1 and 200")
        return sorted(set(v))


class UncertaintyConfig(_Strict):
    level: float = Field(0.95, gt=0.5, lt=1.0)
    wilson: bool = True
    bootstrap: bool = False
    n_resamples: int = Field(2000, ge=100, le=100_000)
    seed: int = Field(20251018, ge=0, le=2**63 - 1,
                      description="NumPy seed; non-negative and within TOML's 64-bit range.")
    bootstrap_bin_metrics: bool = Field(
        False, description="Exploratory ECE intervals with frozen bin edges.")
    max_elements_per_batch: int = Field(4_000_000, ge=10_000)


class GroupConfig(_Strict):
    max_groups: int = Field(12, ge=1, le=100)
    min_group_size: int = Field(30, ge=1)
    max_cardinality: int = Field(50, ge=2, le=10_000)


class PrivacyConfig(_Strict):
    mode: Literal["private", "public"] = "private"
    min_cell: int = Field(10, ge=1)
    reveal_group_names: bool = False
    include_fingerprints: bool | None = None  # None -> True in private, False in public
    export_retained_rows: bool = False

    def fingerprints_enabled(self) -> bool:
        if self.include_fingerprints is None:
            return self.mode == "private"
        return self.include_fingerprints


class AnalysisConfig(_Strict):
    """Everything that determines a CalibCheck analysis. Exported with every run."""

    columns: ColumnMapping = Field(default_factory=ColumnMapping)
    positive_label: str | None = None
    missing_policy: Literal["fail", "drop"] = "fail"
    transforms: dict[str, Literal["percent", "logit"]] = Field(
        default_factory=dict,
        description="Explicit, recorded per-column transforms: 'percent' divides by 100, "
                    "'logit' applies the logistic function. Never applied implicitly.")
    clip_eps: float = Field(1e-15, ge=MIN_CLIP_EPS, lt=1e-3)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    reference_prevalence: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Prevalence from training/development data, used as a fixed reference.")
    binning: BinningConfig = Field(default_factory=BinningConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    dependence: Dependence = "independent"
    groups: GroupConfig = Field(default_factory=GroupConfig)
    time_window: Literal["D", "W", "M", "Q", "Y"] | None = None
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    reference_model: str | None = None
    pairwise: bool = False
    primary_metric: Literal["brier", "log_loss"] | None = None
    max_file_mb: float = Field(500.0, gt=0)
    max_columns: int = Field(2000, ge=1)

    @model_validator(mode="after")
    def _check_reference_model(self) -> AnalysisConfig:
        if self.reference_model is not None and self.columns.probs and \
                self.reference_model not in self.columns.probs:
            raise ValueError(
                f"reference_model {self.reference_model!r} is not among the probability columns")
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Parse a TOML configuration file into a plain dict (no code execution)."""
    p = Path(path)
    try:
        with p.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {p.name}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Configuration file is not valid TOML: {exc}") from exc
    if "calibcheck" in data and isinstance(data["calibcheck"], dict) and len(data) == 1:
        data = data["calibcheck"]
    return data


def resolve_config(file_values: dict[str, Any] | None = None,
                   overrides: dict[str, Any] | None = None) -> AnalysisConfig:
    """Build an :class:`AnalysisConfig` from defaults < file values < overrides.

    ``overrides`` uses the same nested shape as the TOML file; ``None`` leaves mean
    "not specified" and are removed before merging so they do not mask file values.
    """
    merged = _deep_merge({}, file_values or {})
    merged = _deep_merge(merged, _drop_none(overrides or {}))
    try:
        return AnalysisConfig.model_validate(merged)
    except ValidationError as exc:
        msgs = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            msgs.append(f"{loc}: {err['msg']}")
        raise ConfigError("Invalid configuration:\n  " + "\n  ".join(msgs)) from exc


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            sub = _drop_none(v)
            if sub:
                out[k] = sub
        else:
            out[k] = v
    return out


def config_to_toml(cfg: AnalysisConfig) -> str:
    """Serialise a resolved configuration as TOML (for reproduction from the CLI).

    Keys that are not TOML bare keys (``A-Za-z0-9_-``), such as column names containing dots,
    spaces, quotes or Unicode, are written as quoted keys, so a literal key like ``model.v1``
    never turns into nested tables. Strings are escaped per TOML 1.0, including all control
    characters. ``None`` values are omitted because TOML has no null; every optional field
    defaults to ``None``, so the reloaded configuration is identical. Values that TOML cannot
    represent (lone surrogates, integers outside 64 bits) raise ``ValueError`` instead of
    being altered.
    """
    data = cfg.model_dump(mode="json")
    return _to_toml(data)


_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n",
            "\f": "\\f", "\r": "\\r"}
_INT64 = (-(2**63), 2**63 - 1)


def _toml_string(s: str) -> str:
    """A TOML basic string with every required escape."""
    out = []
    for ch in s:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ch < " " or ch == "\x7f":
            out.append(f"\\u{ord(ch):04X}")
        elif "\ud800" <= ch <= "\udfff":
            raise ValueError(f"cannot write {s!r} to TOML: it contains a lone surrogate "
                             f"(U+{ord(ch):04X}), which is not valid Unicode text")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _toml_key(key: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"TOML keys must be strings, got {type(key).__name__}")
    return key if _BARE_KEY.fullmatch(key) else _toml_string(key)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        if not _INT64[0] <= v <= _INT64[1]:
            raise ValueError(f"cannot write integer {v} to TOML: outside the 64-bit range")
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)  # shortest repr round-trips exactly
    if isinstance(v, str):
        return _toml_string(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        items = [f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items() if x is not None]
        return "{" + ", ".join(items) + "}"
    raise TypeError(f"cannot serialise {type(v).__name__} to TOML")


def _to_toml(data: dict[str, Any], path: tuple[str, ...] = ()) -> str:
    """Scalars first, then one ``[table]`` per nested dict; each path segment is quoted
    independently, so literal dots in keys stay literal."""
    blocks: list[str] = []
    scalars = [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in data.items()
               if v is not None and not isinstance(v, dict)]
    if path:
        header = "[" + ".".join(_toml_key(p) for p in path) + "]"
        blocks.append("\n".join([header, *scalars]))
    elif scalars:
        blocks.append("\n".join(scalars))
    for k, v in data.items():
        if isinstance(v, dict):
            blocks.append(_to_toml(v, (*path, k)).strip())
    return "\n\n".join(b for b in blocks if b) + "\n"
