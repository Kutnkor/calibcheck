"""Typed, JSON-serialisable result objects.

Every number that leaves the scoring engine travels inside one of these models. Undefined
values are represented as ``None`` together with a ``status`` and a ``reason`` code; NaN and
infinity never appear in exported JSON.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from calibcheck._version import SCHEMA_VERSION
from calibcheck.codes import explain
from calibcheck.config import AnalysisConfig

Status = Literal["ok", "unavailable", "failed", "skipped"]
Severity = Literal["info", "warning", "error"]
Direction = Literal["lower_is_better", "higher_is_better", "descriptive"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _finite_or_none(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class Finding(_Model):
    """A structured validation warning, error, or evidence-based finding."""

    code: str
    severity: Severity
    message: str
    field: str | None = None
    count: int | None = None
    model: str | None = None
    population: str | None = None
    caveat: str | None = None
    evidence: dict[str, float | int | str | None] = Field(default_factory=dict)

    @property
    def explanation(self) -> str:
        return explain(self.code)


class Interval(_Model):
    """An uncertainty interval for an aggregate statistic (never for an individual's probability)."""

    status: Status
    method: str
    level: float
    low: float | None = None
    high: float | None = None
    reason: str | None = None
    n_resamples: int | None = None
    n_effective: int | None = None
    n_failed: int | None = None
    seed: int | None = None
    interpretation: str = ""

    _fix_low = field_validator("low", "high", mode="before")(_finite_or_none)


class MetricValue(_Model):
    """One statistic with its scale, availability and population."""

    name: str
    label: str
    category: Literal["overall", "calibration", "discrimination", "threshold", "descriptive",
                      "reference", "skill"]
    value: float | None
    status: Status
    reason: str | None = None
    scale: str
    direction: Direction
    n: int
    population: str
    interval: Interval | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    _fix_value = field_validator("value", mode="before")(_finite_or_none)

    @property
    def available(self) -> bool:
        return self.status == "ok" and self.value is not None


class BinStat(_Model):
    index: int
    lower: float
    upper: float
    closed: Literal["left", "both"]
    count: int
    n_pos: int
    mean_pred: float | None
    observed_rate: float | None
    signed_gap: float | None = Field(
        description="mean_pred - observed_rate; positive means predictions exceed the observed rate")
    abs_gap: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_status: Status = "skipped"
    ci_reason: str | None = None
    low_support: bool = False


class BinningResult(_Model):
    strategy: Literal["uniform", "quantile"]
    n_bins_requested: int
    n_bins_actual: int
    n_nonempty: int
    edges: list[float]
    edge_rule: str
    edge_source: str
    bins: list[BinStat]
    min_support_nonempty: int | None
    support_threshold: int
    ci_method: str | None = None
    ci_level: float | None = None


class SensitivityRow(_Model):
    strategy: Literal["uniform", "quantile"]
    n_bins_requested: int
    n_bins_actual: int
    n_nonempty: int
    min_support: int | None
    ece: float | None
    max_abs_gap: float | None
    rms_gap: float | None
    is_primary: bool


class ClipInfo(_Model):
    eps: float
    n_clipped_low: int
    n_clipped_high: int
    applies_to: list[str]
    note: str = ("Clipping is a numerical stabilisation used only inside logarithms/logits; "
                 "Brier loss, bins and summaries use the original probabilities.")


class ProbabilitySummary(_Model):
    min: float
    max: float
    mean: float
    n_unique: int
    n_zero: int
    n_one: int
    histogram_edges: list[float]
    histogram_counts: list[int]


class ModelResult(_Model):
    name: str
    display_name: str
    n: int
    population: str
    metrics: dict[str, MetricValue]
    binning: BinningResult
    sensitivity: list[SensitivityRow] = Field(default_factory=list)
    clipping: ClipInfo
    summary: ProbabilitySummary
    findings: list[Finding] = Field(default_factory=list)


class ComparisonResult(_Model):
    reference: str
    candidate: str
    metric: str
    definition: str
    delta: float | None
    status: Status
    reason: str | None = None
    favours: Literal["candidate", "reference", "neither", "undetermined"]
    n: int
    population: str
    interval: Interval | None = None
    edge_policy: str | None = None


class GroupRow(_Model):
    name: str
    n: int | None
    n_pos: int | None
    prevalence: float | None
    suppressed: bool = False
    suppression_reason: str | None = None
    low_support: bool = False
    per_model: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    prevalence_ci_low: float | None = None
    prevalence_ci_high: float | None = None
    bins: dict[str, list[BinStat]] = Field(
        default_factory=dict, description="Per-model bin tables on the shared edges.")


class GroupResult(_Model):
    column: str
    n_groups_total: int
    n_groups_shown: int
    n_groups_examined: int
    other_bucket: bool
    missing_category_present: bool
    edges: list[float]
    edge_policy: str
    rows: list[GroupRow]
    interpretation: str = ("Descriptive subgroup performance differences on this data. "
                           "They do not establish causality, discrimination, or legal fairness.")
    exploratory: bool = True


class TimeWindowRow(_Model):
    window: str
    n: int | None
    n_pos: int | None
    prevalence: float | None
    per_model: dict[str, dict[str, float | None]]
    suppressed: bool = False
    suppression_reason: str | None = None


class TimeWindowResult(_Model):
    column: str
    frequency: str
    timezone_handling: str
    n_missing: int
    rows: list[TimeWindowRow]
    suppression_policy: str | None = None
    interpretation: str = ("Prevalence, prediction distribution and calibration can each change "
                           "over time; a single plot does not identify causal drift. Intervals for "
                           "dependent windows are disabled.")


class Exclusion(_Model):
    reason: str
    count: int
    field: str | None = None


class DatasetOverview(_Model):
    n_rows_input: int
    n_rows_retained: int
    n_rows_excluded: int
    exclusions: list[Exclusion] = Field(default_factory=list)
    missing_policy: Literal["fail", "drop"]
    n_pos: int
    n_neg: int
    prevalence: float | None
    label_column: str
    positive_label: str
    negative_label: str | None
    label_mapping: dict[str, int]
    probability_meaning: str
    models: list[str]
    id_column: str | None
    id_kind: Literal["provided", "positional"]
    metadata_fields: dict[str, str | None]
    group_columns: list[str]
    prob_ranges: dict[str, dict[str, float | int]]
    analysis_population: str


class InputRecord(_Model):
    role: str
    name: str | None = None
    path: str | None = None
    sha256: str | None = None
    n_rows: int
    n_columns: int


class RunManifest(_Model):
    schema_version: str = SCHEMA_VERSION
    calibcheck_version: str
    python_version: str
    platform: str
    dependencies: dict[str, str]
    analysis_mode: str
    seed: int | None
    privacy_mode: Literal["private", "public"]
    inputs: list[InputRecord]
    filtering: list[Exclusion]
    protocol: dict[str, Any] = Field(default_factory=dict)
    results_digest: str | None = None
    volatile: dict[str, str] = Field(
        default_factory=dict, description="Timestamps etc. Excluded from results_digest.")


class AnalysisResult(_Model):
    """The full structured output of one analysis run."""

    schema_version: str = SCHEMA_VERSION
    kind: Literal["single", "comparison", "recalibration", "inspect"]
    title: str = "CalibCheck report"
    synthetic: bool = False
    data_note: str | None = None
    config: AnalysisConfig
    dataset: DatasetOverview
    validation: list[Finding] = Field(default_factory=list)
    models: list[ModelResult] = Field(default_factory=list)
    comparisons: list[ComparisonResult] = Field(default_factory=list)
    groups: list[GroupResult] = Field(default_factory=list)
    time_windows: TimeWindowResult | None = None
    findings: list[Finding] = Field(default_factory=list)
    protocol: dict[str, Any] = Field(default_factory=dict)
    manifest: RunManifest | None = None

    def model(self, name: str) -> ModelResult:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)

    def metrics_table(self) -> list[dict[str, Any]]:
        """Flat rows (model, metric, value, status, ...) for notebooks and CSV export."""
        rows = []
        for m in self.models:
            for mv in m.metrics.values():
                rows.append({
                    "model": m.name, "metric": mv.name, "label": mv.label,
                    "category": mv.category, "value": mv.value, "status": mv.status,
                    "reason": mv.reason, "n": mv.n, "population": mv.population,
                    "ci_low": mv.interval.low if mv.interval else None,
                    "ci_high": mv.interval.high if mv.interval else None,
                    "ci_method": mv.interval.method if mv.interval else None,
                })
        return rows


class InspectResult(_Model):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["inspect"] = "inspect"
    validation_only: bool = True
    valid: bool
    dataset: DatasetOverview | None
    findings: list[Finding]
    config: AnalysisConfig
