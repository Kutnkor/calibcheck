"""Loading and strict validation of prediction files.

The input contract (v0.1): a CSV (or a pandas DataFrame through the same adapter) in wide
format with one outcome column, one or more *explicitly selected* probability columns giving
P(Y = positive_label), and optional ID / entity / timestamp / split / subgroup columns.

Nothing is silently repaired: invalid values either stop the run (default ``fail`` policy)
or, under the explicit ``drop`` policy, remove whole rows with every exclusion counted.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibcheck.config import AnalysisConfig
from calibcheck.errors import InputValidationError
from calibcheck.schemas import DatasetOverview, Exclusion, Finding, InputRecord

MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "<na>", "nat"})
MISSING_GROUP = "(missing)"


@dataclass(frozen=True)
class PredictionData:
    """Validated, analysis-ready arrays. Construct with :func:`load_predictions`."""

    y: np.ndarray                      # int8, 1 = declared positive label
    probs: dict[str, np.ndarray]       # float64 in [0, 1], original valid values
    ids: np.ndarray                    # object array of str
    id_kind: str                       # "provided" | "positional"
    positive_label: str
    negative_label: str | None
    label_column: str
    id_column: str | None = None
    entity: np.ndarray | None = None
    timestamps: pd.Series | None = None
    timezone_handling: str | None = None
    n_timestamp_missing: int = 0
    split: np.ndarray | None = None
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    n_rows_input: int = 0
    input_record: InputRecord | None = None
    transforms: dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def models(self) -> list[str]:
        return list(self.probs)

    def subset(self, mask: np.ndarray, *, note: str | None = None) -> PredictionData:
        """Return the rows selected by a boolean mask (for data roles, splits, groups)."""
        mask = np.asarray(mask, dtype=bool)
        ts = self.timestamps
        if ts is not None:
            ts = ts[mask].reset_index(drop=True)
        return replace(
            self,
            y=self.y[mask],
            probs={k: v[mask] for k, v in self.probs.items()},
            ids=self.ids[mask],
            entity=None if self.entity is None else self.entity[mask],
            timestamps=ts,
            split=None if self.split is None else self.split[mask],
            groups={k: v[mask] for k, v in self.groups.items()},
        )

    def with_probs(self, probs: dict[str, np.ndarray]) -> PredictionData:
        for name, p in probs.items():
            if p.shape != self.y.shape:
                raise ValueError(f"probability array {name!r} has the wrong length")
        return replace(self, probs=dict(probs))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _is_missing_str(values: pd.Series) -> np.ndarray:
    s = values.astype(str).str.strip().str.lower()
    return np.asarray(s.isin(MISSING_TOKENS) | values.isna(), dtype=bool)


def canonical_label(value: Any) -> str:
    """Canonical string form of a label: numeric integral values map to e.g. '1', '0'."""
    s = str(value).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    if np.isfinite(f) and float(f).is_integer():
        return str(int(f))
    return s


def _label_series(raw: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (canonical label strings, missing mask)."""
    missing = _is_missing_str(raw)
    text = raw.astype(str).str.strip()
    mapping = {u: canonical_label(u) for u in pd.unique(text[~missing])}
    labels = text.map(mapping).where(~missing, "").to_numpy(dtype=object)
    return labels, missing


def _text_column(raw: pd.Series, missing_value: str = MISSING_GROUP) -> tuple[np.ndarray, np.ndarray]:
    """Stripped text values with missing entries replaced by a visible category."""
    miss = _is_missing_str(raw)
    vals = raw.astype(str).str.strip().where(~miss, missing_value).to_numpy(dtype=object)
    return vals, miss


def _parse_probs(raw: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (float64 values, missing mask, nonnumeric mask)."""
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
        vals = np.array(raw.to_numpy(dtype="float64", na_value=np.nan), copy=True)
        missing = np.isnan(vals) & raw.isna().to_numpy()
        return vals, missing, np.zeros(len(raw), dtype=bool)
    missing = _is_missing_str(raw)
    stripped = raw.astype(str).str.strip()
    vals = np.array(pd.to_numeric(stripped.where(~missing, None), errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan), copy=True)
    lowered = stripped.str.lower()
    explicit_nonfinite = lowered.isin({"inf", "+inf", "-inf", "infinity", "-infinity"}).to_numpy()
    vals[explicit_nonfinite] = np.where(lowered[explicit_nonfinite].str.startswith("-"),
                                        -np.inf, np.inf)
    nonnumeric = ~missing & np.isnan(vals) & ~explicit_nonfinite
    return vals, missing, nonnumeric


def _read_csv(path: Path, cfg: AnalysisConfig) -> tuple[pd.DataFrame, list[Finding], bytes]:
    if not path.exists():
        raise InputValidationError(f"Input file not found: {path.name}", code="INPUT_UNREADABLE")
    if not path.is_file():
        raise InputValidationError("Input must be a readable CSV file, not a directory.", code="INPUT_UNREADABLE")
    try:
        size_mb = path.stat().st_size / 1e6
    except OSError as exc:
        raise InputValidationError("Cannot inspect the input file.", code="INPUT_UNREADABLE") from exc
    if size_mb > cfg.max_file_mb:
        raise InputValidationError(
            f"Input file is {size_mb:.1f} MB, above the configured limit of {cfg.max_file_mb} MB "
            "(raise max_file_mb explicitly to analyse it).", code="INPUT_TOO_LARGE")
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise InputValidationError("Cannot read the input file; check access permissions.",
                                   code="INPUT_UNREADABLE") from exc
    df, findings = read_csv_bytes(raw_bytes, cfg)
    return df, findings, raw_bytes


def read_csv_bytes(raw_bytes: bytes, cfg: AnalysisConfig) -> tuple[pd.DataFrame, list[Finding]]:
    """Parse CSV bytes with every column as text (no type guessing, no NA coercion)."""
    findings: list[Finding] = []
    if len(raw_bytes) / 1e6 > cfg.max_file_mb:
        raise InputValidationError(
            f"Input is {len(raw_bytes) / 1e6:.1f} MB, above the configured limit of "
            f"{cfg.max_file_mb} MB.", code="INPUT_TOO_LARGE")
    try:
        text = raw_bytes.decode("utf-8-sig", errors="strict") if raw_bytes else ""
    except UnicodeDecodeError as exc:
        raise InputValidationError("The file is not UTF-8 encoded text.",
                                   code="INPUT_UNREADABLE") from exc
    if not text.strip():
        raise InputValidationError("The file is empty.", code="INPUT_EMPTY")
    try:
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise InputValidationError(f"Could not parse the CSV header: {exc}",
                                   code="INPUT_UNREADABLE") from exc
    header = [h.strip() for h in header]
    if not header or any(not h for h in header):
        raise InputValidationError("CSV column headers must be nonempty; name every column.",
                                   code="INPUT_EMPTY_HEADER")
    dupes = sorted(k for k, c in Counter(header).items() if c > 1)
    if dupes:
        raise InputValidationError(
            f"Duplicate column headers: {', '.join(repr(d) for d in dupes)}.",
            code="INPUT_DUPLICATE_HEADER")
    if len(header) > cfg.max_columns:
        raise InputValidationError(
            f"The file has {len(header)} columns, above the limit of {cfg.max_columns}.",
            code="INPUT_TOO_MANY_COLUMNS")
    try:
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != len(header):
                raise InputValidationError(
                    f"CSV record ending at line {reader.line_num} has {len(row)} fields; "
                    f"the header has {len(header)}. Fix the row width before analysis.",
                    code="INPUT_SHAPE")
    except csv.Error as exc:
        raise InputValidationError("Malformed CSV quoting; provide a valid CSV file.",
                                   code="INPUT_UNREADABLE") from exc
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_filter=False,
                         skipinitialspace=False)
    except (pd.errors.ParserError, ValueError) as exc:
        raise InputValidationError(f"Could not parse the CSV body: {exc}",
                                   code="INPUT_UNREADABLE") from exc
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        raise InputValidationError("The file has a header but no data rows.", code="INPUT_EMPTY")
    return df, findings


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------

def load_predictions(source: str | Path | pd.DataFrame | bytes, cfg: AnalysisConfig, *,
                     role: str = "evaluation", name: str | None = None) -> PredictionData:
    """Load and validate predictions according to ``cfg``.

    Raises :class:`InputValidationError` (with structured findings) when the input violates
    the contract. Returns validated data with warnings for valid-but-limited inputs.
    """
    if isinstance(source, bytes | bytearray):
        df, upload_findings = read_csv_bytes(bytes(source), cfg)
        record = InputRecord(role=role, name=name or "<upload>",
                             sha256=hashlib.sha256(source).hexdigest(), n_rows=len(df),
                             n_columns=df.shape[1])
        return validate_frame(df, cfg, findings=upload_findings, record=record)
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        df.columns = [str(c).strip() for c in df.columns]
        dupes = sorted(k for k, c in Counter(df.columns).items() if c > 1)
        if dupes:
            raise InputValidationError(f"Duplicate column headers: {dupes}",
                                       code="INPUT_DUPLICATE_HEADER")
        if len(df) == 0:
            raise InputValidationError("The data frame has no rows.", code="INPUT_EMPTY")
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy().tobytes()
        ).hexdigest()
        record = InputRecord(role=role, name="<DataFrame>", sha256=digest, n_rows=len(df),
                             n_columns=df.shape[1])
        findings: list[Finding] = []
    else:
        path = Path(source)
        df, findings, raw = _read_csv(path, cfg)
        record = InputRecord(role=role, name=path.name, path=str(path),
                             sha256=hashlib.sha256(raw).hexdigest(), n_rows=len(df),
                             n_columns=df.shape[1])
    return validate_frame(df, cfg, findings=findings, record=record)


def validate_frame(df: pd.DataFrame, cfg: AnalysisConfig, *, findings: list[Finding] | None = None,
                   record: InputRecord | None = None) -> PredictionData:
    findings = list(findings or [])
    errors: list[Finding] = []
    cols = cfg.columns
    n_input = len(df)

    # ---- schema mapping ---------------------------------------------------------------
    if not cols.probs:
        raise InputValidationError(
            "No probability column selected. Name them explicitly (e.g. --prob p_model_a); "
            "CalibCheck does not guess probability columns from numeric columns.",
            code="INPUT_NO_PROB_COLUMNS")
    role_cols = cols.role_columns()
    usage = Counter(c for _, c in role_cols)
    ambiguous = sorted(c for c, k in usage.items() if k > 1)
    if ambiguous:
        raise InputValidationError(
            f"Column(s) mapped to more than one role: {', '.join(map(repr, ambiguous))}.",
            code="INPUT_AMBIGUOUS_MAPPING")
    missing_cols = [(r, c) for r, c in role_cols if c not in df.columns]
    if missing_cols:
        msg = "; ".join(f"{r} column {c!r}" for r, c in missing_cols)
        avail = ", ".join(repr(c) for c in list(df.columns)[:30])
        raise InputValidationError(f"Missing required column(s): {msg}. Available: {avail}",
                                   code="INPUT_MISSING_COLUMN")

    # ---- labels -----------------------------------------------------------------------
    labels, label_missing = _label_series(df[cols.label])
    distinct = sorted(set(labels[~label_missing].tolist()))
    if len(distinct) > 2:
        shown = ", ".join(map(repr, distinct[:6])) + (" ..." if len(distinct) > 6 else "")
        raise InputValidationError(
            f"Outcome column {cols.label!r} has {len(distinct)} distinct values ({shown}). "
            "CalibCheck v0.1 supports binary outcomes only; multiclass calibration is on the "
            "roadmap.", code="LABEL_NOT_BINARY")
    if not distinct:
        raise InputValidationError(f"Outcome column {cols.label!r} has no non-missing values.",
                                   code="LABEL_MISSING")
    if cfg.positive_label is not None:
        pos = canonical_label(cfg.positive_label)
        if pos not in distinct:
            if len(distinct) == 2:
                raise InputValidationError(
                    f"Declared positive label {cfg.positive_label!r} does not match the observed "
                    f"labels {distinct}.", code="LABEL_POSITIVE_NOT_FOUND")
            findings.append(Finding(
                code="LABEL_POSITIVE_NOT_FOUND", severity="warning", field=cols.label,
                message=(f"Declared positive label {pos!r} does not occur; every retained "
                         f"outcome is the negative label {distinct[0]!r}.")))
        neg_candidates = [d for d in distinct if d != pos]
        neg = neg_candidates[0] if neg_candidates else None
    else:
        if set(distinct) <= {"0", "1"}:
            pos, neg = "1", "0"
        else:
            raise InputValidationError(
                f"Outcome labels {distinct} are not 0/1, so the positive label must be declared "
                "explicitly (e.g. --positive-label yes).", code="LABEL_POSITIVE_UNDECLARED")
    y_full = (labels == pos).astype(np.int8)

    # ---- probabilities ----------------------------------------------------------------
    drop = cfg.missing_policy == "drop"
    invalid_any = label_missing.copy()
    exclusions: list[Exclusion] = []
    if label_missing.any():
        f = Finding(code="LABEL_MISSING", severity="warning" if drop else "error",
                    field=cols.label, count=int(label_missing.sum()),
                    message=f"{int(label_missing.sum())} row(s) have a missing outcome in "
                            f"{cols.label!r}.")
        (findings if drop else errors).append(f)
        exclusions.append(Exclusion(reason="LABEL_MISSING", count=int(label_missing.sum()),
                                    field=cols.label))

    probs_full: dict[str, np.ndarray] = {}
    transforms_applied: dict[str, str] = {}
    for col in cols.probs:
        vals, miss, nonnum = _parse_probs(df[col])
        tr = cfg.transforms.get(col)
        if tr is not None:
            ok = ~miss & ~nonnum & np.isfinite(vals)
            if tr == "percent":
                vals = np.where(ok, vals / 100.0, vals)
                desc = "divided by 100 (declared percent scale)"
            else:
                with np.errstate(over="ignore"):
                    vals = np.where(ok, 1.0 / (1.0 + np.exp(-vals)), vals)
                desc = "logistic function applied (declared logit scale)"
            transforms_applied[col] = tr
            findings.append(Finding(code="TRANSFORM_APPLIED", severity="info", field=col,
                                    message=f"Column {col!r}: {desc}, as explicitly configured."))
        nonfinite = ~miss & ~nonnum & ~np.isfinite(vals)
        finite = np.isfinite(vals)
        out_of_range = finite & ((vals < 0.0) | (vals > 1.0))
        problems = [("PROB_MISSING", miss, "missing"), ("PROB_NONNUMERIC", nonnum, "non-numeric"),
                    ("PROB_NONFINITE", nonfinite, "NaN/infinite"),
                    ("PROB_OUT_OF_RANGE", out_of_range, "outside [0, 1]")]
        for code, mask, what in problems:
            k = int(mask.sum())
            if not k:
                continue
            msg = f"Column {col!r}: {k} value(s) {what}."
            if code == "PROB_OUT_OF_RANGE":
                fin = vals[finite]
                if fin.size and fin.min() >= 0 and fin.max() <= 100 and fin.max() > 1:
                    msg += (" The values look like percentages; CalibCheck does not divide by 100 "
                            "automatically. Declare transforms = {col = 'percent'} if that is the "
                            "intended meaning.")
                elif fin.size and fin.min() < 0:
                    msg += (" Negative values suggest scores or logits, which are not "
                            "probabilities; declare a 'logit' transform only if that is correct.")
            f = Finding(code=code, severity="warning" if drop else "error", field=col, count=k,
                        message=msg)
            (findings if drop else errors).append(f)
            exclusions.append(Exclusion(reason=code, count=k, field=col))
            invalid_any |= mask
        probs_full[col] = vals

    # ---- IDs ---------------------------------------------------------------------------
    if cols.id is not None:
        raw_ids = df[cols.id]
        id_missing = _is_missing_str(raw_ids)
        ids_full = raw_ids.astype(str).str.strip().to_numpy(dtype=object)
        if id_missing.any():
            f = Finding(code="ID_MISSING", severity="warning" if drop else "error",
                        field=cols.id, count=int(id_missing.sum()),
                        message=f"{int(id_missing.sum())} row(s) have a missing ID in {cols.id!r}.")
            (findings if drop else errors).append(f)
            exclusions.append(Exclusion(reason="ID_MISSING", count=int(id_missing.sum()),
                                        field=cols.id))
            invalid_any |= id_missing
        present = ids_full[~id_missing]
        counts = Counter(present.tolist())
        dup_keys = [k for k, c in counts.items() if c > 1]
        if dup_keys:
            n_dup_rows = sum(counts[k] for k in dup_keys)
            errors.append(Finding(
                code="ID_DUPLICATE", severity="error", field=cols.id, count=n_dup_rows,
                message=f"{len(dup_keys)} ID value(s) occur more than once ({n_dup_rows} rows). "
                        "IDs must be unique; duplicates are never dropped automatically."))
        id_kind = "provided"
    else:
        width = max(6, len(str(n_input)))
        ids_full = np.array([f"row{i:0{width}d}" for i in range(n_input)], dtype=object)
        id_kind = "positional"
        findings.append(Finding(
            code="ID_POSITIONAL", severity="info",
            message="No ID column given; deterministic positional IDs were created. They identify "
                    "rows within this file only and cannot align separate files."))

    if errors:
        raise InputValidationError(
            "Input validation failed. Fix the issues below or use the explicit drop-invalid "
            "policy (--drop-invalid) where appropriate.", findings=errors + findings,
            code=errors[0].code)

    keep = ~invalid_any
    n_retained = int(keep.sum())
    if n_retained == 0:
        raise InputValidationError("No rows remain after excluding invalid rows.",
                                   code="UNAVAILABLE_EMPTY", findings=findings)
    if exclusions:
        findings.append(Finding(
            code="ROWS_EXCLUDED", severity="warning", count=int((~keep).sum()),
            message=f"Excluded {int((~keep).sum())} of {n_input} rows under the explicit "
                    f"drop-invalid policy; {n_retained} retained. Every model is evaluated on the "
                    "same retained rows."))

    y = y_full[keep]
    probs = {k: v[keep].astype(np.float64) for k, v in probs_full.items()}
    ids = ids_full[keep]

    # ---- optional metadata ---------------------------------------------------------------
    entity = None
    if cols.entity is not None:
        entity, e_miss = _text_column(df[cols.entity][keep])
        if e_miss.any():
            findings.append(Finding(code="GROUP_MISSING_VALUES", severity="warning",
                                    field=cols.entity, count=int(e_miss.sum()),
                                    message=f"{int(e_miss.sum())} row(s) have a missing entity ID."))
    split = None
    if cols.split is not None:
        split, _ = _text_column(df[cols.split][keep])

    groups: dict[str, np.ndarray] = {}
    for gcol in cols.groups:
        vals_g, g_miss = _text_column(df[gcol][keep])
        n_levels = int(pd.unique(vals_g).size)
        if n_levels > cfg.groups.max_cardinality:
            raise InputValidationError(
                f"Subgroup column {gcol!r} has {n_levels} distinct values (limit "
                f"{cfg.groups.max_cardinality}). High-cardinality or identifying fields need "
                "explicit handling: bin them first or raise groups.max_cardinality deliberately.",
                code="GROUP_HIGH_CARDINALITY")
        if g_miss.any():
            findings.append(Finding(
                code="GROUP_MISSING_VALUES", severity="info", field=gcol, count=int(g_miss.sum()),
                message=f"{int(g_miss.sum())} row(s) have no value in {gcol!r}; they appear as "
                        f"{MISSING_GROUP!r} rather than being deleted."))
        groups[gcol] = vals_g

    timestamps = None
    tz_note = None
    n_ts_missing = 0
    if cols.timestamp is not None:
        t_raw = df[cols.timestamp][keep].reset_index(drop=True)
        t_miss = _is_missing_str(t_raw)
        try:
            parsed = pd.to_datetime(t_raw.where(~pd.Series(t_miss), None), utc=True,
                                    errors="coerce", format="mixed")
        except (ValueError, TypeError):
            parsed = pd.Series(pd.NaT, index=t_raw.index, dtype="datetime64[ns, UTC]")
        bad = (~t_miss) & parsed.isna().to_numpy()
        if bad.any():
            findings.append(Finding(
                code="TIME_UNPARSEABLE", severity="warning", field=cols.timestamp,
                count=int(bad.sum()),
                message=f"{int(bad.sum())} timestamp(s) in {cols.timestamp!r} could not be parsed; "
                        "time-window views are disabled rather than silently dropping rows."))
        else:
            timestamps = parsed
            tz_note = ("Timestamps parsed with pandas.to_datetime(utc=True): offset-aware values "
                       "converted to UTC; naive values interpreted as UTC.")
            n_ts_missing = int(t_miss.sum())
            if n_ts_missing:
                findings.append(Finding(
                    code="TIME_MISSING", severity="info", field=cols.timestamp, count=n_ts_missing,
                    message=f"{n_ts_missing} row(s) have no timestamp; they are omitted from "
                            "time-window views only."))

    # ---- descriptive warnings -----------------------------------------------------------------
    n_pos = int(y.sum())
    if n_pos in (0, len(y)):
        findings.append(Finding(
            code="LABEL_SINGLE_CLASS", severity="warning", field=cols.label,
            message=f"All {len(y)} retained outcomes belong to one class "
                    f"({'positive' if n_pos else 'negative'}). Brier and log loss remain defined; "
                    "ROC-AUC and average precision are unavailable."))
    for col, p in probs.items():
        uniq = np.unique(p)
        if uniq.size == 1:
            findings.append(Finding(code="PROB_CONSTANT", severity="warning", field=col,
                                    model=col, message=f"Column {col!r} is constant "
                                    f"({uniq[0]:.6g}); it carries no ranking information."))
        elif uniq.size == 2:
            findings.append(Finding(
                code="PROB_TWO_VALUED", severity="warning", field=col, model=col,
                message=f"Column {col!r} takes only two values ({uniq[0]:.6g}, {uniq[1]:.6g}). If "
                        "these are hard class predictions rather than probabilities, calibration "
                        "analysis is limited."))
        n0, n1 = int((p == 0.0).sum()), int((p == 1.0).sum())
        if n0 or n1:
            findings.append(Finding(
                code="PROB_ENDPOINTS", severity="info", field=col, model=col, count=n0 + n1,
                message=f"Column {col!r}: {n0} value(s) exactly 0 and {n1} exactly 1. Brier loss "
                        f"uses them unchanged; log loss clips them to [eps, 1-eps] with "
                        f"eps={cfg.clip_eps:g}."))

    return PredictionData(
        y=y, probs=probs, ids=ids, id_kind=id_kind, positive_label=pos, negative_label=neg,
        label_column=cols.label, id_column=cols.id, entity=entity, timestamps=timestamps,
        timezone_handling=tz_note, n_timestamp_missing=n_ts_missing, split=split, groups=groups,
        findings=findings, exclusions=exclusions, n_rows_input=n_input, input_record=record,
        transforms=transforms_applied)


def overview(data: PredictionData, cfg: AnalysisConfig, *,
             population: str = "analysis population") -> DatasetOverview:
    n_pos = int(data.y.sum())
    n = data.n
    ranges: dict[str, dict[str, float | int]] = {}
    for k, p in data.probs.items():
        ranges[k] = {"min": float(p.min()), "max": float(p.max()), "mean": float(p.mean()),
                     "n_zero": int((p == 0).sum()), "n_one": int((p == 1).sum()),
                     "n_unique": int(np.unique(p).size)}
    mapping = {data.positive_label: 1}
    if data.negative_label is not None:
        mapping[data.negative_label] = 0
    return DatasetOverview(
        n_rows_input=data.n_rows_input, n_rows_retained=n,
        n_rows_excluded=data.n_rows_input - n, exclusions=data.exclusions,
        missing_policy=cfg.missing_policy, n_pos=n_pos, n_neg=n - n_pos,
        prevalence=n_pos / n if n else None, label_column=data.label_column,
        positive_label=data.positive_label, negative_label=data.negative_label,
        label_mapping=mapping,
        probability_meaning=f"P({data.label_column} = {data.positive_label!r})",
        models=data.models, id_column=data.id_column, id_kind=data.id_kind,
        metadata_fields={"entity": cfg.columns.entity, "timestamp": cfg.columns.timestamp,
                         "split": cfg.columns.split},
        group_columns=list(data.groups), prob_ranges=ranges,
        analysis_population=f"{population}: {n} of {data.n_rows_input} input rows")


# --------------------------------------------------------------------------------------
# alignment of separately supplied files
# --------------------------------------------------------------------------------------

def align_by_id(a: PredictionData, b: PredictionData, *, allow_intersection: bool = False,
                suffixes: tuple[str, str] = ("", "_b")) -> tuple[PredictionData, list[Finding]]:
    """Pair two validated prediction sets by unique row ID (never by position).

    Rejects positional IDs, contradictory outcomes and unmatched rows (unless
    ``allow_intersection`` is set, in which case the loss is reported).
    """
    if a.id_kind != "provided" or b.id_kind != "provided":
        raise InputValidationError(
            "Separate files can only be paired by explicit, unique ID columns; positional "
            "alignment is never used.", code="ALIGN_DUPLICATE_KEY")
    if a.positive_label != b.positive_label:
        raise InputValidationError(
            f"Files declare different positive labels ({a.positive_label!r} vs "
            f"{b.positive_label!r}).", code="ALIGN_LABEL_CONFLICT")
    ia = pd.Index(a.ids)
    ib = pd.Index(b.ids)
    common = ia.intersection(ib, sort=False)
    findings: list[Finding] = []
    only_a = len(ia) - len(common)
    only_b = len(ib) - len(common)
    if only_a or only_b:
        f = Finding(code="ALIGN_UNMATCHED", severity="warning" if allow_intersection else "error",
                    count=only_a + only_b,
                    message=f"{only_a} row(s) only in the first file and {only_b} only in the "
                            f"second; {len(common)} matched.")
        if not allow_intersection:
            raise InputValidationError(
                "Files do not contain the same IDs. Re-run with an explicit intersection "
                "(--intersection) to compare on the matched rows only.", findings=[f],
                code="ALIGN_UNMATCHED")
        findings.append(f)
    if len(common) == 0:
        raise InputValidationError("The files share no IDs.", code="ALIGN_UNMATCHED")
    pa = ia.get_indexer(common)
    pb = ib.get_indexer(common)
    conflict = a.y[pa] != b.y[pb]
    if conflict.any():
        raise InputValidationError(
            f"{int(conflict.sum())} matched ID(s) have contradictory outcomes between the files.",
            code="ALIGN_LABEL_CONFLICT")
    metadata = [("entity", a.entity, b.entity), ("split", a.split, b.split),
                ("timestamp", a.timestamps, b.timestamps)]
    metadata += [(name, a.groups[name], b.groups[name]) for name in sorted(a.groups.keys() & b.groups.keys())]
    for name, left, right in metadata:
        if left is None or right is None:
            continue
        lhs = pd.Series(np.asarray(left)[pa]).reset_index(drop=True)
        rhs = pd.Series(np.asarray(right)[pb]).reset_index(drop=True)
        if not lhs.equals(rhs):
            raise InputValidationError(
                f"Matched IDs have conflicting {name!r} metadata; resolve the conflict before comparison.",
                code="ALIGN_METADATA_CONFLICT")
    sa = a.subset(np.isin(np.arange(a.n), pa))
    order = np.argsort(pa)
    # sa rows are in a's original order; reorder b to match
    pb_sorted = pb[order]
    probs = dict(sa.probs)
    for name, p in b.probs.items():
        key = name if name not in probs else f"{name}{suffixes[1]}"
        if key in probs:
            raise InputValidationError(f"Probability column name collision: {name!r}",
                                       code="INPUT_AMBIGUOUS_MAPPING")
        probs[key] = p[pb_sorted]
    return replace(sa, probs=probs, findings=sa.findings + findings), findings


def load_probabilities_only(path: str | Path, prob_col: str, cfg: AnalysisConfig, *,
                            id_col: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a file for ``apply``: outcomes are not needed. Returns (raw frame, probabilities).

    The raw frame keeps every original column as text so the output preserves IDs exactly.
    """
    df, _, _ = _read_csv(Path(path), cfg)
    if prob_col not in df.columns:
        raise InputValidationError(f"Probability column {prob_col!r} not found.",
                                   code="INPUT_MISSING_COLUMN")
    vals, miss, nonnum = _parse_probs(df[prob_col])
    bad = miss | nonnum | ~np.isfinite(vals) | (np.isfinite(vals) & ((vals < 0) | (vals > 1)))
    if bad.any():
        raise InputValidationError(
            f"Column {prob_col!r}: {int(bad.sum())} value(s) are missing or not valid "
            "probabilities in [0, 1]. Nothing was written.", code="PROB_OUT_OF_RANGE")
    if id_col is not None:
        if id_col not in df.columns:
            raise InputValidationError(f"ID column {id_col!r} not found.",
                                       code="INPUT_MISSING_COLUMN")
        ids = df[id_col].astype(str).str.strip()
        if _is_missing_str(df[id_col]).any() or ids.duplicated().any():
            raise InputValidationError(f"ID column {id_col!r} has missing or duplicate values.",
                                       code="ID_DUPLICATE")
    return df, vals
