"""Export: versioned JSON, CSV tables, PNG figures, a terminal summary and an offline HTML report.

Exports are written into one output directory with an ``export_manifest.json`` listing every
file and its SHA-256. A ``_INCOMPLETE`` marker exists while writing and remains (together with
``"complete": false`` in the manifest) if any step fails, so a partial directory can never
masquerade as a finished report. Existing directories are only replaced when they hold a
previous CalibCheck export and ``overwrite`` is requested; only files listed in that export's
manifest are removed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import pandas as pd

from calibcheck._version import SCHEMA_VERSION, __version__
from calibcheck.codes import explain
from calibcheck.config import config_to_toml
from calibcheck.errors import OutputError
from calibcheck.findings import follow_ups
from calibcheck.metrics import REGISTRY
from calibcheck.schemas import AnalysisResult, InspectResult

MANIFEST_NAME = "export_manifest.json"
MARKER = "_INCOMPLETE"
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


# --------------------------------------------------------------------------------------
# safe output handling
# --------------------------------------------------------------------------------------

def safe_name(name: str, max_len: int = 60) -> str:
    """Filesystem-safe slug for user-controlled names (model names, group columns)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "unnamed"
    return s[:max_len]


@dataclass
class OutputDir:
    root: Path
    files: dict[str, str] = field(default_factory=dict)
    _used: set[str] = field(default_factory=set)

    def _resolve(self, rel: str) -> Path:
        target = (self.root / rel).resolve()
        if self.root.resolve() not in target.parents and target != self.root.resolve():
            raise OutputError(f"Refusing to write outside the output directory: {rel!r}")
        return target

    def unique(self, rel: str) -> str:
        base, dot, ext = rel.rpartition(".")
        cand, i = rel, 2
        while cand.lower() in self._used:
            cand = f"{base}_{i}.{ext}" if dot else f"{rel}_{i}"
            i += 1
        self._used.add(cand.lower())
        return cand

    def write_bytes(self, rel: str, data: bytes) -> str:
        rel = self.unique(rel)
        path = self._resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files[rel] = hashlib.sha256(data).hexdigest()
        return rel

    def write_text(self, rel: str, text: str) -> str:
        return self.write_bytes(rel, text.encode("utf-8"))


def prepare_output_dir(path: str | Path, *, overwrite: bool = False) -> OutputDir:
    root = Path(path)
    if root.exists() and not root.is_dir():
        raise OutputError(f"Output path exists and is not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        manifest = root / MANIFEST_NAME
        if not manifest.exists():
            raise OutputError(
                f"Output directory {root} is not empty and does not contain a previous "
                "CalibCheck export; choose a new or empty directory.")
        if not overwrite:
            raise OutputError(f"Output directory {root} already contains a CalibCheck export; "
                              "pass --overwrite to replace it.")
        try:
            old = json.loads(manifest.read_text(encoding="utf-8"))
            listed = list(old.get("files", {}))
        except (json.JSONDecodeError, OSError) as exc:
            raise OutputError(f"Cannot read the previous export manifest: {exc}") from exc
        out = OutputDir(root)
        for rel in [*listed, MANIFEST_NAME, MARKER]:
            p = out._resolve(rel)
            if p.is_file():
                p.unlink()
        for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    return OutputDir(root)


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection in *text* cells (numbers are untouched)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(FORMULA_PREFIXES) else v)
    out.columns = ["'" + c if isinstance(c, str) and c.startswith(FORMULA_PREFIXES) else c
                   for c in out.columns]
    return out


def to_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, allow_nan=False, ensure_ascii=False)


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------

def bins_frame(res: AnalysisResult) -> pd.DataFrame:
    rows = []
    for m in res.models:
        for b in m.binning.bins:
            d = b.model_dump()
            d["model"] = m.name
            rows.append(d)
    cols = ["model", "index", "lower", "upper", "closed", "count", "n_pos", "mean_pred",
            "observed_rate", "signed_gap", "abs_gap", "ci_low", "ci_high", "ci_status",
            "ci_reason", "low_support"]
    return pd.DataFrame(rows, columns=cols)


def sensitivity_frame(res: AnalysisResult) -> pd.DataFrame:
    rows = [{"model": m.name, **s.model_dump()} for m in res.models for s in m.sensitivity]
    return pd.DataFrame(rows)


def comparisons_frame(res: AnalysisResult) -> pd.DataFrame:
    rows = []
    for c in res.comparisons:
        d = c.model_dump(exclude={"interval"})
        iv = c.interval
        d.update({"ci_low": iv.low if iv else None, "ci_high": iv.high if iv else None,
                  "ci_method": iv.method if iv else None,
                  "ci_status": iv.status if iv else "skipped",
                  "ci_reason": iv.reason if iv else None})
        rows.append(d)
    return pd.DataFrame(rows)


def groups_frame(res: AnalysisResult) -> pd.DataFrame:
    rows = []
    for g in res.groups:
        for r in g.rows:
            for m, stats in (r.per_model or {"": {}}).items():
                rows.append({"column": g.column, "group": r.name, "n": r.n, "n_pos": r.n_pos,
                             "prevalence": r.prevalence, "prevalence_ci_low": r.prevalence_ci_low,
                             "prevalence_ci_high": r.prevalence_ci_high,
                             "suppressed": r.suppressed, "low_support": r.low_support,
                             "model": m, **{k: v for k, v in stats.items()}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------------------

def _f(v: float | None, d: int = 4) -> str:
    return "unavailable" if v is None else f"{v:.{d}f}"


def summary_text(res: AnalysisResult) -> str:
    d = res.dataset
    lines = [f"CalibCheck {__version__} — {res.title}"]
    if res.synthetic:
        lines.append("SYNTHETIC DATA")
    lines.append(f"Rows analysed: {d.n_rows_retained} of {d.n_rows_input} "
                 f"(excluded {d.n_rows_excluded}); positives {d.n_pos}, negatives {d.n_neg}; "
                 f"prevalence {_f(d.prevalence, 3)}")
    lines.append(f"Probability meaning: {d.probability_meaning}")
    for m in res.models:
        lines.append(f"\nModel {m.display_name} (n={m.n})")
        for key in ("brier", "log_loss", "ece", "roc_auc", "mean_error", "brier_skill"):
            mv = m.metrics[key]
            val = _f(mv.value) if mv.available else f"unavailable ({mv.reason})"
            if mv.interval and mv.interval.status == "ok":
                val += (f"  [{round(mv.interval.level * 100)}% {mv.interval.method}: "
                        f"{_f(mv.interval.low)}, {_f(mv.interval.high)}]")
            extra = ""
            if key == "ece":
                extra = f"  ({m.binning.strategy}, {m.binning.n_bins_actual} bins)"
            lines.append(f"  {mv.label:<34} {val}{extra}")
    if res.comparisons:
        lines.append("\nPaired comparisons (candidate - reference, same rows)")
        for c in res.comparisons:
            iv = c.interval
            ci = (f" [{_f(iv.low)}, {_f(iv.high)}]" if iv is not None and iv.status == "ok"
                  else "")
            lines.append(f"  {c.metric:<9} {c.candidate} vs {c.reference}: "
                         f"{_f(c.delta)}{ci} -> {c.favours}")
    if res.findings:
        lines.append("\nFindings")
        for f in res.findings:
            lines.append(f"  [{f.code}] {f.message}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------

_env = jinja2.Environment(
    loader=jinja2.PackageLoader("calibcheck", "templates"),
    autoescape=jinja2.select_autoescape(["html", "j2"], default=True),
    undefined=jinja2.StrictUndefined,
)


def _fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int | np.integer):
        return f"{int(v):,}"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


_env.filters["fmt"] = _fmt
_env.filters["explain"] = explain


def render_html(res: AnalysisResult, figures: dict[str, tuple[bytes, str]]) -> str:
    tmpl = _env.get_template("report.html.j2")
    figs = {k: (base64.b64encode(png).decode("ascii"), alt) for k, (png, alt) in figures.items()}
    order = ["brier", "log_loss", "brier_skill", "reference_brier", "ece", "max_abs_gap",
             "max_abs_gap_supported", "rms_gap", "mean_error", "mean_pred", "prevalence",
             "roc_auc", "average_precision", "accuracy"]
    return tmpl.render(res=res, figs=figs, version=__version__, schema_version=SCHEMA_VERSION,
                       registry=REGISTRY, metric_order=order, follow_ups=follow_ups(res),
                       config_toml=config_to_toml(res.config),
                       manifest_json=to_json(res.manifest.model_dump(mode="json"))
                       if res.manifest else "{}")


# --------------------------------------------------------------------------------------
# main export
# --------------------------------------------------------------------------------------

def render_report(res: AnalysisResult, out_dir: str | Path, *, overwrite: bool = False,
                  figures: bool = True, retained_ids: np.ndarray | None = None) -> dict[str, Any]:
    """Write all report artifacts. Returns the export manifest."""
    from calibcheck import plotting

    out = prepare_output_dir(out_dir, overwrite=overwrite)
    marker = out.root / MARKER
    marker.write_text("Export in progress or failed. Do not use these files.\n", encoding="utf-8")
    manifest: dict[str, Any] = {"tool": "calibcheck", "calibcheck_version": __version__,
                                "schema_version": SCHEMA_VERSION, "complete": False,
                                "files": {}}
    try:
        out.write_text("report.json", to_json(res.model_dump(mode="json")))
        out.write_text("config.toml", config_to_toml(res.config))
        out.write_text("summary.txt", summary_text(res))
        metrics_df = pd.DataFrame(res.metrics_table())
        out.write_text("tables/metrics.csv", csv_safe(metrics_df).to_csv(index=False))
        out.write_text("tables/bins.csv", csv_safe(bins_frame(res)).to_csv(index=False))
        sens = sensitivity_frame(res)
        if not sens.empty:
            out.write_text("tables/binning_sensitivity.csv", csv_safe(sens).to_csv(index=False))
        if res.comparisons:
            out.write_text("tables/comparisons.csv",
                           csv_safe(comparisons_frame(res)).to_csv(index=False))
        if res.groups:
            out.write_text("tables/groups.csv", csv_safe(groups_frame(res)).to_csv(index=False))
        if res.time_windows is not None:
            tw = pd.DataFrame([{"window": r.window, "n": r.n, "n_pos": r.n_pos,
                                "prevalence": r.prevalence, "suppressed": r.suppressed,
                                "suppression_reason": r.suppression_reason, "model": m, **s}
                               for r in res.time_windows.rows for m, s in r.per_model.items()])
            out.write_text("tables/time_windows.csv", csv_safe(tw).to_csv(index=False))
        findings_df = pd.DataFrame([f.model_dump(exclude={"evidence"}) for f in
                                    res.validation + res.findings])
        out.write_text("tables/findings.csv", csv_safe(findings_df).to_csv(index=False))
        if retained_ids is not None:
            if res.config.privacy.mode == "public":
                raise OutputError("Retained-row export is disabled in public privacy mode.")
            out.write_text("retained_rows.csv", csv_safe(pd.DataFrame(
                {"row_id": retained_ids})).to_csv(index=False))

        figs: dict[str, tuple[bytes, str]] = {}
        if figures:
            models = res.models
            chunks = [models[i:i + plotting.MAX_OVERLAY]
                      for i in range(0, len(models), plotting.MAX_OVERLAY)]
            cidx = {m.name: i for i, m in enumerate(models)}
            for j, chunk in enumerate(chunks):
                png = plotting.to_png(plotting.reliability_figure(
                    chunk, color_index={m.name: cidx[m.name] % plotting.MAX_OVERLAY
                                        for m in chunk}))
                key = "reliability" if j == 0 else f"reliability_{j + 1}"
                out.write_bytes(f"figures/{key}.png", png)
                figs[key] = (png, plotting.reliability_alt_text(chunk))
            for g in res.groups:
                for m in models[:plotting.MAX_OVERLAY]:
                    fig = plotting.group_figure(g, m)
                    if fig is None:
                        continue
                    png = plotting.to_png(fig)
                    key = f"groups_{safe_name(g.column)}_{safe_name(m.name)}"
                    rel = out.write_bytes(f"figures/{key}.png", png)
                    figs[Path(rel).stem] = (png, f"Reliability by {g.column} for {m.display_name}; "
                                                 "see the subgroup table for exact values.")
            if res.time_windows is not None and res.time_windows.rows:
                png = plotting.to_png(plotting.time_figure(res.time_windows, models))
                out.write_bytes("figures/time_windows.png", png)
                figs["time_windows"] = (png, "Observed prevalence and mean prediction per time "
                                             "window; exact values in the time-window table.")
        out.write_text("report.html", render_html(res, figs))
        manifest["files"] = out.files
        manifest["complete"] = True
        (out.root / MANIFEST_NAME).write_text(to_json(manifest), encoding="utf-8")
        marker.unlink()
    except Exception as exc:
        manifest["files"] = out.files
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (out.root / MANIFEST_NAME).write_text(to_json(manifest), encoding="utf-8")
        raise
    return manifest


def write_inspect(res: InspectResult) -> str:
    return to_json(res.model_dump(mode="json"))
