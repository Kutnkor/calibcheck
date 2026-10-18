"""The ``calibcheck`` command-line interface.

Exit codes: 0 success (warnings may be shown), 1 invalid input / configuration / artifact,
2 usage error, 3 data-role protocol violation, 4 unsafe output request.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from calibcheck._version import ARTIFACT_SCHEMA_VERSION, SCHEMA_VERSION, __version__
from calibcheck.errors import CalibCheckError, OutputError, ProtocolError

app = typer.Typer(
    name="calibcheck", no_args_is_help=True, add_completion=False,
    help="How trustworthy are your model's probabilities? Local-first calibration diagnostics "
         "for binary classifiers. Run `calibcheck demo` for a complete synthetic walkthrough.")
console = Console(stderr=False, highlight=False)
err = Console(stderr=True, highlight=False)

# ------------------------------------------------------------------------------------------
# shared options
# ------------------------------------------------------------------------------------------
Label = Annotated[str | None, typer.Option("--label", help="Outcome column (default y_true).")]
Probs = Annotated[list[str] | None, typer.Option("--prob", help="Probability column for "
                  "P(label = positive). Repeat for several models. Never guessed.")]
IdCol = Annotated[str | None, typer.Option("--id", help="Unique row-ID column (kept as text).")]
PosLabel = Annotated[str | None, typer.Option("--positive-label", help="Value of the outcome "
                     "column that counts as the positive event (required unless labels are 0/1).")]
Groups = Annotated[list[str] | None, typer.Option("--group", help="Categorical subgroup column "
                   "(repeatable).")]
Entity = Annotated[str | None, typer.Option("--entity", help="Entity / cluster ID column.")]
Timestamp = Annotated[str | None, typer.Option("--timestamp", help="Timestamp column.")]
ConfigOpt = Annotated[Path | None, typer.Option("--config", help="TOML configuration file; "
                      "command-line options override it.")]
Bins = Annotated[int | None, typer.Option("--bins", help="Number of bins (default 10).")]
Strategy = Annotated[str | None, typer.Option("--strategy", help="uniform | quantile.")]
MinSupport = Annotated[int | None, typer.Option("--min-support", help="Bin-size warning "
                       "threshold (heuristic, default 30).")]
Boot = Annotated[bool | None, typer.Option("--bootstrap/--no-bootstrap", help="Percentile "
                 "bootstrap intervals for aggregate losses and paired differences.")]
BootBins = Annotated[bool | None, typer.Option("--bootstrap-bins/--no-bootstrap-bins",
                     help="Exploratory ECE intervals with frozen bin edges.")]
NRes = Annotated[int | None, typer.Option("--n-resamples", help="Bootstrap replicates "
                 "(default 2000).")]
Seed = Annotated[int | None, typer.Option("--seed", help="Random seed for resampling.")]
Level = Annotated[float | None, typer.Option("--level", help="Interval level (default 0.95).")]
Dependence = Annotated[str | None, typer.Option("--dependence", help="independent | "
                       "entity_clustered | time_dependent | unknown. Non-independent "
                       "declarations disable IID intervals.")]
RefPrev = Annotated[float | None, typer.Option("--reference-prevalence", help="Prevalence from "
                    "training/development data for the constant reference predictor.")]
Drop = Annotated[bool | None, typer.Option("--drop-invalid/--fail-invalid", help="Drop rows "
                 "with missing/invalid outcomes or probabilities (recorded) instead of failing.")]
Eps = Annotated[float | None, typer.Option("--clip-eps", help="Clipping epsilon for log loss "
                "and logits (default 1e-15).")]
Privacy = Annotated[str | None, typer.Option("--privacy", help="private | public. Public omits "
                    "paths, fingerprints, row IDs and group names, and suppresses small cells.")]
Reveal = Annotated[bool | None, typer.Option("--reveal-group-names/--hide-group-names",
                   help="Show real group names in public mode.")]
TimeWin = Annotated[str | None, typer.Option("--time-window", help="D | W | M | Q | Y.")]
Transform = Annotated[list[str] | None, typer.Option("--transform", help="Explicit transform "
                      "COLUMN=percent|logit (recorded).")]
Out = Annotated[Path | None, typer.Option("--out", "-o", help="Output directory.")]
Overwrite = Annotated[bool, typer.Option("--overwrite", help="Replace a previous CalibCheck "
                      "export in the output directory.")]
JsonOut = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON to stdout.")]
Quiet = Annotated[bool, typer.Option("--quiet", "-q", help="Suppress the terminal summary.")]
NoFig = Annotated[bool, typer.Option("--no-figures", help="Skip PNG figures (faster).")]
Title = Annotated[str | None, typer.Option("--title", help="Report title.")]


def _overrides(*, label: str | None = None, probs: list[str] | None = None,
               id_col: str | None = None, positive_label: str | None = None,
               groups: list[str] | None = None, entity: str | None = None,
               timestamp: str | None = None, bins: int | None = None,
               strategy: str | None = None, min_support: int | None = None,
               bootstrap: bool | None = None, bootstrap_bins: bool | None = None,
               n_resamples: int | None = None, seed: int | None = None,
               level: float | None = None, dependence: str | None = None,
               reference_prevalence: float | None = None, drop: bool | None = None,
               clip_eps: float | None = None, privacy: str | None = None,
               reveal: bool | None = None, time_window: str | None = None,
               transforms: list[str] | None = None, reference_model: str | None = None,
               pairwise: bool | None = None) -> dict[str, Any]:
    tr = None
    if transforms:
        tr = {}
        for t in transforms:
            col, _, kind = t.partition("=")
            if kind not in ("percent", "logit"):
                raise typer.BadParameter(f"--transform expects COLUMN=percent|logit, got {t!r}")
            tr[col] = kind
    return {
        "columns": {"label": label, "probs": probs or None, "id": id_col, "groups": groups or None,
                    "entity": entity, "timestamp": timestamp},
        "positive_label": positive_label,
        "missing_policy": None if drop is None else ("drop" if drop else "fail"),
        "transforms": tr, "clip_eps": clip_eps, "reference_prevalence": reference_prevalence,
        "binning": {"n_bins": bins, "strategy": strategy, "min_support": min_support},
        "uncertainty": {"bootstrap": bootstrap, "bootstrap_bin_metrics": bootstrap_bins,
                        "n_resamples": n_resamples, "seed": seed, "level": level},
        "dependence": dependence, "time_window": time_window,
        "privacy": {"mode": privacy, "reveal_group_names": reveal},
        "reference_model": reference_model, "pairwise": pairwise,
    }


def _resolve(config: Path | None, overrides: dict[str, Any]):
    from calibcheck.config import load_config_file, resolve_config
    file_vals = load_config_file(config) if config else {}
    return resolve_config(file_vals, overrides)


def _fail(exc: CalibCheckError) -> None:
    code = 3 if isinstance(exc, ProtocolError) else 4 if isinstance(exc, OutputError) else 1
    err.print(f"[bold red]Error[/] \\[{exc.code}] {escape(str(exc))}")
    raise typer.Exit(code)


def _print_findings(findings: list, title: str = "Validation") -> None:
    if not findings:
        return
    err.print(f"[bold]{title}[/]")
    for f in findings:
        style = {"error": "red", "warning": "yellow", "info": "cyan"}[f.severity]
        err.print(f"  [{style}]{f.severity:<7}[/] \\[{f.code}] {escape(f.message)}")


def _add(t: Table, *cells: str) -> None:
    t.add_row(*[escape(str(c)) for c in cells])


def _summary_table(res) -> Table:
    t = Table(title=escape(f"{res.title}{'  (SYNTHETIC)' if res.synthetic else ''}"), show_lines=False)
    t.add_column("Metric")
    for m in res.models:
        t.add_column(escape(m.display_name), justify="right")
    for key in ("brier", "log_loss", "brier_skill", "ece", "max_abs_gap", "mean_error",
                "roc_auc", "average_precision"):
        row = [res.models[0].metrics[key].label]
        for m in res.models:
            mv = m.metrics[key]
            if not mv.available:
                row.append(f"n/a ({mv.reason})")
                continue
            s = f"{mv.value:.4f}"
            if mv.interval is not None and mv.interval.status == "ok":
                s += f" [{mv.interval.low:.4f}, {mv.interval.high:.4f}]"
            row.append(s)
        t.add_row(*[escape(c) for c in row])
    return t


def _emit(res, out: Path | None, *, overwrite: bool, quiet: bool, as_json: bool,
          figures: bool, retained=None) -> None:
    from calibcheck.report import render_report, summary_text, to_json
    manifest = None
    if out is not None:
        manifest = render_report(res, out, overwrite=overwrite, figures=figures,
                                 retained_ids=retained)
    if as_json:
        console.print_json(to_json(res.model_dump(mode="json")))
        return
    if quiet:
        if out is not None:
            console.print(str(out / "report.html"))
        return
    console.print(_summary_table(res))
    b = res.models[0].binning
    console.print(f"ECE uses {b.strategy} binning with {b.n_bins_actual} bins "
                  f"({b.n_nonempty} nonempty). Intervals, where shown, are "
                  f"{int(res.config.uncertainty.level * 100)}% percentile bootstrap (IID).")
    text = summary_text(res)
    if "\nFindings" in text:
        console.print("\n[bold]Findings[/]")
        for f in res.findings:
            console.print(f"  • {escape(f.message)}")
    if out is not None and manifest is not None:
        console.print(f"\n[green]Report written[/] ({len(manifest['files'])} files): "
                      f"{out / 'report.html'}")


# ------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Show the CalibCheck version and schema versions."""
    console.print(f"calibcheck {__version__} (report schema {SCHEMA_VERSION}, artifact schema "
                  f"{ARTIFACT_SCHEMA_VERSION})")


@app.command()
def inspect(
    file: Annotated[Path, typer.Argument(help="Prediction CSV.")],
    label: Label = None, prob: Probs = None, id: IdCol = None, positive_label: PosLabel = None,
    group: Groups = None, entity: Entity = None, timestamp: Timestamp = None,
    drop_invalid: Drop = None, transform: Transform = None, config: ConfigOpt = None,
    json_out: JsonOut = False,
) -> None:
    """Validate a file and summarise it. No fitting, no resampling (validation only)."""
    from calibcheck.analysis import inspect_predictions
    from calibcheck.report import write_inspect
    try:
        cfg = _resolve(config, _overrides(label=label, probs=prob, id_col=id,
                                          positive_label=positive_label, groups=group,
                                          entity=entity, timestamp=timestamp, drop=drop_invalid,
                                          transforms=transform))
        res = inspect_predictions(file, cfg)
    except CalibCheckError as exc:
        _fail(exc)
        return
    if json_out:
        console.print_json(write_inspect(res))
    else:
        console.print("[bold]CalibCheck inspect[/] — validation only, not a statistical report")
        if res.dataset is not None:
            d = res.dataset
            t = Table(show_header=False)
            t.add_column("k")
            t.add_column("v")
            _add(t, "Rows retained / input", f"{d.n_rows_retained} / {d.n_rows_input}")
            _add(t, "Outcome", f"{d.label_column}: positive = {d.positive_label!r}, "
                                 f"negative = {d.negative_label!r}")
            _add(t, "Positives / negatives", f"{d.n_pos} / {d.n_neg} (prevalence "
                                               f"{d.prevalence:.3f})")
            _add(t, "Probability meaning", d.probability_meaning)
            _add(t, "Row IDs", f"{d.id_kind}" + (f" ({d.id_column})" if d.id_column else ""))
            for name, r in d.prob_ranges.items():
                _add(t, f"{name}", f"range [{r['min']:.4g}, {r['max']:.4g}], mean "
                                     f"{r['mean']:.4g}, {r['n_unique']} distinct, "
                                     f"{r['n_zero']} zeros, {r['n_one']} ones")
            if d.group_columns:
                _add(t, "Subgroup columns", ", ".join(d.group_columns))
            console.print(t)
        _print_findings(res.findings, "Findings")
    if not res.valid:
        raise typer.Exit(1)


@app.command()
def report(
    file: Annotated[Path, typer.Argument(help="Prediction CSV.")],
    label: Label = None, prob: Probs = None, id: IdCol = None, positive_label: PosLabel = None,
    group: Groups = None, entity: Entity = None, timestamp: Timestamp = None,
    bins: Bins = None, strategy: Strategy = None, min_support: MinSupport = None,
    bootstrap: Boot = None, bootstrap_bins: BootBins = None, n_resamples: NRes = None,
    seed: Seed = None, level: Level = None, dependence: Dependence = None,
    reference_prevalence: RefPrev = None, drop_invalid: Drop = None, clip_eps: Eps = None,
    privacy: Privacy = None, reveal_group_names: Reveal = None, time_window: TimeWin = None,
    transform: Transform = None, config: ConfigOpt = None, out: Out = None,
    overwrite: Overwrite = False, json_out: JsonOut = False, quiet: Quiet = False,
    no_figures: NoFig = False, title: Title = None,
    artifact: Annotated[list[Path] | None, typer.Option(
        "--fitted-artifact", help="Calibrator artifact(s) whose calibration rows must not "
                                  "appear in this evaluation file (overlap check).")] = None,
    export_retained: Annotated[bool, typer.Option("--export-retained", help="Write the IDs of "
                               "retained rows (private mode only).")] = False,
) -> None:
    """Evaluate one probability column (or several on the same rows) and export a report."""
    from calibcheck.analysis import analyze_data
    from calibcheck.data import load_predictions
    try:
        cfg = _resolve(config, _overrides(
            label=label, probs=prob, id_col=id, positive_label=positive_label, groups=group,
            entity=entity, timestamp=timestamp, bins=bins, strategy=strategy,
            min_support=min_support, bootstrap=bootstrap, bootstrap_bins=bootstrap_bins,
            n_resamples=n_resamples, seed=seed, level=level, dependence=dependence,
            reference_prevalence=reference_prevalence, drop=drop_invalid, clip_eps=clip_eps,
            privacy=privacy, reveal=reveal_group_names, time_window=time_window,
            transforms=transform))
        data = load_predictions(file, cfg)
        protocol = _artifact_checks(artifact, data)
        res = analyze_data(data, cfg, title=title, protocol=protocol)
        if not json_out:
            _print_findings([f for f in data.findings if f.severity != "info"])
        _emit(res, out, overwrite=overwrite, quiet=quiet, as_json=json_out,
              figures=not no_figures, retained=data.ids if export_retained else None)
    except CalibCheckError as exc:
        _fail(exc)


def _artifact_checks(paths: list[Path] | None, data) -> dict[str, Any]:
    if not paths:
        return {}
    from calibcheck.calibration import load_artifact
    from calibcheck.protocol import check_artifact_overlap
    checks = []
    for p in paths:
        art = load_artifact(p)
        fs = check_artifact_overlap(art, data)
        checks.append({"artifact_method": art.method, "model_association": art.model_association,
                       "result": [f.model_dump(include={"code", "severity", "message"})
                                  for f in fs]})
    return {"artifact_overlap_checks": checks}


@app.command()
def compare(
    file: Annotated[Path, typer.Argument(help="Prediction CSV with two or more probability "
                                              "columns on the same rows.")],
    label: Label = None, prob: Probs = None, id: IdCol = None, positive_label: PosLabel = None,
    group: Groups = None, bins: Bins = None, strategy: Strategy = None,
    min_support: MinSupport = None, bootstrap: Boot = None, bootstrap_bins: BootBins = None,
    n_resamples: NRes = None, seed: Seed = None, level: Level = None,
    dependence: Dependence = None, drop_invalid: Drop = None, clip_eps: Eps = None,
    privacy: Privacy = None, reveal_group_names: Reveal = None,
    reference: Annotated[str | None, typer.Option("--reference", help="Reference model; every "
                         "other model is compared with it.")] = None,
    pairwise: Annotated[bool | None, typer.Option("--pairwise/--no-pairwise", help="All "
                        "pairwise comparisons (explicit request for >2 models).")] = None,
    other: Annotated[Path | None, typer.Option("--other", help="Second CSV, joined by unique "
                     "ID (requires --id).")] = None,
    other_prob: Annotated[list[str] | None, typer.Option("--other-prob", help="Probability "
                          "column(s) in --other.")] = None,
    intersection: Annotated[bool, typer.Option("--intersection", help="Allow comparing on the "
                            "matched IDs only (reported).")] = False,
    config: ConfigOpt = None, out: Out = None, overwrite: Overwrite = False,
    json_out: JsonOut = False, quiet: Quiet = False, no_figures: NoFig = False,
    title: Title = None,
    artifact: Annotated[list[Path] | None, typer.Option("--fitted-artifact", help="Artifact "
                        "overlap check (see report).")] = None,
) -> None:
    """Paired comparison of probability columns on identical rows."""
    from calibcheck.analysis import analyze_data
    from calibcheck.data import align_by_id, load_predictions
    from calibcheck.errors import InputValidationError
    try:
        ov = _overrides(label=label, probs=prob, id_col=id, positive_label=positive_label,
                        groups=group, bins=bins, strategy=strategy, min_support=min_support,
                        bootstrap=bootstrap, bootstrap_bins=bootstrap_bins,
                        n_resamples=n_resamples, seed=seed, level=level, dependence=dependence,
                        drop=drop_invalid, clip_eps=clip_eps, privacy=privacy,
                        reveal=reveal_group_names, pairwise=pairwise)
        cfg = _resolve(config, ov)
        data = load_predictions(file, cfg)
        if other is not None:
            if not other_prob:
                raise InputValidationError("--other needs --other-prob.",
                                           code="INPUT_NO_PROB_COLUMNS")
            cfg2 = cfg.model_copy(deep=True)
            cfg2.columns.probs = list(other_prob)
            cfg2.columns.groups = []
            data2 = load_predictions(other, cfg2, role="evaluation (second file)")
            data, _align = align_by_id(data, data2, allow_intersection=intersection)
            cfg.columns.probs = data.models
            data_records = [data2.input_record] if data2.input_record else []
        else:
            data_records = []
        if len(data.models) < 2:
            raise InputValidationError("compare needs at least two probability columns.",
                                       code="INPUT_NO_PROB_COLUMNS")
        if reference is not None:
            cfg.reference_model = reference
        if len(data.models) > 2 and cfg.reference_model is None and not cfg.pairwise:
            err.print("[yellow]More than two models: pass --reference NAME or --pairwise to "
                      "request paired differences; showing the model table only.[/]")
        protocol = _artifact_checks(artifact, data)
        res = analyze_data(data, cfg, kind="comparison", title=title or "CalibCheck comparison",
                           protocol=protocol, extra_inputs=data_records)
        _emit(res, out, overwrite=overwrite, quiet=quiet, as_json=json_out,
              figures=not no_figures)
    except CalibCheckError as exc:
        _fail(exc)


@app.command()
def fit(
    file: Annotated[Path, typer.Argument(help="CALIBRATION data (never the final test file).")],
    method: Annotated[str, typer.Option("--method", help="identity | sigmoid | isotonic | "
                      "temperature")] = "sigmoid",
    label: Label = None, prob: Annotated[str | None, typer.Option("--prob", help="Probability "
                                         "column to recalibrate.")] = None,
    id: IdCol = None, positive_label: PosLabel = None, entity: Entity = None,
    timestamp: Timestamp = None, drop_invalid: Drop = None, clip_eps: Eps = None,
    config: ConfigOpt = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Artifact JSON path.")] = Path(
        "calibrator.json"),
    overwrite: Overwrite = False,
    no_id_digests: Annotated[bool, typer.Option("--no-id-digests", help="Do not store salted "
                             "ID digests (disables later overlap checks).")] = False,
) -> None:
    """Fit a recalibrator on calibration data and write a JSON artifact.

    The in-sample fit is NOT held-out performance; evaluate with `recalibrate` or `compare` on
    separate test rows.
    """
    from calibcheck.calibration import fit_calibrator, save_artifact
    from calibcheck.data import load_predictions
    try:
        cfg = _resolve(config, _overrides(label=label, probs=[prob] if prob else None, id_col=id,
                                          positive_label=positive_label, entity=entity,
                                          timestamp=timestamp, drop=drop_invalid,
                                          clip_eps=clip_eps))
        if len(cfg.columns.probs) != 1:
            raise CalibCheckError("fit needs exactly one --prob column.", code="CONFIG_INVALID")
        data = load_predictions(file, cfg, role="calibration")
        model = cfg.columns.probs[0]
        ts = str(data.timestamps.max()) if data.timestamps is not None else None
        art = fit_calibrator(
            data.probs[model], data.y, method, eps=cfg.clip_eps, model_association=model,
            positive_label=data.positive_label, label_column=data.label_column, ids=data.ids,
            id_kind=data.id_kind, entities=data.entity, timestamp_max=ts,
            input_sha256=data.input_record.sha256 if data.input_record else None,
            store_id_digests=not no_id_digests)
        path = save_artifact(art, out, overwrite=overwrite)
    except CalibCheckError as exc:
        _fail(exc)
        return
    console.print(f"[green]Fitted[/] {method} on {data.n} calibration rows "
                  f"({int(data.y.sum())} events) for column {model!r}.")
    if art.method != "isotonic":
        console.print(f"  parameters: {art.params}")
    else:
        console.print(f"  isotonic breakpoints: {len(art.params['x'])}")
    for w in art.diagnostics.warnings:
        err.print(f"  [yellow]warning[/] \\[{w.code}] {escape(w.message)}")
    console.print(f"  artifact: {path}")
    console.print("  This is an in-sample fit. Judge it on separate test rows.")


@app.command()
def apply(
    file: Annotated[Path, typer.Argument(help="Prediction CSV (outcomes not required).")],
    artifact: Annotated[Path, typer.Option("--artifact", help="Calibrator JSON artifact.")],
    prob: Annotated[str, typer.Option("--prob", help="Probability column to transform.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output CSV path.")],
    id: IdCol = None,
    column: Annotated[str | None, typer.Option("--column", help="Name of the new column "
                      "(default <prob>__<method>).")] = None,
    overwrite: Overwrite = False,
    replace_column: Annotated[bool, typer.Option("--replace-column", help="Allow replacing an "
                              "existing column with the same name.")] = False,
    force_association: Annotated[bool, typer.Option("--force-association", help="Apply even "
                                 "though the artifact was fitted for a differently named "
                                 "column.")] = False,
) -> None:
    """Apply a fitted calibrator artifact, writing a new probability column."""
    from calibcheck.calibration import apply_calibrator, load_artifact
    from calibcheck.config import AnalysisConfig
    from calibcheck.data import load_probabilities_only
    from calibcheck.report import csv_safe
    try:
        if out.resolve() == file.resolve():
            raise OutputError("--out must differ from the input file.")
        if out.exists() and not overwrite:
            raise OutputError(f"{out} exists; pass --overwrite to replace it.")
        art = load_artifact(artifact)
        if art.model_association != prob and not force_association:
            raise CalibCheckError(
                f"The artifact was fitted for column {art.model_association!r}, not {prob!r}. "
                "Names alone cannot establish compatibility; pass --force-association only if "
                "this is the same model's output.", code="APPLY_ASSOCIATION_MISMATCH")
        df, p = load_probabilities_only(file, prob, AnalysisConfig(), id_col=id)
        q, notes = apply_calibrator(art, p)
        name = column or f"{prob}__{art.method}"
        if name in df.columns and not replace_column:
            raise OutputError(f"Column {name!r} already exists; choose --column or pass "
                              "--replace-column.")
        df[name] = [f"{v:.17g}" for v in q]
        out.parent.mkdir(parents=True, exist_ok=True)
        csv_safe(df).to_csv(out, index=False)
    except CalibCheckError as exc:
        _fail(exc)
        return
    for n in notes:
        err.print(f"  [cyan]info[/] \\[{n.code}] {escape(n.message)}")
    console.print(f"[green]Applied[/] {art.method} calibrator to {len(q)} rows -> column "
                  f"{name!r} in {out}")


@app.command()
def recalibrate(
    calibration: Annotated[Path, typer.Option("--calibration", help="Calibration CSV (fits the "
                           "map; may include an inner selection).")],
    test: Annotated[Path, typer.Option("--test", help="Untouched test CSV for final "
                    "evaluation.")],
    prob: Annotated[str, typer.Option("--prob", help="Probability column in both files.")],
    method: Annotated[list[str] | None, typer.Option("--method", help="Predeclared methods "
                      "(repeatable; default sigmoid and isotonic).")] = None,
    mode: Annotated[str, typer.Option("--mode", help="exploratory (show all methods) | selected "
                    "(choose on inner calibration folds, then one test evaluation).")] = "exploratory",
    selection_metric: Annotated[str, typer.Option("--selection-metric")] = "log_loss",
    temporal: Annotated[bool, typer.Option("--temporal", help="Require calibration rows to "
                        "precede test rows in time; chronological inner folds.")] = False,
    label: Label = None, id: IdCol = None, positive_label: PosLabel = None,
    entity: Entity = None, timestamp: Timestamp = None, group: Groups = None,
    bins: Bins = None, strategy: Strategy = None, bootstrap: Boot = None,
    bootstrap_bins: BootBins = None, n_resamples: NRes = None, seed: Seed = None,
    dependence: Dependence = None, clip_eps: Eps = None, privacy: Privacy = None,
    config: ConfigOpt = None, out: Out = None, overwrite: Overwrite = False,
    json_out: JsonOut = False, quiet: Quiet = False, no_figures: NoFig = False,
    title: Title = None,
) -> None:
    """Fit on calibration rows, apply to test rows, and compare raw vs recalibrated (paired)."""
    from calibcheck.calibration import save_artifact
    from calibcheck.data import load_predictions
    from calibcheck.protocol import evaluate_recalibration
    try:
        cfg = _resolve(config, _overrides(
            label=label, probs=[prob], id_col=id, positive_label=positive_label, entity=entity,
            timestamp=timestamp, groups=group, bins=bins, strategy=strategy, bootstrap=bootstrap,
            bootstrap_bins=bootstrap_bins, n_resamples=n_resamples, seed=seed,
            dependence=dependence, clip_eps=clip_eps, privacy=privacy))
        cal_cfg = cfg.model_copy(deep=True)
        cal_cfg.columns.groups = []
        cal = load_predictions(calibration, cal_cfg, role="calibration")
        tst = load_predictions(test, cfg, role="test")
        if mode not in ("exploratory", "selected"):
            raise CalibCheckError("--mode must be exploratory or selected", code="CONFIG_INVALID")
        res, arts = evaluate_recalibration(cal, tst, cfg, model=prob, methods=method,
                                           mode=mode, selection_metric=selection_metric,  # type: ignore[arg-type]
                                           temporal=temporal, title=title)
        if out is not None:
            _emit(res, out, overwrite=overwrite, quiet=quiet, as_json=json_out,
                  figures=not no_figures)
            for m, a in arts.items():
                save_artifact(a, out / "artifacts" / f"{m}.json", overwrite=True)
        else:
            _emit(res, None, overwrite=False, quiet=quiet, as_json=json_out,
                  figures=False)
    except CalibCheckError as exc:
        _fail(exc)


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", "-o", help="Directory for demo data and "
                   "reports.")] = Path("calibcheck-demo"),
    n: Annotated[int, typer.Option("--n", help="Total synthetic rows (split 50/50 into "
                 "calibration and test).")] = 6000,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    overwrite: Overwrite = False,
    n_resamples: NRes = 2000,
) -> None:
    """Run the full synthetic 90%-vs-65% walkthrough (offline, deterministic)."""
    import numpy as np

    from calibcheck.analysis import analyze_data
    from calibcheck.calibration import apply_calibrator, fit_calibrator, save_artifact
    from calibcheck.config import resolve_config
    from calibcheck.data import load_predictions
    from calibcheck.protocol import evaluate_recalibration
    from calibcheck.report import render_report
    from calibcheck.synthetic import SYNTHETIC_NOTE, headline
    try:
        if out.exists() and any(out.iterdir()) and not overwrite:
            raise OutputError(f"{out} is not empty; pass --overwrite or choose another "
                              "directory.")
        sc = headline(n=n, seed=seed)
        data_dir = out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        cal_df = sc.frame[sc.frame["split"] == "calibration"].drop(columns=["latent_p_true"])
        test_df = sc.frame[sc.frame["split"] == "test"].drop(columns=["latent_p_true"])
        cal_df.to_csv(data_dir / "calibration.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)
        (data_dir / "README.txt").write_text(
            f"{SYNTHETIC_NOTE}\nScenario: {sc.description}\nParameters: {sc.params}\n"
            f"Seed: {seed}\nExpected interpretation: {sc.expected_interpretation}\n",
            encoding="utf-8")
        cfg = resolve_config({}, {"columns": {"label": "y_true", "probs": ["p_model"],
                                              "id": "row_id", "groups": ["segment"]},
                                  "uncertainty": {"bootstrap": True, "n_resamples": n_resamples,
                                                  "seed": seed}})
        console.print(f"[bold]CalibCheck demo[/] — {SYNTHETIC_NOTE}")
        console.print(f"Scenario: {sc.description}")
        tst = load_predictions(data_dir / "test.csv", cfg, role="test")
        res = analyze_data(tst, cfg, title="Demo: raw model on test rows (synthetic)",
                           synthetic=True, data_note=SYNTHETIC_NOTE)
        render_report(res, out / "reports" / "raw", overwrite=overwrite)
        bin9 = res.models[0].binning.bins[-1]
        console.print(f"\n1) Raw model, test rows: bin [0.9, 1.0] has {bin9.count} rows, mean "
                      f"prediction {bin9.mean_pred:.3f}, observed {bin9.observed_rate:.3f} "
                      f"(95% Wilson {bin9.ci_low:.3f}–{bin9.ci_high:.3f}); generator setting "
                      f"for the flagged cohort: {sc.params['cohort_true']}.")
        cal_cfg = cfg.model_copy(deep=True)
        cal_cfg.columns.groups = []
        cal = load_predictions(data_dir / "calibration.csv", cal_cfg, role="calibration")
        art_dir = out / "artifacts"
        for m in ("sigmoid", "isotonic"):
            art = fit_calibrator(cal.probs["p_model"], cal.y, m, model_association="p_model",
                                 ids=cal.ids, id_kind=cal.id_kind)
            save_artifact(art, art_dir / f"{m}.json", overwrite=True)
        iso = fit_calibrator(cal.probs["p_model"], cal.y, "isotonic",
                             model_association="p_model", store_id_digests=False)
        q, _ = apply_calibrator(iso, test_df["p_model"].to_numpy(dtype=float))
        applied = test_df.copy()
        applied["p_model__isotonic"] = q
        applied.to_csv(data_dir / "test_with_isotonic.csv", index=False)
        console.print(f"2) Fitted sigmoid and isotonic maps on {cal.n} calibration rows -> "
                      f"{art_dir}. Isotonic maps 0.90 to {float(apply_calibrator(iso, np.array([0.9]))[0][0]):.3f}.")
        rres, _ = evaluate_recalibration(cal, tst, cfg, model="p_model",
                                         methods=["sigmoid", "isotonic"], mode="exploratory",
                                         title="Demo: raw vs recalibrated on untouched test rows "
                                               "(synthetic)", synthetic=True,
                                         data_note=SYNTHETIC_NOTE)
        render_report(rres, out / "reports" / "recalibration", overwrite=overwrite)
        console.print("3) Held-out paired comparison (test rows, candidate - raw):")
        for c in rres.comparisons:
            if c.metric in ("brier", "log_loss") and c.delta is not None:
                iv = c.interval
                ci = (f" [{iv.low:+.4f}, {iv.high:+.4f}]" if iv is not None and
                      iv.status == "ok" else "")
                console.print(f"   {c.candidate:<20} {c.metric:<9} {c.delta:+.4f}{ci} -> "
                              f"{c.favours}")
        console.print(f"\nReports: {out / 'reports' / 'raw' / 'report.html'}\n"
                      f"         {out / 'reports' / 'recalibration' / 'report.html'}")
    except CalibCheckError as exc:
        _fail(exc)


@app.command()
def example(
    out: Annotated[Path, typer.Argument(help="Where to write the bundled example CSV.")] = Path(
        "example_predictions.csv"),
    overwrite: Overwrite = False,
) -> None:
    """Write the small bundled example (synthetic: two models, 1,000 rows, a channel column)."""
    from importlib.resources import files
    if out.exists() and not overwrite:
        err.print(f"{out} exists; pass --overwrite to replace it.")
        raise typer.Exit(4)
    data = files("calibcheck").joinpath("data/example_predictions.csv").read_bytes()
    out.write_bytes(data)
    console.print(f"Wrote {escape(str(out))} (SYNTHETIC). Columns: row_id, y_true, p_model, "
                  "p_model_b, channel. Try:\n  calibcheck report "
                  f"{escape(str(out))} --prob p_model --id row_id --group channel --out "
                  "reports/example")


@app.command("app")
def app_cmd(
    port: Annotated[int, typer.Option("--port")] = 8501,
    host: Annotated[str, typer.Option("--host", help="Bind address. The default keeps the app "
                    "on this machine; binding elsewhere sends uploads to this host over the "
                    "network.")] = "127.0.0.1",
    max_upload_mb: Annotated[int, typer.Option("--max-upload-mb")] = 200,
) -> None:
    """Launch the optional local Streamlit interface (pip install 'calibcheck[app]')."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        err.print("Streamlit is not installed. Install the optional extra: "
                  "pip install 'calibcheck[app]'")
        raise typer.Exit(1) from None
    script = Path(__file__).parent / "app" / "streamlit_app.py"
    if host not in ("127.0.0.1", "localhost", "::1"):
        err.print(f"[yellow]Binding to {host}: anyone who can reach it can upload data to this "
                  "machine.[/]")
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), "--server.address", host,
           "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false", "--server.maxUploadSize", str(max_upload_mb),
           "--server.fileWatcherType", "none", "--client.toolbarMode", "minimal"]
    console.print(f"Starting CalibCheck app on http://{host}:{port} (Ctrl+C to stop)")
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def schema(
    out: Annotated[Path, typer.Option("--out", "-o", help="Directory for JSON Schema files.")]
    = Path("schema"),
) -> None:
    """Write JSON Schemas for reports and calibrator artifacts."""
    from calibcheck.calibration import CalibratorArtifact
    from calibcheck.schemas import AnalysisResult, InspectResult
    out.mkdir(parents=True, exist_ok=True)
    for name, model in (("report", AnalysisResult), ("inspect", InspectResult),
                        ("calibrator", CalibratorArtifact)):
        (out / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(), indent=2), encoding="utf-8")
    console.print(f"Wrote schemas to {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
