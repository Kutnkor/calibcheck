"""High-level library API: inspect, analyse and compare prediction data.

Both the CLI and the Streamlit app call these functions; neither reimplements metrics,
filtering or binning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibcheck.binning import make_edges
from calibcheck.config import AnalysisConfig, resolve_config
from calibcheck.data import PredictionData, load_predictions, overview
from calibcheck.errors import InputValidationError
from calibcheck.findings import generate_findings
from calibcheck.groups import (
    analyze_groups,
    analyze_time_windows,
    suppress_small_cells,
    suppress_small_windows,
)
from calibcheck.provenance import build_manifest
from calibcheck.schemas import AnalysisResult, InspectResult
from calibcheck.scoring import compare_models, comparison_pairs, edge_description, score_model

Source = str | Path | pd.DataFrame | PredictionData


def _config(config: AnalysisConfig | dict[str, Any] | None, overrides: dict[str, Any]) -> AnalysisConfig:
    base = config.model_dump(mode="json") if isinstance(config, AnalysisConfig) else dict(config or {})
    return resolve_config(base, overrides)


def _load(source: Source, cfg: AnalysisConfig, role: str = "evaluation") -> PredictionData:
    if isinstance(source, PredictionData):
        return source
    return load_predictions(source, cfg, role=role)


def inspect_predictions(source: str | Path | pd.DataFrame,
                        config: AnalysisConfig | dict[str, Any] | None = None,
                        **overrides: Any) -> InspectResult:
    """Validate a prediction file and summarise it, without fitting or resampling."""
    cfg = _config(config, overrides)
    try:
        data = load_predictions(source, cfg)
    except InputValidationError as exc:
        from calibcheck.schemas import Finding
        fs = exc.findings or [Finding(code=exc.code, severity="error", message=str(exc))]
        if not any(f.severity == "error" for f in fs):
            fs = [Finding(code=exc.code, severity="error", message=str(exc).split("\n")[0]), *fs]
        return InspectResult(valid=False, dataset=None, findings=fs, config=cfg)
    return InspectResult(valid=True, dataset=overview(data, cfg), findings=data.findings,
                         config=cfg)


def shared_edges_for(data: PredictionData, cfg: AnalysisConfig, models: list[str],
                     reference: str | None) -> tuple[dict[str, np.ndarray], dict[str, str], str]:
    """Edges per model plus the declared edge policy.

    Uniform edges are identical for all models. With quantile binning and more than one
    model, edges are estimated once from the reference model and shared, so curves are
    directly comparable; the source is recorded.
    """
    b = cfg.binning
    pop = "analysis population"
    if b.strategy == "uniform":
        e = make_edges(np.empty(0), "uniform", b.n_bins)
        desc = edge_description("uniform", b.n_bins, None, pop, data.n)
        return {m: e for m in models}, {m: desc for m in models}, desc
    if len(models) == 1:
        m = models[0]
        desc = edge_description("quantile", b.n_bins, m, pop, data.n)
        return {m: make_edges(data.probs[m], "quantile", b.n_bins)}, {m: desc}, desc
    ref = reference or models[0]
    e = make_edges(data.probs[ref], "quantile", b.n_bins)
    desc = ("shared " + edge_description("quantile", b.n_bins, ref, pop, data.n) +
            "; applied unchanged to every model")
    return {m: e for m in models}, {m: desc for m in models}, desc


def analyze_data(data: PredictionData, cfg: AnalysisConfig, *, kind: str | None = None,
                 display_names: dict[str, str] | None = None, title: str | None = None,
                 synthetic: bool = False, data_note: str | None = None,
                 protocol: dict[str, Any] | None = None,
                 extra_inputs: list | None = None) -> AnalysisResult:
    """Core orchestration on validated data (used by every surface)."""
    models = data.models
    pairs = comparison_pairs(models, cfg)
    reference = cfg.reference_model or (pairs[0][0] if pairs else None)
    edges, edge_src, policy = shared_edges_for(data, cfg, models, reference)
    names = display_names or {}
    pop = "analysis population"
    results = {m: score_model(data.y, data.probs[m], name=m, cfg=cfg, population=pop,
                              edges=edges[m], edge_source=edge_src[m],
                              display_name=names.get(m, m)) for m in models}
    comps = compare_models(data.y, data.probs, pairs, cfg=cfg, population=pop,
                           shared_edges=edges[reference] if reference else edges[models[0]],
                           edge_policy=policy, results=results) if pairs else []
    groups = analyze_groups(data, cfg, edges, policy) if data.groups else []
    time_windows = analyze_time_windows(data, cfg)
    if cfg.privacy.mode == "public":
        groups = suppress_small_cells(groups, cfg.privacy.min_cell,
                                      cfg.privacy.reveal_group_names)
        time_windows = suppress_small_windows(time_windows, cfg.privacy.min_cell)
    res = AnalysisResult(
        kind=kind or ("comparison" if len(models) > 1 else "single"),
        title=title or "CalibCheck report", synthetic=synthetic, data_note=data_note,
        config=cfg, dataset=overview(data, cfg), validation=data.findings,
        models=list(results.values()), comparisons=comps, groups=groups,
        time_windows=time_windows, protocol=protocol or {})
    res.findings = generate_findings(res)
    inputs = [data.input_record] if data.input_record else []
    inputs += list(extra_inputs or [])
    res.manifest = build_manifest(res, inputs, mode=res.kind, protocol=protocol)
    return res


def analyze_predictions(source: Source, config: AnalysisConfig | dict[str, Any] | None = None,
                        **overrides: Any) -> AnalysisResult:
    """Validate and analyse one or more probability columns on the same rows.

    Example::

        res = analyze_predictions("preds.csv", columns={"label": "y_true", "probs": ["p_a"],
                                                        "id": "row_id"})
        res.model("p_a").metrics["brier"].value
    """
    cfg = _config(config, overrides)
    data = _load(source, cfg)
    return analyze_data(data, cfg)


def compare_predictions(source: Source, config: AnalysisConfig | dict[str, Any] | None = None,
                        **overrides: Any) -> AnalysisResult:
    """Paired comparison of two or more probability columns that share outcomes and IDs."""
    cfg = _config(config, overrides)
    if len(cfg.columns.probs) < 2 and not isinstance(source, PredictionData):
        raise InputValidationError("compare needs at least two probability columns.",
                                   code="INPUT_NO_PROB_COLUMNS")
    data = _load(source, cfg)
    return analyze_data(data, cfg, kind="comparison")
