"""The shared scoring engine.

``score_model`` and ``compare_models`` accept validated arrays plus an
:class:`~calibcheck.config.AnalysisConfig` and return structured results. They perform no
I/O, plotting or UI work, and are reused by single-model reports, comparisons, subgroup
views and held-out recalibration evaluations, so every surface applies identical formulas.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from calibcheck import uncertainty as unc
from calibcheck.binning import (
    assign_bins,
    bin_summaries,
    bin_table,
    ece_from_assignments,
    make_edges,
    sensitivity_table,
)
from calibcheck.config import AnalysisConfig
from calibcheck.metrics import brier_losses, clip_info, core_metrics, log_losses
from calibcheck.schemas import ComparisonResult, Interval, ModelResult, ProbabilitySummary

HIST_BINS = 20


def edge_description(strategy: str, n_bins: int, source: str | None, population: str, n: int) -> str:
    if strategy == "uniform":
        return f"fixed equal-width edges k/{n_bins} over [0, 1]"
    who = f"predictions of {source!r}" if source else "the predictions"
    return f"empirical quantiles (inverted CDF) of {who} on {population} (n={n})"


def _wilson_settings(cfg: AnalysisConfig) -> tuple[float | None, str | None]:
    if not cfg.uncertainty.wilson:
        return None, "UNAVAILABLE_DISABLED"
    block = unc.dependence_block(cfg.dependence)
    if block:
        return None, block
    return cfg.uncertainty.level, None


def score_model(y: np.ndarray, p: np.ndarray, *, name: str, cfg: AnalysisConfig,
                population: str = "analysis population", edges: np.ndarray | None = None,
                edge_source: str | None = None, display_name: str | None = None,
                sensitivity: bool | None = None) -> ModelResult:
    """Score one probability column on one population."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    n = int(y.size)
    bcfg = cfg.binning
    metrics = core_metrics(y, p, eps=cfg.clip_eps, threshold=cfg.threshold,
                           population=population, reference_prevalence=cfg.reference_prevalence)
    if edges is None:
        edges = make_edges(p, bcfg.strategy, bcfg.n_bins)
        edge_source = edge_description(bcfg.strategy, bcfg.n_bins, name, population, n)
    wl, wreason = _wilson_settings(cfg)
    binning = bin_table(y, p, edges, strategy=bcfg.strategy, n_bins_requested=bcfg.n_bins,
                        edge_source=edge_source or "declared edges",
                        min_support=bcfg.min_support, wilson_level=wl, wilson_reason=wreason)
    metrics.update(bin_summaries(binning, n, population))

    _attach_intervals(metrics, y, p, cfg, edges)

    do_sens = bcfg.sensitivity if sensitivity is None else sensitivity
    sens = sensitivity_table(y, p, bins=bcfg.sensitivity_bins,
                             strategies=list(bcfg.sensitivity_strategies),
                             primary=(bcfg.strategy, bcfg.n_bins),
                             min_support=bcfg.min_support) if do_sens and n else []
    hist_edges = np.arange(HIST_BINS + 1) / HIST_BINS
    hist = np.bincount(assign_bins(p, hist_edges), minlength=HIST_BINS)
    summary = ProbabilitySummary(
        min=float(p.min()), max=float(p.max()), mean=float(p.mean()),
        n_unique=int(np.unique(p).size), n_zero=int((p == 0).sum()), n_one=int((p == 1).sum()),
        histogram_edges=[float(e) for e in hist_edges], histogram_counts=[int(c) for c in hist])
    return ModelResult(name=name, display_name=display_name or name, n=n, population=population,
                       metrics=metrics, binning=binning, sensitivity=sens,
                       clipping=clip_info(p, cfg.clip_eps), summary=summary)


def _attach_intervals(metrics: dict, y: np.ndarray, p: np.ndarray, cfg: AnalysisConfig,
                      edges: np.ndarray) -> None:
    u = cfg.uncertainty
    targets = ["brier", "log_loss", "mean_error"]
    if not u.bootstrap:
        return
    block = unc.dependence_block(cfg.dependence)
    if block:
        for t in targets + (["ece"] if u.bootstrap_bin_metrics else []):
            metrics[t].interval = unc.unavailable_interval(
                block, u.level, note=f"Declared dependence: {cfg.dependence}. An IID bootstrap "
                                     "would misstate uncertainty, so no interval is reported.")
        return
    n = y.size
    if n < 2:
        return
    cols = np.column_stack([brier_losses(y, p), log_losses(y, p, cfg.clip_eps),
                            p - y.astype(np.float64)])
    reps = unc.bootstrap_means(cols, u)
    for j, t in enumerate(targets):
        if metrics[t].available:
            metrics[t].interval = unc.percentile_interval(reps[:, j], u.level, seed=u.seed,
                                                          method_note=unc.IID_NOTE)
    if u.bootstrap_bin_metrics and metrics["ece"].available:
        a = assign_bins(p, edges)
        er = unc.frozen_edge_ece_replicates(y, [p], [a], [len(edges) - 1], u)
        iv = unc.percentile_interval(er[:, 0], u.level, seed=u.seed, method_note=(
            "Exploratory: percentile bootstrap of binned ECE with bin edges frozen at their "
            "full-sample values (conditional interpretation). The absolute value makes ECE "
            "biased upward in small bins, so treat this interval as approximate."))
        iv.reason = iv.reason or "UNC_ECE_EXPLORATORY"
        metrics["ece"].interval = iv


# --------------------------------------------------------------------------------------
# paired comparisons
# --------------------------------------------------------------------------------------

def comparison_pairs(models: list[str], cfg: AnalysisConfig) -> list[tuple[str, str]]:
    """(reference, candidate) pairs. Larger collections need an explicit request."""
    if len(models) < 2:
        return []
    if cfg.reference_model is not None:
        return [(cfg.reference_model, m) for m in models if m != cfg.reference_model]
    if cfg.pairwise:
        return list(combinations(models, 2))
    if len(models) == 2:
        return [(models[0], models[1])]
    return []


DEFINITIONS = {
    "brier": "delta = Brier(candidate) - Brier(reference); negative favours the candidate",
    "log_loss": "delta = LogLoss(candidate) - LogLoss(reference); negative favours the candidate",
    "ece": "delta = ECE(candidate) - ECE(reference) on shared edges; negative favours the candidate",
    "roc_auc": "delta = AUC(candidate) - AUC(reference); positive favours the candidate",
}


def compare_models(y: np.ndarray, probs: dict[str, np.ndarray], pairs: list[tuple[str, str]], *,
                   cfg: AnalysisConfig, population: str, shared_edges: np.ndarray,
                   edge_policy: str, results: dict[str, ModelResult]) -> list[ComparisonResult]:
    """Paired differences on identical rows; intervals resample the same row indices."""
    y = np.asarray(y, dtype=np.int8)
    n = int(y.size)
    u = cfg.uncertainty
    block = unc.dependence_block(cfg.dependence)
    out: list[ComparisonResult] = []
    for ref, cand in pairs:
        pr, pc = probs[ref], probs[cand]
        rows = []
        d_bs = brier_losses(y, pc) - brier_losses(y, pr)
        d_ll = log_losses(y, pc, cfg.clip_eps) - log_losses(y, pr, cfg.clip_eps)
        ivs: dict[str, Interval | None] = {"brier": None, "log_loss": None, "ece": None}
        if u.bootstrap and block is None and n >= 2:
            reps = unc.bootstrap_means(np.column_stack([d_bs, d_ll]), u)
            ivs["brier"] = unc.percentile_interval(reps[:, 0], u.level, seed=u.seed,
                                                   method_note="Paired " + unc.IID_NOTE)
            ivs["log_loss"] = unc.percentile_interval(reps[:, 1], u.level, seed=u.seed,
                                                      method_note="Paired " + unc.IID_NOTE)
            if u.bootstrap_bin_metrics:
                k = len(shared_edges) - 1
                er = unc.frozen_edge_ece_replicates(
                    y, [pr, pc], [assign_bins(pr, shared_edges), assign_bins(pc, shared_edges)],
                    [k, k], u)
                iv = unc.percentile_interval(er[:, 1] - er[:, 0], u.level, seed=u.seed,
                                             method_note="Exploratory paired ECE difference, "
                                                         "frozen shared edges.")
                iv.reason = iv.reason or "UNC_ECE_EXPLORATORY"
                ivs["ece"] = iv
            else:
                ivs["ece"] = unc.unavailable_interval(
                    "UNAVAILABLE_DISABLED", u.level,
                    note="ECE bootstrap is opt-in (--bootstrap-bins) because it is exploratory.")
        elif u.bootstrap:
            for key in ivs:
                ivs[key] = unc.unavailable_interval(block or "UNAVAILABLE_EMPTY", u.level)
        # ECE on shared edges
        k = len(shared_edges) - 1
        e_ref = _ece(y, pr, shared_edges, k)
        e_cand = _ece(y, pc, shared_edges, k)
        rows.append(("brier", float(d_bs.mean()), ivs["brier"], "lower"))
        rows.append(("log_loss", float(d_ll.mean()), ivs["log_loss"], "lower"))
        rows.append(("ece", e_cand - e_ref, ivs["ece"], "lower"))
        auc_r = results[ref].metrics["roc_auc"]
        auc_c = results[cand].metrics["roc_auc"]
        for metric, delta, civ, better in rows:
            out.append(_comparison(ref, cand, metric, delta, civ, better, n, population,
                                   edge_policy if metric == "ece" else None))
        if auc_r.available and auc_c.available:
            out.append(_comparison(ref, cand, "roc_auc", float(auc_c.value - auc_r.value),  # type: ignore[operator]
                                   unc.unavailable_interval("UNAVAILABLE_NOT_SUPPORTED", u.level,
                                                            note="No AUC difference interval in "
                                                                 "v0.1.") if u.bootstrap else None,
                                   "higher", n, population, None))
        else:
            out.append(ComparisonResult(reference=ref, candidate=cand, metric="roc_auc",
                                        definition=DEFINITIONS["roc_auc"], delta=None,
                                        status="unavailable", reason="UNAVAILABLE_SINGLE_CLASS",
                                        favours="undetermined", n=n, population=population))
    return out


def _ece(y: np.ndarray, p: np.ndarray, edges: np.ndarray, k: int) -> float:
    return ece_from_assignments(y.astype(np.float64), p, assign_bins(p, edges), k)


def _comparison(ref: str, cand: str, metric: str, delta: float, iv: Interval | None,
                better: str, n: int, population: str, edge_policy: str | None) -> ComparisonResult:
    favours = "undetermined"
    if iv is not None and iv.status == "ok" and iv.low is not None and iv.high is not None:
        if iv.low > 0 or iv.high < 0:
            positive = iv.low > 0
            cand_better = (not positive) if better == "lower" else positive
            favours = "candidate" if cand_better else "reference"
        else:
            favours = "neither"
    return ComparisonResult(reference=ref, candidate=cand, metric=metric,
                            definition=DEFINITIONS[metric], delta=delta, status="ok",
                            favours=favours, n=n, population=population,
                            interval=iv, edge_policy=edge_policy)
