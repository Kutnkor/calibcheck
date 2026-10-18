"""Deterministic probability binning, bin statistics, Wilson intervals and ECE-type summaries.

Boundary semantics (all strategies): bins are left-closed / right-open, ``[e_k, e_{k+1})``,
except the final bin which is closed, ``[e_{K-1}, e_K]``. Edges always span [0, 1] so every
valid probability (including exactly 0 and 1) is assigned to exactly one bin.

Uniform edges are computed as ``k / K`` (not ``linspace``) so that e.g. 0.3 is the nearest
double to 3/10 and an input of exactly 0.3 lands in the bin starting at 0.3.

Note: scikit-learn's ``calibration_curve`` assigns a value equal to an interior edge to the
*lower* bin (``searchsorted(..., side='left')``); CalibCheck uses left-closed bins. The two
agree whenever no prediction lies exactly on an interior edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from calibcheck.metrics import _mv, unavailable
from calibcheck.schemas import BinningResult, BinStat, MetricValue, SensitivityRow

EDGE_RULE = "left-closed [lower, upper), final bin closed [lower, 1.0]"


def uniform_edges(n_bins: int) -> np.ndarray:
    return np.arange(n_bins + 1, dtype=np.float64) / n_bins


def quantile_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Approximately equal-frequency edges with honest tie handling.

    Interior edges are empirical quantiles (``method='inverted_cdf'``, so every edge is an
    observed value). Duplicate edges collapse, and edges at or below the minimum prediction
    are dropped because they would only create empty bins. No noise is ever added.
    Constant predictions yield a single bin ``[0, 1]``.
    """
    p = np.asarray(p, dtype=np.float64)
    if n_bins <= 1 or p.size == 0:
        return np.array([0.0, 1.0])
    qs = np.arange(1, n_bins, dtype=np.float64) / n_bins
    interior = np.unique(np.quantile(p, qs, method="inverted_cdf"))
    interior = interior[(interior > p.min()) & (interior > 0.0) & (interior < 1.0)]
    return np.concatenate([[0.0], interior, [1.0]])


def assign_bins(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index for every probability under the left-closed rule (last bin closed)."""
    p = np.asarray(p, dtype=np.float64)
    k = len(edges) - 1
    idx = np.searchsorted(edges, p, side="right") - 1
    return np.clip(idx, 0, k - 1).astype(np.int64)


def wilson_interval(k: np.ndarray, n: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for binomial proportions (vectorised; n > 0 required).

    Matches ``scipy.stats.binomtest(k, n).proportion_ci(level, method='wilson')``.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    z = float(norm.ppf(0.5 + level / 2.0))
    z2 = z * z
    denom = n + z2
    center = (k + z2 / 2.0) / denom
    half = (z / denom) * np.sqrt(k * (n - k) / n + z2 / 4.0)
    low = np.where(k == 0, 0.0, np.clip(center - half, 0.0, 1.0))
    high = np.where(k == n, 1.0, np.clip(center + half, 0.0, 1.0))
    return low, high


def bin_table(y: np.ndarray, p: np.ndarray, edges: np.ndarray, *, strategy: str,
              n_bins_requested: int, edge_source: str, min_support: int,
              wilson_level: float | None, wilson_reason: str | None = None) -> BinningResult:
    """Per-bin statistics. Empty bins are kept with unavailable estimates."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    k = len(edges) - 1
    idx = assign_bins(p, edges)
    counts = np.bincount(idx, minlength=k)
    pos = np.bincount(idx, weights=y, minlength=k)
    psum = np.bincount(idx, weights=p, minlength=k)
    assert counts.sum() == p.size, "every observation must be assigned to exactly one bin"
    nonempty = counts > 0
    lo = hi = None
    if wilson_level is not None and nonempty.any():
        lo, hi = wilson_interval(pos[nonempty], counts[nonempty], wilson_level)
    bins: list[BinStat] = []
    j = 0
    for b in range(k):
        c = int(counts[b])
        if c == 0:
            bins.append(BinStat(index=b, lower=float(edges[b]), upper=float(edges[b + 1]),
                                closed="both" if b == k - 1 else "left", count=0, n_pos=0,
                                mean_pred=None, observed_rate=None, signed_gap=None,
                                abs_gap=None, ci_status="unavailable",
                                ci_reason="UNAVAILABLE_EMPTY", low_support=True))
            continue
        mp = float(psum[b] / c)
        orate = float(pos[b] / c)
        if lo is not None and hi is not None:
            ci = {"ci_low": float(lo[j]), "ci_high": float(hi[j]), "ci_status": "ok"}
        else:
            ci = {"ci_status": "unavailable" if wilson_reason else "skipped",
                  "ci_reason": wilson_reason or "UNAVAILABLE_DISABLED"}
        j += 1
        bins.append(BinStat(index=b, lower=float(edges[b]), upper=float(edges[b + 1]),
                            closed="both" if b == k - 1 else "left", count=c,
                            n_pos=round(pos[b]), mean_pred=mp, observed_rate=orate,
                            signed_gap=mp - orate, abs_gap=abs(mp - orate),
                            low_support=c < min_support, **ci))
    supports = counts[nonempty]
    return BinningResult(
        strategy=strategy,
        n_bins_requested=n_bins_requested, n_bins_actual=k, n_nonempty=int(nonempty.sum()),
        edges=[float(e) for e in edges], edge_rule=EDGE_RULE, edge_source=edge_source,
        bins=bins, min_support_nonempty=int(supports.min()) if supports.size else None,
        support_threshold=min_support,
        ci_method="Wilson score (independent observations, pointwise)" if lo is not None else None,
        ci_level=wilson_level if lo is not None else None)


def bin_summaries(binning: BinningResult, n: int, population: str) -> dict[str, MetricValue]:
    """ECE, max gap (all and supported bins), RMS gap from one declared partition."""
    ne = [b for b in binning.bins if b.count > 0]
    part = (f"{binning.strategy}, {binning.n_bins_actual} bins "
            f"({binning.n_nonempty} nonempty; {binning.n_bins_requested} requested)")
    common: dict[str, Any] = {"partition": part, "edges": binning.edges}
    if not ne or n == 0:
        return {name: unavailable(name, "UNAVAILABLE_EMPTY", n, population, **common)
                for name in ("ece", "max_abs_gap", "max_abs_gap_supported", "rms_gap")}
    w = np.array([b.count for b in ne], dtype=np.float64) / n
    gaps = np.array([b.abs_gap for b in ne], dtype=np.float64)
    ece = float((w * gaps).sum())
    rms = float(np.sqrt((w * gaps**2).sum()))
    imax = int(np.argmax(gaps))
    out = {
        "ece": _mv("ece", ece, n, population, estimand="binned positive-class ECE",
                   bins_used="all nonempty bins", **common),
        "rms_gap": _mv("rms_gap", rms, n, population, bins_used="all nonempty bins",
                       weighting="bin support", **common),
        "max_abs_gap": _mv("max_abs_gap", float(gaps[imax]), n, population,
                           bins_used="all nonempty bins", bin_index=ne[imax].index,
                           bin_range=[ne[imax].lower, ne[imax].upper],
                           bin_support=ne[imax].count, **common),
    }
    sup = [b for b in ne if b.count >= binning.support_threshold]
    excluded_n = int(sum(b.count for b in ne if b.count < binning.support_threshold))
    if sup:
        g2 = np.array([b.abs_gap for b in sup], dtype=np.float64)
        i2 = int(np.argmax(g2))
        out["max_abs_gap_supported"] = _mv(
            "max_abs_gap_supported", float(g2[i2]), n, population,
            bins_used=f"nonempty bins with count >= {binning.support_threshold}",
            bins_included=len(sup), bins_excluded=len(ne) - len(sup),
            observations_excluded=excluded_n, bin_index=sup[i2].index,
            bin_range=[sup[i2].lower, sup[i2].upper], bin_support=sup[i2].count, **common)
    else:
        out["max_abs_gap_supported"] = unavailable(
            "max_abs_gap_supported", "UNAVAILABLE_NO_SUPPORTED_BINS", n, population,
            observations_excluded=excluded_n, **common)
    return out


def make_edges(p: np.ndarray, strategy: str, n_bins: int) -> np.ndarray:
    if strategy == "uniform":
        return uniform_edges(n_bins)
    if strategy == "quantile":
        return quantile_edges(p, n_bins)
    raise ValueError(f"unknown binning strategy {strategy!r}")


def ece_from_assignments(y: np.ndarray, p: np.ndarray, idx: np.ndarray, k: int) -> float:
    n = y.size
    counts = np.bincount(idx, minlength=k)
    pos = np.bincount(idx, weights=y, minlength=k)
    psum = np.bincount(idx, weights=p, minlength=k)
    m = counts > 0
    return float(np.abs(pos[m] - psum[m]).sum() / n)


def sensitivity_table(y: np.ndarray, p: np.ndarray, *, bins: list[int], strategies: list[str],
                      primary: tuple[str, int], min_support: int) -> list[SensitivityRow]:
    """Bin-based diagnostics under several predeclared partitions (a robustness check)."""
    rows = []
    y = np.asarray(y, dtype=np.float64)
    combos = [(s, b) for s in strategies for b in bins]
    if primary not in combos:
        combos.append(primary)
    for strategy, nb in combos:
        edges = make_edges(p, strategy, nb)
        bt = bin_table(y, p, edges, strategy=strategy, n_bins_requested=nb, edge_source="",
                       min_support=min_support, wilson_level=None)
        s = bin_summaries(bt, int(y.size), "")
        rows.append(SensitivityRow(
            strategy=strategy,
            n_bins_requested=nb, n_bins_actual=bt.n_bins_actual, n_nonempty=bt.n_nonempty,
            min_support=bt.min_support_nonempty, ece=s["ece"].value,
            max_abs_gap=s["max_abs_gap"].value, rms_gap=s["rms_gap"].value,
            is_primary=(strategy, nb) == primary))
    return rows
