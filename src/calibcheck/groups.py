"""Descriptive subgroup and time-window views built on the shared scoring primitives.

Groups are ordered by size (then name), never by estimated error, so the display does not
turn the worst-looking tiny slice into a headline. Every examined group is counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from calibcheck.binning import assign_bins, bin_table, ece_from_assignments, wilson_interval
from calibcheck.config import AnalysisConfig
from calibcheck.data import MISSING_GROUP, PredictionData
from calibcheck.metrics import brier_losses, log_losses
from calibcheck.schemas import GroupResult, GroupRow, TimeWindowResult, TimeWindowRow
from calibcheck.scoring import _wilson_settings

OTHER_GROUP = "(other)"


def _row_stats(y: np.ndarray, p: np.ndarray, eps: float, edges: np.ndarray) -> dict[str, float | None]:
    n = y.size
    out: dict[str, float | None] = {
        "brier": float(brier_losses(y, p).mean()),
        "log_loss": float(log_losses(y, p, eps).mean()),
        "mean_pred": float(p.mean()),
        "mean_error": float(p.mean() - y.mean()),
        "ece": ece_from_assignments(y.astype(np.float64), p, assign_bins(p, edges),
                                    len(edges) - 1),
    }
    n_pos = int(y.sum())
    if 0 < n_pos < n:
        out["roc_auc"] = float(roc_auc_score(y, p))
    else:
        out["roc_auc"] = None
    return out


def analyze_groups(data: PredictionData, cfg: AnalysisConfig,
                   edges: dict[str, np.ndarray], edge_policy: str) -> list[GroupResult]:
    results = []
    gcfg = cfg.groups
    wl, _ = _wilson_settings(cfg)
    for col, values in data.groups.items():
        levels, counts = np.unique(values.astype(str), return_counts=True)
        order = sorted(range(len(levels)), key=lambda i: (-counts[i], levels[i]))
        shown = [levels[i] for i in order[: gcfg.max_groups]]
        rest = [levels[i] for i in order[gcfg.max_groups:]]
        buckets: list[tuple[str, np.ndarray]] = [(g, values == g) for g in shown]
        if rest:
            buckets.append((OTHER_GROUP, np.isin(values, rest)))
        rows: list[GroupRow] = []
        for name, mask in buckets:
            y = data.y[mask]
            n = int(mask.sum())
            n_pos = int(y.sum())
            row = GroupRow(name=name, n=n, n_pos=n_pos, prevalence=n_pos / n if n else None,
                           low_support=n < gcfg.min_group_size)
            if wl is not None and n:
                lo, hi = wilson_interval(np.array([n_pos]), np.array([n]), wl)
                row.prevalence_ci_low, row.prevalence_ci_high = float(lo[0]), float(hi[0])
            for m, p in data.probs.items():
                pm = p[mask]
                row.per_model[m] = _row_stats(y, pm, cfg.clip_eps, edges[m])
                bt = bin_table(y, pm, edges[m], strategy=cfg.binning.strategy,
                               n_bins_requested=cfg.binning.n_bins, edge_source=edge_policy,
                               min_support=cfg.binning.min_support, wilson_level=wl)
                row.bins[m] = bt.bins
            rows.append(row)
        results.append(GroupResult(
            column=col, n_groups_total=len(levels), n_groups_shown=len(shown),
            n_groups_examined=len(levels), other_bucket=bool(rest),
            missing_category_present=MISSING_GROUP in set(levels.tolist()),
            edges=[float(e) for e in next(iter(edges.values()))], edge_policy=edge_policy,
            rows=rows))
    return results


def flag_small_cells(cells: list[tuple[int, int]], min_cell: int) -> list[tuple[int, str]]:
    """The public small-cell policy shared by subgroups and time windows.

    ``cells`` holds (n, n_pos) per cell. A cell is flagged when n < min_cell or when all its
    outcomes are identical (which would reveal every member's outcome). If exactly one cell is
    flagged, the next-smallest cell is flagged too (complementary suppression) so the hidden
    values cannot be recovered by subtracting the published cells from the overall totals.
    Returns (index, reason) pairs.
    """
    flagged: list[tuple[int, str]] = []
    for i, (n, n_pos) in enumerate(cells):
        if n < min_cell:
            flagged.append((i, f"n < {min_cell}"))
        elif n_pos in (0, n):
            flagged.append((i, "all outcomes identical"))
    if len(flagged) == 1:
        remaining = [i for i in range(len(cells)) if i != flagged[0][0]]
        if remaining:
            j = min(remaining, key=lambda i: (cells[i][0], i))
            flagged.append((j, "complementary suppression"))
    return flagged


def suppress_small_cells(groups: list[GroupResult], min_cell: int,
                         reveal_names: bool) -> list[GroupResult]:
    """Public-export suppression of subgroup cells (policy: :func:`flag_small_cells`)."""
    out = []
    for g in groups:
        g = g.model_copy(deep=True)
        flagged = flag_small_cells([(r.n or 0, r.n_pos or 0) for r in g.rows], min_cell)
        for i, why in flagged:
            r = g.rows[i]
            r.suppressed = True
            r.suppression_reason = f"UNAVAILABLE_SUPPRESSED ({why})"
            r.n_pos = None
            r.prevalence = None
            r.prevalence_ci_low = r.prevalence_ci_high = None
            r.per_model = {m: dict.fromkeys(v) for m, v in r.per_model.items()}
            r.bins = {}
            if r.n is not None and r.n < min_cell:
                r.n = None  # the count itself is withheld
        if not reveal_names:
            k = 0
            for r in g.rows:
                if r.name in (MISSING_GROUP, OTHER_GROUP):
                    continue
                k += 1
                r.name = f"group {k}"
        out.append(g)
    return out


def analyze_time_windows(data: PredictionData, cfg: AnalysisConfig) -> TimeWindowResult | None:
    if data.timestamps is None or cfg.time_window is None:
        return None
    ts = data.timestamps
    ok = ts.notna().to_numpy()
    freq = {"D": "D", "W": "W", "M": "M", "Q": "Q", "Y": "Y"}[cfg.time_window]
    naive = ts[ok].dt.tz_convert("UTC").dt.tz_localize(None)
    periods = naive.dt.to_period(freq).astype(str).to_numpy()
    idx = np.flatnonzero(ok)
    rows = []
    for w in sorted(set(periods.tolist()), key=lambda s: pd.Period(s, freq=freq)):
        sel = idx[periods == w]
        y = data.y[sel]
        per = {}
        for m, p in data.probs.items():
            pm = p[sel]
            per[m] = {"mean_pred": float(pm.mean()), "brier": float(brier_losses(y, pm).mean()),
                      "log_loss": float(log_losses(y, pm, cfg.clip_eps).mean())}
        rows.append(TimeWindowRow(window=w, n=int(sel.size), n_pos=int(y.sum()),
                                  prevalence=float(y.mean()), per_model=per))
    return TimeWindowResult(column=cfg.columns.timestamp or "", frequency=cfg.time_window,
                            timezone_handling=data.timezone_handling or "",
                            n_missing=int((~ok).sum()), rows=rows)


def suppress_small_windows(tw: TimeWindowResult | None, min_cell: int) -> TimeWindowResult | None:
    """Public-export suppression of time windows, using the same policy as subgroups.

    Suppressed windows keep their label (a calendar period, not a record) but lose the event
    count, prevalence and every per-model statistic, which become ``null``. The row count is
    withheld too when it is below ``min_cell``. Nothing is replaced by zero.
    """
    if tw is None:
        return None
    tw = tw.model_copy(deep=True)
    flagged = flag_small_cells([(r.n or 0, r.n_pos or 0) for r in tw.rows], min_cell)
    for i, why in flagged:
        r = tw.rows[i]
        r.suppressed = True
        r.suppression_reason = f"UNAVAILABLE_SUPPRESSED ({why})"
        r.n_pos = None
        r.prevalence = None
        r.per_model = {m: dict.fromkeys(v) for m, v in r.per_model.items()}
        if r.n is not None and r.n < min_cell:
            r.n = None
    tw.suppression_policy = (f"public mode: windows with fewer than {min_cell} rows or identical "
                             "outcomes are suppressed, plus complementary suppression when only "
                             "one window would be hidden")
    return tw
