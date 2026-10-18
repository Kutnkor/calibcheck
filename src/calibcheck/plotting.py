"""Static Matplotlib figures rendered from structured results (never from raw data).

Encoding choices: fixed [0, 1] axes, the y = x reference line, point area proportional to
bin support, pointwise Wilson bars labelled as bin-rate intervals, and an aligned histogram of
predictions so sparse regions are visible. Empty bins are omitted from the curve, never
interpolated. Model identity uses colour *and* marker shape; at most three models share one
panel (the validated colour-vision-safe set), larger collections use separate panels.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from calibcheck.schemas import BinStat, GroupResult, ModelResult, TimeWindowResult

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
MAX_OVERLAY = 3


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _shorten(name: str, n: int = 32) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _points(bins: Sequence[BinStat]) -> tuple[list[float], list[float], list[int], list[float], list[float]]:
    xs, ys, ns, lo, hi = [], [], [], [], []
    for b in bins:
        if b.count == 0 or b.mean_pred is None or b.observed_rate is None:
            continue
        xs.append(b.mean_pred)
        ys.append(b.observed_rate)
        ns.append(b.count)
        lo.append(b.observed_rate - b.ci_low if b.ci_low is not None else 0.0)
        hi.append(b.ci_high - b.observed_rate if b.ci_high is not None else 0.0)
    return xs, ys, ns, lo, hi


def _sizes(ns: Sequence[int], total_max: int) -> list[float]:
    return [18 + 220 * (n / total_max) ** 0.5 for n in ns] if total_max else []


def reliability_figure(models: Sequence[ModelResult], *, title: str | None = None,
                       color_index: dict[str, int] | None = None) -> Figure:
    models = list(models)[:MAX_OVERLAY]
    color_index = color_index or {m.name: i for i, m in enumerate(models)}
    fig = plt.figure(figsize=(6.6, 7.4), dpi=110, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)
    _style(ax)
    _style(axh)
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK2, linewidth=1.2, label="y = x (perfect calibration)")
    nmax = max((b.count for m in models for b in m.binning.bins), default=1)
    any_ci = False
    for m in models:
        i = color_index.get(m.name, 0) % len(COLORS)
        xs, ys, ns, lo, hi = _points(m.binning.bins)
        has_ci = any(b.ci_status == "ok" for b in m.binning.bins)
        any_ci |= has_ci
        if has_ci:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=COLORS[i], elinewidth=1.2,
                        capsize=2.5, alpha=0.9, zorder=2)
        ax.plot(xs, ys, color=COLORS[i], linewidth=1.6, zorder=3)
        ax.scatter(xs, ys, s=_sizes(ns, nmax), color=COLORS[i], marker=MARKERS[i],
                   edgecolor=SURFACE, linewidth=1.2, zorder=4, label=_shorten(m.display_name))
        widths = [b - a for a, b in zip(m.summary.histogram_edges[:-1], m.summary.histogram_edges[1:], strict=True)]
        offset = (i - (len(models) - 1) / 2) * (widths[0] / max(len(models), 1))
        axh.bar([a + w / 2 + offset for a, w in zip(m.summary.histogram_edges[:-1], widths, strict=True)],
                m.summary.histogram_counts, width=widths[0] / max(len(models), 1) * 0.92,
                color=COLORS[i], alpha=0.85, label=_shorten(m.display_name))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed positive fraction in bin", color=INK, fontsize=10)
    plt.setp(ax.get_xticklabels(), visible=False)
    axh.set_xlabel("Mean predicted probability of the positive class", color=INK, fontsize=10)
    axh.set_ylabel("Rows", color=INK, fontsize=9)
    axh.set_xlim(0, 1)
    b0 = models[0].binning
    sub = (f"{b0.strategy} bins: {b0.n_bins_actual} ({b0.n_nonempty} nonempty); "
           "point area ∝ bin count; empty bins omitted")
    if any_ci:
        sub += (f"\nbars: pointwise {round((b0.ci_level or 0.95) * 100)}% Wilson intervals "
                "for the observed bin rate (not individual-probability intervals)")
    fig.text(0.02, 0.985, (title + "\n" if title else "") + sub, fontsize=8.5, color=INK2,
             va="top", ha="left")
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.93),
                     ncol=2, fontsize=8.5, frameon=False, markerscale=0.6)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.08)
    return fig


def group_figure(group: GroupResult, model: ModelResult, max_panels: int = 6) -> Figure | None:
    rows = [r for r in group.rows if not r.suppressed and r.bins.get(model.name)][:max_panels]
    if not rows:
        return None
    k = len(rows)
    cols = min(3, k)
    nrows = (k + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.0 * cols, 3.0 * nrows + 0.4), dpi=110,
                             squeeze=False, facecolor=SURFACE)
    nmax = max((b.count for r in rows for b in r.bins[model.name]), default=1)
    for ax, r in zip(axes.ravel(), rows, strict=False):
        _style(ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK2, linewidth=1)
        xs, ys, ns, lo, hi = _points(r.bins[model.name])
        if any(b.ci_status == "ok" for b in r.bins[model.name]):
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=COLORS[0], elinewidth=1,
                        capsize=2)
        ax.plot(xs, ys, color=COLORS[0], linewidth=1.4)
        ax.scatter(xs, ys, s=_sizes(ns, nmax), color=COLORS[0], edgecolor=SURFACE, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{_shorten(r.name, 24)} (n={r.n})", fontsize=9, color=INK)
    for ax in axes.ravel()[k:]:
        ax.set_visible(False)
    fig.suptitle(f"{_shorten(model.display_name)} by {_shorten(group.column, 24)} — shared edges; "
                 "x: mean prediction, y: observed rate", fontsize=9, color=INK2)
    fig.tight_layout()
    return fig


def time_figure(tw: TimeWindowResult, models: Sequence[ModelResult]) -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=110, facecolor=SURFACE)
    _style(ax)
    x = list(range(len(tw.rows)))

    def _val(v: float | None) -> float:
        return np.nan if v is None else float(v)  # suppressed/undefined -> gap, never zero

    ax.plot(x, [_val(r.prevalence) for r in tw.rows], color=INK, linewidth=2, marker="D",
            markersize=5, label="Observed prevalence")
    for i, m in enumerate(list(models)[:MAX_OVERLAY]):
        ax.plot(x, [_val(r.per_model[m.name]["mean_pred"]) for r in tw.rows], color=COLORS[i],
                marker=MARKERS[i], linewidth=1.6, markersize=5,
                label=f"Mean prediction: {_shorten(m.display_name, 24)}")
    ax.set_xticks(x)
    ax.set_xticklabels([r.window + (" (suppressed)" if r.suppressed else "") for r in tw.rows],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability", color=INK, fontsize=9)
    ax.set_title(f"Per-{tw.frequency} windows (descriptive; no intervals for dependent windows)",
                 fontsize=9, color=INK2, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def to_png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


def reliability_alt_text(models: Sequence[ModelResult]) -> str:
    parts = []
    for m in models:
        ne = [b for b in m.binning.bins if b.count]
        worst = max(ne, key=lambda b: b.abs_gap or 0.0) if ne else None
        s = (f"{m.display_name}: {m.binning.n_nonempty} nonempty bins over {m.n} rows")
        if worst is not None:
            s += (f"; largest gap in [{worst.lower:.2f}, {worst.upper:.2f}] with mean prediction "
                  f"{worst.mean_pred:.3f} vs observed {worst.observed_rate:.3f} (n={worst.count})")
        parts.append(s + ".")
    return "Reliability diagram. " + " ".join(parts)
