"""Seeded, batched observation-level bootstrap.

Estimand: the sampling distribution of an aggregate statistic over the *fixed* evaluation
predictions, under independent and identically distributed observations. Intervals are
percentile intervals. They are conditional on the fitted model (and any fitted recalibrator)
and do not capture training or method-selection variability.

Memory: indices are drawn in batches of ``max(1, max_elements // n)`` replicates so memory
scales with the batch, not with ``n * n_resamples``.

Pairing: :func:`bootstrap_means` resamples the *same* row indices for every column, so
differences between models are computed replicate-by-replicate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from calibcheck.config import UncertaintyConfig
from calibcheck.schemas import Interval

IID_NOTE = ("Percentile bootstrap over observations (IID assumption), conditional on the fixed "
            "predictions; not an interval for any individual's event probability.")


def _batches(n: int, n_resamples: int, max_elements: int) -> Iterator[int]:
    size = max(1, max_elements // max(n, 1))
    done = 0
    while done < n_resamples:
        b = min(size, n_resamples - done)
        yield b
        done += b


def resample_indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    dtype = np.int32 if n < 2**31 - 1 else np.int64
    return rng.integers(0, n, size=(b, n), dtype=dtype)


def bootstrap_means(columns: np.ndarray, ucfg: UncertaintyConfig) -> np.ndarray:
    """Replicate means of each column of ``columns`` (shape n x k), paired across columns.

    Returns an array of shape (n_resamples, k).
    """
    cols = np.asarray(columns, dtype=np.float64)
    if cols.ndim == 1:
        cols = cols[:, None]
    n, k = cols.shape
    rng = np.random.default_rng(ucfg.seed)
    out = np.empty((ucfg.n_resamples, k), dtype=np.float64)
    pos = 0
    for b in _batches(n, ucfg.n_resamples, ucfg.max_elements_per_batch):
        idx = resample_indices(rng, n, b)
        for j in range(k):
            out[pos:pos + b, j] = cols[:, j][idx].mean(axis=1)
        pos += b
    return out


def bootstrap_statistic(n: int, stat: Callable[[np.ndarray], np.ndarray],
                        ucfg: UncertaintyConfig, k: int = 1) -> np.ndarray:
    """General batched bootstrap: ``stat`` maps an index matrix (b x n) to (b x k) values.

    Undefined replicates must be returned as NaN; they are counted, never silently replaced.
    """
    rng = np.random.default_rng(ucfg.seed)
    out = np.empty((ucfg.n_resamples, k), dtype=np.float64)
    pos = 0
    for b in _batches(n, ucfg.n_resamples, ucfg.max_elements_per_batch):
        idx = resample_indices(rng, n, b)
        out[pos:pos + b] = np.asarray(stat(idx), dtype=np.float64).reshape(b, k)
        pos += b
    return out


def percentile_interval(replicates: np.ndarray, level: float, *, seed: int, method_note: str,
                        min_effective_fraction: float = 0.99) -> Interval:
    reps = np.asarray(replicates, dtype=np.float64)
    finite = np.isfinite(reps)
    n_eff = int(finite.sum())
    n_failed = int(reps.size - n_eff)
    if n_eff == 0 or n_eff < min_effective_fraction * reps.size:
        return Interval(status="failed", method="percentile bootstrap", level=level,
                        reason="UNC_BOOTSTRAP_FAILED_REPLICATES", n_resamples=int(reps.size),
                        n_effective=n_eff, n_failed=n_failed, seed=seed,
                        interpretation=method_note)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps[finite], [a, 1.0 - a])
    reason = None
    if n_failed:
        reason = "UNC_BOOTSTRAP_FAILED_REPLICATES"
    if float(hi) == float(lo):
        reason = "UNC_BOOTSTRAP_DEGENERATE"
    return Interval(status="ok", method="percentile bootstrap", level=level, low=float(lo),
                    high=float(hi), reason=reason, n_resamples=int(reps.size),
                    n_effective=n_eff, n_failed=n_failed, seed=seed, interpretation=method_note)


def unavailable_interval(reason: str, level: float, method: str = "percentile bootstrap",
                         note: str = "") -> Interval:
    return Interval(status="unavailable", method=method, level=level, reason=reason,
                    interpretation=note)


def dependence_block(dependence: str) -> str | None:
    """Return a reason code if the declared dependence forbids IID interval methods."""
    if dependence == "independent":
        return None
    return "UNAVAILABLE_DEPENDENCE"


def frozen_edge_ece_replicates(y: np.ndarray, probs: list[np.ndarray], assignments: list[np.ndarray],
                               k_bins: list[int], ucfg: UncertaintyConfig) -> np.ndarray:
    """Paired bootstrap of binned ECE with bin edges frozen at their full-sample values.

    Returns (n_resamples, len(probs)). Interpretation is conditional on the declared edges.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size

    def stat(idx: np.ndarray) -> np.ndarray:
        b = idx.shape[0]
        res = np.empty((b, len(probs)))
        yb = y[idx]
        offs = np.arange(b)[:, None]
        for j, (p, a, kb) in enumerate(zip(probs, assignments, k_bins, strict=True)):
            flat = (offs * kb + a[idx]).ravel()
            sy = np.bincount(flat, weights=yb.ravel(), minlength=b * kb).reshape(b, kb)
            sp = np.bincount(flat, weights=p[idx].ravel(), minlength=b * kb).reshape(b, kb)
            res[:, j] = np.abs(sy - sp).sum(axis=1) / n
        return res

    return bootstrap_statistic(n, stat, ucfg, k=len(probs))
