"""Deterministic, rule-based findings generated from structured results.

No language model is involved and no data leaves the machine. Each finding carries a stable
code, the supporting statistics, the population, and a caveat. The vocabulary deliberately
avoids verdicts such as "trustworthy", "safe", "unbiased" or "ready for deployment".
"""

from __future__ import annotations

import numpy as np

from calibcheck.binning import wilson_interval
from calibcheck.codes import explain
from calibcheck.schemas import AnalysisResult, BinStat, Finding, GroupResult, ModelResult

LARGE_SPARSE_GAP = 0.10  # heuristic for flagging sparse bins; not a statistical threshold
HIGH_AUC = 0.75          # heuristic for "ranks well" in the ranking-vs-calibration note
SMALL_N = 100            # heuristic for small evaluation sets

BANNED_WORDS = ("trustworthy", "safe ", "unbiased", "ready for deployment", "guarantee",
                "certif", "fair ", "proves", "significant")


def _fmt(x: float | None, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _rng(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {hi:.2f}]"


def _partition(m: ModelResult) -> str:
    b = m.binning
    return f"{b.strategy} binning, {b.n_bins_actual} bins ({b.n_nonempty} nonempty)"


def _separated(b: BinStat) -> bool:
    return (b.ci_status == "ok" and b.ci_low is not None and b.ci_high is not None
            and b.mean_pred is not None and not (b.ci_low <= b.mean_pred <= b.ci_high))


def _merge_runs(bins: list[BinStat]) -> list[list[BinStat]]:
    """Group consecutive (by index among nonempty bins) separated bins with the same sign."""
    runs: list[list[BinStat]] = []
    prev: BinStat | None = None
    for b in bins:
        if b.count == 0:
            continue
        sep = _separated(b) and not b.low_support
        if not sep:
            prev = None
            continue
        sign = np.sign(b.signed_gap or 0.0)
        if prev is not None and np.sign(prev.signed_gap or 0.0) == sign and runs:
            runs[-1].append(b)
        else:
            runs.append([b])
        prev = b
    return runs


def model_findings(m: ModelResult, level: float | None, dependence: str) -> list[Finding]:
    out: list[Finding] = []
    pop = f"{m.population} (n={m.n})"
    bins = m.binning.bins
    supported = [b for b in bins if b.count >= m.binning.support_threshold]
    ci_on = any(b.ci_status == "ok" for b in bins)
    part = _partition(m)

    if m.summary.n_unique == 1:
        out.append(Finding(
            code="FINDING_CONSTANT_PREDICTOR", severity="info", model=m.name, population=pop,
            message=f"{m.display_name} predicts the same probability ({m.summary.min:.3f}) for "
                    "every row. It can show probability bias relative to the observed rate but "
                    "has no ability to rank observations.",
            evidence={"probability": m.summary.min}))

    if ci_on:
        runs = _merge_runs(bins)
        for run in runs:
            n = sum(b.count for b in run)
            k = sum(b.n_pos for b in run)
            mp = sum((b.mean_pred or 0.0) * b.count for b in run) / n
            orate = k / n
            lo, hi = wilson_interval(np.array([k]), np.array([n]), level or 0.95)
            over = mp > orate
            code = "FINDING_OVERPREDICTION" if over else "FINDING_UNDERPREDICTION"
            verb = "exceed" if over else "fall below"
            out.append(Finding(
                code=code, severity="warning", model=m.name, population=pop,
                message=(f"For {m.display_name}, predicted probabilities {verb} the observed event "
                         f"rate in {_rng(run[0].lower, run[-1].upper)}: mean prediction "
                         f"{mp:.3f} vs observed {orate:.3f} ({k} events in {n} rows; "
                         f"{round((level or 0.95) * 100)}% Wilson interval "
                         f"{lo[0]:.3f}–{hi[0]:.3f})."),
                caveat=(f"Pointwise approximate interval assuming independent observations; "
                        f"{part}. With {m.binning.n_nonempty} bins checked, roughly "
                        f"{round((1 - (level or 0.95)) * 100)}% of well-calibrated bins "
                        "would be flagged by chance, so small isolated gaps deserve "
                        "confirmation on new data. Not an interval for any individual's "
                        "probability."),
                evidence={"range_low": run[0].lower, "range_high": run[-1].upper, "n": n,
                          "events": k, "mean_pred": mp, "observed_rate": orate,
                          "ci_low": float(lo[0]), "ci_high": float(hi[0]),
                          "bins": len(run)}))
        if not runs:
            covered = sum(b.count for b in supported)
            out.append(Finding(
                code="FINDING_NO_CLEAR_BIN_GAP", severity="info", model=m.name, population=pop,
                message=(f"For {m.display_name}, none of the {len(supported)} bins with at least "
                         f"{m.binning.support_threshold} rows ({covered} of {m.n} rows) has a "
                         "Wilson interval for its observed rate that excludes the bin's mean "
                         "prediction."),
                caveat=("This is absence of evidence for a gap at this resolution, not proof of "
                        "calibration: gaps can be smaller than the intervals can detect, hide "
                        "inside bins, or occur in sparse regions."),
                evidence={"supported_bins": len(supported), "rows_covered": covered}))
    else:
        reason = next((b.ci_reason for b in bins if b.ci_reason), "UNAVAILABLE_DISABLED")
        cand = [b for b in supported if b.abs_gap is not None]
        if cand:
            b = max(cand, key=lambda x: (x.abs_gap or 0.0, -x.index))
            over = (b.signed_gap or 0) > 0
            out.append(Finding(
                code="FINDING_OVERPREDICTION" if over else "FINDING_UNDERPREDICTION",
                severity="info", model=m.name, population=pop,
                message=(f"For {m.display_name}, the largest gap among bins with at least "
                         f"{m.binning.support_threshold} rows is in {_rng(b.lower, b.upper)}: "
                         f"mean prediction {_fmt(b.mean_pred)} vs observed "
                         f"{_fmt(b.observed_rate)} (n={b.count})."),
                caveat=(f"No bin intervals were computed ({reason}; declared dependence: "
                        f"{dependence}), so this gap is descriptive and not compared with "
                        "sampling variability."),
                evidence={"n": b.count, "mean_pred": b.mean_pred, "observed_rate": b.observed_rate}))

    sparse = [b for b in bins if 0 < b.count < m.binning.support_threshold
              and (b.abs_gap or 0) >= LARGE_SPARSE_GAP]
    if sparse:
        tot = sum(b.count for b in sparse)
        out.append(Finding(
            code="FINDING_SPARSE_LARGE_GAP", severity="info", model=m.name, population=pop,
            message=(f"For {m.display_name}, {len(sparse)} sparse bin(s) (fewer than "
                     f"{m.binning.support_threshold} rows each; {tot} rows in total) show gaps of "
                     f"at least {LARGE_SPARSE_GAP:.2f}. Evidence in these regions is weak."),
            caveat="The support threshold is a practical display heuristic, not a statistical law.",
            evidence={"bins": len(sparse), "rows": tot}))

    me = m.metrics["mean_error"]
    if me.available:
        mean_p = m.metrics["mean_pred"].value
        prev = m.metrics["prevalence"].value
        iv = me.interval
        if iv is not None and iv.status == "ok" and iv.low is not None and iv.high is not None:
            ex = iv.low > 0 or iv.high < 0
            msg = (f"{m.display_name}: average prediction {_fmt(mean_p)} vs observed prevalence "
                   f"{_fmt(prev)} (difference {me.value:+.3f}; "
                   f"{round(iv.level * 100)}% bootstrap interval {iv.low:+.3f} to "
                   f"{iv.high:+.3f}).")
            msg += (" The interval excludes zero." if ex else
                    " The interval includes zero, so aggregate bias is not distinguished from "
                    "sampling noise.")
        else:
            msg = (f"{m.display_name}: average prediction {_fmt(mean_p)} vs observed prevalence "
                   f"{_fmt(prev)} (difference {me.value:+.3f}; no interval computed).")
        out.append(Finding(
            code="FINDING_AGGREGATE_BIAS", severity="info", model=m.name, population=pop,
            message=msg, caveat="Positive and negative local gaps can cancel in this aggregate.",
            evidence={"mean_pred": mean_p, "prevalence": prev, "difference": me.value}))

    auc = m.metrics["roc_auc"]
    gaps = [f for f in out if f.code in ("FINDING_OVERPREDICTION", "FINDING_UNDERPREDICTION")
            and f.severity == "warning"]
    if auc.available and (auc.value or 0) >= HIGH_AUC and gaps:
        out.append(Finding(
            code="FINDING_RANKING_VS_CALIBRATION", severity="info", model=m.name, population=pop,
            message=(f"{m.display_name} ranks observations well (ROC-AUC {auc.value:.3f}) while "
                     f"{len(gaps)} probability range(s) show calibration gaps. Good ranking does "
                     "not imply that the probabilities match observed frequencies."),
            evidence={"roc_auc": auc.value, "gap_ranges": len(gaps)}))
    if m.n < SMALL_N:
        out.append(Finding(
            code="UNC_SMALL_SUPPORT", severity="warning", model=m.name, population=pop,
            message=(f"Only {m.n} rows were evaluated. Even a reliability curve that looks "
                     "straight can coexist with wide uncertainty at this size."),
            evidence={"n": m.n}))
    return out


def comparison_findings(res: AnalysisResult) -> list[Finding]:
    out: list[Finding] = []
    by_pair: dict[tuple[str, str], list] = {}
    names = {m.name: m.display_name for m in res.models}
    for c in res.comparisons:
        by_pair.setdefault((c.reference, c.candidate), []).append(c)
        if c.status != "ok" or c.delta is None or c.metric == "roc_auc":
            continue
        cand, ref = names.get(c.candidate, c.candidate), names.get(c.reference, c.reference)
        label = {"brier": "Brier loss", "log_loss": "log loss", "ece": "binned ECE"}[c.metric]
        iv = c.interval
        pop = f"{c.population} (n={c.n}, identical rows for both)"
        if c.favours in ("candidate", "reference") and iv is not None:
            winner = cand if c.favours == "candidate" else ref
            loser = ref if c.favours == "candidate" else cand
            out.append(Finding(
                code="FINDING_COMPARISON_FAVORS", severity="info", population=pop,
                message=(f"On {label}, {winner} is lower than {loser}: {c.definition.split(';')[0]}"
                         f" = {c.delta:+.4f} ({round(iv.level * 100)}% paired bootstrap "
                         f"interval {iv.low:+.4f} to {iv.high:+.4f})."),
                caveat=("Interval resamples the same rows for both models and is conditional on "
                        "the fixed predictions." + (" ECE intervals are exploratory." if
                                                    c.metric == "ece" else "")),
                evidence={"delta": c.delta, "ci_low": iv.low, "ci_high": iv.high, "n": c.n,
                          "metric": c.metric}))
        elif c.favours == "neither" and iv is not None:
            out.append(Finding(
                code="FINDING_COMPARISON_INCONCLUSIVE", severity="info", population=pop,
                message=(f"Insufficient evidence to distinguish {cand} from {ref} on {label}: "
                         f"difference {c.delta:+.4f}, {round(iv.level * 100)}% paired "
                         f"interval {iv.low:+.4f} to {iv.high:+.4f} includes zero."),
                evidence={"delta": c.delta, "ci_low": iv.low, "ci_high": iv.high, "n": c.n,
                          "metric": c.metric}))
        else:
            why = (explain(iv.reason).rstrip(".") if iv is not None and iv.reason
                   else "bootstrap not requested")
            why = why[:1].lower() + why[1:]
            out.append(Finding(
                code="FINDING_COMPARISON_NO_INTERVAL", severity="info", population=pop,
                message=(f"{label}: {cand} minus {ref} = {c.delta:+.4f} on the same rows. No "
                         f"paired interval was computed ({why}); the magnitude alone does not "
                         "show whether the difference exceeds sampling variability."),
                evidence={"delta": c.delta, "n": c.n, "metric": c.metric}))
    for (ref, cand), cs in by_pair.items():
        signs = {}
        for c in cs:
            if c.status != "ok" or c.delta is None or c.delta == 0:
                continue
            better_cand = c.delta < 0 if c.metric != "roc_auc" else c.delta > 0
            signs[c.metric] = better_cand
        if len(set(signs.values())) > 1:
            pro = [k for k, v in signs.items() if v]
            con = [k for k, v in signs.items() if not v]
            out.append(Finding(
                code="FINDING_METRICS_DISAGREE", severity="info",
                message=(f"Metrics disagree for {names.get(cand, cand)} vs "
                         f"{names.get(ref, ref)}: point estimates favour the candidate on "
                         f"{', '.join(pro)} and the reference on {', '.join(con)}. No overall "
                         "winner is declared."),
                evidence={"favour_candidate": ", ".join(pro), "favour_reference": ", ".join(con)}))
    return out


def group_findings(groups: list[GroupResult], res: AnalysisResult, min_size: int) -> list[Finding]:
    out: list[Finding] = []
    for g in groups:
        eligible = [r for r in g.rows if not r.suppressed and r.n is not None and r.n >= min_size]
        for m in res.models:
            over, under = [], []
            gaps = []
            for r in eligible:
                pm = r.per_model.get(m.name, {}).get("mean_pred")
                if pm is None or r.prevalence is None:
                    continue
                gaps.append((r.name, pm - r.prevalence, r.n))
                if r.prevalence_ci_low is not None and r.prevalence_ci_high is not None:
                    if pm > r.prevalence_ci_high:
                        over.append(r.name)
                    elif pm < r.prevalence_ci_low:
                        under.append(r.name)
            if len(gaps) >= 2:
                lo = min(gaps, key=lambda t: t[1])
                hi = max(gaps, key=lambda t: t[1])
                out.append(Finding(
                    code="FINDING_GROUP_RANGE", severity="info", model=m.name,
                    population=f"{len(eligible)} of {g.n_groups_examined} examined groups in "
                               f"{g.column!r} with n >= {min_size}",
                    message=(f"{m.display_name} by {g.column}: mean prediction minus prevalence "
                             f"ranges from {lo[1]:+.3f} ({lo[0]}, n={lo[2]}) to {hi[1]:+.3f} "
                             f"({hi[0]}, n={hi[2]})."),
                    caveat=("Descriptive and exploratory: groups were not predeclared as "
                            "confirmatory hypotheses, and differences do not establish causality "
                            "or a legal fairness conclusion."),
                    evidence={"groups_examined": g.n_groups_examined, "eligible": len(eligible)}))
            me = m.metrics["mean_error"]
            agg_small = me.available and abs(me.value or 0) < 0.5 * min(
                [abs(t[1]) for t in gaps if t[0] in over + under] or [1.0])
            if over and under and agg_small:
                out.append(Finding(
                    code="FINDING_GROUP_CANCELLATION", severity="warning", model=m.name,
                    population=f"groups of {g.column!r}",
                    message=(f"{m.display_name}: predictions exceed the observed rate in "
                             f"{', '.join(over)} and fall below it in {', '.join(under)} "
                             f"(group Wilson intervals exclude the group mean prediction), while "
                             f"the overall difference is only {me.value:+.3f}. Opposite subgroup "
                             "gaps can cancel in aggregate."),
                    caveat="Descriptive subgroup comparison; not a causal or fairness finding.",
                    evidence={"over": ", ".join(over), "under": ", ".join(under),
                              "aggregate_difference": me.value}))
    return out


def coverage_findings(res: AnalysisResult) -> list[Finding]:
    d = res.dataset
    out = []
    if d.n_pos == 0 or d.n_neg == 0:
        out.append(Finding(
            code="FINDING_LIMITED_COVERAGE", severity="warning",
            message=("Only one outcome class is present. Brier and log loss are reported, ranking "
                     "metrics are unavailable, and the report cannot show how probabilities "
                     "relate to the missing class."),
            evidence={"n_pos": d.n_pos, "n_neg": d.n_neg}))
    return out


def generate_findings(res: AnalysisResult) -> list[Finding]:
    """All findings for a result object, in a fixed order (deterministic)."""
    level = res.config.uncertainty.level if res.config.uncertainty.wilson else None
    out = coverage_findings(res)
    for m in res.models:
        out += model_findings(m, level, res.config.dependence)
    out += comparison_findings(res)
    out += group_findings(res.groups, res, res.config.groups.min_group_size)
    return out


def follow_ups(res: AnalysisResult) -> list[str]:
    """Suggested next steps, kept separate from the numerical evidence."""
    codes = {f.code for f in res.findings}
    out = []
    if codes & {"FINDING_OVERPREDICTION", "FINDING_UNDERPREDICTION"} and res.kind != "recalibration":
        out.append("If probabilities will be used as probabilities, consider fitting a recalibrator "
                   "on separate calibration data (calibcheck fit) and judging it on untouched test "
                   "rows (calibcheck recalibrate). Recalibration is a candidate, not an automatic "
                   "improvement.")
    if "FINDING_SPARSE_LARGE_GAP" in codes or "UNC_SMALL_SUPPORT" in codes:
        out.append("Collect more evaluation data in sparse probability ranges before drawing "
                   "conclusions there.")
    if not res.config.uncertainty.bootstrap and res.config.dependence == "independent":
        out.append("Re-run with --bootstrap for intervals on aggregate losses and paired "
                   "differences.")
    if res.config.dependence != "independent":
        out.append(f"Observations were declared {res.config.dependence}; dependence-aware "
                   "intervals (cluster or block bootstrap) are on the roadmap and were not "
                   "computed.")
    if res.groups:
        out.append("Subgroup results are exploratory; confirm any pattern of interest with "
                   "predeclared groups on separate data.")
    return out
