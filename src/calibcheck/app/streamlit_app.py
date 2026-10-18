"""CalibCheck local browser interface (optional extra: pip install 'calibcheck[app]').

Every number shown here comes from the same library calls as the CLI
(``load_predictions`` → ``analyze_data`` / ``evaluate_recalibration``). Uploaded data lives
only in this browser session's memory on the machine running the app; nothing is persisted,
cached globally or sent elsewhere, and "Clear loaded data" removes it from the session.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from calibcheck import __version__
from calibcheck.analysis import analyze_data
from calibcheck.config import AnalysisConfig, config_to_toml, resolve_config
from calibcheck.data import load_predictions, read_csv_bytes
from calibcheck.errors import CalibCheckError
from calibcheck.metrics import REGISTRY
from calibcheck.plotting import reliability_alt_text, reliability_figure
from calibcheck.protocol import evaluate_recalibration
from calibcheck.report import render_report, to_json
from calibcheck.synthetic import SYNTHETIC_NOTE, headline

st.set_page_config(page_title="CalibCheck", page_icon="📏", layout="wide")

SS = st.session_state
KEY_ORDER = ["brier", "log_loss", "brier_skill", "ece", "max_abs_gap", "mean_error", "roc_auc",
             "average_precision", "accuracy"]


# --------------------------------------------------------------------------------------
# data selection
# --------------------------------------------------------------------------------------

def _demo_bytes() -> tuple[bytes, bytes]:
    sc = headline()
    df = sc.frame.drop(columns="latent_p_true")
    test = df[df.split == "test"].to_csv(index=False).encode()
    cal = df[df.split == "calibration"].to_csv(index=False).encode()
    return test, cal


def clear_session() -> None:
    for k in list(SS.keys()):
        del SS[k]


def header() -> None:
    st.title("CalibCheck")
    st.markdown("**How trustworthy are your model's probabilities?** Check whether predicted "
                "probabilities match observed event frequencies, with uncertainty, paired "
                "comparisons and separate-data recalibration.")
    st.caption(f"v{__version__} · runs locally · Uploaded files are processed in memory by "
               "the machine running this app. If someone hosts this app on a server, that "
               "server receives your upload. No telemetry, no external services.")


def choose_data() -> tuple[bytes, str, bool] | None:
    st.subheader("1 · Choose data")
    src = st.radio("Data source", ["Synthetic demo (90% predicted vs 65% observed cohort)",
                                   "Upload a prediction CSV"], key="source",
                   horizontal=True)
    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button("Clear loaded data", help="Remove uploaded data and results from this "
                                               "session."):
            clear_session()
            st.rerun()
    if src.startswith("Synthetic"):
        if "demo" not in SS:
            SS["demo"] = _demo_bytes()
        with c1:
            st.info(SYNTHETIC_NOTE)
        return SS["demo"][0], "demo_test.csv", True
    with c1:
        up = st.file_uploader("Prediction CSV (one row per observation; outcome column plus "
                              "probability columns for the positive class)", type=["csv"],
                              key="upload")
    if up is None:
        st.stop()
    return up.getvalue(), up.name, False


# --------------------------------------------------------------------------------------
# column mapping & assumptions → AnalysisConfig (same model as the CLI)
# --------------------------------------------------------------------------------------

def map_columns(raw: bytes, is_demo: bool) -> dict[str, Any]:
    st.subheader("2 · Map columns")
    try:
        df, _ = read_csv_bytes(raw, AnalysisConfig())
    except CalibCheckError as exc:
        st.error(f"[{exc.code}] {exc}")
        st.stop()
    cols = list(df.columns)
    st.caption(f"{len(df):,} rows · {len(cols)} columns. Probability columns are never guessed: "
               "choose them explicitly.")
    none = "(none)"
    c1, c2, c3 = st.columns(3)
    label = c1.selectbox("Outcome column", cols,
                         index=cols.index("y_true") if "y_true" in cols else 0, key="label")
    default_probs = ["p_model"] if is_demo and "p_model" in cols else []
    probs = c2.multiselect("Probability column(s): P(outcome = positive)",
                           [c for c in cols if c != label], default=default_probs, key="probs")
    id_opts = [none] + [c for c in cols if c not in probs and c != label]
    id_col = c3.selectbox("Row-ID column (optional, needed for alignment and role checks)",
                          id_opts, index=id_opts.index("row_id") if "row_id" in id_opts else 0,
                          key="id")
    values = sorted({str(v).strip() for v in df[label] if str(v).strip() != ""})[:10]
    c4, c5, c6 = st.columns(3)
    pos_default = "1" if set(values) <= {"0", "1"} and "1" in values else (values[0] if values
                                                                            else "")
    pos = c4.selectbox("Positive label (the event the probabilities refer to)", values,
                       index=values.index(pos_default) if pos_default in values else 0,
                       key="positive")
    used = {label, *probs, id_col}
    groups = c5.multiselect("Subgroup columns (optional, categorical)",
                            [c for c in cols if c not in used],
                            default=["segment"] if is_demo else [], key="groups")
    ts_opts = [none] + [c for c in cols if c not in used]
    ts = c6.selectbox("Timestamp column (optional)", ts_opts, key="timestamp")
    if not probs:
        st.warning("Select at least one probability column to continue.")
        st.stop()
    return {"columns": {"label": label, "probs": probs, "id": None if id_col == none else id_col,
                        "groups": groups, "timestamp": None if ts == none else ts},
            "positive_label": pos}


def assumptions(base: dict[str, Any]) -> AnalysisConfig:
    st.subheader("3 · Confirm assumptions")
    c1, c2 = st.columns(2)
    dep = c1.radio("Observation dependence", ["independent", "entity_clustered",
                                              "time_dependent", "unknown"], key="dependence",
                   help="Interval methods here assume independent observations. Any other "
                        "declaration disables them rather than producing misleading "
                        "precision.")
    missing = c2.radio("Invalid or missing values", ["fail", "drop"], key="missing",
                       format_func=lambda v: {"fail": "Stop with an error (default)",
                                              "drop": "Drop affected rows (recorded)"}[v])
    with st.expander("Advanced statistical settings"):
        a1, a2, a3 = st.columns(3)
        strategy = a1.selectbox("Binning", ["uniform", "quantile"], key="strategy")
        n_bins = a2.number_input("Bins", 2, 50, 10, key="bins")
        min_support = a3.number_input("Small-bin warning threshold (heuristic)", 1, 1000, 30,
                                      key="min_support")
        b1, b2, b3 = st.columns(3)
        boot = b1.checkbox("Bootstrap intervals (aggregate losses, paired differences)",
                           value=True, key="boot")
        n_res = b2.number_input("Bootstrap replicates", 100, 20000, 1000, step=100,
                                key="n_res")
        seed = b3.number_input("Seed", 0, 2**31 - 1, 20251018, key="seed")
        d1, d2, d3 = st.columns(3)
        eps = d1.select_slider("Log-loss clipping ε", [1e-15, 1e-12, 1e-9, 1e-6], value=1e-15,
                               key="eps")
        privacy = d2.radio("Report privacy", ["private", "public"], key="privacy",
                           help="Public omits paths, fingerprints and group names and "
                                "suppresses small cells.")
        tw = d3.selectbox("Time windows", [None, "D", "W", "M", "Q", "Y"], key="tw",
                          format_func=lambda v: "off" if v is None else v)
    try:
        return resolve_config(base, {
            "dependence": dep, "missing_policy": missing,
            "binning": {"strategy": strategy, "n_bins": int(n_bins),
                        "min_support": int(min_support)},
            "uncertainty": {"bootstrap": bool(boot), "n_resamples": int(n_res),
                            "seed": int(seed)},
            "clip_eps": float(eps), "privacy": {"mode": privacy}, "time_window": tw})
    except CalibCheckError as exc:
        st.error(str(exc))
        st.stop()
    raise AssertionError("unreachable")


def cli_command(cfg: AnalysisConfig, filename: str) -> str:
    c = cfg.columns
    parts = ["calibcheck", "report" if len(c.probs) == 1 else "compare", filename,
             "--label", c.label]
    for p in c.probs:
        parts += ["--prob", p]
    if c.id:
        parts += ["--id", c.id]
    if cfg.positive_label is not None:
        parts += ["--positive-label", cfg.positive_label]
    for g in c.groups:
        parts += ["--group", g]
    parts += ["--bins", str(cfg.binning.n_bins), "--strategy", cfg.binning.strategy,
              "--dependence", cfg.dependence]
    if cfg.uncertainty.bootstrap:
        parts += ["--bootstrap", "--n-resamples", str(cfg.uncertainty.n_resamples), "--seed",
                  str(cfg.uncertainty.seed)]
    if cfg.missing_policy == "drop":
        parts.append("--drop-invalid")
    parts += ["--privacy", cfg.privacy.mode, "--out", "calibcheck-report"]
    return " ".join(shlex.quote(p) for p in parts)


def _key(raw: bytes, cfg: AnalysisConfig, extra: str = "") -> str:
    h = hashlib.sha256(raw)
    h.update(cfg.model_dump_json().encode())
    h.update(extra.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------

def _fmt(v: float | None, d: int = 4) -> str:
    return "unavailable" if v is None else f"{v:.{d}f}"


def score_table(res) -> pd.DataFrame:
    rows = []
    for key in KEY_ORDER:
        row = {"Metric": REGISTRY[key].label, "Category": REGISTRY[key].category}
        for m in res.models:
            mv = m.metrics[key]
            if not mv.available:
                row[m.display_name] = f"unavailable ({mv.reason})"
                continue
            s = _fmt(mv.value)
            if mv.interval is not None and mv.interval.status == "ok":
                s += f"  [{_fmt(mv.interval.low)}, {_fmt(mv.interval.high)}]"
            elif mv.interval is not None:
                s += f"  (interval: {mv.interval.reason})"
            row[m.display_name] = s
        rows.append(row)
    return pd.DataFrame(rows)


def show_results(res, raw: bytes, filename: str) -> None:
    d = res.dataset
    st.subheader("4 · Results")
    if res.synthetic:
        st.caption("SYNTHETIC DATA")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows analysed", f"{d.n_rows_retained:,}")
    m2.metric("Rows excluded", f"{d.n_rows_excluded:,}")
    m3.metric("Positives", f"{d.n_pos:,}")
    m4.metric("Prevalence", _fmt(d.prevalence, 3))
    m5.metric("Models", len(res.models))
    st.caption(f"Each probability means {d.probability_meaning}.")
    tabs = st.tabs(["Evidence", "Reliability", "Sensitivity", "Comparisons", "Subgroups",
                    "Validation", "Methodology"])
    with tabs[0]:
        st.markdown("Rule-based findings computed from the statistics. Each names its "
                    "population and caveat. None certifies a model.")
        for f in res.findings:
            box = st.warning if f.severity == "warning" else st.info
            box(f"**{f.code}** — {f.message}" + (f"\n\n_{f.caveat}_" if f.caveat else ""))
        st.markdown("**Key scores**")
        st.dataframe(score_table(res), hide_index=True, use_container_width=True)
    with tabs[1]:
        names = [m.display_name for m in res.models]
        chosen = st.multiselect("Models in the diagram (max 3 per panel)", names,
                                default=names[:3], max_selections=3, key="diagram_models")
        sel = [m for m in res.models if m.display_name in chosen]
        if sel:
            fig = reliability_figure(sel)
            st.pyplot(fig, clear_figure=True)
            st.caption(reliability_alt_text(sel))
        for m in sel:
            st.markdown(f"**Bins — {m.display_name}** · edges: {m.binning.edge_source}; "
                        f"rule: {m.binning.edge_rule}")
            bins = pd.DataFrame([{
                "range": f"[{b.lower:.3f}, {b.upper:.3f}{']' if b.closed == 'both' else ')'}",
                "rows": b.count, "events": b.n_pos, "mean prediction": b.mean_pred,
                "observed rate": b.observed_rate, "gap (pred − obs)": b.signed_gap,
                "interval low": b.ci_low, "interval high": b.ci_high,
                "interval": "Wilson (pointwise, IID)" if b.ci_status == "ok" else b.ci_reason,
                "support": "empty" if b.count == 0 else ("small" if b.low_support else "ok"),
            } for b in m.binning.bins])
            st.dataframe(bins, hide_index=True, use_container_width=True)
    with tabs[2]:
        st.markdown("Robustness check over predeclared partitions; the primary partition "
                    "stays fixed.")
        st.dataframe(pd.DataFrame([{"model": m.display_name, **s.model_dump()}
                                   for m in res.models for s in m.sensitivity]),
                     hide_index=True, use_container_width=True)
    with tabs[3]:
        if not res.comparisons:
            st.write("Select two or more probability columns for paired comparisons.")
        else:
            st.markdown("Differences use identical rows. Intervals resample the same row "
                        "indices for both models.")
            st.dataframe(pd.DataFrame([{
                "metric": c.metric, "candidate": c.candidate, "reference": c.reference,
                "difference": c.delta, "interval low": c.interval.low if c.interval else None,
                "interval high": c.interval.high if c.interval else None,
                "favours": c.favours, "rows": c.n} for c in res.comparisons]),
                hide_index=True, use_container_width=True)
    with tabs[4]:
        if not res.groups:
            st.write("No subgroup columns selected.")
        for g in res.groups:
            st.markdown(f"**{g.column}** — {g.interpretation} {g.n_groups_examined} group(s) "
                        "examined, ordered by size.")
            st.dataframe(pd.DataFrame([{
                "group": r.name, "rows": r.n, "prevalence": r.prevalence,
                "suppressed": r.suppressed, "small": r.low_support,
                **{f"{m}: {k}": v for m, s in r.per_model.items() for k, v in s.items()
                   if k in ("mean_pred", "brier", "ece")}} for r in g.rows]),
                hide_index=True, use_container_width=True)
    with tabs[5]:
        for f in res.validation:
            st.write(f"`{f.code}` {f.message}")
        if not res.validation:
            st.write("No validation warnings.")
    with tabs[6]:
        st.markdown(
            "- **Calibration** compares probabilities with observed frequencies (target "
            "E[Y | P] = P). **Discrimination** (ROC-AUC) describes ranking only.\n"
            "- Brier and log loss are overall scores; a lower value does not by itself prove "
            "better calibration.\n- Binned ECE depends on the partition shown with it.\n"
            "- Wilson bars are approximate pointwise intervals for bin rates, not intervals "
            "for individual probabilities.\n- Subgroup differences are descriptive.")
        with st.expander("Formulas"):
            for key in KEY_ORDER:
                st.markdown(f"**{REGISTRY[key].label}** — {REGISTRY[key].description}")

    st.subheader("6 · Export")
    st.code(cli_command(res.config, filename), language="bash")
    c1, c2, c3 = st.columns(3)
    c1.download_button("report.json", to_json(res.model_dump(mode="json")),
                       file_name="calibcheck_report.json", mime="application/json")
    c2.download_button("config.toml", config_to_toml(res.config), file_name="calibcheck.toml")
    with tempfile.TemporaryDirectory(prefix="calibcheck_") as tmp:  # removed immediately
        render_report(res, Path(tmp) / "r")
        html = (Path(tmp) / "r" / "report.html").read_bytes()
    c3.download_button("Offline HTML report", html, file_name="calibcheck_report.html",
                       mime="text/html")


def recalibration_section(cfg: AnalysisConfig, test_raw: bytes, is_demo: bool) -> None:
    st.subheader("5 · Optional: recalibrate with separate data")
    st.caption("A recalibrator must be fitted on calibration rows that are separate from the "
               "evaluation rows. Results are shown on the evaluation (test) rows only.")
    if is_demo:
        cal_raw = SS["demo"][1]
        st.write("Demo calibration rows: a disjoint half of the synthetic cohort.")
    else:
        up = st.file_uploader("Calibration CSV (same columns, different rows)", type=["csv"],
                              key="cal_upload")
        if up is None:
            return
        cal_raw = up.getvalue()
    model = st.selectbox("Probability column to recalibrate", cfg.columns.probs, key="recal_model")
    methods = st.multiselect("Predeclared methods", ["sigmoid", "isotonic", "temperature"],
                             default=["sigmoid", "isotonic"], key="recal_methods")
    mode = st.radio("Mode", ["exploratory", "selected"], horizontal=True, key="recal_mode",
                    help="Exploratory shows every method on the test rows. Selected chooses "
                         "one method on inner folds of the calibration data, then evaluates "
                         "it once.")
    key = _key(test_raw + cal_raw, cfg, f"{model}|{methods}|{mode}")
    if st.button("Fit on calibration rows and evaluate on test rows", type="primary",
                 key="recal_run"):
        try:
            rcfg = cfg.model_copy(deep=True)
            rcfg.columns.probs = [model]
            cal_cfg = rcfg.model_copy(deep=True)
            cal_cfg.columns.groups = []
            with st.spinner("Fitting and evaluating…"):
                cal = load_predictions(cal_raw, cal_cfg, role="calibration")
                test = load_predictions(test_raw, rcfg, role="test")
                res, arts = evaluate_recalibration(cal, test, rcfg, model=model,
                                                   methods=methods, mode=mode,  # type: ignore[arg-type]
                                                   synthetic=is_demo)
            SS["recal"] = {"key": key, "res": res,
                           "arts": {m: a.model_dump(mode="json") for m, a in arts.items()}}
        except CalibCheckError as exc:
            SS.pop("recal", None)
            st.error(f"[{exc.code}] {exc}")
            return
    rec = SS.get("recal")
    if rec is None:
        return
    if rec["key"] != key:
        st.warning("Settings changed since the last recalibration run. Results are hidden until "
                   "you run it again.")
        return
    res = rec["res"]
    st.json(res.protocol.get("fit", {}), expanded=False)
    st.pyplot(reliability_figure(res.models[:3]), clear_figure=True)
    st.caption(reliability_alt_text(res.models[:3]))
    st.dataframe(score_table(res), hide_index=True, use_container_width=True)
    for f in res.findings:
        (st.warning if f.severity == "warning" else st.info)(f"**{f.code}** — {f.message}")
    for m, a in rec["arts"].items():
        st.download_button(f"Download {m} artifact (JSON)", json.dumps(a, indent=2),
                           file_name=f"calibrator_{m}.json", key=f"dl_{m}")


def main() -> None:
    header()
    chosen = choose_data()
    if chosen is None:
        return
    raw, filename, is_demo = chosen
    base = map_columns(raw, is_demo)
    cfg = assumptions(base)
    key = _key(raw, cfg)
    if st.button("Run analysis", type="primary", key="run"):
        try:
            with st.spinner("Validating and computing…"):
                data = load_predictions(raw, cfg, name=filename)
                res = analyze_data(data, cfg, synthetic=is_demo,
                                   data_note=SYNTHETIC_NOTE if is_demo else None,
                                   title="CalibCheck report" + (" (synthetic demo)"
                                                                if is_demo else ""))
            SS["result"] = {"key": key, "res": res}
        except CalibCheckError as exc:
            SS.pop("result", None)
            st.error(f"Validation failed [{exc.code}]: {str(exc).splitlines()[0]}")
            for f in exc.findings:
                if f.severity == "error":
                    st.write(f"• `{f.code}` {f.message}")
    stored = SS.get("result")
    if stored is None:
        st.info("Configure the analysis above, then press **Run analysis**.")
        return
    if stored["key"] != key:
        st.warning("Data or settings changed since the last run. Results are hidden until you "
                   "run the analysis again.")
        return
    show_results(stored["res"], raw, filename)
    recalibration_section(cfg, raw, is_demo)


main()
