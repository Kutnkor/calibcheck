"""Executes experiments/PROTOCOL.md and writes experiments/results/ (generated only).

Usage:
    python experiments/run_experiments.py            # studies A and B (offline)
    python experiments/run_experiments.py --with-adult   # also downloads UCI Adult from OpenML
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from calibcheck import synthetic
from calibcheck.binning import bin_summaries, bin_table, uniform_edges
from calibcheck.calibration import apply_calibrator, fit_calibrator
from calibcheck.metrics import core_metrics
from calibcheck.provenance import dependency_versions

OUT = Path(__file__).parent / "results"
METHODS = ["sigmoid", "isotonic", "temperature"]
EPS = 1e-15
TOL = 1e-12  # differences below this are floating-point noise, not "worse"


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    m = core_metrics(y, p, eps=EPS, threshold=0.5, population="test")
    bt = bin_table(y, p, uniform_edges(10), strategy="uniform", n_bins_requested=10,
                   edge_source="uniform", min_support=30, wilson_level=None)
    s = bin_summaries(bt, int(y.size), "test")
    return {"brier": m["brier"].value, "log_loss": m["log_loss"].value, "ece": s["ece"].value,
            "roc_auc": m["roc_auc"].value}


def evaluate(study: str, dataset: str, base: str, seed: int, p_cal, y_cal, p_test, y_test) -> list[dict]:
    rows = []
    raw = score(y_test, p_test)
    common = {"study": study, "dataset": dataset, "base": base, "seed": seed,
              "n_cal": int(y_cal.size), "n_test": int(y_test.size),
              "test_prevalence": float(y_test.mean())}
    rows.append({**common, "method": "raw", "status": "ok", **raw,
                 "d_brier": 0.0, "d_log_loss": 0.0, "d_ece": 0.0, "error": ""})
    for method in METHODS:
        try:
            art = fit_calibrator(p_cal, y_cal, method, eps=EPS, model_association=base,
                                 store_id_digests=False)
            q, _ = apply_calibrator(art, p_test)
            s = score(y_test, q)
            rows.append({**common, "method": method, "status": "ok", **s,
                         "d_brier": s["brier"] - raw["brier"],
                         "d_log_loss": s["log_loss"] - raw["log_loss"],
                         "d_ece": s["ece"] - raw["ece"],
                         "warnings": ";".join(w.code for w in art.diagnostics.warnings),
                         "error": ""})
        except Exception as exc:  # recorded, never hidden
            rows.append({**common, "method": method, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows


def study_a(seeds: range) -> list[dict]:
    rows = []
    for name in ["calibrated", "near_calibrated", "overconfident", "underconfident",
                 "intercept_bias", "imbalanced", "headline"]:
        for seed in seeds:
            n = 6000 if name == "headline" else 4000
            sc = synthetic.generate(name, n=n, seed=seed)
            df = sc.frame
            cal, test = df[df.split == "calibration"], df[df.split == "test"]
            rows += evaluate("A_synthetic_recalibration", name, "p_model", seed,
                             cal.p_model.to_numpy(float), cal.y_true.to_numpy(np.int8),
                             test.p_model.to_numpy(float), test.y_true.to_numpy(np.int8))
    return rows


def base_models(numeric: list[str] | None = None, categorical: list[str] | None = None):
    if categorical:
        pre = ColumnTransformer([("num", StandardScaler(), numeric),
                                 ("cat", OneHotEncoder(handle_unknown="ignore",
                                                       min_frequency=20), categorical)])
        lr = make_pipeline(pre, LogisticRegression(max_iter=2000))
        pre2 = ColumnTransformer([("num", "passthrough", numeric),
                                  ("cat", OneHotEncoder(handle_unknown="ignore",
                                                        min_frequency=20), categorical)])
        rf = make_pipeline(pre2, RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                        random_state=0))
    else:
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    return {"logistic_regression": lr, "random_forest": rf}


def study_b_dataset(name: str, X, y: np.ndarray, seeds: range, numeric=None, categorical=None) -> list[dict]:
    rows = []
    for seed in seeds:
        idx = np.arange(len(y))
        tr, rest = train_test_split(idx, train_size=0.5, stratify=y, random_state=seed)
        cal, te = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=seed)
        Xs = (lambda ix: X.iloc[ix]) if isinstance(X, pd.DataFrame) else (lambda ix: X[ix])
        prev = float(y[tr].mean())
        rows += evaluate("B_base_predictors", name, "constant_train_prevalence", seed,
                         np.full(len(cal), prev), y[cal], np.full(len(te), prev), y[te])
        for bname, model in base_models(numeric, categorical).items():
            try:
                model.set_params(**({"randomforestclassifier__random_state": seed}
                                    if "randomforestclassifier" in getattr(model, "named_steps", {})
                                    else {"random_state": seed} if bname == "random_forest" else {}))
                model.fit(Xs(tr), y[tr])
                pos_col = list(model.classes_).index(1)
                p_cal = model.predict_proba(Xs(cal))[:, pos_col]
                p_te = model.predict_proba(Xs(te))[:, pos_col]
                rows += evaluate("B_base_predictors", name, bname, seed, p_cal, y[cal], p_te, y[te])
            except Exception as exc:
                rows.append({"study": "B_base_predictors", "dataset": name, "base": bname,
                             "seed": seed, "method": "all", "status": "failed",
                             "error": f"{type(exc).__name__}: {exc}"})
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[(df.status == "ok") & (df.method != "raw")]
    g = ok.groupby(["study", "dataset", "base", "method"])
    out = g.agg(runs=("seed", "count"), n_test=("n_test", "median"),
                mean_d_brier=("d_brier", "mean"), sd_d_brier=("d_brier", "std"),
                min_d_brier=("d_brier", "min"), max_d_brier=("d_brier", "max"),
                worse_brier=("d_brier", lambda s: int((s > TOL).sum())),
                mean_d_log_loss=("d_log_loss", "mean"),
                worse_log_loss=("d_log_loss", lambda s: int((s > TOL).sum())),
                mean_d_ece=("d_ece", "mean")).reset_index()
    raw = df[(df.status == "ok") & (df.method == "raw")].groupby(
        ["study", "dataset", "base"]).agg(raw_brier=("brier", "mean"), raw_ece=("ece", "mean"),
                                          raw_auc=("roc_auc", "mean")).reset_index()
    return out.merge(raw, on=["study", "dataset", "base"], how="left")


def to_markdown(df: pd.DataFrame, cols: list[str]) -> str:
    def fmt(c, v):
        if isinstance(v, float | np.floating):
            if np.isnan(v):
                return "—"
            if c in ("runs", "n_test") or c.startswith("worse"):
                return f"{int(v)}"
            return f"{v:+.4f}" if c.startswith(("mean_d", "sd_d", "min_d", "max_d")) else f"{v:.4f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-adult", action="store_true")
    ap.add_argument("--quick", action="store_true", help="3 seeds (smoke run; not the study)")
    args = ap.parse_args()
    seeds = range(1, 4) if args.quick else range(1, 21)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = study_a(seeds)
    ta = time.perf_counter() - t0
    Xs, ys = make_classification(n_samples=6000, n_features=10, n_informative=5, flip_y=0.1,
                                 class_sep=0.8, random_state=2026)
    rows += study_b_dataset("synthetic_tabular", Xs, ys.astype(np.int8), seeds)
    bc = load_breast_cancer()
    y_bc = (bc.target == 0).astype(np.int8)  # positive event: malignant (sklearn target 0)
    rows += study_b_dataset("breast_cancer", bc.data, y_bc, seeds)
    adult_note = "not run (opt-in --with-adult)"
    if args.with_adult:
        try:
            from sklearn.datasets import fetch_openml
            ad = fetch_openml(data_id=1590, as_frame=True)
            X = ad.data
            y = (ad.target.astype(str).str.strip().str.rstrip(".") == ">50K").astype(np.int8).to_numpy()
            cat = [c for c in X.columns if str(X[c].dtype) == "category"]
            num = [c for c in X.columns if c not in cat]
            X = X.copy()
            for c in cat:
                X[c] = X[c].astype(str)
            rows += study_b_dataset("adult", X, y, range(1, 11) if not args.quick else seeds,
                                    numeric=num, categorical=cat)
            adult_note = f"run: OpenML data_id 1590, {len(y)} rows, prevalence {y.mean():.4f}"
        except Exception as exc:
            adult_note = f"FAILED: {type(exc).__name__}: {exc}"
            traceback.print_exc()
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runs.csv", index=False)
    summ = summarise(df)
    summ.to_csv(OUT / "summary.csv", index=False)
    failures = df[df.status == "failed"]
    cols = ["dataset", "base", "method", "runs", "n_test", "raw_brier", "mean_d_brier",
            "sd_d_brier", "min_d_brier", "max_d_brier", "worse_brier", "mean_d_log_loss",
            "worse_log_loss", "mean_d_ece"]
    env = dependency_versions()
    md = [
        "# Experiment results (generated by experiments/run_experiments.py)",
        "",
        "Protocol: [PROTOCOL.md](../PROTOCOL.md). Every row is generated from `runs.csv`. "
        "d_* = method − raw on identical test rows; negative favours the method. `worse_*` "
        "counts runs where the method was worse than raw by more than 1e-12. These describe the listed scenarios "
        "only.",
        "",
        f"- Seeds: {seeds.start}–{seeds.stop - 1}{' (QUICK SMOKE RUN, not the declared study)' if args.quick else ''}",
        f"- Scheduled runs (rows incl. raw): {len(df)}; failed rows: {len(failures)}",
        f"- Adult: {adult_note}",
        f"- Wall-clock: {elapsed:.1f} s total (study A {ta:.1f} s)",
        f"- Hardware: {platform.machine()} {platform.processor() or ''} / {platform.system()} "
        f"{platform.release()}; Python {sys.version.split()[0]}",
        f"- Libraries: {', '.join(f'{k} {v}' for k, v in env.items())}",
        "",
        "## Study A — synthetic recalibration (base model outputs are known distortions)",
        "",
        to_markdown(summ[summ.study == "A_synthetic_recalibration"], cols),
        "",
        "## Study B — base predictors, train / calibration / test roles",
        "",
        to_markdown(summ[summ.study == "B_base_predictors"], cols),
        "",
        "## Failures",
        "",
        "None." if failures.empty else to_markdown(failures, ["study", "dataset", "base",
                                                             "seed", "method", "error"]),
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
