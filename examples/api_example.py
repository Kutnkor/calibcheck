"""Library API walkthrough (runs offline on synthetic data).

Shows: an explicit positive-label mapping, subgroup diagnostics, and separate
calibration/test data for recalibration. Run:  python examples/api_example.py
"""

from __future__ import annotations

import pandas as pd

from calibcheck import (
    analyze_predictions,
    apply_calibrator,
    evaluate_recalibration,
    fit_calibrator,
    load_predictions,
    resolve_config,
)
from calibcheck.synthetic import generate

# 1. Positive-label mapping: outcomes are "event"/"no_event"; p is P(outcome = "event").
sc = generate("group_cancellation", n=3000, seed=5)
df = sc.frame.drop(columns="latent_p_true")
df["outcome"] = df["y_true"].map({1: "event", 0: "no_event"})
res = analyze_predictions(
    df, columns={"label": "outcome", "probs": ["p_model"], "id": "row_id", "groups": ["region"]},
    positive_label="event", uncertainty={"bootstrap": True, "n_resamples": 500, "seed": 1})
print(res.dataset.probability_meaning, res.dataset.label_mapping)
print(pd.DataFrame(res.metrics_table())[["model", "metric", "value", "ci_low", "ci_high"]].head(6))

# 2. Subgroup diagnostics (descriptive): per-group rows as a table, no browser needed.
for g in res.groups:
    print(pd.DataFrame([{"group": r.name, "n": r.n, "prevalence": r.prevalence,
                         "mean_pred": r.per_model["p_model"]["mean_pred"]} for r in g.rows]))
for f in res.findings:
    print(f.code, "-", f.message)

# 3. Separate calibration / test data.
cfg = resolve_config({"columns": {"label": "y_true", "probs": ["p_model"], "id": "row_id"},
                      "uncertainty": {"bootstrap": True, "n_resamples": 500}})
over = generate("overconfident", n=4000, seed=2).frame.drop(columns="latent_p_true")
cal = load_predictions(over[over.split == "calibration"], cfg, role="calibration")
test = load_predictions(over[over.split == "test"], cfg, role="test")
art = fit_calibrator(cal.probs["p_model"], cal.y, "sigmoid", model_association="p_model",
                     ids=cal.ids)
q, _ = apply_calibrator(art, test.probs["p_model"])  # labels are not needed to apply
print("sigmoid parameters fitted on calibration rows:", art.params)
recal, _ = evaluate_recalibration(cal, test, cfg, model="p_model", methods=["sigmoid"])
for c in recal.comparisons:
    if c.metric == "brier":
        print(f"{c.candidate} vs {c.reference}: delta Brier {c.delta:+.4f} "
              f"[{c.interval.low:+.4f}, {c.interval.high:+.4f}] -> {c.favours}")
