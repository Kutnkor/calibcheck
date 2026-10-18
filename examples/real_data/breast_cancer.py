"""Real-data example: Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn.

Source: UCI Machine Learning Repository, DOI 10.24432/C5DW2B, licence CC BY 4.0
(https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The copy used
here ships with scikit-learn (no download). Positive event: *malignant* (scikit-learn target 0),
declared explicitly; probabilities are P(diagnosis = malignant).

Roles: stratified 50% train / 25% calibration / 25% test (seed below). Preprocessing
(StandardScaler) is fitted on training rows only, inside the pipeline. The script writes
calibration.csv and test.csv; evaluate them with the CalibCheck CLI (see README.md).

Usage:  python examples/real_data/breast_cancer.py [OUT_DIR] [--seed 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="examples/real_data/output")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    bc = load_breast_cancer()
    diagnosis = np.where(bc.target == 0, "malignant", "benign")
    y = (diagnosis == "malignant").astype(int)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=0.5, stratify=y, random_state=a.seed)
    cal, te = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=a.seed)
    models = {
        "p_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "p_forest": RandomForestClassifier(n_estimators=300, random_state=a.seed),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = {"calibration": pd.DataFrame({"row_id": [f"bc{i:04d}" for i in cal],
                                           "diagnosis": diagnosis[cal]}),
              "test": pd.DataFrame({"row_id": [f"bc{i:04d}" for i in te],
                                    "diagnosis": diagnosis[te]})}
    for name, model in models.items():
        model.fit(bc.data[tr], y[tr])
        pos = list(model.classes_).index(1)
        frames["calibration"][name] = model.predict_proba(bc.data[cal])[:, pos].round(6)
        frames["test"][name] = model.predict_proba(bc.data[te])[:, pos].round(6)
    for role, df in frames.items():
        df.to_csv(out / f"{role}.csv", index=False)
        print(f"wrote {out / f'{role}.csv'} ({len(df)} rows, {int((df.diagnosis == 'malignant').sum())} malignant)")
    print(f"training rows: {len(tr)} (not written; the base models were fitted on them only)")


if __name__ == "__main__":
    main()
