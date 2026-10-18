# Examples

All examples run offline unless stated otherwise. Everything here is synthetic except
`real_data/`.

| Path | What it shows |
|---|---|
| `example_predictions.csv` | Small synthetic file (1,000 rows; two models; `channel` subgroup). The same file is bundled in the package (`calibcheck example`). |
| `calibcheck.toml` | A configuration file for that example. |
| `demo_output/` | Output of `calibcheck demo --out examples/demo_output` (seed 7): data, artifacts, and two complete reports. **Representative report:** `demo_output/reports/recalibration/report.html`. |
| `api_example.py` | Python API: positive-label mapping, subgroup tables, separate calibration and test data. |
| `consume_report.py` | Reading `report.json` safely: schema check, `null` handling. |
| `real_data/breast_cancer.py` | Breast Cancer Wisconsin (Diagnostic, CC BY 4.0, bundled with scikit-learn). Train / calibration / test roles. The output is written to `real_data/output/` and is not committed. |
| `counterexamples/` | Invalid or limited inputs with expected behaviour (`NOTES.md`). |

Real-data walkthrough (executed 2026-10-18; results vary with `--seed`):

```bash
python examples/real_data/breast_cancer.py                 # writes calibration.csv / test.csv
calibcheck recalibrate --calibration examples/real_data/output/calibration.csv \
  --test examples/real_data/output/test.csv --label diagnosis --positive-label malignant \
  --prob p_forest --id row_id --method sigmoid --method isotonic --bootstrap --seed 1 \
  --out examples/real_data/output/recalibration_report
```

With seed 1 there are 143 test rows (53 malignant). Every paired Brier and log-loss
difference interval for sigmoid and isotonic included zero, so the report says "insufficient
evidence to distinguish". This negative result is the expected outcome at this sample size.
