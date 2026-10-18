# Python API

All public names are importable from `calibcheck`. The functions return pydantic models that
serialise with `.model_dump(mode="json")`. Plotting and report writing are separate and
optional.

```python
from calibcheck import (
    analyze_predictions,   # validate + analyse one or more probability columns on the same rows
    compare_predictions,   # same, but requires >= 2 columns and labels the result "comparison"
    inspect_predictions,   # validation-only overview (no fitting, no resampling)
    load_predictions,      # CSV path | bytes | DataFrame -> PredictionData (validated arrays)
    analyze_data,          # analyse an already validated PredictionData
    fit_calibrator, apply_calibrator, save_artifact, load_artifact,
    evaluate_recalibration,  # fit on calibration rows, compare on test rows (paired)
    render_report,         # write JSON/CSV/PNG/HTML to a directory
    resolve_config, AnalysisConfig,
)
```

## Configuration

Every analysis function accepts `config=` (an `AnalysisConfig` or a nested dict) plus keyword
overrides that use the same nested shape:

```python
res = analyze_predictions(
    "preds.csv",
    columns={"label": "outcome", "probs": ["p_a"], "id": "row_id", "groups": ["region"]},
    positive_label="event",
    binning={"strategy": "quantile", "n_bins": 10},
    uncertainty={"bootstrap": True, "n_resamples": 2000, "seed": 1},
)
```

## Results

```python
m = res.model("p_a")
m.metrics["brier"].value, m.metrics["brier"].interval     # value + Interval or None
m.metrics["roc_auc"].status, m.metrics["roc_auc"].reason  # e.g. "unavailable", "UNAVAILABLE_SINGLE_CLASS"
m.binning.edges, m.binning.bins[9].observed_rate, m.binning.bins[9].ci_low
res.comparisons            # paired differences, candidate - reference
res.groups, res.findings   # descriptive subgroup tables; rule-based findings
import pandas as pd
pd.DataFrame(res.metrics_table())
```

## Exceptions

`CalibCheckError` is the base class. Subclasses: `InputValidationError` (with `.findings`),
`ConfigError`, `ProtocolError`, `ArtifactError`, `FittingError` and `OutputError`. Each has a
stable `.code`.

## Separate calibration and test data

```python
cfg = resolve_config({"columns": {"label": "y_true", "probs": ["p"], "id": "row_id"}})
cal = load_predictions("calibration.csv", cfg, role="calibration")
test = load_predictions("test.csv", cfg, role="test")
art = fit_calibrator(cal.probs["p"], cal.y, "isotonic", model_association="p", ids=cal.ids)
q, notes = apply_calibrator(art, test.probs["p"])         # no labels needed
result, artifacts = evaluate_recalibration(cal, test, cfg, model="p",
                                           methods=["sigmoid", "isotonic"], mode="exploratory")
```

A complete, runnable version is in [`examples/api_example.py`](../examples/api_example.py).
