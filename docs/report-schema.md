# Machine-readable outputs

## report.json (schema 1.0)

The JSON Schema is in [`docs/schema/report.schema.json`](schema/report.schema.json); regenerate
it with `calibcheck schema --out docs/schema`. Top-level fields:

| Field | Content |
|---|---|
| `schema_version` | `"1.0"`. Consumers should check the major version. |
| `kind` | `single`, `comparison` or `recalibration` |
| `synthetic`, `data_note` | Whether the data are synthetic, plus a note |
| `config` | The fully resolved `AnalysisConfig` |
| `dataset` | Rows in, retained and excluded (with reasons), label mapping, probability meaning, ranges |
| `validation` | Structured findings: `code`, `severity`, `message`, `field`, `count` |
| `models[]` | `metrics{name → MetricValue}`, `binning` (edges, rule, source, bins), `sensitivity`, `clipping`, `summary` |
| `comparisons[]` | `reference`, `candidate`, `metric`, `definition`, `delta`, `interval`, `favours`, `n`, `population` |
| `groups[]`, `time_windows` | Descriptive subgroup and time views. In public mode, suppressed rows have `suppressed: true`, a `suppression_reason`, and `null` for counts, rates and per-model statistics. |
| `findings[]` | Rule-based findings with `evidence` and `caveat` |
| `protocol` | Data roles, mode, fit status and overlap checks (recalibration) |
| `manifest` | Versions, seed, inputs (paths and hashes omitted in public mode), filtering, `results_digest` and `volatile.created_utc` |

`MetricValue` has `value` (`null` when undefined), `status` (`ok`, `unavailable`, `failed` or
`skipped`), `reason` (a stable code), `scale`, `direction`, `n`, `population`, an optional
`interval` and `details`. The JSON never contains `NaN` or `Infinity`.

`results_digest` is a SHA-256 over everything except `manifest`. Two runs with the same data,
configuration and seed have the same digest, while timestamps live only in `manifest.volatile`.

Codes and their explanations are listed in `calibcheck/codes.py`. Codes may be added in minor
versions, but an existing code keeps its meaning.

## CSV tables

`tables/metrics.csv` has the columns model, metric, label, category, value, status, reason, n,
population, ci_low, ci_high and ci_method. `tables/bins.csv` has one row per (model, bin),
including empty bins. Text cells that begin with `= + - @` or a tab or carriage return are
prefixed with `'` so spreadsheets treat them as text; numeric columns are unchanged.

## Calibrator artifact (schema 1.0)

See [`docs/schema/calibrator.schema.json`](schema/calibrator.schema.json). An artifact holds
the method, parameters (`a`/`b`, `T`, or isotonic `x`/`y` breakpoints), `input_transform`,
`output_rule`, `clip_eps`, the label mapping, `model_association`, fit diagnostics and
provenance (counts, salted ID digests). Artifacts with a different major schema version are
rejected.

## Example consumer

[`examples/consume_report.py`](../examples/consume_report.py) checks the schema version and
treats `null` as unavailable.
