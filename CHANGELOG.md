# Changelog

All notable changes are listed here. Report-schema and artifact-schema changes are called out
explicitly so downstream consumers can detect them.

## Unreleased fixes (pre-0.1.0)

* **Privacy:** public mode now applies the subgroup small-cell policy to time windows: fewer
  than `min_cell` rows, identical outcomes, and complementary suppression. Report schema 1.0
  change: `TimeWindowRow.n`, `n_pos` and `prevalence` are now nullable. The new
  `TimeWindowRow.suppressed`/`suppression_reason` and `TimeWindowResult.suppression_policy`
  fields are added. `tables/time_windows.csv` gains `suppressed` and `suppression_reason`
  columns.
* **Validation:** `fit_calibrator` validates `p`, `y`, `ids`, `entities` and `eps` before any
  type conversion. Labels such as 0.9 or 1.9 and probabilities outside [0, 1] now raise
  `InputValidationError` instead of being cast. `apply_calibrator` uses the same probability
  check (no string, boolean or 2-D coercion), and its invalid-input error is now
  `InputValidationError` rather than `ArtifactError`.

* **Configuration export:** `config.toml` (written with every report and offered by the app)
  now serialises keys per TOML 1.0. Keys that are not bare keys, such as column names with
  dots (`model.v1`), spaces, quotes, backslashes, Unicode or control characters, are written as
  quoted keys, so they no longer reload as nested tables or fail to parse. All string escapes,
  including control characters, follow the spec. Floats round-trip exactly, including `inf`.
  Values TOML cannot hold (lone surrogates, integers outside 64 bits) raise `ValueError`
  instead of being altered. `uncertainty.seed` is now validated as 0 ≤ seed ≤ 2⁶³−1, the
  range both NumPy and TOML accept, so bad seeds fail at configuration time rather than during
  bootstrap or export. No new dependency: the writer is a small in-house serializer, and
  `tomllib` remains the only parser.

## 0.1.0 — unreleased (prepared locally 2026-10-18; no public release)

* First version.
* Report schema **1.0** and calibrator artifact schema **1.0**.
* Binary input. Validation, metrics, binning, Wilson intervals, IID percentile bootstrap,
  paired comparisons, subgroups, time windows, recalibration (identity, logit-affine sigmoid,
  isotonic, temperature), data-role protocol, CLI, Streamlit app, and JSON, CSV and HTML export.
