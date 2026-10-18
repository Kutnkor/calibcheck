# Extension boundaries and roadmap

Nothing on this page is implemented in v0.1 unless it says so.

## Where things belong

| Change | Module | Contract |
|---|---|---|
| New input format (e.g. Parquet) | `data.py` | Produce the same `PredictionData`. Reuse `validate_frame`, so every rule applies unchanged. |
| New aggregate metric | `metrics.py` (`REGISTRY` + `core_metrics`) | Return a `MetricValue` with status and reason. Add hand-checked and reference tests. |
| New bin summary | `binning.py` (`bin_summaries`) | Must name its partition and the bins it used. |
| New interval method | `uncertainty.py` | Declare the estimand and the dependence it supports. Return `Interval` with `status`/`reason`. Respect `dependence_block`. |
| New calibrator | `calibration.py` | Add to `METHODS`, `TRANSFORMS` and `validate_artifact`. JSON-only parameters. Test for non-decreasing output in [0, 1] and persistence equality. |
| New finding | `findings.py` + a code in `codes.py` | Deterministic. Include evidence, population and caveat. Avoid the words in `BANNED_WORDS`. |
| Report section | `templates/report.html.j2` | Autoescaped, no external resources. |

## Roadmap (not implemented)

1. **Beta calibration** (Kull et al., 2017). This is a three-parameter map on ln p and −ln(1 − p)
   with a, b ≥ 0. Acceptance: matches a reference implementation on fixtures, artifact
   round-trip, monotonicity property test.
2. **Dependence-aware intervals.** Cluster bootstrap that resamples entities, and moving-block
   bootstrap for time order. Acceptance: a written estimand, a coverage simulation on the
   `clustered` and `temporal` scenarios, and IID refusal logic kept for other declarations.
3. **Parquet input** behind an optional `pyarrow` extra. Acceptance: identical results to CSV
   on the bundled example, and leading-zero IDs preserved.
4. **Observation weights.** Requires a weighted estimand for every metric, weighted Wilson or
   bootstrap alternatives, and weighted bins. This is not a partial feature.
5. **Multiclass calibration.** Classwise and top-label diagnostics are *different estimands*.
   They must not be produced by renaming binary outputs.
6. **Confirmatory subgroup workflow.** Predeclared groups, evaluation on separate data, and a
   defined testing family before any multiplicity adjustment.
7. **Monitoring.** Scheduled re-evaluation on new windows. Out of scope for a local analysis
   tool unless it remains file-based.
