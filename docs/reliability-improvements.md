# Reliability improvements — 2026-10-18

Twenty-five independent changes address reproducible failures or unsafe edge cases.
Each code fix has regression coverage; the NumPy typing fix was identified from the initial GitHub CI logs.

| No. | Improvement |
|---|---|
| 1 | support NumPy reduction typing across CI Python versions |
| 2 | reject clipping epsilons that produce infinite endpoint losses |
| 3 | reject ambiguous duplicate isotonic interpolation knots |
| 4 | refuse calibrators from optimizers that did not converge |
| 5 | validate calibrator provenance counts and probability ranges |
| 6 | validate calibrator artifacts before saving them |
| 7 | publish calibrator files atomically without clobber races |
| 8 | report unreadable artifact paths as domain errors |
| 9 | enforce artifact size limits for in-memory uploads |
| 10 | reject duplicate keys and nonstandard constants in artifact JSON |
| 11 | detect entity leakage when evaluating saved calibrators |
| 12 | require complete timestamps for temporal evaluation protocols |
| 13 | validate Python evaluation and selection options explicitly |
| 14 | explain insufficient rows and entities during method selection |
| 15 | reject conflicting metadata when pairing prediction files |
| 16 | require two models when comparing prevalidated prediction data |
| 17 | reject malformed CSV row widths before pandas inference |
| 18 | reject unnamed CSV columns instead of inventing headers |
| 19 | handle unreadable CSV paths without raw filesystem tracebacks |
| 20 | wrap configuration filesystem failures in ConfigError |
| 21 | reject transformations that would otherwise be silently ignored |
| 22 | preflight owned export manifests before overwriting report files |
| 23 | neutralize whitespace-prefixed spreadsheet formulas in CSV exports |
| 24 | preserve full float64 precision when exporting recalibrated probabilities |
| 25 | preserve correctly rounded CSV probability values |

## Verification

* Full source suite: **307 passed** on Python 3.11/macOS, including 62 additional regression cases.
* Ruff: `ruff check src tests examples experiments benchmarks` passed.
* Types: `mypy` passed for all 24 source modules.
* Wheel and source distribution build successfully; platform installation checks run in GitHub CI.
* Updated JSON Schemas are generated from the runtime models.

Regression coverage is in `tests/numerical/test_artifact_integrity.py`,
`tests/protocol/test_protocol_integrity.py`, `test_input_integrity.py`, and
`tests/integration/test_export_integrity.py`, alongside the existing numerical and workflow tests.

## User-visible behavior

CSV decimals close to zero or one are preserved as correctly rounded float64 values.
Applied probabilities use 17 significant digits so reading the CSV does not lose precision.
Invalid configuration, metadata conflicts and malformed files fail with actionable domain errors.
Non-converged fits produce no reusable calibrator. Temporal protocols need complete timestamps.
Saved calibrators are validated and written atomically, and report overwrite validates every
manifest path before deleting any previous output.
