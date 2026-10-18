# v0.1 acceptance record (2026-10-18)

Each requirement maps to evidence that was **executed or inspected** for this release
candidate. The environment was macOS arm64 (Apple M4) with Python 3.11.1 (development venv) and
Python 3.13.7 (clean wheel install). Anything that was not run is listed under "Not verified".

## Product checklist

| Requirement | Evidence |
|---|---|
| Accepts a documented binary CSV | `calibcheck report examples/example_predictions.csv --config examples/calibcheck.toml` produced a complete report. Format: README and [cli.md](cli.md). |
| Rejects invalid probabilities, labels, IDs and headers | `tests/protocol/test_validation.py` (30 tests). `examples/counterexamples/*` gave exit code 1 for missing outcomes, invalid or percent probabilities and duplicate IDs. |
| Preserves label semantics | `test_label_rules`, `test_label_encoding_symmetry`. The real-data example declares `--positive-label malignant`. |
| Computes the core metrics | `tests/numerical/test_metrics.py`: hand-checked values plus scikit-learn 1.9.1 cross-checks. |
| Inspectable bins and reliability plots | `tests/numerical/test_binning.py`, `test_properties.py`. Figures were inspected visually in `docs/images/`. |
| Supported uncertainty | Wilson matches SciPy to 1e-12. Bootstrap agrees with `scipy.stats.bootstrap` within Monte Carlo error. Dependence refusal is tested (`test_uncertainty.py`). |
| Compares aligned models | `test_comparison_direction_and_paired_interval`, `test_alignment_by_id_never_by_position`, CLI `compare` (same file and `--other`). |
| Fits and applies recalibrators with separate roles | `tests/numerical/test_calibration.py`, `tests/protocol/test_roles.py`, and CLI `fit`/`apply`/`recalibrate` in `tests/integration/test_cli.py`. |
| JSON and offline HTML | `test_full_export_is_complete_and_offline` (no external URLs or scripts). A clean wheel install ran `report` and `demo` under `sandbox-exec` with **network denied**, writing to a directory name with spaces. |
| CLI and optional local app | 11 CLI subcommands are tested. `tests/interface/test_app.py` (3 tests via `streamlit.testing`). The app was launched with `calibcheck app` and exercised in a browser: Run analysis, then the Reliability tab. |
| Installs from a built wheel | `python -m build` produced sdist and wheel. A clean Python 3.13 venv installed the wheel, and the full suite (245 tests) passed against it. |
| Deterministic demo | `calibcheck demo --out examples/demo_output` (seed 7) gave identical numbers on repeated runs (`launch/demo_transcript.txt`). |
| Tests, method docs, contribution materials | `tests/` (245), [methodology.md](methodology.md), CONTRIBUTING, CODE_OF_CONDUCT, `.github/` templates |

## Scientific checklist

| Requirement | Where enforced or tested |
|---|---|
| Brier is not labelled as pure calibration | `metrics.REGISTRY` descriptions, the HTML "Key scores" note and the methodology section on terminology |
| ECE names its partition | `details.partition` on every ECE value. The CLI and HTML show the strategy and actual bins. |
| Bars are not individual-probability guarantees | Figure subtitle, HTML notes, finding caveats |
| Single-class data produces no ranking metrics | `test_single_class_keeps_losses_marks_ranking_unavailable`, `test_json_has_no_nan_tokens_and_null_reasons` |
| Same eligible rows and pairing | The drop policy applies one common population (`test_drop_policy_records_exclusions_and_common_population`). The paired bootstrap uses the same indices (`test_pairing_identical_models_zero_difference`). |
| Fitting and selection never use test outcomes | `test_changing_test_labels_cannot_change_fit_or_selection`, overlap refusal tests |
| Synthetic vs observed labelled | `synthetic` flag and `data_note` in reports. `SYNTHETIC_NOTE` in demo output. `PROTOCOL.md` and `SUMMARY.md` name each dataset. |
| Reported results come from runs | `experiments/results/*` and `benchmarks/results.md` are script-generated. README and post numbers are copied from `launch/demo_transcript.txt` and `SUMMARY.md`. |
| No verdict language | `test_findings_vocabulary_and_determinism` over all 14 scenarios |
| Unsupported uncertainty is unavailable | `test_dependence_declared_refuses_iid_intervals`, ROC-AUC delta interval `UNAVAILABLE_NOT_SUPPORTED` |

## Post-review fixes (2026-10-18)

| Fix | Evidence |
|---|---|
| Public time-window suppression | `tests/protocol/test_time_window_privacy.py`: policy unit test, nulls (not zeros), complementary suppression, private mode unchanged, and JSON/CSV/HTML/plot consistency |
| TOML config round trip | `tests/protocol/test_config_toml.py`: 16 awkward keys, defaults, nested/list/bool/float/inf values, optional-field check, a 200-example property test, loud failure for unrepresentable values. Integration: `test_exported_config_reproduces_report_for_dotted_column` reproduces identical results from the exported `config.toml`. Before the fix, 18 of these tests failed. |
| `fit_calibrator` input validation | `tests/numerical/test_calibration.py`: 60 parametrised rejections across 4 methods, lossless encodings accepted, `ids`/`eps` checks, `apply_calibrator` without coercion |

## Lint and type checks

`ruff check src tests examples experiments benchmarks` and `mypy` both pass.

## Not verified

* **GitHub Actions CI** (`.github/workflows/ci.yml`) has never run on GitHub, so Linux and
  Windows have not been exercised.
* **Python 3.12** was not tested locally. 3.11 and 3.13 were.
* **Cancelling a long bootstrap in the app** relies on Streamlit's own stop and rerun behaviour.
  There is no dedicated progress bar or cancel button, and this was not tested.
* **Video recording** of the demo was not produced (`launch/record_demo.sh` and the storyboard
  are provided instead).
* **Public release**: nothing was pushed or published. The PyPI name `calibcheck` was
  unregistered when checked on 2026-10-18, which does not reserve it.
