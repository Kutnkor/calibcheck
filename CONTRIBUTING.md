# Contributing to CalibCheck

Thank you for helping. CalibCheck is a small, individually maintained project. Responses may
take a while, and there is no guaranteed support schedule.

## Before you start

* For anything larger than a fix, open an issue first so we can agree on the estimand, the
  assumptions and the tests.
* **Never attach private data.** Bug reports need a *sanitised, minimal* reproduction: a few
  synthetic rows (see `calibcheck.synthetic`) or a hand-written CSV that shows the problem.

## Workflow

1. `pip install -e '.[dev,app]'`
2. Add tests next to similar ones (see [docs/development.md](docs/development.md)). Numerical
   changes need a hand-checkable expected value and, where possible, a reference check against
   scikit-learn or SciPy under a stated convention.
3. `ruff check src tests examples experiments benchmarks && mypy && pytest`
4. Update the docs. Any change to the report or artifact schema needs a version bump and a
   `CHANGELOG.md` entry.

## Where code belongs

See the table in [docs/extending.md](docs/extending.md). The CLI (`cli.py`) and the app
(`app/streamlit_app.py`) must not reimplement metrics, filtering or binning. They call the
library.

## Scientific ground rules

* Do not report a number where a value is undefined. Use `status` plus a reason code.
* Do not add verdict language ("trustworthy", "safe", "unbiased") to findings.
* Do not fit or select anything on evaluation outcomes.
* Mark proposals as unimplemented until they ship with tests.

## Good first tasks (with acceptance criteria)

* **Parquet input**: identical `report.json` results (excluding manifest) to the CSV path on
  `examples/example_predictions.csv`, and leading-zero IDs preserved.
* **AUC difference interval**: a paired bootstrap of ΔAUC with degenerate (single-class)
  replicates counted. The test must check the interval contains the point estimate.
* **Report table of contents**: an in-page navigation list in the HTML template. Still offline,
  escaped and with no scripts.

By contributing you agree that your contribution is licensed under the MIT License and that you
will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
