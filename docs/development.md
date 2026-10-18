# Development guide

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,app]'
pytest                       # all tests (~10 s)
pytest -m "not slow"         # skip bootstrap cross-check and app tests
ruff check src tests examples experiments benchmarks
mypy                         # strict-ish type checks on src/calibcheck
python -m build              # sdist + wheel into dist/
python experiments/run_experiments.py [--with-adult]   # ~40 s; regenerates experiments/results
python benchmarks/bench.py                             # ~1-2 min; regenerates benchmarks/results.md
```

## Test layout

* `tests/numerical`: hand-checkable values, scikit-learn and SciPy reference checks, bootstrap
  behaviour, calibrators, and Hypothesis property tests.
* `tests/protocol`: the input contract and the data-role protocol (overlap, label isolation,
  groups, time).
* `tests/integration`: CLI workflows, API and CLI agreement, report integrity, synthetic
  scenarios and findings vocabulary.
* `tests/interface`: the Streamlit app via `streamlit.testing`.

Randomised tests use fixed seeds. Floating-point comparisons use explicit tolerances, and
tests never require bitwise equality across different reduction orders.

## Environment used for v0.1

macOS arm64 (Apple M4), Python 3.11.1, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6,
scikit-learn 1.9.1, matplotlib 3.11.2, pydantic 2.14.1, typer 0.27.3, streamlit 1.66.0.
Compatibility decisions:

* scikit-learn 1.9 `log_loss` has no `eps` argument and clips at machine epsilon; CalibCheck
  keeps its own documented ε.
* `brier_score_loss` gained `scale_by_half` in 1.7; the tests pass it explicitly.
* pandas 3 returns read-only arrays from `to_numpy()`, so CalibCheck copies before mutating.
