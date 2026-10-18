# CalibCheck methodology (v0.1)

This document states exactly what CalibCheck computes, under which assumptions, and what it
cannot establish. It describes the behaviour shipped in v0.1. Roadmap items are marked as
**not implemented**.

## 1. Target and scope

For binary outcomes, let *Y* ∈ {0, 1} indicate the declared **positive label** and *P* be the
predicted probability of the positive class. A forecaster is calibrated when

  E[Y | P] = P.

CalibCheck estimates aspects of this relationship from finite data. It cannot observe any
individual's true event probability. Every interval it reports describes an *aggregate*
statistic (a bin rate, a mean loss, a paired difference) and never one observation's
probability.

The input probability always means P(Y = positive_label), including values below 0.5. It is never
reinterpreted as "confidence in the predicted class" (top-label confidence).

In scope for v0.1: binary outcomes, probability-only input, CSV files and data frames.
**Not implemented**: multiclass or top-label calibration, regression uncertainty, observation
weights, and dependence-aware intervals (cluster or block bootstrap).

## 2. Terminology contract

| Term | Meaning in CalibCheck | What it does *not* mean |
|---|---|---|
| Calibration | Agreement between predicted probabilities and observed frequencies | Good ranking |
| Discrimination | Ranking ability (ROC-AUC, average precision) | Probability accuracy |
| Overall probabilistic score | Brier loss, log loss: these mix calibration and discrimination | A pure calibration measure |
| Threshold accuracy | Fraction correct at an explicit threshold (default 0.5) | The central calibration result |
| Uncertainty | Sampling variability of an aggregate statistic under stated assumptions | Uncertainty about an individual |
| Drift | Changes in prevalence, prediction distribution or calibration over time windows | A causal explanation |

A model can rank well and still assign poor probabilities. Section 11 includes a synthetic
example with ROC-AUC above 0.85 and large calibration gaps. A lower Brier or log loss does not by itself
show better calibration.

## 3. Input contract and validation

* The wide CSV has one outcome column, one or more **explicitly named** probability columns,
  and optional ID, entity, timestamp, split and subgroup columns. Probability columns are never
  guessed from numeric columns.
* The positive label must be declared unless the labels are exactly {0, 1}. Integral numeric
  labels are canonicalised, so `1.0` becomes `1`. More than two distinct labels are rejected (multiclass is out of
  scope).
* Probabilities must be finite and in [0, 1]. Exact 0 and 1 are valid. Percentages and logits
  are **never** converted implicitly; the explicit `--transform col=percent|logit` is recorded.
* Row IDs are read as text, so leading zeros survive, and they must be unique and non-missing.
  Without an ID column, deterministic positional IDs are created and flagged. They cannot be
  used to align separate files.
* Missing data policy: `fail` (default) stops on any missing or invalid outcome or probability.
  `drop` removes whole rows, records every reason and count, and evaluates all models on the
  **same retained rows**.
* Subgroup values that are missing appear as a visible `(missing)` category. Columns with more
  distinct values than `groups.max_cardinality` (default 50) are rejected unless the limit is
  raised deliberately.
* Validation messages give fields and counts and never print raw rows.

## 4. Metric definitions

All metrics are computed in float64 on the declared population (`n` is reported with each).

| Metric | Definition | Notes |
|---|---|---|
| Brier loss | BS = (1/n) Σ (pᵢ − yᵢ)² | Binary [0, 1] scale. This equals scikit-learn `brier_score_loss(..., scale_by_half=True)` for binary input (the `scale_by_half` option exists from scikit-learn 1.7). It uses the **original** probabilities. |
| Log loss | −(1/n) Σ [yᵢ ln qᵢ + (1 − yᵢ) ln(1 − qᵢ)], qᵢ = clip(pᵢ, ε, 1 − ε) | Natural log. ε defaults to 1e-15 and is recorded with the clipped counts. scikit-learn 1.9 clips at float64 machine epsilon and has no `eps` argument; the tests compare the two under matching ε. |
| Reference Brier | Brier loss of a constant predictor r | r is the training or development prevalence if supplied (`--reference-prevalence`). Otherwise r is the evaluation prevalence, labelled as a *descriptive* reference that uses held-out outcomes. |
| Brier skill score | 1 − BS / BS_ref | Unavailable (`UNAVAILABLE_ZERO_REFERENCE`) when BS_ref = 0. |
| ROC-AUC, average precision | scikit-learn definitions (ties count ½ in AUC) | Both classes are required, otherwise the value is `UNAVAILABLE_SINGLE_CLASS`. A constant predictor gives AUC 0.5 and is flagged. |
| Mean prediction error | mean(p) − mean(y) | Positive means predictions exceed prevalence. Local gaps can cancel in this aggregate. |
| Threshold accuracy | mean(1[p ≥ t] = y), t = 0.5 by default | Secondary view. |

Undefined values are exported as `null` with a reason code, never as NaN or as a plausible
number.

## 5. Binning, reliability diagrams and bin summaries

**Edges.** Bins are left-closed and right-open, `[e_k, e_{k+1})`, except that the last bin
is closed, `[e_{K−1}, 1]`. Edges always span [0, 1], so every valid probability is assigned
exactly once. This invariant is property-tested.

* *Uniform* edges are computed as k/K. For example, 0.3 is the nearest double to 3/10, so an
  input of exactly 0.3 falls in `[0.3, 0.4)`.
* *Quantile* edges are empirical quantiles with `method="inverted_cdf"`, so every edge is an
  observed value. Duplicate edges collapse and edges at or below the minimum are dropped. Random
  noise is never added to break ties. The requested and actual bin counts are both reported,
  and constant predictions produce a single bin. The population used to estimate the edges is recorded.
* Difference from scikit-learn: `calibration_curve` assigns a value equal to an interior edge
  to the *lower* bin (`searchsorted(..., side="left")`). The two agree whenever no prediction
  lies exactly on an interior edge, and a test documents this.

**Per-bin statistics.** Each bin reports count, events, mean prediction, observed rate,
signed gap = mean prediction − observed rate, and absolute gap. Empty bins stay in the tables
with unavailable estimates. They are omitted from curves and never interpolated.

**Reliability diagram.** The x-axis is the mean predicted positive-class probability per bin
and the y-axis is the observed positive fraction. The diagram shows the y = x line, fixed
[0, 1] axes, marker area proportional to support, and an aligned histogram of predictions. No
smoothing is applied. Identity is encoded by colour and marker shape, with at most three models
per panel.

**Binned positive-class ECE.**
ECE = Σ_b (n_b / n) · |ȳ_b − p̄_b| over nonempty bins of the declared partition. This is a
finite-sample diagnostic that depends on the partition. It is **not** top-label confidence ECE
and not the population calibration error; see Vaicenavicius et al. (2019) on how evaluation
choices affect calibration estimates. ECE is always shown with its strategy and actual bin
count. No ECE threshold certifies a model.

**Other summaries.** Using the same partition, CalibCheck also reports:

* the maximum absolute gap, with the support of the bin that attains it;
* the maximum absolute gap over supported bins only (count ≥ `min_support`, default 30), with
  the excluded bins and rows disclosed;
* the support-weighted RMS gap.

These summaries are never combined into a composite score.

**Sensitivity.** A fixed grid (5, 10 and 20 bins × uniform and quantile) is reported as a
robustness check. The primary partition stays fixed in each saved report, and the grid is not
used for selection.

## 6. Uncertainty

**Wilson intervals for bin rates.** Under an independent-observation approximation, each
nonempty bin gets a pointwise Wilson score interval (Wilson, 1927). All-positive and
all-negative bins are included, and empty bins get none. The implementation matches
`scipy.stats.binomtest(k, n).proportion_ci(method="wilson")` to 1e-12 in tests. These are
pointwise intervals and not simultaneous bands. With K bins, roughly (1 − level) · K
well-calibrated bins are expected to be flagged by chance, and findings say so.

**Bootstrap for aggregate losses.** CalibCheck uses a seeded observation-level percentile
bootstrap (Efron & Tibshirani, 1993) for Brier loss, log loss and mean prediction error. The
defaults are 2,000 replicates, level 0.95, `numpy.random.default_rng(seed)`, and batches of
`max(1, 4,000,000 // n)` replicates so that memory does not grow as n × replicates. Failed or
degenerate replicates are counted and reported. On a fixture, the intervals agree with
`scipy.stats.bootstrap(method="percentile")` up to Monte Carlo error. Percentile intervals can
under-cover in small samples; this is a stated limitation.

**Estimand and conditioning.** The bootstrap intervals describe sampling variability of the
statistic over evaluation rows, **conditional on the fixed predictions**, and so on the fitted
base model and any fitted recalibrator. They do not include training variability, recalibrator
fitting variability or method-selection variability. Repeated-fit variability is studied in
`experiments/`.

**ECE bootstrap (opt-in, exploratory).** The bin edges are frozen at their full-sample values,
so the interval is conditional on the partition. The absolute value biases binned ECE upward
in small bins, so these intervals are labelled as approximate. A variant that re-estimates the
edges is **not implemented**.

**Dependence.** Users declare one of `independent` (the default, recorded as an assumption),
`entity_clustered`, `time_dependent` or `unknown`. Any declaration other than independent
disables Wilson and bootstrap intervals with the reason `UNAVAILABLE_DEPENDENCE`, while descriptive
results remain. CalibCheck never silently falls back to an IID bootstrap. Cluster and block
bootstraps are **not implemented**.

**Small samples.** Counts and event counts are shown next to every estimate. The
`min_support` threshold is a display heuristic, not a statistical law.

## 7. Paired comparisons

Models are compared only on identical rows: one file with several probability columns, or
separate files joined by unique ID with matching outcomes. Separate files require
`--intersection` if any rows are unmatched, and the number of lost rows is reported. For each
metric:

  Δ = metric(candidate) − metric(reference). For losses, negative favours the candidate.

The intervals resample the *same row indices* for both models: the bootstrap is applied to the
per-row loss differences. They are never built by comparing two separate intervals.

* **favours candidate / reference**: the paired interval excludes 0.
* **neither**: the interval includes 0, reported as "insufficient evidence to distinguish".
* **undetermined**: no interval was computed.

When metrics disagree, CalibCheck states that and declares no overall winner. With more than
two models, comparisons require an explicit `--reference` or `--pairwise` request. ECE
differences use shared edges: uniform, or quantile edges estimated from the declared reference
model. ROC-AUC differences are reported without an interval in v0.1.

## 8. Subgroups and time windows

Subgroups come from explicitly selected categorical columns. Groups are ordered by **size**,
not error. The display limit pools the remaining groups into `(other)`, the number of examined
groups is reported, and all groups share the global bin edges. Findings are descriptive and
exploratory. They do not establish causality, discrimination or legal fairness. Multiple-testing
adjustments are **not implemented**, because v0.1 defines no confirmatory testing family.

In public export mode, a cell is suppressed if it has fewer than `min_cell` rows (default 10) or
if all its outcomes are identical, which would reveal every member's outcome. When exactly one
cell is suppressed, the next-smallest cell is also suppressed so totals cannot reveal it. Group
names are replaced by `group k` unless `--reveal-group-names` is set.

Time windows follow the same public-mode policy, with the same thresholds and complementary
rule. A suppressed window keeps its calendar label, but its event count, prevalence and every
per-model statistic become `null`, with `suppressed: true` and a reason. Its row count is also
withheld when it is below `min_cell`. The window becomes a gap in the plot, never a zero. This
happens in the analysis layer, so JSON, CSV, HTML, figures and the app's downloads all agree.
Private mode keeps full detail.

Time windows (D/W/M/Q/Y) use timestamps parsed with `pandas.to_datetime(utc=True)`: offset-aware
values are converted to UTC and naive values are read as UTC. Any unparseable timestamp disables
the view rather than silently dropping rows. Window views are descriptive, and no intervals are
computed for dependent windows.

## 9. Recalibration and the data-role protocol

**Roles.** The base model is trained on *training* data, which CalibCheck never sees and cannot
audit. Its predictions on *calibration* rows fit the recalibrator. Final evaluation uses
untouched *test* rows.

* IDs that appear in both calibration and test raise `ProtocolError` (exit code 3). Entities
  shared across roles raise an error when entity IDs are given. Under `--temporal`, every
  calibration timestamp must precede every test timestamp.
* Positional IDs make overlap unverifiable, and the report says so
  (`ROLE_PROVENANCE_UNKNOWN`). It never claims independence.
* Artifacts store **salted** SHA-256 prefixes of calibration IDs. `--fitted-artifact` refuses
  to evaluate on rows that were used for fitting. Hashes are not anonymisation, and storing them
  can be disabled with `--no-id-digests`.

**Modes.**

* *Exploratory* evaluates every predeclared method on the test rows and labels the result as
  such. The best observed result is not an unbiased estimate for a method chosen because it
  looked best.
* *Selected* chooses a method by pooled out-of-fold log loss (or Brier) on inner folds of the
  **calibration** data, then freezes it and evaluates once on the test rows. The folds are KFold
  (shuffled with the run seed), GroupKFold by entity when entities exist, or chronological
  expanding windows under `--temporal`. Ties go to the simpler method, in the order
  identity < temperature < sigmoid < isotonic.

Tests verify that changing test labels changes neither the fitted parameters nor the selected
method.

**Methods.**

* `identity`: q = p. This is the baseline.
* `sigmoid` (*logit-affine sigmoid calibration*): q = σ(a·logit(clip(p, ε)) + b). It is fitted by
  maximum likelihood with L-BFGS-B, a ∈ [0, 50], b ∈ [−50, 50], starting from (1, 0), with
  gtol 1e-9, maxiter 1000 and no regularisation. The slope a = 0 is allowed; it is the
  constant-map boundary, and the map is non-decreasing rather than strictly increasing. Warnings
  cover non-convergence, parameters at a bound and perfectly separated calibration data. This
  is Platt's sigmoid (Platt, 1999) applied to the *logit of a probability*, not to raw SVM
  decision values.
* `isotonic`: scikit-learn `IsotonicRegression(y_min=0, y_max=1, increasing=True,
  out_of_bounds="clip")`, based on Zadrozny & Elkan (2002). The breakpoints are stored and
  applied with `numpy.interp` (linear between breakpoints, constant outside them), which a test
  checks against `IsotonicRegression.predict`. Ties in the output can change ranking metrics,
  and this is reported rather than hidden.
* `temperature`: q = σ(logit(clip(p, ε)) / T), T ∈ [0.01, 100], fitted with bounded Brent
  minimisation on log T. It is an adaptation of Guo et al. (2017) to binary probability input.
  With no intercept, it cannot fix a base-rate shift; Study A in `experiments/` shows this for
  `intercept_bias`.
* Beta calibration (Kull et al., 2017) is **not implemented**. It is unrelated to beta posterior
  intervals.

**Artifacts** are versioned JSON with no executable content. They record method, parameters,
input transform, ε, label mapping, model association, fit diagnostics and provenance. Loading
rejects pickle or joblib bytes, unknown schema versions, unknown fields and invalid parameters
(negative slope, non-monotone or out-of-range breakpoints). `apply` needs no outcomes, refuses
a mismatched model association unless `--force-association` is given, and never overwrites
files or columns without an explicit flag.

## 10. Findings engine

Findings are deterministic rules over the structured results. No language model is used and
no data leaves the machine. Each finding carries a stable code, the supporting numbers, the
population and a caveat. Neighbouring well-supported bins whose Wilson interval excludes the
mean prediction, with the same sign, are merged into one range finding. Tests check that the
findings vocabulary avoids verdict words such as "trustworthy", "safe", "unbiased" and "ready
for deployment".

## 11. What uploaded predictions cannot reveal

* Whether the base model saw the calibration or test rows during training.
* Whether rows are truly independent. Dependence must be declared.
* Whether the evaluation population matches the deployment population.
* Whether a subgroup difference is caused by group membership.

## 12. Limitations of v0.1

* Percentile bootstrap intervals are approximate and can under-cover for small n or skewed
  losses.
* Wilson bin intervals are pointwise and assume independence.
* Binned ECE depends on the partition and is biased upward for sparse bins.
* No dependence-aware intervals, weights, multiclass support or confirmatory subgroup testing.
* Validation copies text columns in memory; at 1,000,000 rows it peaked near 390 MB of traced
  memory on the benchmark machine (`benchmarks/results.md`).
* The real-data example (569 rows, about 143 test rows) is too small for definitive comparisons, and the
  report says "insufficient evidence" there.

## References

* Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly
  Weather Review*, 78(1), 1–3.
* DeGroot, M. H., & Fienberg, S. E. (1983). The comparison and evaluation of forecasters.
  *Journal of the Royal Statistical Society: Series D*, 32(1–2), 12–22.
* Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
* Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
  networks. *ICML*, PMLR 70, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html
* Kull, M., Silva Filho, T., & Flach, P. (2017). Beta calibration: a well-founded and easily
  implemented improvement on logistic calibration for binary classifiers. *AISTATS*, PMLR 54,
  623–631. https://proceedings.mlr.press/v54/kull17a.html
* Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated
  probabilities using Bayesian binning. *AAAI*.
* Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised
  learning. *ICML*.
* Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods. In *Advances in Large Margin Classifiers*. MIT Press.
* Vaicenavicius, J., Widmann, D., Andersson, C., Lindsten, F., Roll, J., & Schön, T. (2019).
  Evaluating model calibration in classification. *AISTATS*, PMLR 89, 3459–3467.
  https://proceedings.mlr.press/v89/vaicenavicius19a.html
* Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
  *Journal of the American Statistical Association*, 22(158), 209–212.
* Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass
  probability estimates. *KDD*.
* scikit-learn documentation for `brier_score_loss`, `log_loss`, `calibration_curve` and
  `IsotonicRegression`, checked against the installed version 1.9.1 by inspecting signatures
  and source. SciPy documentation for `BinomTestResult.proportion_ci` and `bootstrap`, checked
  against the installed version 1.17.1.

The PMLR entries (Guo, Kull, Vaicenavicius) and the UCI dataset pages were checked online on
2026-10-18. The other references are standard citations that were not re-fetched for this
release.
