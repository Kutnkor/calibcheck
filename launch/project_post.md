# Draft technical post (not published)

## When "90%" means 65%: checking probabilities, not just rankings

Many models produce probabilities that people read as probabilities. Examples include a
fraud score of 0.9, or a risk of 30%. Ranking metrics such as ROC-AUC say nothing about whether
those numbers match reality. CalibCheck is a small open-source Python tool for that question.

**What it does.** You give it a CSV with outcomes and one or more probability columns. It
validates the file strictly: you name the positive label and the probability columns, and
invalid values are never silently repaired. It then reports:

* Brier and log loss, with explicit conventions;
* binned positive-class ECE, with the partition named;
* reliability diagrams with exact bin edges and pointwise Wilson intervals for bin rates;
* seeded bootstrap intervals for losses, and *paired* intervals for model differences;
* descriptive subgroup and time-window views.

**Recalibration with data roles.** A recalibrator (logit-affine sigmoid, isotonic or
temperature) is fitted on calibration rows and judged only on untouched test rows. ID overlap
stops the run, and method selection uses inner folds of the calibration data.

**A synthetic example** (`calibcheck demo`, seed 7). A cohort is scored at 0.90 while its
outcomes are drawn from Bernoulli(0.65). On 3,000 test rows, the [0.9, 1.0] bin holds 937
rows with an observed rate of 0.653 (95% Wilson 0.622–0.683). An isotonic map fitted on 3,000
separate rows lowers test Brier loss by 0.0173 (paired 95% interval 0.0129–0.0214).

**And the results that do not flatter it.** In the bundled experiments (20 seeds per
scenario), isotonic recalibration of an already-calibrated model *raised* test log loss in
20 of 20 runs. On the 569-row Breast Cancer Wisconsin dataset (random forest, seed 1, 143 test rows),
every paired Brier and log-loss interval for the sigmoid and isotonic recalibrators included zero. CalibCheck reports "insufficient evidence to distinguish these approaches"
instead of naming a winner.

**Design choices worth arguing about:**

* undefined values are `null` with a reason code, never NaN;
* declaring dependent observations disables IID intervals rather than faking precision;
* findings come from deterministic rules, not a language model;
* the HTML report is a single offline file.

The methodology document lists every formula, assumption and limitation:
https://github.com/Kutnkor/calibcheck/blob/main/docs/methodology.md
