"""Registry of stable warning, error, unavailability and finding codes.

Codes are part of the public JSON contract: they may be added, but an existing code keeps
its meaning across minor versions. ``EXPLANATIONS`` gives the generic, readable meaning of
each code; individual findings add run-specific detail.
"""

from __future__ import annotations

EXPLANATIONS: dict[str, str] = {
    # --- input / schema validation -------------------------------------------------------
    "ALIGN_METADATA_CONFLICT": "Matched rows disagree on shared entity, split, timestamp or subgroup metadata.",
    "INPUT_EMPTY_HEADER": "CSV headers must name every column; blank headers are not inferred.",
    "INPUT_EMPTY": "The file contains no data rows.",
    "INPUT_UNREADABLE": "The file could not be parsed as CSV.",
    "INPUT_DUPLICATE_HEADER": "Two or more columns share the same header name.",
    "INPUT_MISSING_COLUMN": "A column named in the mapping is not present in the file.",
    "INPUT_AMBIGUOUS_MAPPING": "The same column is mapped to more than one role.",
    "INPUT_NO_PROB_COLUMNS": "No probability column was selected; CalibCheck never guesses them.",
    "INPUT_TOO_MANY_COLUMNS": "The file exceeds the configured column limit.",
    "INPUT_SHAPE": "Arrays passed to the API must be one-dimensional and of matching length.",
    "INPUT_TOO_LARGE": "The file exceeds the configured size limit.",
    "LABEL_MISSING": "Outcome values are missing.",
    "LABEL_NOT_BINARY": "The outcome column has more than two distinct values (multiclass is out of scope).",
    "LABEL_POSITIVE_UNDECLARED": "The positive label must be declared because the labels are not 0/1.",
    "LABEL_POSITIVE_NOT_FOUND": "The declared positive label does not occur and cannot be matched to a label value.",
    "LABEL_SINGLE_CLASS": "Only one outcome class is present; ranking metrics are unavailable.",
    "PROB_MISSING": "Probability values are missing.",
    "PROB_NONNUMERIC": "Probability values could not be parsed as numbers.",
    "PROB_NONFINITE": "Probability values are NaN or infinite.",
    "PROB_OUT_OF_RANGE": "Probability values lie outside [0, 1].",
    "PROB_LOOKS_LIKE_PERCENT": "Values look like percentages (0-100). They are not divided by 100 automatically.",
    "PROB_TWO_VALUED": "The probability column takes only two distinct values; it may contain hard predictions.",
    "PROB_CONSTANT": "The probability column is constant; it carries no ranking information.",
    "PROB_ENDPOINTS": "Some probabilities are exactly 0 or 1; log loss uses the declared clipping rule.",
    "ID_MISSING": "Row identifiers are missing.",
    "ID_DUPLICATE": "Row identifiers are not unique.",
    "ID_POSITIONAL": "No ID column was given; positional IDs were generated and cannot align separate files.",
    "GROUP_MISSING_VALUES": "Some subgroup values are missing and are shown as a visible '(missing)' category.",
    "GROUP_HIGH_CARDINALITY": "The subgroup column has too many distinct values for a readable, privacy-aware analysis.",
    "GROUP_SPARSE": "Some subgroups have fewer observations than the display threshold.",
    "TIME_UNPARSEABLE": "Some timestamps could not be parsed; time-window views are disabled.",
    "TIME_MISSING": "Some timestamps are missing; those rows are absent from time-window views only.",
    "ROWS_EXCLUDED": "Rows were excluded under the explicit drop-invalid policy.",
    "TRANSFORM_APPLIED": "An explicit, recorded transformation was applied to a probability column.",
    # --- alignment / roles ---------------------------------------------------------------
    "ALIGN_UNMATCHED": "Some rows have no matching ID in the other file.",
    "ALIGN_LABEL_CONFLICT": "The same ID has contradictory outcomes in different files.",
    "ALIGN_DUPLICATE_KEY": "Duplicate IDs make the join many-to-many.",
    "ROLE_OVERLAP": "The same IDs appear in data roles that must be disjoint.",
    "ROLE_ENTITY_OVERLAP": "The same entities appear in data roles that must be disjoint.",
    "ROLE_PROVENANCE_UNKNOWN": "Role separation cannot be verified (no trustworthy IDs); independence is not established.",
    "ROLE_TEMPORAL_ORDER": "Calibration data is not strictly earlier than evaluation data.",
    "ROLE_BASE_MODEL_UNVERIFIABLE": "CalibCheck cannot verify how the base model was trained from predictions alone.",
    # --- metric availability -------------------------------------------------------------
    "UNAVAILABLE_SINGLE_CLASS": "Requires both outcome classes in the population.",
    "UNAVAILABLE_ZERO_REFERENCE": "The reference Brier loss is zero, so the skill score is undefined.",
    "UNAVAILABLE_EMPTY": "The population is empty.",
    "UNAVAILABLE_NO_SUPPORTED_BINS": "No bin meets the support threshold for this secondary summary.",
    "UNAVAILABLE_DISABLED": "This computation was not requested.",
    "UNAVAILABLE_DEPENDENCE": "The declared dependence structure conflicts with the independence assumption of this interval method.",
    "UNAVAILABLE_SUPPRESSED": "Suppressed in public export because the cell is small or reveals individual outcomes.",
    "UNAVAILABLE_NOT_SUPPORTED": "This interval or method is not supported in v0.1.",
    "UNAVAILABLE_FIT_FAILED": "The recalibrator failed to fit; no recalibrated probabilities exist.",
    # --- uncertainty ---------------------------------------------------------------------
    "UNC_BOOTSTRAP_DEGENERATE": "All bootstrap replicates are identical; the interval has zero width.",
    "UNC_BOOTSTRAP_FAILED_REPLICATES": "Some bootstrap replicates produced undefined statistics.",
    "UNC_SMALL_SUPPORT": "Support is small; estimates are imprecise (practical heuristic, not a statistical law).",
    "UNC_ECE_EXPLORATORY": "ECE intervals are exploratory approximations conditional on frozen bin edges.",
    # --- calibration fitting -------------------------------------------------------------
    "FIT_NOT_CONVERGED": "The optimizer did not report convergence.",
    "FIT_AT_BOUND": "A fitted parameter is at its optimization bound.",
    "FIT_SEPARATION": "Calibration outcomes are perfectly separated by the probability; the maximum-likelihood slope is unbounded, so the fitted map is a truncated approximation.",
    "FIT_CONSTANT_MAP": "The fitted map is constant; it removes all ranking information.",
    "FIT_SMALL_CALIBRATION_SET": "The calibration set is small; the fitted map may be unstable.",
    "FIT_SINGLE_CLASS": "The calibration data contains one class only; the map cannot be fitted.",
    "APPLY_OUT_OF_RANGE": "Inputs outside the calibration range were mapped with the documented clipping rule.",
    "APPLY_ASSOCIATION_MISMATCH": "The artifact's declared model association differs from the selected column.",
    # --- findings ------------------------------------------------------------------------
    "FINDING_OVERPREDICTION": "Predicted probabilities exceed the observed event rate in a probability range.",
    "FINDING_UNDERPREDICTION": "Predicted probabilities fall below the observed event rate in a probability range.",
    "FINDING_NO_CLEAR_BIN_GAP": "No well-supported bin shows a gap that its approximate interval separates from its mean prediction.",
    "FINDING_SPARSE_LARGE_GAP": "A large gap occurs in a sparsely populated bin; evidence is weak.",
    "FINDING_AGGREGATE_BIAS": "Average predicted probability differs from the observed prevalence.",
    "FINDING_RANKING_VS_CALIBRATION": "Ranking quality and probability calibration tell different stories.",
    "FINDING_CONSTANT_PREDICTOR": "A constant predictor can reveal probability bias but has no ranking ability.",
    "FINDING_LIMITED_COVERAGE": "Narrow outcome coverage limits what this report can establish.",
    "FINDING_COMPARISON_FAVORS": "A paired difference favours one model on a named metric and population.",
    "FINDING_COMPARISON_INCONCLUSIVE": "Insufficient evidence to distinguish the models on a named metric.",
    "FINDING_COMPARISON_NO_INTERVAL": "A paired difference was computed without an interval.",
    "FINDING_METRICS_DISAGREE": "Metrics disagree about which model is better; no overall winner is declared.",
    "FINDING_GROUP_CANCELLATION": "Subgroup gaps in opposite directions can cancel in aggregate.",
    "FINDING_GROUP_RANGE": "Descriptive spread of calibration gaps across subgroups.",
    "FINDING_RECAL_UNCHANGED": "Recalibration did not change probabilities materially.",
}


def explain(code: str) -> str:
    """Return the generic explanation for ``code`` (or a fallback for unknown codes)."""
    return EXPLANATIONS.get(code, "No explanation registered for this code.")
