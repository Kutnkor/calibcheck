"""CalibCheck: how trustworthy are your model's probabilities?

Public API (stable names for v0.1)::

    from calibcheck import (analyze_predictions, compare_predictions, inspect_predictions,
                            fit_calibrator, apply_calibrator, evaluate_recalibration,
                            render_report, load_predictions, AnalysisConfig)
"""

from calibcheck._version import ARTIFACT_SCHEMA_VERSION, SCHEMA_VERSION, __version__
from calibcheck.analysis import (
    analyze_data,
    analyze_predictions,
    compare_predictions,
    inspect_predictions,
)
from calibcheck.calibration import (
    CalibratorArtifact,
    apply_calibrator,
    fit_calibrator,
    load_artifact,
    save_artifact,
)
from calibcheck.config import AnalysisConfig, resolve_config
from calibcheck.data import PredictionData, align_by_id, load_predictions
from calibcheck.errors import (
    ArtifactError,
    CalibCheckError,
    ConfigError,
    FittingError,
    InputValidationError,
    OutputError,
    ProtocolError,
)
from calibcheck.protocol import evaluate_recalibration, select_method
from calibcheck.report import render_report, summary_text
from calibcheck.schemas import AnalysisResult, Finding, InspectResult, MetricValue

__all__ = [
    "ARTIFACT_SCHEMA_VERSION", "SCHEMA_VERSION", "AnalysisConfig", "AnalysisResult",
    "ArtifactError", "CalibCheckError", "CalibratorArtifact", "ConfigError", "Finding",
    "FittingError", "InputValidationError", "InspectResult", "MetricValue", "OutputError",
    "PredictionData", "ProtocolError", "__version__", "align_by_id", "analyze_data",
    "analyze_predictions", "apply_calibrator", "compare_predictions", "evaluate_recalibration",
    "fit_calibrator", "inspect_predictions", "load_artifact", "load_predictions",
    "render_report", "resolve_config", "save_artifact", "select_method", "summary_text",
]
