"""Stable exception types raised by CalibCheck.

Every exception carries a stable ``code`` (see :mod:`calibcheck.codes`) so callers can
branch on the failure without parsing messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calibcheck.schemas import Finding


class CalibCheckError(Exception):
    """Base class for all CalibCheck errors."""

    code: str = "CALIBCHECK_ERROR"

    def __init__(self, message: str, *, code: str | None = None,
                 findings: list[Finding] | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.findings: list[Finding] = list(findings or [])

    def __str__(self) -> str:
        base = super().__str__()
        if not self.findings:
            return base
        lines = [base]
        for f in self.findings:
            if f.severity == "error":
                lines.append(f"  - [{f.code}] {f.message}")
        return "\n".join(lines)


class InputValidationError(CalibCheckError):
    """The input data violates the CalibCheck input contract."""

    code = "INPUT_INVALID"


class ConfigError(CalibCheckError):
    """The configuration is malformed or internally inconsistent."""

    code = "CONFIG_INVALID"


class ProtocolError(CalibCheckError):
    """A requested workflow would violate the data-role protocol (e.g. leakage)."""

    code = "PROTOCOL_VIOLATION"


class ArtifactError(CalibCheckError):
    """A calibrator artifact is malformed, unsupported or incompatible."""

    code = "ARTIFACT_INVALID"


class FittingError(CalibCheckError):
    """A recalibrator could not be fitted."""

    code = "FIT_FAILED"


class OutputError(CalibCheckError):
    """Writing outputs would be unsafe (collision, traversal, overwrite)."""

    code = "OUTPUT_UNSAFE"


class CalibCheckInternalWarning(UserWarning):
    """Warning category for internal numerical issues (should not normally occur)."""
