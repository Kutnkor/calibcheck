"""Recalibration maps fitted on separate calibration data, persisted as inspectable JSON.

Methods (binary, probability-only input):

* ``identity`` — q = p. The baseline every other method must be compared with.
* ``sigmoid`` — *logit-affine sigmoid calibration*: q = sigmoid(a * logit(clip(p)) + b),
  maximum likelihood with a >= 0 (non-decreasing; a = 0 is the constant-map boundary). This
  is Platt-style scaling applied to the logit of a probability, not to raw decision scores.
* ``isotonic`` — non-decreasing step-wise linear map from scikit-learn's IsotonicRegression
  (y in [0, 1], ``out_of_bounds='clip'``), stored as breakpoints and applied with linear
  interpolation (``numpy.interp``), constant beyond the fitted range.
* ``temperature`` — q = sigmoid(logit(clip(p)) / T), T > 0. One parameter, no intercept, so
  it cannot correct a shift in the base rate.

Artifacts never contain executable content; they are validated before use.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from calibcheck._version import ARTIFACT_SCHEMA_VERSION, __version__
from calibcheck.config import MIN_CLIP_EPS
from calibcheck.errors import ArtifactError, FittingError, InputValidationError, OutputError
from calibcheck.schemas import Finding

Method = Literal["identity", "sigmoid", "isotonic", "temperature"]
METHODS: tuple[str, ...] = ("identity", "sigmoid", "isotonic", "temperature")
SIMPLICITY = {m: i for i, m in enumerate(("identity", "temperature", "sigmoid", "isotonic"))}

A_MAX = 50.0
B_MAX = 50.0
LOG_T_BOUNDS = (math.log(1e-2), math.log(1e2))
SMALL_CALIBRATION_N = 200  # practical warning heuristic, not a theorem
MAX_ARTIFACT_BYTES = 20_000_000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FitProvenance(_Strict):
    n: int
    n_pos: int
    prob_range: list[float]
    input_sha256: str | None = None
    id_kind: str
    id_digest_salt: str | None = None
    id_digests: list[str] | None = None
    entity_digests: list[str] | None = None
    timestamp_max: str | None = None
    note: str = ("ID digests are salted SHA-256 prefixes used only to detect role overlap; "
                 "they are not anonymisation.")


class FitDiagnostics(_Strict):
    status: Literal["ok", "failed"]
    converged: bool
    n_iter: int | None = None
    objective: float | None = None
    message: str = ""
    warnings: list[Finding] = Field(default_factory=list)


class CalibratorArtifact(_Strict):
    artifact_type: Literal["calibcheck.calibrator"] = "calibcheck.calibrator"
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    method: Method
    params: dict[str, Any]
    input_transform: str
    output_rule: str
    clip_eps: float
    positive_label: str
    label_column: str
    model_association: str = Field(description="The probability column / model this map was fitted for.")
    fitted_on: FitProvenance
    fit_config: dict[str, Any] = Field(default_factory=dict)
    diagnostics: FitDiagnostics
    calibcheck_version: str = __version__
    created_utc: str = ""


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q) - np.log1p(-q)


def _as_probabilities(p: Any, what: str = "p", *, allow_empty: bool = False) -> np.ndarray:
    """Validate probabilities *before* any lossy conversion, then return float64.

    Requires a one-dimensional, nonempty, real-numeric array (booleans, strings, objects and
    complex values are rejected rather than coerced) with finite values in [0, 1].
    """
    raw = np.asarray(p)
    if raw.dtype.kind not in "iuf":
        raise InputValidationError(
            f"{what} must be a numeric array of probabilities (got dtype {raw.dtype}); values "
            "are not coerced.", code="PROB_NONNUMERIC")
    if raw.ndim != 1:
        raise InputValidationError(f"{what} must be one-dimensional (got shape {raw.shape}).",
                                   code="INPUT_SHAPE")
    if raw.size == 0 and not allow_empty:
        raise InputValidationError(f"{what} is empty.", code="INPUT_EMPTY")
    vals = raw.astype(np.float64)
    if raw.dtype.kind == "f" and not np.all(np.isfinite(raw)):
        k = int((~np.isfinite(raw)).sum())
        raise InputValidationError(f"{what}: {k} value(s) are NaN or infinite.",
                                   code="PROB_NONFINITE")
    bad = (raw < 0) | (raw > 1)
    if bad.any():
        raise InputValidationError(
            f"{what}: {int(bad.sum())} value(s) lie outside [0, 1]; they are not clipped or "
            "rescaled.", code="PROB_OUT_OF_RANGE")
    return vals


def _as_binary_labels(y: Any, n: int) -> np.ndarray:
    """Validate outcomes as exactly 0/1 (or boolean) *before* casting to int8."""
    raw = np.asarray(y)
    if raw.dtype.kind not in "biuf":
        raise InputValidationError(
            f"y must be numeric 0/1 or boolean (got dtype {raw.dtype}); map labels to 0/1 "
            "explicitly with the declared positive label.", code="LABEL_NOT_BINARY")
    if raw.ndim != 1:
        raise InputValidationError(f"y must be one-dimensional (got shape {raw.shape}).",
                                   code="INPUT_SHAPE")
    if raw.size != n:
        raise InputValidationError(f"y has {raw.size} values but p has {n}.",
                                   code="INPUT_SHAPE")
    if raw.dtype.kind != "b":
        if raw.dtype.kind == "f" and not np.all(np.isfinite(raw)):
            raise InputValidationError("y contains NaN or infinite values.", code="LABEL_MISSING")
        bad = (raw != 0) & (raw != 1)
        if bad.any():
            raise InputValidationError(
                f"y: {int(bad.sum())} value(s) are not exactly 0 or 1; labels are never rounded "
                "or truncated.", code="LABEL_NOT_BINARY")
    return raw.astype(np.int8)


def _check_aligned(name: str, values: Any, n: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size != n:
        raise InputValidationError(f"{name} must be one-dimensional with {n} values "
                                   f"(got shape {arr.shape}).", code="INPUT_SHAPE")
    return arr


def _digest(salt: str, values: np.ndarray) -> list[str]:
    return sorted(hashlib.sha256((salt + str(v)).encode()).hexdigest()[:20] for v in values)


def digests_for(salt: str, values: np.ndarray) -> set[str]:
    return set(_digest(salt, values))


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------

def _fit_sigmoid(z: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], FitDiagnostics]:
    yf = y.astype(np.float64)
    n = yf.size

    def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = theta
        s = a * z + b
        nll = float((np.logaddexp(0.0, s) - yf * s).sum() / n)
        r = expit(s) - yf
        return nll, np.array([float((r * z).sum() / n), float(r.sum() / n)])

    res = minimize(fun, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, A_MAX), (-B_MAX, B_MAX)],
                   options={"maxiter": 1000, "gtol": 1e-9, "ftol": 1e-12})
    a, b = (float(v) for v in res.x)
    warns: list[Finding] = []
    if a <= 1e-10:
        warns.append(Finding(code="FIT_CONSTANT_MAP", severity="warning",
                             message="Fitted slope a = 0: the map is constant and removes all "
                                     "ranking information."))
    if a >= A_MAX - 1e-6 or abs(b) >= B_MAX - 1e-6:
        warns.append(Finding(code="FIT_AT_BOUND", severity="warning",
                             message=f"A parameter reached its bound (a={a:.4g}, b={b:.4g}); the "
                                     "calibration data may be (nearly) separable."))
    if not res.success:
        warns.append(Finding(code="FIT_NOT_CONVERGED", severity="warning",
                             message=f"Optimizer did not report convergence: {res.message}"))
    return {"a": a, "b": b}, FitDiagnostics(
        status="ok", converged=bool(res.success), n_iter=int(res.nit), objective=float(res.fun),
        message=str(res.message), warnings=warns)


def _fit_temperature(z: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], FitDiagnostics]:
    yf = y.astype(np.float64)

    def nll(log_t: float) -> float:
        s = z / math.exp(log_t)
        return float((np.logaddexp(0.0, s) - yf * s).mean())

    res = minimize_scalar(nll, bounds=LOG_T_BOUNDS, method="bounded",
                          options={"xatol": 1e-10, "maxiter": 500})
    t = float(math.exp(res.x))
    warns = []
    if abs(res.x - LOG_T_BOUNDS[0]) < 1e-6 or abs(res.x - LOG_T_BOUNDS[1]) < 1e-6:
        warns.append(Finding(code="FIT_AT_BOUND", severity="warning",
                             message=f"Temperature at its bound (T={t:.4g})."))
    if not res.success:
        warns.append(Finding(code="FIT_NOT_CONVERGED", severity="warning",
                             message=f"Optimizer did not report convergence: {res.message}"))
    return {"T": t}, FitDiagnostics(status="ok", converged=bool(res.success),
                                    n_iter=int(res.nfev), objective=float(res.fun),
                                    message=str(res.message), warnings=warns)


def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], FitDiagnostics]:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(p, y.astype(np.float64))
    x = [float(v) for v in iso.X_thresholds_]
    yy = [float(v) for v in iso.y_thresholds_]
    warns = []
    if len(set(yy)) == 1:
        warns.append(Finding(code="FIT_CONSTANT_MAP", severity="warning",
                             message="The isotonic map is constant; it removes all ranking "
                                     "information."))
    return {"x": x, "y": yy}, FitDiagnostics(status="ok", converged=True, n_iter=None,
                                             objective=None, message="pool adjacent violators",
                                             warnings=warns)


TRANSFORMS = {
    "identity": ("p (unchanged)", "q = p"),
    "sigmoid": ("z = logit(clip(p, eps, 1-eps))", "q = sigmoid(a*z + b), a >= 0"),
    "temperature": ("z = logit(clip(p, eps, 1-eps))", "q = sigmoid(z / T), T > 0"),
    "isotonic": ("p (unchanged)", "q = linear interpolation of breakpoints (x, y); constant "
                 "outside [x_min, x_max]"),
}


def fit_calibrator(p: np.ndarray, y: np.ndarray, method: str, *, eps: float = 1e-15,
                   model_association: str, positive_label: str = "1",
                   label_column: str = "y_true", ids: np.ndarray | None = None,
                   id_kind: str = "provided", entities: np.ndarray | None = None,
                   timestamp_max: str | None = None, input_sha256: str | None = None,
                   store_id_digests: bool = True) -> CalibratorArtifact:
    """Fit a recalibration map on *calibration* data only and return a validated artifact."""
    if method not in METHODS:
        raise FittingError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}.")
    if not (isinstance(eps, int | float) and math.isfinite(eps) and MIN_CLIP_EPS <= eps < 1e-3):
        raise InputValidationError(f"eps must be a finite number in [float64 epsilon, 1e-3); got {eps!r}.",
                                   code="CONFIG_INVALID")
    p = _as_probabilities(p)
    y = _as_binary_labels(y, p.size)
    ids = _check_aligned("ids", ids, p.size)
    entities = _check_aligned("entities", entities, p.size)
    n_pos = int(y.sum())
    if method != "identity" and n_pos in (0, y.size):
        raise FittingError("Calibration data contains a single outcome class; a recalibration "
                           "map cannot be fitted meaningfully.", code="FIT_SINGLE_CLASS")
    if method == "identity":
        params: dict[str, Any] = {}
        diag = FitDiagnostics(status="ok", converged=True, message="no parameters")
    elif method == "sigmoid":
        params, diag = _fit_sigmoid(_logit(p, eps), y)
        if p[y == 0].max() < p[y == 1].min():
            diag.warnings.append(Finding(
                code="FIT_SEPARATION", severity="warning",
                message="Every positive has a higher probability than every negative in the "
                        "calibration data; the maximum-likelihood slope is unbounded, so the "
                        "fitted map depends on the optimizer's stopping rule."))
    elif method == "temperature":
        params, diag = _fit_temperature(_logit(p, eps), y)
    else:
        params, diag = _fit_isotonic(p, y)
    if not diag.converged:
        raise FittingError("Optimizer did not converge; no calibrator artifact was created.")
    if method != "identity" and y.size < SMALL_CALIBRATION_N:
        diag.warnings.append(Finding(
            code="FIT_SMALL_CALIBRATION_SET", severity="warning",
            message=f"Only {y.size} calibration rows; the fitted map may be unstable "
                    f"(practical heuristic: fewer than {SMALL_CALIBRATION_N})."))
    salt = secrets.token_hex(16) if store_id_digests and ids is not None else None
    prov = FitProvenance(
        n=int(y.size), n_pos=n_pos, prob_range=[float(p.min()), float(p.max())],
        input_sha256=input_sha256, id_kind=id_kind, id_digest_salt=salt,
        id_digests=_digest(salt, ids) if salt and ids is not None and id_kind == "provided" else None,
        entity_digests=_digest(salt, np.unique(entities)) if salt and entities is not None else None,
        timestamp_max=timestamp_max)
    tin, tout = TRANSFORMS[method]
    art = CalibratorArtifact(
        method=method, params=params, input_transform=tin, output_rule=tout,
        clip_eps=eps, positive_label=positive_label, label_column=label_column,
        model_association=model_association, fitted_on=prov,
        fit_config={"bounds": {"a": [0.0, A_MAX], "b": [-B_MAX, B_MAX]} if method == "sigmoid"
                    else ({"T": [1e-2, 1e2]} if method == "temperature" else {}),
                    "initialization": {"a": 1.0, "b": 0.0} if method == "sigmoid" else None,
                    "regularization": "none",
                    "optimizer": {"sigmoid": "L-BFGS-B (gtol 1e-9, maxiter 1000)",
                                  "temperature": "bounded Brent on log T (xatol 1e-10)",
                                  "isotonic": "sklearn IsotonicRegression (PAVA)",
                                  "identity": "none"}[method]},
        diagnostics=diag, created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    validate_artifact(art)
    return art


# --------------------------------------------------------------------------------------
# applying
# --------------------------------------------------------------------------------------

def apply_calibrator(art: CalibratorArtifact, p: np.ndarray) -> tuple[np.ndarray, list[Finding]]:
    """Map probabilities through a fitted artifact. Does not need outcomes."""
    validate_artifact(art)
    p = _as_probabilities(p, allow_empty=True)
    findings: list[Finding] = []
    m = art.method
    if m == "identity":
        q = p.copy()
    elif m == "sigmoid":
        q = expit(art.params["a"] * _logit(p, art.clip_eps) + art.params["b"])
    elif m == "temperature":
        q = expit(_logit(p, art.clip_eps) / art.params["T"])
    else:
        x = np.asarray(art.params["x"], dtype=np.float64)
        yv = np.asarray(art.params["y"], dtype=np.float64)
        q = np.interp(p, x, yv)
        k = int(((p < x[0]) | (p > x[-1])).sum())
        if k:
            findings.append(Finding(
                code="APPLY_OUT_OF_RANGE", severity="info", count=k,
                message=f"{k} input(s) fall outside the calibration range [{x[0]:.4g}, "
                        f"{x[-1]:.4g}] and were mapped to the nearest end value."))
    return np.clip(q, 0.0, 1.0), findings


# --------------------------------------------------------------------------------------
# persistence & validation
# --------------------------------------------------------------------------------------

def validate_artifact(art: CalibratorArtifact) -> None:
    major = art.schema_version.split(".")[0]
    if major != ARTIFACT_SCHEMA_VERSION.split(".")[0]:
        raise ArtifactError(f"Unsupported artifact schema version {art.schema_version!r} "
                            f"(this CalibCheck reads {ARTIFACT_SCHEMA_VERSION}).")
    if not (MIN_CLIP_EPS <= art.clip_eps < 1e-3):
        raise ArtifactError("clip_eps must be in [float64 epsilon, 1e-3).")
    if art.diagnostics.status != "ok" or not art.diagnostics.converged:
        raise ArtifactError("The artifact records a failed or non-converged fit and cannot be applied.")
    p = art.params

    def _finite(v: Any) -> bool:
        return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)

    if art.method == "identity":
        if p:
            raise ArtifactError("identity calibrator must have no parameters.")
    elif art.method == "sigmoid":
        if set(p) != {"a", "b"} or not all(_finite(p[k]) for k in ("a", "b")):
            raise ArtifactError("sigmoid calibrator needs finite parameters 'a' and 'b'.")
        if p["a"] < 0:
            raise ArtifactError("sigmoid slope 'a' must be >= 0 (non-decreasing map).")
    elif art.method == "temperature":
        if set(p) != {"T"} or not _finite(p["T"]) or p["T"] <= 0:
            raise ArtifactError("temperature calibrator needs a finite parameter T > 0.")
    elif art.method == "isotonic":
        if set(p) != {"x", "y"}:
            raise ArtifactError("isotonic calibrator needs breakpoint lists 'x' and 'y'.")
        x, yv = p["x"], p["y"]
        if (not isinstance(x, list) or not isinstance(yv, list) or len(x) != len(yv)
                or len(x) == 0 or not all(_finite(v) for v in x + yv)):
            raise ArtifactError("isotonic breakpoints must be equal-length, non-empty, finite.")
        if any(b <= a for a, b in itertools.pairwise(x)) or any(
                b < a for a, b in itertools.pairwise(yv)):
            raise ArtifactError("isotonic x must be strictly increasing and y non-decreasing.")
        if min(x) < 0 or max(x) > 1 or min(yv) < 0 or max(yv) > 1:
            raise ArtifactError("isotonic breakpoints must lie in [0, 1].")


def save_artifact(art: CalibratorArtifact, path: str | Path, *, overwrite: bool = False) -> Path:
    out = Path(path)
    if out.exists() and not overwrite:
        raise OutputError(f"{out} already exists; pass --overwrite to replace it.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art.model_dump(mode="json"), indent=2, allow_nan=False),
                   encoding="utf-8")
    return out


def load_artifact(path: str | Path) -> CalibratorArtifact:
    """Load and validate a JSON artifact. Pickle/joblib files are never accepted."""
    pth = Path(path)
    if not pth.exists():
        raise ArtifactError(f"Artifact not found: {pth.name}")
    if pth.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ArtifactError("Artifact file is unreasonably large.")
    raw = pth.read_bytes()
    if raw[:2] in (b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        raise ArtifactError("This looks like a pickle/joblib file. CalibCheck only loads its "
                            "documented JSON artifacts and never unpickles uploads.")
    return parse_artifact(raw)


def parse_artifact(raw: bytes | str) -> CalibratorArtifact:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"Artifact is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("artifact_type") != "calibcheck.calibrator":
        raise ArtifactError("Not a CalibCheck calibrator artifact.")
    sv = str(data.get("schema_version", ""))
    if sv.split(".")[0] != ARTIFACT_SCHEMA_VERSION.split(".")[0]:
        raise ArtifactError(f"Unsupported artifact schema version {sv!r}.")
    try:
        art = CalibratorArtifact.model_validate(data)
    except ValidationError as exc:
        raise ArtifactError(f"Malformed artifact: {exc.errors()[0]['msg']} at "
                            f"{'.'.join(map(str, exc.errors()[0]['loc']))}") from exc
    validate_artifact(art)
    return art
