"""Run manifests, result digests and privacy redaction.

Content fingerprints (SHA-256 of input bytes) support reproducibility checks. A hash is not
anonymisation: low-entropy inputs can be confirmed by re-hashing a guess. Public mode
therefore omits paths, fingerprints, row IDs and group names unless explicitly requested.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from calibcheck._version import __version__
from calibcheck.schemas import AnalysisResult, InputRecord, RunManifest

DEPENDENCIES = ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "jinja2", "pydantic",
                "typer", "rich")


def dependency_versions() -> dict[str, str]:
    out = {}
    for d in DEPENDENCIES:
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = "not installed"
    return out


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def results_digest(res: AnalysisResult) -> str:
    """SHA-256 over the statistical results only (manifest and volatile metadata excluded)."""
    payload = res.model_dump(mode="json", exclude={"manifest"})
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def build_manifest(res: AnalysisResult, inputs: list[InputRecord], *, mode: str,
                   protocol: dict[str, Any] | None = None) -> RunManifest:
    cfg = res.config
    public = cfg.privacy.mode == "public"
    fp = cfg.privacy.fingerprints_enabled()
    recs = []
    for r in inputs:
        r = r.model_copy()
        if public:
            r.path = None
            r.name = None if not cfg.privacy.reveal_group_names else r.name
        if not fp:
            r.sha256 = None
        recs.append(r)
    return RunManifest(
        calibcheck_version=__version__, python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.machine()}",
        dependencies=dependency_versions(), analysis_mode=mode,
        seed=cfg.uncertainty.seed if cfg.uncertainty.bootstrap else None,
        privacy_mode=cfg.privacy.mode, inputs=recs, filtering=res.dataset.exclusions,
        protocol=protocol or {}, results_digest=results_digest(res),
        volatile={"created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")})
