from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibcheck.config import AnalysisConfig, resolve_config


@pytest.fixture
def cfg_factory():
    def make(probs=("p",), **kw) -> AnalysisConfig:
        base = {"columns": {"label": "y", "probs": list(probs), "id": "id"}}
        return resolve_config(base, kw)
    return make


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["001", "002", "003", "004"],
        "y": [1, 0, 1, 0],
        "p": [0.9, 0.2, 0.6, 0.4],
    })


@pytest.fixture
def write_csv(tmp_path: Path):
    def w(name: str, text: str) -> Path:
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p
    return w


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
