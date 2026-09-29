from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from templar_fraud_lab.data.synthetic import generate_fixture  # noqa: E402
from templar_fraud_lab.data.temporal import derive_temporal_features  # noqa: E402
from templar_fraud_lab.schema import ROW_ID, TIMESTAMP  # noqa: E402


@pytest.fixture(autouse=True)
def _repo_cwd():
    """Configs use repo-relative paths; run every test from the repository root."""
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.fixture(scope="session")
def small_frame():
    frame = generate_fixture(n_rows=1500, seed=1)
    frame = frame.sort_values([TIMESTAMP, ROW_ID], kind="mergesort").reset_index(drop=True)
    return derive_temporal_features(frame)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
