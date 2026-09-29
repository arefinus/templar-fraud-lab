"""Dataset adapters. Real datasets are never shipped; adapters read from a data
root supplied by the user and raise actionable errors when the data is absent."""

from __future__ import annotations

from templar_fraud_lab.data.baf import load_baf
from templar_fraud_lab.data.ieee_cis import load_ieee_cis
from templar_fraud_lab.data.synthetic import (
    FIXTURE_VERSION,
    generate_fixture,
    load_fixture,
)
from templar_fraud_lab.data.temporal import derive_temporal_features

__all__ = [
    "FIXTURE_VERSION",
    "derive_temporal_features",
    "generate_fixture",
    "load_baf",
    "load_fixture",
    "load_ieee_cis",
]
