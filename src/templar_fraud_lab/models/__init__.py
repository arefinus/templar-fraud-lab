"""Neural encoder (SAINT-style) and its trainer."""

from __future__ import annotations

from templar_fraud_lab.models.saint import SaintConfig, SaintEncoder, historical_context_mask
from templar_fraud_lab.models.train_saint import (
    TrainResult,
    encode_in_context_windows,
    load_checkpoint,
    save_checkpoint,
    train_saint,
)

__all__ = [
    "SaintConfig", "SaintEncoder", "TrainResult", "encode_in_context_windows",
    "historical_context_mask", "load_checkpoint", "save_checkpoint", "train_saint",
]
