"""Forward-chaining (expanding-window) k-fold splitter for time-ordered rows.

Rows are cut into k+1 contiguous chronological blocks; fold j trains on blocks
0..j and validates on block j+1. No validation row ever precedes a training row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    index: int
    train_positions: np.ndarray
    val_positions: np.ndarray


def forward_chaining_folds(n_rows: int, n_splits: int = 5, min_train: int | None = None) -> list[Fold]:
    """Positions refer to a frame already sorted by (ts, row_id)."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if n_rows < n_splits + 1:
        raise ValueError(f"need at least {n_splits + 1} rows for {n_splits} folds, got {n_rows}")
    edges = np.linspace(0, n_rows, n_splits + 2, dtype=int)
    folds: list[Fold] = []
    for j in range(n_splits):
        train_end, val_end = edges[j + 1], edges[j + 2]
        if min_train is not None and train_end < min_train:
            continue
        folds.append(Fold(index=j,
                          train_positions=np.arange(0, train_end),
                          val_positions=np.arange(train_end, val_end)))
    if not folds:
        raise ValueError("min_train excluded every fold")
    return folds


def assert_forward_only(folds: list[Fold], ts: np.ndarray) -> None:
    for fold in folds:
        if ts[fold.train_positions].max() > ts[fold.val_positions].min():
            raise AssertionError(f"fold {fold.index}: a training row is later than a validation row")
