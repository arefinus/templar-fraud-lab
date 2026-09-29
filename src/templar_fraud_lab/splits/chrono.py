"""Chronological 70/15/15 split plus a final-10% late slice.

The late slice is a same-dataset chronological slice, not an external set. Whether
it overlaps the 15% test split is an ambiguity in the source paper; we measure
and export the intersection rather than assume it (`late_slice_overlap.json`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from templar_fraud_lab.schema import ROW_ID, TIMESTAMP


@dataclass(frozen=True)
class ChronoSplit:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    late_ids: tuple[str, ...]
    fractions: tuple[float, float, float]
    late_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "fractions": list(self.fractions),
            "late_fraction": self.late_fraction,
            "counts": {
                "train": len(self.train_ids), "val": len(self.val_ids),
                "test": len(self.test_ids), "late": len(self.late_ids),
            },
            "train": list(self.train_ids), "val": list(self.val_ids),
            "test": list(self.test_ids), "late": list(self.late_ids),
        }


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values([TIMESTAMP, ROW_ID], kind="mergesort")


def chronological_split(
    frame: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    late_fraction: float = 0.10,
) -> ChronoSplit:
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if not 0 < late_fraction < 1:
        raise ValueError("late_fraction must be in (0, 1)")
    ordered = _ordered(frame)
    ids = ordered[ROW_ID].to_numpy()
    n = len(ids)
    n_train = int(np.floor(fractions[0] * n))
    n_val = int(np.floor(fractions[1] * n))
    n_late = int(np.ceil(late_fraction * n))
    return ChronoSplit(
        train_ids=tuple(ids[:n_train]),
        val_ids=tuple(ids[n_train:n_train + n_val]),
        test_ids=tuple(ids[n_train + n_val:]),
        late_ids=tuple(ids[n - n_late:]),
        fractions=fractions,
        late_fraction=late_fraction,
    )


def assert_strict_chronology(frame: pd.DataFrame, split: ChronoSplit) -> None:
    """Every train ts <= every val ts <= every test ts. Raises on violation."""
    ts = frame.set_index(ROW_ID)[TIMESTAMP]
    tr, va, te = (ts.loc[list(ids)] for ids in (split.train_ids, split.val_ids, split.test_ids))
    if len(tr) and len(va) and tr.max() > va.min():
        raise AssertionError("train contains a row later than the earliest validation row")
    if len(va) and len(te) and va.max() > te.min():
        raise AssertionError("validation contains a row later than the earliest test row")


def late_slice_overlap(split: ChronoSplit) -> dict[str, object]:
    late, test, val, train = (set(split.late_ids), set(split.test_ids),
                              set(split.val_ids), set(split.train_ids))
    inter = late & test
    return {
        "late_rows": len(late),
        "test_rows": len(test),
        "late_in_test": len(inter),
        "late_in_val": len(late & val),
        "late_in_train": len(late & train),
        "share_of_late_inside_test": (len(inter) / len(late)) if late else 0.0,
        "share_of_test_that_is_late": (len(inter) / len(test)) if test else 0.0,
        "interpretation": (
            "The late slice is the final 10% of rows in time; the test split is the final "
            "15%. With these fractions the late slice is a strict subset of the test split, "
            "so late-slice metrics are NOT independent of test metrics. Measured, not assumed."
        ),
    }


def write_overlap_report(split: ChronoSplit, path) -> None:
    from pathlib import Path

    Path(path).write_text(json.dumps(late_slice_overlap(split), indent=2) + "\n", encoding="utf-8")


def split_manifest_hash(split: ChronoSplit) -> str:
    payload = json.dumps(split.as_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
