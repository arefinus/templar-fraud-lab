"""Temporal feature derivation using only the past of each entity.

`time_since_last_card_event` for a row uses only strictly earlier rows of the
same entity, so it can be computed once over the whole frame without leaking
future information into any split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from templar_fraud_lab.schema import ENTITY_ID, TIMESTAMP

SECONDS_PER_DAY = 86_400
NO_HISTORY_SENTINEL = -1.0


def derive_temporal_features(frame: pd.DataFrame, epoch_offset_seconds: int = 0) -> pd.DataFrame:
    """Return a copy with `hour`, `dow` and `time_since_last_card_event`.

    `ts` is treated as seconds; `epoch_offset_seconds` shifts the origin so the
    fixture's first day is a Monday (dow 0). Rows must be sorted chronologically
    before calling so the "previous event" is genuinely earlier."""
    if not frame[TIMESTAMP].is_monotonic_increasing:
        raise ValueError("frame must be sorted by ts before deriving temporal features")
    out = frame.copy()
    seconds = out[TIMESTAMP].to_numpy(dtype=np.int64) + epoch_offset_seconds
    out["hour"] = (seconds % SECONDS_PER_DAY) // 3600
    out["dow"] = (seconds // SECONDS_PER_DAY) % 7
    prev = out.groupby(ENTITY_ID, sort=False)[TIMESTAMP].shift(1)
    gap = (out[TIMESTAMP] - prev).astype(float)
    out["time_since_last_card_event"] = gap.fillna(NO_HISTORY_SENTINEL)
    return out
