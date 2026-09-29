"""Smoothed target encoding, out-of-fold in forward order.

For training rows the encoding of row i uses only the labels of strictly earlier
rows of the same category (expanding window, shrunk to the prior with strength
`m`). This is the strictest reading of "out-of-fold in forward order": no row sees
its own label or any later label. Validation and test rows use statistics from
the whole training frame. The paper does not say whether its OOF scheme is
block-wise or expanding; see docs/paper-implementation-audit.md, ambiguity 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from templar_fraud_lab.schema import MISSING_TOKEN


def _keys(series: pd.Series) -> np.ndarray:
    return series.astype(object).where(series.notna(), MISSING_TOKEN).astype(str).to_numpy()


def forward_oof_target_encode(
    categories: pd.Series, labels: pd.Series, smoothing: float = 20.0
) -> tuple[np.ndarray, dict[str, tuple[int, int]], float]:
    """Return (encoded values for training rows, per-category (count, positives), prior).

    `categories` and `labels` must already be in chronological order."""
    keys = _keys(categories)
    y = labels.to_numpy(dtype=np.float64)
    prior = float(y.mean()) if len(y) else 0.0
    counts: dict[str, int] = {}
    positives: dict[str, int] = {}
    encoded = np.empty(len(y), dtype=np.float64)
    for i, (k, yi) in enumerate(zip(keys, y)):
        n, s = counts.get(k, 0), positives.get(k, 0)
        encoded[i] = (s + smoothing * prior) / (n + smoothing)
        counts[k] = n + 1
        positives[k] = s + int(yi)
    stats = {k: (counts[k], positives.get(k, 0)) for k in counts}
    return encoded, stats, prior


def target_encode_apply(
    categories: pd.Series,
    stats: dict[str, tuple[int, int]],
    prior: float,
    smoothing: float = 20.0,
) -> np.ndarray:
    keys = _keys(categories)
    out = np.empty(len(keys), dtype=np.float64)
    for i, k in enumerate(keys):
        n, s = stats.get(k, (0, 0))
        out[i] = (s + smoothing * prior) / (n + smoothing)
    return out
