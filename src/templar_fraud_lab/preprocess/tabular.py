"""Train-only tabular preprocessing.

Steps, mirroring the paper's Table 2 as read by us: median imputation of numeric
columns (train medians), an explicit missing token for categoricals, one-hot for
low-cardinality categoricals, frequency encoding for high-cardinality ones, and a
RobustScaler fitted on training rows. Unknown categories at transform time map to
all-zero one-hot columns and frequency 0. The fitted state is a plain dict so it
can be hashed and compared (tests assert it is unchanged when test rows change).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from templar_fraud_lab.schema import MISSING_TOKEN, FeatureSchema


class TabularPreprocessor:
    def __init__(self, schema: FeatureSchema, extra_numeric: tuple[str, ...] = ()) -> None:
        self.schema = schema
        self.extra_numeric = tuple(extra_numeric)
        self.medians: dict[str, float] = {}
        self.levels: dict[str, list[str]] = {}
        self.frequencies: dict[str, dict[str, float]] = {}
        self.numeric_ranges: dict[str, tuple[float, float]] = {}
        self.scaler: RobustScaler | None = None
        self.feature_names: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------ fitting
    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return self.schema.numeric + self.extra_numeric

    def fit(self, train: pd.DataFrame) -> "TabularPreprocessor":
        if len(train) == 0:
            raise ValueError("cannot fit preprocessor on an empty training frame")
        for col in self.numeric_columns:
            series = pd.to_numeric(train[col], errors="coerce")
            med = float(series.median()) if series.notna().any() else 0.0
            self.medians[col] = med
            self.numeric_ranges[col] = (float(series.min()), float(series.max()))
        for col in self.schema.low_card:
            vals = train[col].astype(object).where(train[col].notna(), MISSING_TOKEN)
            self.levels[col] = sorted(set(vals.astype(str)))
        for col in self.schema.high_card:
            vals = train[col].astype(object).where(train[col].notna(), MISSING_TOKEN).astype(str)
            counts = vals.value_counts()
            self.frequencies[col] = {k: float(v) / len(vals) for k, v in counts.items()}
        self.feature_names = self._build_feature_names()
        unscaled = self._encode(train)
        self.scaler = RobustScaler().fit(unscaled)
        self._fitted = True
        return self

    def _build_feature_names(self) -> list[str]:
        names = list(self.numeric_columns)
        for col in self.schema.low_card:
            names.extend(f"{col}={lvl}" for lvl in self.levels[col])
        names.extend(f"{col}__freq" for col in self.schema.high_card)
        return names

    # ---------------------------------------------------------------- transform
    def _encode(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for col in self.numeric_columns:
            series = pd.to_numeric(frame[col], errors="coerce").fillna(self.medians[col])
            blocks.append(series.to_numpy(dtype=np.float64)[:, None])
        for col in self.schema.low_card:
            vals = frame[col].astype(object).where(frame[col].notna(), MISSING_TOKEN).astype(str)
            levels = self.levels[col]
            onehot = np.zeros((len(frame), len(levels)), dtype=np.float64)
            index = {lvl: i for i, lvl in enumerate(levels)}
            for r, v in enumerate(vals.to_numpy()):
                j = index.get(v)
                if j is not None:
                    onehot[r, j] = 1.0
            blocks.append(onehot)
        for col in self.schema.high_card:
            vals = frame[col].astype(object).where(frame[col].notna(), MISSING_TOKEN).astype(str)
            freq = self.frequencies[col]
            blocks.append(np.array([freq.get(v, 0.0) for v in vals], dtype=np.float64)[:, None])
        return np.hstack(blocks)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.scaler is None:
            raise RuntimeError("preprocessor must be fitted before transform")
        return self.scaler.transform(self._encode(frame)).astype(np.float32)

    def unknown_categories(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        """Categories present in `frame` but absent from the training levels."""
        out: dict[str, list[str]] = {}
        for col in self.schema.low_card:
            vals = set(frame[col].dropna().astype(str))
            unknown = sorted(vals - set(self.levels[col]))
            if unknown:
                out[col] = unknown
        return out

    # -------------------------------------------------------------------- state
    def state(self) -> dict[str, object]:
        if self.scaler is None:
            raise RuntimeError("not fitted")
        return {
            "medians": self.medians,
            "levels": self.levels,
            "frequencies": {c: dict(sorted(f.items())) for c, f in self.frequencies.items()},
            "numeric_ranges": {c: list(r) for c, r in self.numeric_ranges.items()},
            "scaler_center": self.scaler.center_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "feature_names": self.feature_names,
        }

    def state_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode()).hexdigest()
