"""One-dimensional optimal-transport calibration, plus isotonic as a comparator.

What the OT map solves. Let S be the empirical distribution of validation scores
and R the empirical distribution of binned label rates (validation rows are cut
into K quantile bins with at least `min_support` rows each; bin k contributes its
fraud rate r_k with mass n_k/n). The 1-D optimal transport map from S to R under
any convex cost is the monotone quantile-matching map T = F_R^{-1} o F_S. We
realise it at K knots (knot k: x = the bin's score quantile, y = the k-th SMALLEST
bin rate, i.e. rates sorted so the map is monotone) and interpolate piecewise-
linearly, clamped to [0, 1]. The mean of T(s) over validation equals the
validation prevalence exactly, because T pushes S onto R.

How it differs from isotonic regression. Isotonic minimises squared error to the
labels under a monotonicity constraint and POOLS adjacent violating bins into
one flat step. Quantile matching never pools: when bin rates are out of order,
it REARRANGES them (assigns the sorted rates to the ordered bins) so the output
distribution of calibrated scores matches R. Isotonic preserves the pooled
conditional means; OT preserves the marginal distribution of rates. On a
perfectly ranked validation set the two coincide.

The paper's exact construction is not stated beyond "1-D optimal transport map,
256 quantile knots, monotone piecewise-linear, at least 200 observations per
interval"; see docs/paper-implementation-audit.md, ambiguity 8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class OtCalibrator:
    knots_x: tuple[float, ...]
    knots_y: tuple[float, ...]
    n_bins: int
    min_support: int

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        out = np.interp(s, self.knots_x, self.knots_y)
        return np.clip(out, 0.0, 1.0)

    def as_dict(self) -> dict[str, object]:
        return {"method": "ot-quantile-matching", "knots_x": list(self.knots_x),
                "knots_y": list(self.knots_y), "n_bins": self.n_bins, "min_support": self.min_support}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def _bin_edges(n: int, k: int, min_support: int) -> np.ndarray:
    k_eff = max(1, min(k, n // max(min_support, 1)))
    return np.linspace(0, n, k_eff + 1, dtype=int)


def fit_ot_calibrator(scores: np.ndarray, labels: np.ndarray, *, n_knots: int = 256,
                      min_support: int = 200) -> OtCalibrator:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(s) != len(y) or len(s) == 0:
        raise ValueError("scores and labels must be non-empty and the same length")
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    edges = _bin_edges(len(s), n_knots, min_support)
    xs, rates = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xs.append(float(np.median(s_sorted[a:b])))
        rates.append(float(y_sorted[a:b].mean()))
    xs_arr = np.array(xs)
    # ensure strictly increasing x for interpolation (ties collapse to one knot)
    keep = np.concatenate([[True], np.diff(xs_arr) > 0])
    xs_arr = xs_arr[keep]
    rates_arr = np.array(rates)[keep]
    ys_sorted = np.sort(rates_arr)  # quantile matching: k-th bin gets k-th smallest rate
    return OtCalibrator(tuple(xs_arr.tolist()), tuple(ys_sorted.tolist()), len(xs_arr), min_support)


@dataclass(frozen=True)
class IsotonicCalibrator:
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        return np.clip(np.interp(s, self.x_thresholds, self.y_thresholds), 0.0, 1.0)


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, labels)
    return IsotonicCalibrator(tuple(iso.X_thresholds_.tolist()), tuple(iso.y_thresholds_.tolist()))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
                               ) -> tuple[float, list[dict[str, float]]]:
    """Equal-width ECE with per-bin counts. Returns (ece, bins)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 0, 1)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        count = int(m.sum())
        if count == 0:
            bins.append({"lower": float(edges[b]), "upper": float(edges[b + 1]), "count": 0,
                         "mean_predicted": None, "observed_rate": None})
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += count / len(p) * abs(conf - acc)
        bins.append({"lower": float(edges[b]), "upper": float(edges[b + 1]), "count": count,
                     "mean_predicted": conf, "observed_rate": acc})
    return float(ece), bins
