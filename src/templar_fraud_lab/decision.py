"""Cost-sensitive threshold search and the three-way decision.

Eq. 11 as we read it: t* = argmin_t C_FP * FP(t) + C_FN * FN(t) + C_R * R(t) over
the grid [0.001, 0.500] in steps of 0.001. Tie-break: the LOWEST threshold among
those attaining the minimum cost (deterministic). The cost matrix is a DEMO file
(`configs/demo_costs.yaml`); the paper does not state units or values.

Review routing: a row is `review` when the verifier failed on its escalation
path, or when its disagreement proxy exceeds `review_uncertainty`. Otherwise the
row is `fraud` if p >= t* and `legitimate` if p < t*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID_LOW, GRID_HIGH, GRID_STEP = 0.001, 0.500, 0.001
DECISION_FRAUD, DECISION_LEGIT, DECISION_REVIEW = "fraud", "legitimate", "review"


@dataclass(frozen=True)
class CostMatrix:
    c_fp: float
    c_fn: float
    c_review: float
    label: str = "demo"

    def __post_init__(self) -> None:
        if min(self.c_fp, self.c_fn, self.c_review) < 0:
            raise ValueError("costs must be non-negative")


def threshold_grid() -> np.ndarray:
    n = int(round((GRID_HIGH - GRID_LOW) / GRID_STEP)) + 1
    return np.round(GRID_LOW + GRID_STEP * np.arange(n), 3)


def total_cost(y: np.ndarray, decisions: np.ndarray, costs: CostMatrix) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    fp = int(((decisions == DECISION_FRAUD) & (y == 0)).sum())
    fn = int(((decisions == DECISION_LEGIT) & (y == 1)).sum())
    r = int((decisions == DECISION_REVIEW).sum())
    return {"fp": fp, "fn": fn, "review": r,
            "cost": costs.c_fp * fp + costs.c_fn * fn + costs.c_review * r}


def decide(p: np.ndarray, threshold: float, review_mask: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    out = np.where(p >= threshold, DECISION_FRAUD, DECISION_LEGIT).astype(object)
    out[np.asarray(review_mask, dtype=bool)] = DECISION_REVIEW
    return out


def review_mask(verifier_failed: np.ndarray, uncertainty: np.ndarray, review_uncertainty: float) -> np.ndarray:
    return np.asarray(verifier_failed, dtype=bool) | (np.asarray(uncertainty, dtype=np.float64) > review_uncertainty)


@dataclass(frozen=True)
class ThresholdSearch:
    threshold: float
    cost: float
    grid_costs: tuple[float, ...]
    n_ties: int


def search_threshold(p: np.ndarray, y: np.ndarray, review: np.ndarray, costs: CostMatrix) -> ThresholdSearch:
    """Exhaustive search on the fixed grid; review rows carry C_R at every threshold."""
    grid = threshold_grid()
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    review = np.asarray(review, dtype=bool)
    free = ~review
    r_cost = costs.c_review * int(review.sum())
    pos, neg = (y == 1) & free, (y == 0) & free
    costs_grid = np.empty(len(grid))
    for i, t in enumerate(grid):
        flagged = p >= t
        fp = int((flagged & neg).sum())
        fn = int((~flagged & pos).sum())
        costs_grid[i] = costs.c_fp * fp + costs.c_fn * fn + r_cost
    best = costs_grid.min()
    ties = np.flatnonzero(np.isclose(costs_grid, best))
    return ThresholdSearch(float(grid[ties[0]]), float(best), tuple(costs_grid.tolist()), int(len(ties)))
