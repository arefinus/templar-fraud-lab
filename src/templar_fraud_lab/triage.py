"""CatBoost triage over [z, x] and the fast/escalation routing rule.

Two routing rules are implemented behind a config switch because the paper
gives both (Eq. 5 and Section 2.7) without reconciling them:

  eq5:  fast if p_triage < gamma_low AND u < t_u, else escalation.
  band: fast if p_triage < low OR p_triage > high, escalation inside [low, high].

Uncertainty proxy u for eq5 (config `routing.uncertainty`):
  rank_disagreement (default): |F_triage(p_triage) - F_z(p_z)| where F_* are
      empirical CDFs of each score fitted on TRAINING rows. The two sources
      available before escalation are compared on the rank scale, so a poorly
      calibrated encoder does not force every row into escalation.
  disagreement: |p_triage - p_z| on the raw probability scale.
  margin: 1 - 2|p_triage - 0.5|; note that for p < 0.5 this equals 2p, so the
      eq5 rule degenerates to p_triage < min(gamma_low, t_u / 2).

The stacking disagreement of Eq. 9 needs p_escalation, which does not exist
before routing, so it cannot be the routing uncertainty; see audit A04.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

ROUTE_FAST, ROUTE_ESCALATION = "fast", "escalation"
UNCERTAINTY_MODES = ("rank_disagreement", "disagreement", "margin")


@dataclass(frozen=True)
class TriageConfig:
    iterations: int = 200
    depth: int = 4
    learning_rate: float = 0.1
    l2_leaf_reg: float = 5.0
    random_strength: float = 0.3
    bagging_temperature: float = 0.5
    seed: int = 42


@dataclass(frozen=True)
class RoutingConfig:
    rule: str = "eq5"
    gamma_low: float = 0.18
    gamma_high: float = 0.72
    t_u: float = 0.25
    uncertainty: str = "rank_disagreement"

    def __post_init__(self) -> None:
        if self.rule not in {"eq5", "band"}:
            raise ValueError("rule must be 'eq5' or 'band'")
        if self.uncertainty not in UNCERTAINTY_MODES:
            raise ValueError(f"uncertainty must be one of {UNCERTAINTY_MODES}")
        if self.rule == "band" and self.gamma_high - self.gamma_low < 0.20:
            raise ValueError("band thresholds must be separated by at least 0.20")


@dataclass(frozen=True)
class ScoreReference:
    """Sorted training scores used as empirical CDFs for rank disagreement."""

    triage_sorted: np.ndarray
    z_sorted: np.ndarray

    @staticmethod
    def fit(p_triage_train: np.ndarray, p_z_train: np.ndarray) -> "ScoreReference":
        return ScoreReference(np.sort(np.asarray(p_triage_train, dtype=np.float64)),
                              np.sort(np.asarray(p_z_train, dtype=np.float64)))


def _ecdf(sorted_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_ref, values, side="right") / max(len(sorted_ref), 1)


def class_weight_ratio(y: np.ndarray) -> float:
    pos = float(np.asarray(y).sum())
    if pos == 0:
        raise ValueError("training fold has no positive rows; cannot set class weight")
    return (len(y) - pos) / pos


def train_triage(z: np.ndarray, x: np.ndarray, y: np.ndarray, cfg: TriageConfig) -> CatBoostClassifier:
    features = np.hstack([z, x]).astype(np.float32)
    model = CatBoostClassifier(
        iterations=cfg.iterations, depth=cfg.depth, learning_rate=cfg.learning_rate,
        l2_leaf_reg=cfg.l2_leaf_reg, random_strength=cfg.random_strength,
        bagging_temperature=cfg.bagging_temperature, random_seed=cfg.seed,
        scale_pos_weight=class_weight_ratio(y), boosting_type="Ordered",
        verbose=False, allow_writing_files=False, thread_count=1,
    )
    model.fit(features, np.asarray(y).astype(int))
    return model


def predict_triage(model: CatBoostClassifier, z: np.ndarray, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.hstack([z, x]).astype(np.float32))[:, 1]


def uncertainty(p_triage: np.ndarray, p_z: np.ndarray, cfg: RoutingConfig,
                reference: ScoreReference | None = None) -> np.ndarray:
    p_triage = np.asarray(p_triage, dtype=np.float64)
    p_z = np.asarray(p_z, dtype=np.float64)
    if cfg.uncertainty == "rank_disagreement":
        if reference is None:
            raise ValueError("rank_disagreement needs a ScoreReference fitted on training rows")
        return np.abs(_ecdf(reference.triage_sorted, p_triage) - _ecdf(reference.z_sorted, p_z))
    if cfg.uncertainty == "disagreement":
        return np.abs(p_triage - p_z)
    return 1.0 - 2.0 * np.abs(p_triage - 0.5)


def route(p_triage: np.ndarray, u: np.ndarray, cfg: RoutingConfig) -> np.ndarray:
    p = np.asarray(p_triage, dtype=np.float64)
    if cfg.rule == "eq5":
        fast = (p < cfg.gamma_low) & (np.asarray(u) < cfg.t_u)
    else:
        fast = (p < cfg.gamma_low) | (p > cfg.gamma_high)
    return np.where(fast, ROUTE_FAST, ROUTE_ESCALATION).astype(object)


def save_triage(model: CatBoostClassifier, path: str | Path) -> None:
    model.save_model(str(path), format="json")


def load_triage(path: str | Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path), format="json")
    return model
