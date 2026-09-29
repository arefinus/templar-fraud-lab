"""Evaluation metrics. Both average precision and trapezoidal PR-AUC are reported
because the two differ and the paper does not say which "AUC-PR" it used."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from templar_fraud_lab.calibration import expected_calibration_error
from templar_fraud_lab.decision import DECISION_FRAUD, DECISION_LEGIT, DECISION_REVIEW


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auroc": None, "average_precision": None, "pr_auc_trapezoidal": None}
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "pr_auc_trapezoidal": float(auc(recall, precision)),
    }


def decision_metrics(y: np.ndarray, decisions: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y).astype(int)
    d = np.asarray(decisions)
    scored = d != DECISION_REVIEW
    tp = int(((d == DECISION_FRAUD) & (y == 1)).sum())
    fp = int(((d == DECISION_FRAUD) & (y == 0)).sum())
    fn = int(((d == DECISION_LEGIT) & (y == 1)).sum())
    tn = int(((d == DECISION_LEGIT) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1, "scored_rows": int(scored.sum()),
            "review_rows": int((~scored).sum())}


def route_shares(routes: np.ndarray, verifier_status: np.ndarray) -> dict[str, float]:
    routes = np.asarray(routes)
    n = max(len(routes), 1)
    return {
        "fast_share": float((routes == "fast").mean()) if len(routes) else 0.0,
        "escalation_share": float((routes == "escalation").mean()) if len(routes) else 0.0,
        "verifier_fail_share": float((np.asarray(verifier_status) == "fail").sum() / n),
        "verifier_pass_share": float((np.asarray(verifier_status) == "pass").sum() / n),
    }


def calibration_block(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, object]:
    ece, bins = expected_calibration_error(p, y, n_bins)
    return {"ece": ece, "n_bins": n_bins, "bins": bins}
