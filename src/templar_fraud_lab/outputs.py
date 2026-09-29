"""Writers for predictions, metrics and reliability outputs of a pipeline run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from templar_fraud_lab.decision import total_cost
from templar_fraud_lab.metrics import calibration_block, decision_metrics, ranking_metrics, route_shares
from templar_fraud_lab.pipeline import PipelineState
from templar_fraud_lab.schema import LABEL, ROW_ID, TIMESTAMP

PREDICTION_COLUMNS = [
    ROW_ID, TIMESTAMP, LABEL, "split", "p_z", "p_triage", "route_uncertainty", "route",
    "verification_status", "reason_codes", "p_escalation", "p_stacked", "disagreement",
    "p_calibrated", "p_isotonic", "review", "decision",
]


def predictions_frame(s: PipelineState, name: str) -> pd.DataFrame:
    f = s.frames[name]
    return pd.DataFrame({
        ROW_ID: f[ROW_ID].to_numpy(), TIMESTAMP: f[TIMESTAMP].to_numpy(), LABEL: s.y[name], "split": name,
        "p_z": s.p_z[name], "p_triage": s.p_triage[name], "route_uncertainty": s.u_route[name],
        "route": s.route[name], "verification_status": s.ver_status[name],
        "reason_codes": ["|".join(c) for c in s.ver_codes[name]], "p_escalation": s.p_esc[name],
        "p_stacked": s.p_stacked[name], "disagreement": s.disagreement[name], "p_calibrated": s.p_cal[name],
        "p_isotonic": s.p_iso[name], "review": s.review[name].astype(int), "decision": s.decision[name],
    })[PREDICTION_COLUMNS]


def write_predictions(s: PipelineState, path: Path) -> pd.DataFrame:
    frame = pd.concat([predictions_frame(s, n) for n in ("val", "test", "late")], ignore_index=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.6f")
    return frame


def slice_metrics(s: PipelineState, name: str, ece_bins: int) -> dict[str, Any]:
    y = s.y[name]
    ver = np.asarray(s.ver_status[name])
    out: dict[str, Any] = {
        "rows": int(len(y)), "fraud_rows": int(y.sum()), "prevalence": float(y.mean()) if len(y) else None,
        "ranking_calibrated": ranking_metrics(y, s.p_cal[name]),
        "ranking_stacked": ranking_metrics(y, s.p_stacked[name]),
        "ranking_triage_only": ranking_metrics(y, s.p_triage[name]),
        "ranking_encoder_only": ranking_metrics(y, s.p_z[name]),
        "decision": decision_metrics(y, s.decision[name]),
        "routes": route_shares(s.route[name], ver),
        "calibration_before_ot": calibration_block(y, s.p_stacked[name], ece_bins),
        "calibration_after_ot": calibration_block(y, s.p_cal[name], ece_bins),
        "calibration_isotonic_comparator": calibration_block(y, s.p_iso[name], ece_bins),
        "cost": total_cost(y, s.decision[name], s.costs),
        "decision_shares": {k: float((s.decision[name] == k).mean()) for k in ("fraud", "legitimate", "review")},
    }
    return out


def build_metrics(s: PipelineState) -> dict[str, Any]:
    ece_bins = s.cfg.calibration.ece_bins
    return {
        "evidence_status": "demo",
        "note": ("Computed by the pipeline on synthetic fixture v1, seed "
                 f"{s.cfg.seed}, mode {s.cfg.mode}. Not a benchmark result."),
        "seed": s.cfg.seed, "mode": s.cfg.mode,
        "threshold": s.threshold,
        "threshold_search": {k: v for k, v in s.info["threshold_search"].items() if k != "grid_costs"},
        "cost_matrix": {"c_fp": s.costs.c_fp, "c_fn": s.costs.c_fn, "c_review": s.costs.c_review, "label": s.costs.label},
        "stacking_weights": dict(zip(s.stacking_fit.sources, s.stacking_fit.weights)),
        "stacking_iterations": s.stacking_fit.iterations,
        "routing_rule": s.cfg.routing.rule,
        "routing_comparison_test_escalation_share": s.info["routing_comparison_test"],
        "saint": {"parameters": s.info["saint_parameters"], "train_seconds": s.info["saint_train_seconds"],
                  "best_epoch": s.saint_history.best_epoch, "epochs_run": s.saint_history.epochs_run,
                  "best_val_ap": s.saint_history.best_val_ap},
        "latency_ms_per_row": {"fast": s.info.get("latency_fast_ms_per_row"),
                               "escalation": s.info.get("latency_escalation_ms_per_row"),
                               "note": "wall-clock on the machine that ran the demo; synthetic fixture; not a benchmark"},
        "template_bank": {"size": len(s.bank), "origin": s.bank.origin},
        "unknown_categories_in_test": s.info["unknown_categories_in_test"],
        "feature_count": len(s.info["feature_names"]),
        "slices": {n: slice_metrics(s, n, ece_bins) for n in ("val", "test", "late")},
    }


def write_metrics(s: PipelineState, path: Path) -> dict[str, Any]:
    metrics = build_metrics(s)
    path.write_text(json.dumps(metrics, indent=2, default=_default) + "\n", encoding="utf-8")
    return metrics


def write_reliability(metrics: dict[str, Any], path: Path) -> None:
    test = metrics["slices"]["test"]
    payload = {
        "schemaVersion": 1, "mode": "synthetic-demo", "slice": "test",
        "note": metrics["note"],
        "before_ot": test["calibration_before_ot"],
        "after_ot": test["calibration_after_ot"],
        "isotonic_comparator": test["calibration_isotonic_comparator"],
    }
    path.write_text(json.dumps(payload, indent=2, default=_default) + "\n", encoding="utf-8")


def write_threshold_curve(s: PipelineState, path: Path) -> None:
    from templar_fraud_lab.decision import threshold_grid

    grid = threshold_grid()
    costs = s.info["threshold_search"]["grid_costs"]
    pd.DataFrame({"threshold": grid, "val_cost": costs}).to_csv(path, index=False, lineterminator="\n")


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and o != o:
        return None
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")
