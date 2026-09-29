"""Re-evaluate a saved predictions file with the stated protocol.

Protocol: metrics on the `test` slice and the `late` slice separately; ranking
metrics on the calibrated score; decision metrics on the recorded three-way
decision; cost from the supplied cost matrix; ECE with 10 equal-width bins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from templar_fraud_lab.config import load_costs
from templar_fraud_lab.decision import total_cost
from templar_fraud_lab.metrics import calibration_block, decision_metrics, ranking_metrics, route_shares
from templar_fraud_lab.outputs import PREDICTION_COLUMNS
from templar_fraud_lab.schema import LABEL

PROTOCOL = ("chronological 70/15/15 split; late = final 10% rows; ranking on p_calibrated; "
            "decision metrics on recorded decisions; ECE 10 equal-width bins; cost from supplied matrix")


def evaluate_predictions(predictions: Path, costs_path: Path, out: Path) -> dict[str, Any]:
    frame = pd.read_csv(predictions)
    missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"predictions file missing columns {missing}")
    costs = load_costs(costs_path)
    slices: dict[str, Any] = {}
    for name in ("test", "late"):
        part = frame[frame["split"] == name]
        if part.empty:
            continue
        y = part[LABEL].to_numpy()
        d = part["decision"].to_numpy(dtype=object)
        slices[name] = {
            "rows": int(len(part)), "fraud_rows": int(y.sum()),
            "ranking_calibrated": ranking_metrics(y, part["p_calibrated"].to_numpy()),
            "decision": decision_metrics(y, d),
            "routes": route_shares(part["route"].to_numpy(dtype=object), part["verification_status"].to_numpy(dtype=object)),
            "calibration_after_ot": calibration_block(y, part["p_calibrated"].to_numpy()),
            "cost": total_cost(y, d, costs),
        }
    report = {"evidence_status": "demo", "protocol": PROTOCOL, "rows": int(len(frame)),
              "cost_matrix": {"c_fp": costs.c_fp, "c_fn": costs.c_fn, "c_review": costs.c_review, "label": costs.label},
              "slices": slices}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=_np) + "\n", encoding="utf-8")
    return report


def _np(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    raise TypeError(type(o).__name__)
