"""Paper-implementation audit summary: the eleven ambiguities, each with what this
repository does and, where a run exists, the measured quantity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AMBIGUITIES: tuple[dict[str, str], ...] = (
    {"id": "A01", "title": "Temporal overlap of the final-10% slice with the 15% test split",
     "repo": "Measured and exported to late_slice_overlap.json; late slice is a subset of test here."},
    {"id": "A02", "title": "Calendar and entity availability in BAF (month only; no card/IP/device)",
     "repo": "BAF adapter uses month as the only chronological key and no entity graph."},
    {"id": "A03", "title": "Calibration sample count (100,000 stated vs 88,581 = 15% of IEEE-CIS)",
     "repo": "Calibrator is fitted on the validation split only; the count is whatever the split yields."},
    {"id": "A04", "title": "Routing rule: Eq. 5 (p < gamma_low and u < T_u) vs low/high band thresholds",
     "repo": "Both implemented behind routing.rule; default eq5; escalation shares under both are exported."},
    {"id": "A05", "title": "Fit scope of stacking and calibration (validation split vs pooled OOF)",
     "repo": "Both fitted on the 15% validation split, never on test; no pooled OOF."},
    {"id": "A06", "title": "Intersample attention leakage across rows",
     "repo": "Historical-context mask forbids attention to later timestamps; windows are chronological."},
    {"id": "A07", "title": "Out-of-fold target encoding scheme",
     "repo": "Expanding-window forward OOF on training rows; full-train statistics for val/test."},
    {"id": "A08", "title": "What the 1-D optimal transport calibration is",
     "repo": "Quantile matching of binned label rates with K knots and min support; isotonic as comparator."},
    {"id": "A09", "title": "Template search (MCTS with K=64 over 50,000 rows)",
     "repo": "Hand-written 16-template demo bank plus a small MCTS that appends a few templates."},
    {"id": "A10", "title": "Cost matrix units and values",
     "repo": "Demo cost matrix in configs/demo_costs.yaml; the paper does not state units."},
    {"id": "A11", "title": "Upstream substitutes (card key, graph features, FraudGT baseline)",
     "repo": "Community card key for IEEE-CIS; no graph features; no FraudGT; documented as not implemented."},
)


def build_audit(output_dir: Path) -> dict[str, Any]:
    measured: dict[str, Any] = {}
    overlap = output_dir / "late_slice_overlap.json"
    metrics = output_dir / "metrics.json"
    if overlap.exists():
        measured["A01"] = json.loads(overlap.read_text(encoding="utf-8"))
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        measured["A04"] = m.get("routing_comparison_test_escalation_share")
        measured["A05"] = {"calibration_fit_rows": m["slices"]["val"]["rows"], "stacking_weights": m.get("stacking_weights")}
        measured["A08"] = {"ece_test_before_ot": m["slices"]["test"]["calibration_before_ot"]["ece"],
                           "ece_test_after_ot": m["slices"]["test"]["calibration_after_ot"]["ece"],
                           "ece_test_isotonic": m["slices"]["test"]["calibration_isotonic_comparator"]["ece"]}
        measured["A09"] = m.get("template_bank")
        measured["A10"] = {"cost_matrix": m.get("cost_matrix"), "threshold": m.get("threshold")}
    items = []
    for amb in AMBIGUITIES:
        items.append({**amb, "evidence_status": "demo" if amb["id"] in measured else "documented-only",
                      "measured": measured.get(amb["id"])})
    return {"schemaVersion": 1, "evidence_status": "demo",
            "note": "Measured values come from the synthetic fixture demo run and are not benchmark results.",
            "ambiguities": items, "long_form": "docs/paper-implementation-audit.md"}


def format_audit(audit: dict[str, Any]) -> str:
    lines = ["Paper implementation audit (short form; long form in docs/paper-implementation-audit.md)", ""]
    for item in audit["ambiguities"]:
        lines.append(f"{item['id']}  {item['title']}")
        lines.append(f"      repo: {item['repo']}")
        if item["measured"] is not None:
            lines.append(f"      measured ({item['evidence_status']}): {json.dumps(item['measured'], default=str)[:220]}")
    return "\n".join(lines)
