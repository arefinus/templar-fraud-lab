"""Browser demo trace: a handful of test transactions walked stage by stage.

Shape: {schemaVersion, mode: "synthetic-demo", scenarioId, steps: [{id, stage,
inputRefs, outputRefs, status, reasonCodes}]}. Each step also carries a
`transactionId` and a small `detail` object; consumers may ignore them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from templar_fraud_lab.pipeline import PipelineState
from templar_fraud_lab.schema import ROW_ID

SCHEMA_VERSION = 1


def _f(x: Any) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 6)


def _pick(status: np.ndarray, route: np.ndarray, want: str, k: int, rng: np.random.Generator) -> list[int]:
    if want == "fast":
        pool = np.flatnonzero(route == "fast")
    elif want == "pass":
        pool = np.flatnonzero(status == "pass")
    else:
        pool = np.flatnonzero(status == "fail")
    if len(pool) == 0:
        return []
    return sorted(int(i) for i in rng.choice(pool, size=min(k, len(pool)), replace=False))


def select_sample_rows(s: PipelineState, seed: int = 7) -> list[int]:
    """Three fast rows, three verifier-pass rows (preferring ones decided fraud so the
    trace shows every decision class), two verifier-fail rows."""
    rng = np.random.default_rng(seed)
    status = np.asarray(s.ver_status["test"])
    route = s.route["test"]
    fraud_pass = [int(i) for i in np.flatnonzero((status == "pass") & (s.decision["test"] == "fraud"))]
    picks = _pick(status, route, "fast", 3, rng)
    picks += sorted(int(i) for i in rng.choice(fraud_pass, size=min(2, len(fraud_pass)), replace=False)) if fraud_pass else []
    picks += _pick(status, route, "pass", 3 - min(2, len(fraud_pass)), rng)
    picks += _pick(status, route, "fail", 2, rng)
    return list(dict.fromkeys(picks))


def _steps_for_row(s: PipelineState, i: int) -> list[dict[str, Any]]:
    f = s.frames["test"]
    tx = str(f[ROW_ID].iloc[i])
    route = str(s.route["test"][i])
    ver = str(s.ver_status["test"][i])
    codes = list(s.ver_codes["test"][i])
    decision = str(s.decision["test"][i])
    steps: list[dict[str, Any]] = []

    def add(stage: str, inputs: list[str], outputs: list[str], status: str, reason: list[str], detail: dict[str, Any]) -> None:
        steps.append({"id": f"{tx}/{stage}", "transactionId": tx, "stage": stage, "inputRefs": inputs,
                      "outputRefs": outputs, "status": status, "reasonCodes": reason, "detail": detail})

    add("ingest", [f"tx:{tx}"], [f"features:{tx}"], "ok", [],
        {"amount": _f(f["amount"].iloc[i]), "merchant_category": str(f["merchant_category"].iloc[i]),
         "device_type": str(f["device_type"].iloc[i]), "country_match": int(f["country_match"].iloc[i])})
    add("preprocess", [f"features:{tx}"], [f"x:{tx}"], "ok", [], {"feature_count": int(s.x["test"].shape[1])})
    add("encoder", [f"x:{tx}", "context:earlier-or-equal-rows"], [f"z:{tx}", f"p_z:{tx}"], "ok", [],
        {"p_z": _f(s.p_z["test"][i]), "context_policy": "historical (ts_j <= ts_i)"})
    add("triage", [f"z:{tx}", f"x:{tx}"], [f"p_triage:{tx}"], "ok", [], {"p_triage": _f(s.p_triage["test"][i])})
    add("routing", [f"p_triage:{tx}", f"p_z:{tx}"], [f"route:{tx}"], "ok", [f"ROUTE_{route.upper()}"],
        {"rule": s.cfg.routing.rule, "uncertainty": _f(s.u_route["test"][i]), "route": route})
    if route == "escalation":
        fired = [r.template_id for r in s.rationales["test"][i]]
        add("templates", [f"features:{tx}", "bank:template_bank.json"], [f"rationales:{tx}"],
            "ok" if fired else "empty", [c for c in codes if not c.startswith(("SCHEMA", "SUPPORT", "RANGE", "CONSISTENCY", "VERIFIED"))],
            {"fired_templates": fired})
        ver_codes = [c for c in codes if c.startswith(("SCHEMA", "SUPPORT", "RANGE", "CONSISTENCY", "VERIFIED"))]
        add("verifier", [f"rationales:{tx}"], [f"p_escalation:{tx}"] if ver == "pass" else [f"review_flag:{tx}"],
            ver if ver in ("pass", "fail") else "skipped", ver_codes,
            {"verification_status": ver, "p_escalation": _f(s.p_esc["test"][i])})
    add("stacking", [f"p_triage:{tx}", f"p_escalation:{tx}", f"p_z:{tx}"], [f"p_stacked:{tx}"], "ok", [],
        {"p_stacked": _f(s.p_stacked["test"][i]), "disagreement": _f(s.disagreement["test"][i]),
         "sources_available": int(np.sum(~np.isnan([s.p_triage["test"][i], s.p_esc["test"][i], s.p_z["test"][i]])))})
    add("calibration", [f"p_stacked:{tx}"], [f"p_calibrated:{tx}"], "ok", [], {"p_calibrated": _f(s.p_cal["test"][i])})
    add("decision", [f"p_calibrated:{tx}", f"threshold:{s.threshold}"], [f"decision:{tx}"],
        "review" if decision == "review" else "ok", [f"DECISION_{decision.upper()}"] + (["VERIFIER_FAIL_ROUTED_TO_REVIEW"] if ver == "fail" else []),
        {"decision": decision, "threshold": s.threshold, "label_synthetic": int(s.y["test"][i])})
    return steps


def write_trace(s: PipelineState, path: Path) -> dict[str, Any]:
    rows = select_sample_rows(s)
    steps = [st for i in rows for st in _steps_for_row(s, i)]
    payload = {
        "schemaVersion": SCHEMA_VERSION, "mode": "synthetic-demo",
        "scenarioId": f"fixture-v1-seed{s.cfg.seed}-test-sample",
        "note": "Synthetic fixture rows. Fictional identifiers. Not real transactions.",
        "transactionIds": [str(s.frames["test"][ROW_ID].iloc[i]) for i in rows],
        "steps": steps,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
