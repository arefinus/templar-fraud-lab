"""Template bank: hand-written demo templates and JSON persistence.

The hand-written bank below is labelled `demo`. It is NOT the paper's bank of
K=64 templates discovered by MCTS over a 50,000-row corpus; it exists so the
escalation path, verifier and stacking run end to end on the fixture. A small
MCTS implementation lives in `mcts.py` and can append discovered templates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from templar_fraud_lab.templates.predicate import evaluate, referenced_features, validate

BANK_DEMO = "demo"
BANK_DISCOVERED = "mcts-discovered"


@dataclass(frozen=True)
class Template:
    template_id: str
    reason_code: str
    predicate: dict[str, Any]
    origin: str = BANK_DEMO
    description: str = ""

    @property
    def feature_ids(self) -> tuple[str, ...]:
        return tuple(referenced_features(self.predicate))


@dataclass(frozen=True)
class TemplateBank:
    templates: tuple[Template, ...]
    origin: str
    version: str = "v1"

    def __len__(self) -> int:
        return len(self.templates)

    def by_id(self, template_id: str) -> Template | None:
        return next((t for t in self.templates if t.template_id == template_id), None)

    def ids(self) -> list[str]:
        return [t.template_id for t in self.templates]

    def extended(self, extra: list[Template], origin: str) -> "TemplateBank":
        return TemplateBank(self.templates + tuple(extra), origin, self.version)


def _cmp(op: str, feat: str, const: Any) -> dict[str, Any]:
    return {"op": op, "args": [{"feat": feat}, {"const": const}]}


def _and(*nodes: dict[str, Any]) -> dict[str, Any]:
    return {"op": "and", "args": list(nodes)}


def demo_bank() -> TemplateBank:
    """Sixteen hand-written templates over the fixture schema."""
    t = [
        Template("T01", "HIGH_AMOUNT", _cmp(">", "amount", 250.0), description="amount above 250"),
        Template("T02", "VERY_HIGH_AMOUNT", _cmp(">", "amount", 800.0), description="amount above 800"),
        Template("T03", "COUNTRY_MISMATCH", _cmp("==", "country_match", 0)),
        Template("T04", "EMULATOR_DEVICE", _cmp("==", "device_type", "emulator")),
        Template("T05", "HIGH_IP_RISK", _cmp(">=", "ip_risk_score", 0.5)),
        Template("T06", "BURST_VELOCITY", _cmp(">=", "velocity_1h", 3)),
        Template("T07", "NEW_ACCOUNT", _cmp("<", "account_age_days", 30)),
        Template("T08", "NEW_ACCOUNT_HIGH_AMOUNT",
                 _and(_cmp("<", "account_age_days", 60), _cmp(">", "amount", 120.0))),
        Template("T09", "MISMATCH_AND_RISKY_IP",
                 _and(_cmp("==", "country_match", 0), _cmp(">=", "ip_risk_score", 0.3))),
        Template("T10", "NON_ESSENTIAL_CATEGORY_HIGH_AMOUNT",
                 _and({"op": "not", "args": [{"op": "in", "args": [{"feat": "merchant_category"},
                                                                    {"const": ["grocery", "fuel", "restaurants"]}]}]},
                      _cmp(">", "amount", 150.0)),
                 description="merchant category outside the essentials list and amount above 150"),
        Template("T11", "NIGHT_TRANSACTION",
                 {"op": "or", "args": [_cmp("<", "hour", 5), _cmp(">=", "hour", 23)]}),
        Template("T12", "RAPID_REPEAT",
                 _and(_cmp(">=", "time_since_last_card_event", 0), _cmp("<", "time_since_last_card_event", 600))),
        Template("T13", "EMULATOR_MISMATCH",
                 _and(_cmp("==", "device_type", "emulator"), _cmp("==", "country_match", 0))),
        Template("T14", "AMOUNT_PER_AGE_DAY",
                 {"op": ">", "args": [{"op": "/", "args": [{"feat": "amount"}, {"feat": "account_age_days"}]},
                                      {"const": 5.0}]}),
        Template("T15", "RISK_SUM",
                 {"op": ">", "args": [{"op": "sum", "args": [{"feat": "ip_risk_score"}, {"feat": "velocity_1h"}]},
                                      {"const": 2.5}]}),
        Template("T16", "MISSING_IP_HIGH_AMOUNT",
                 _and({"op": "not", "args": [_cmp(">=", "ip_risk_score", 0.0)]}, _cmp(">", "amount", 100.0))),
    ]
    for tpl in t:
        validate(tpl.predicate)
    return TemplateBank(tuple(t), origin=BANK_DEMO)


def firings_matrix(bank: TemplateBank, frame: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_templates) matrix of 0/1 firings."""
    rows = frame.to_dict(orient="records")
    out = np.zeros((len(rows), len(bank)), dtype=np.float32)
    for j, tpl in enumerate(bank.templates):
        out[:, j] = [1.0 if bool(evaluate(tpl.predicate, r)) else 0.0 for r in rows]
    return out


def save_bank(bank: TemplateBank, path: str | Path) -> None:
    payload = {"version": bank.version, "origin": bank.origin, "evidence_status": "demo",
               "templates": [asdict(t) for t in bank.templates]}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_bank(path: str | Path) -> TemplateBank:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    templates = []
    for t in payload["templates"]:
        validate(t["predicate"])
        templates.append(Template(t["template_id"], t["reason_code"], t["predicate"],
                                  t.get("origin", BANK_DEMO), t.get("description", "")))
    return TemplateBank(tuple(templates), payload["origin"], payload.get("version", "v1"))
