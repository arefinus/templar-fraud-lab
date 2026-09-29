"""Structured rationale produced on the escalation path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from templar_fraud_lab.templates.predicate import referenced_features

VERIFICATION_PENDING = "pending"
VERIFICATION_PASS = "pass"
VERIFICATION_FAIL = "fail"


@dataclass(frozen=True)
class Rationale:
    template_id: str
    feature_ids: tuple[str, ...]
    observed_values: dict[str, Any]
    predicates: dict[str, Any]
    verification_status: str = VERIFICATION_PENDING
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature_ids"] = list(self.feature_ids)
        d["reason_codes"] = list(self.reason_codes)
        return d

    def with_verification(self, status: str, codes: tuple[str, ...]) -> "Rationale":
        return Rationale(self.template_id, self.feature_ids, self.observed_values,
                         self.predicates, status, codes)


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def build_rationale(template_id: str, predicate: dict[str, Any], row: dict[str, Any],
                    reason_code: str) -> Rationale:
    feats = tuple(referenced_features(predicate))
    observed = {f: _jsonable(row.get(f)) for f in feats}
    return Rationale(template_id=template_id, feature_ids=feats, observed_values=observed,
                     predicates=predicate, reason_codes=(reason_code,))
