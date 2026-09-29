"""Structured rationale templates: allowlisted predicate AST, a hand-written demo
bank, and a small Monte Carlo tree search over predicate trees."""

from __future__ import annotations

from templar_fraud_lab.templates.bank import (
    Template,
    TemplateBank,
    demo_bank,
    firings_matrix,
    load_bank,
    save_bank,
)
from templar_fraud_lab.templates.mcts import MctsConfig, discover_templates
from templar_fraud_lab.templates.predicate import (
    ALLOWED_OPS,
    PredicateError,
    evaluate,
    referenced_features,
    validate,
)
from templar_fraud_lab.templates.rationale import Rationale, build_rationale

__all__ = [
    "ALLOWED_OPS", "MctsConfig", "PredicateError", "Rationale", "Template", "TemplateBank",
    "build_rationale", "demo_bank", "discover_templates", "evaluate", "firings_matrix",
    "load_bank", "referenced_features", "save_bank", "validate",
]
