"""Deterministic rationale verifier. No parameters, no learning.

Four checks, all of which must pass: schema, feature support, range, predicate
consistency. A FAIL routes the transaction to `review` and never yields a fraud
or legitimate verdict; the reason codes say which check failed and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from templar_fraud_lab.schema import MISSING_TOKEN
from templar_fraud_lab.templates.bank import TemplateBank
from templar_fraud_lab.templates.predicate import PredicateError, evaluate, referenced_features, validate
from templar_fraud_lab.templates.rationale import VERIFICATION_FAIL, VERIFICATION_PASS, Rationale

RANGE_TOLERANCE = 0.0  # observed numeric values must lie within the training [min, max]

CODE_UNKNOWN_TEMPLATE = "SCHEMA_UNKNOWN_TEMPLATE"
CODE_BAD_PREDICATE = "SCHEMA_INVALID_PREDICATE"
CODE_PREDICATE_MISMATCH = "SCHEMA_PREDICATE_DIFFERS_FROM_BANK"
CODE_UNSUPPORTED_FEATURE = "SUPPORT_FEATURE_NOT_IN_SCHEMA"
CODE_UNLISTED_FEATURE = "SUPPORT_PREDICATE_USES_UNLISTED_FEATURE"
CODE_MISSING_OBSERVATION = "SUPPORT_OBSERVED_VALUE_MISSING"
CODE_OUT_OF_RANGE = "RANGE_NUMERIC_OUTSIDE_TRAINING"
CODE_UNKNOWN_CATEGORY = "RANGE_CATEGORY_UNSEEN_IN_TRAINING"
CODE_PREDICATE_FALSE = "CONSISTENCY_PREDICATE_DOES_NOT_HOLD"
CODE_PASS = "VERIFIED"


@dataclass(frozen=True)
class VerifierContext:
    """Training-fold facts the verifier checks against."""

    numeric_ranges: dict[str, tuple[float, float]]
    categorical_levels: dict[str, list[str]]
    known_features: frozenset[str]


@dataclass(frozen=True)
class VerifierResult:
    status: str
    reason_codes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == VERIFICATION_PASS


def _check_schema(r: Rationale, bank: TemplateBank, codes: list[str]) -> None:
    tpl = bank.by_id(r.template_id)
    if tpl is None:
        codes.append(CODE_UNKNOWN_TEMPLATE)
        return
    try:
        validate(r.predicates)
    except PredicateError:
        codes.append(CODE_BAD_PREDICATE)
        return
    if tpl.predicate != r.predicates:
        codes.append(CODE_PREDICATE_MISMATCH)


def _check_support(r: Rationale, ctx: VerifierContext, codes: list[str]) -> None:
    for f in r.feature_ids:
        if f not in ctx.known_features:
            codes.append(f"{CODE_UNSUPPORTED_FEATURE}:{f}")
    try:
        used = referenced_features(r.predicates)
    except (KeyError, TypeError):
        return
    for f in used:
        if f not in r.feature_ids:
            codes.append(f"{CODE_UNLISTED_FEATURE}:{f}")
        if f not in r.observed_values:
            codes.append(f"{CODE_MISSING_OBSERVATION}:{f}")


def _check_range(r: Rationale, ctx: VerifierContext, codes: list[str]) -> None:
    for f, v in r.observed_values.items():
        if f in ctx.numeric_ranges:
            if v is None:
                continue  # a genuinely missing numeric value is permitted; templates handle None
            lo, hi = ctx.numeric_ranges[f]
            if not (lo - RANGE_TOLERANCE <= float(v) <= hi + RANGE_TOLERANCE):
                codes.append(f"{CODE_OUT_OF_RANGE}:{f}")
        elif f in ctx.categorical_levels:
            level = MISSING_TOKEN if v is None else str(v)
            if level not in ctx.categorical_levels[f]:
                codes.append(f"{CODE_UNKNOWN_CATEGORY}:{f}")


def _check_consistency(r: Rationale, codes: list[str]) -> None:
    try:
        holds = bool(evaluate(r.predicates, dict(r.observed_values)))
    except (KeyError, TypeError, ValueError):
        holds = False
    if not holds:
        codes.append(CODE_PREDICATE_FALSE)


def verify(r: Rationale, bank: TemplateBank, ctx: VerifierContext) -> VerifierResult:
    codes: list[str] = []
    _check_schema(r, bank, codes)
    if codes and codes[0] in (CODE_UNKNOWN_TEMPLATE, CODE_BAD_PREDICATE):
        return VerifierResult(VERIFICATION_FAIL, tuple(codes))
    _check_support(r, ctx, codes)
    _check_range(r, ctx, codes)
    _check_consistency(r, codes)
    if codes:
        return VerifierResult(VERIFICATION_FAIL, tuple(codes))
    return VerifierResult(VERIFICATION_PASS, (CODE_PASS,))


def context_from_preprocessor(pre: Any) -> VerifierContext:
    """Build a VerifierContext from a fitted TabularPreprocessor."""
    known = set(pre.numeric_columns) | set(pre.schema.low_card) | set(pre.schema.high_card)
    levels = dict(pre.levels)
    for col in pre.schema.high_card:
        levels[col] = sorted(pre.frequencies[col])
    return VerifierContext(numeric_ranges=dict(pre.numeric_ranges), categorical_levels=levels,
                           known_features=frozenset(known))
