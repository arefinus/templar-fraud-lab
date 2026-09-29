"""Escalation path: template firings, structured rationales, verification and the
escalation probability.

For every escalated row, each fired template yields a Rationale. The row passes
only if every rationale passes the verifier and at least one template fired. On
pass, p_escalation is the probability of a logistic model over the firing vector
fitted on training rows. On fail, p_escalation is unavailable (NaN) and the row
is flagged for review. If nothing fires, p_escalation is unavailable but the row
is not a verifier failure ("none").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from templar_fraud_lab.templates.bank import TemplateBank, firings_matrix
from templar_fraud_lab.templates.rationale import (
    VERIFICATION_FAIL,
    VERIFICATION_PASS,
    Rationale,
    build_rationale,
)
from templar_fraud_lab.verifier import VerifierContext, verify

STATUS_NONE = "none"  # not escalated, or escalated with no template firing


@dataclass(frozen=True)
class EscalationOutcome:
    verification_status: str
    reason_codes: tuple[str, ...]
    p_escalation: float
    rationales: tuple[Rationale, ...]


def fit_escalation_model(bank: TemplateBank, train: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    firings = firings_matrix(bank, train)
    if y.sum() == 0:
        raise ValueError("training fold has no fraud rows; cannot fit escalation model")
    return LogisticRegression(max_iter=500, class_weight="balanced").fit(firings, y.astype(int))


def escalate_row(row: dict, bank: TemplateBank, ctx: VerifierContext, model: LogisticRegression,
                 firing_row: np.ndarray) -> EscalationOutcome:
    rationales: list[Rationale] = []
    codes: list[str] = []
    failed = False
    for j, tpl in enumerate(bank.templates):
        if firing_row[j] <= 0:
            continue
        r = build_rationale(tpl.template_id, tpl.predicate, row, tpl.reason_code)
        result = verify(r, bank, ctx)
        rationales.append(r.with_verification(result.status, result.reason_codes))
        codes.append(tpl.reason_code)
        if not result.passed:
            failed = True
            codes.extend(result.reason_codes)
    if not rationales:
        return EscalationOutcome(STATUS_NONE, ("NO_TEMPLATE_FIRED",), float("nan"), ())
    if failed:
        return EscalationOutcome(VERIFICATION_FAIL, tuple(dict.fromkeys(codes)), float("nan"), tuple(rationales))
    p = float(model.predict_proba(firing_row[None, :])[0, 1])
    return EscalationOutcome(VERIFICATION_PASS, tuple(dict.fromkeys(codes)), p, tuple(rationales))


def run_escalation(frame: pd.DataFrame, escalated: np.ndarray, bank: TemplateBank, ctx: VerifierContext,
                   model: LogisticRegression) -> tuple[list[str], list[tuple[str, ...]], np.ndarray, list[tuple[Rationale, ...]]]:
    firings = firings_matrix(bank, frame)
    rows = frame.to_dict(orient="records")
    statuses, codes, rationales = [], [], []
    p_esc = np.full(len(frame), np.nan)
    for i, row in enumerate(rows):
        if not escalated[i]:
            statuses.append(STATUS_NONE)
            codes.append(())
            rationales.append(())
            continue
        out = escalate_row(row, bank, ctx, model, firings[i])
        statuses.append(out.verification_status)
        codes.append(out.reason_codes)
        rationales.append(out.rationales)
        p_esc[i] = out.p_escalation
    return statuses, codes, p_esc, rationales
