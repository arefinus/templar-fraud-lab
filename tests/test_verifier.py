from __future__ import annotations

import pytest

from templar_fraud_lab.preprocess.tabular import TabularPreprocessor
from templar_fraud_lab.schema import FIXTURE_SCHEMA
from templar_fraud_lab.templates.bank import demo_bank
from templar_fraud_lab.templates.rationale import Rationale, build_rationale
from templar_fraud_lab.verifier import (
    CODE_OUT_OF_RANGE,
    CODE_PASS,
    CODE_PREDICATE_FALSE,
    CODE_PREDICATE_MISMATCH,
    CODE_UNKNOWN_CATEGORY,
    CODE_UNKNOWN_TEMPLATE,
    CODE_UNLISTED_FEATURE,
    CODE_UNSUPPORTED_FEATURE,
    VerifierContext,
    context_from_preprocessor,
    verify,
)

BANK = demo_bank()
CTX = VerifierContext(
    numeric_ranges={"amount": (1.0, 1000.0), "ip_risk_score": (0.0, 1.0), "country_match": (0, 1),
                    "account_age_days": (1, 2000), "velocity_1h": (0, 6), "hour": (0, 23),
                    "time_since_last_card_event": (-1, 1e7)},
    categorical_levels={"device_type": ["web", "pos", "emulator", "mobile_app"],
                        "merchant_category": ["grocery", "fuel", "travel", "gaming"]},
    known_features=frozenset({"amount", "ip_risk_score", "country_match", "account_age_days", "velocity_1h",
                              "hour", "time_since_last_card_event", "device_type", "merchant_category"}),
)


def rationale(template_id: str, row: dict) -> Rationale:
    tpl = BANK.by_id(template_id)
    return build_rationale(tpl.template_id, tpl.predicate, row, tpl.reason_code)


def test_valid_rationale_passes():
    res = verify(rationale("T09", {"country_match": 0, "ip_risk_score": 0.8}), BANK, CTX)
    assert res.passed and res.reason_codes == (CODE_PASS,)


def test_unknown_template_fails_without_verdict():
    r = Rationale("T99", ("amount",), {"amount": 5.0}, {"op": ">", "args": [{"feat": "amount"}, {"const": 1}]})
    res = verify(r, BANK, CTX)
    assert not res.passed and res.status == "fail" and CODE_UNKNOWN_TEMPLATE in res.reason_codes


def test_predicate_differing_from_bank_fails():
    tpl = BANK.by_id("T01")
    r = Rationale(tpl.template_id, ("amount",), {"amount": 900.0}, {"op": ">", "args": [{"feat": "amount"}, {"const": 1.0}]})
    res = verify(r, BANK, CTX)
    assert CODE_PREDICATE_MISMATCH in res.reason_codes and not res.passed


def test_out_of_range_value_fails():
    res = verify(rationale("T01", {"amount": 50_000.0}), BANK, CTX)
    assert not res.passed and f"{CODE_OUT_OF_RANGE}:amount" in res.reason_codes


def test_unknown_category_fails():
    res = verify(rationale("T10", {"merchant_category": "crypto_exchange", "amount": 400.0}), BANK, CTX)
    assert not res.passed and f"{CODE_UNKNOWN_CATEGORY}:merchant_category" in res.reason_codes


def test_predicate_that_does_not_hold_fails_consistency():
    res = verify(rationale("T01", {"amount": 10.0}), BANK, CTX)
    assert not res.passed and CODE_PREDICATE_FALSE in res.reason_codes


def test_unsupported_and_unlisted_features_fail():
    tpl = BANK.by_id("T03")
    r = Rationale(tpl.template_id, ("ghost_feature",), {"country_match": 0}, tpl.predicate)
    res = verify(r, BANK, CTX)
    assert f"{CODE_UNSUPPORTED_FEATURE}:ghost_feature" in res.reason_codes
    assert f"{CODE_UNLISTED_FEATURE}:country_match" in res.reason_codes


def test_invalid_predicate_ast_is_rejected():
    r = Rationale("T01", ("amount",), {"amount": 5.0}, {"op": "exec", "args": [{"const": "rm"}]})
    res = verify(r, BANK, CTX)
    assert res.status == "fail"


def test_context_from_preprocessor_carries_train_ranges(small_frame):
    pre = TabularPreprocessor(FIXTURE_SCHEMA).fit(small_frame.iloc[:800])
    ctx = context_from_preprocessor(pre)
    assert ctx.numeric_ranges["amount"][0] == pytest.approx(small_frame.iloc[:800]["amount"].min())
    assert "emulator" in ctx.categorical_levels["device_type"]
    assert "email_domain" in ctx.categorical_levels and "amount" in ctx.known_features
