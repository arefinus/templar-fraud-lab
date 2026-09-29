from __future__ import annotations

import numpy as np
import pytest

from templar_fraud_lab.templates.bank import demo_bank, firings_matrix, load_bank, save_bank
from templar_fraud_lab.templates.mcts import MctsConfig, candidate_atoms, discover_templates
from templar_fraud_lab.templates.predicate import (
    PredicateError,
    count_conjuncts,
    depth,
    evaluate,
    referenced_features,
    validate,
)
from templar_fraud_lab.templates.rationale import build_rationale


def cmp(op, feat, const):
    return {"op": op, "args": [{"feat": feat}, {"const": const}]}


@pytest.mark.parametrize("bad", [
    {"op": "exec", "args": [{"const": "x"}]},
    {"op": "__import__", "args": [{"const": "os"}]},
    {"op": "lambda", "args": [{"const": 1}, {"const": 2}]},
    {"call": "open", "args": []},
    {"op": ">", "args": [{"feat": "a"}]},
    {"op": "not", "args": [{"const": 1}, {"const": 2}]},
    {"op": "sum", "args": [{"const": 1}, {"feat": "a"}]},
    {"const": {"nested": "dict"}},
    ["not", "a", "dict"],
])
def test_ast_rejects_disallowed_nodes(bad):
    with pytest.raises(PredicateError):
        validate(bad)


def test_ast_rejects_unknown_feature_when_schema_given():
    with pytest.raises(PredicateError, match="unknown feature"):
        validate(cmp(">", "ghost", 1), allowed_features={"amount"})


def test_ast_rejects_excessive_depth():
    node = {"feat": "a"}
    for _ in range(10):
        node = {"op": "+", "args": [node, {"const": 1}]}
    with pytest.raises(PredicateError, match="deeper"):
        validate(node)


def test_evaluate_comparisons_arithmetic_and_aggregates():
    row = {"a": 10.0, "b": 4.0, "c": "x", "m": None}
    assert evaluate(cmp(">", "a", 5), row) is True
    assert evaluate(cmp("<=", "a", 5), row) is False
    assert evaluate({"op": ">", "args": [{"op": "/", "args": [{"feat": "a"}, {"feat": "b"}]}, {"const": 2}]}, row) is True
    assert evaluate({"op": "sum", "args": [{"feat": "a"}, {"feat": "b"}, {"feat": "m"}]}, row) == 14.0
    assert evaluate({"op": "mean", "args": [{"feat": "a"}, {"feat": "b"}]}, row) == 7.0
    assert evaluate({"op": "in", "args": [{"feat": "c"}, {"const": ["x", "y"]}]}, row) is True
    assert evaluate({"op": "and", "args": [cmp(">", "a", 5), cmp("==", "c", "x")]}, row) is True
    assert evaluate({"op": "not", "args": [cmp("==", "c", "x")]}, row) is False


def test_evaluate_missing_and_division_by_zero_are_false_not_errors():
    row = {"a": None, "b": 0.0, "c": 3.0}
    assert evaluate(cmp(">", "a", 1), row) is False
    assert evaluate({"op": ">", "args": [{"op": "/", "args": [{"feat": "c"}, {"feat": "b"}]}, {"const": 0}]}, row) is False
    assert evaluate(cmp("==", "a", 1), row) is False
    assert evaluate(cmp(">", "zzz", 1), row) is False


def test_demo_bank_has_sixteen_valid_templates_and_fires(small_frame):
    bank = demo_bank()
    assert len(bank) == 16
    assert len(set(bank.ids())) == 16
    for tpl in bank.templates:
        validate(tpl.predicate)
        assert depth(tpl.predicate) <= 3 and count_conjuncts(tpl.predicate) <= 3
    f = firings_matrix(bank, small_frame)
    assert f.shape == (len(small_frame), 16)
    assert f.sum() > 0 and set(np.unique(f)) <= {0.0, 1.0}


def test_bank_json_roundtrip(tmp_path):
    bank = demo_bank()
    save_bank(bank, tmp_path / "bank.json")
    loaded = load_bank(tmp_path / "bank.json")
    assert loaded.ids() == bank.ids()
    assert loaded.templates[3].predicate == bank.templates[3].predicate


def test_rationale_lists_referenced_features_and_observed_values():
    tpl = demo_bank().by_id("T09")
    row = {"country_match": 0, "ip_risk_score": 0.6, "amount": 1.0}
    r = build_rationale(tpl.template_id, tpl.predicate, row, tpl.reason_code)
    assert set(r.feature_ids) == {"country_match", "ip_risk_score"}
    assert r.observed_values == {"country_match": 0, "ip_risk_score": 0.6}
    assert r.verification_status == "pending" and r.reason_codes == ("MISMATCH_AND_RISKY_IP",)
    assert referenced_features(tpl.predicate) == ["country_match", "ip_risk_score"]


def test_mcts_discovers_valid_bounded_templates(small_frame):
    train = small_frame.iloc[:900]
    y = train["is_fraud"].to_numpy()
    cfg = MctsConfig(simulations=6, n_templates=2, max_conjuncts=3, n_folds=2, seed=0)
    atoms = candidate_atoms(train, ("amount", "velocity_1h"), ("device_type",), (0.5, 0.9))
    assert atoms and all(validate(a) is None for a in atoms)
    bank, log = discover_templates(train, y, demo_bank(), ("amount", "velocity_1h"), ("device_type",), cfg)
    assert len(bank) == 18 and len(log) == 2
    for tpl in bank.templates[16:]:
        validate(tpl.predicate)
        assert count_conjuncts(tpl.predicate) <= 3 and depth(tpl.predicate) <= 3
        assert tpl.origin == "mcts-discovered"
    assert all(0.0 <= entry["reward_val_ap"] <= 1.0 for entry in log)
