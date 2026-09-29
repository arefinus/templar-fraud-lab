from __future__ import annotations

import numpy as np
import pytest

from templar_fraud_lab.decision import (
    CostMatrix,
    decide,
    review_mask,
    search_threshold,
    threshold_grid,
    total_cost,
)
from templar_fraud_lab.metrics import decision_metrics, ranking_metrics, route_shares
from templar_fraud_lab.triage import RoutingConfig, ScoreReference, class_weight_ratio, route, uncertainty


def test_threshold_grid_matches_paper_range():
    g = threshold_grid()
    assert len(g) == 500 and g[0] == 0.001 and g[-1] == 0.5
    assert np.allclose(np.diff(g), 0.001)


def test_threshold_search_tie_break_is_lowest_threshold():
    # scores far from the grid: every threshold gives the same confusion counts -> 500-way tie
    p = np.array([0.9, 0.9, 0.0, 0.0])
    y = np.array([1, 1, 0, 0])
    res = search_threshold(p, y, np.zeros(4, dtype=bool), CostMatrix(1, 1, 1))
    assert res.n_ties == 500 and res.threshold == 0.001
    res2 = search_threshold(p, y, np.zeros(4, dtype=bool), CostMatrix(1, 1, 1))
    assert res2.threshold == res.threshold  # deterministic


def test_threshold_search_moves_with_costs():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.1, 2000), 0, 1)
    cheap_fn = search_threshold(p, y, np.zeros(2000, bool), CostMatrix(c_fp=1, c_fn=1, c_review=0))
    dear_fn = search_threshold(p, y, np.zeros(2000, bool), CostMatrix(c_fp=1, c_fn=50, c_review=0))
    assert dear_fn.threshold < cheap_fn.threshold


def test_cost_arithmetic():
    y = np.array([1, 0, 1, 0, 1])
    d = np.array(["fraud", "fraud", "legitimate", "review", "review"], dtype=object)
    out = total_cost(y, d, CostMatrix(c_fp=2.0, c_fn=10.0, c_review=1.5))
    assert out == {"fp": 1, "fn": 1, "review": 2, "cost": 2.0 + 10.0 + 3.0}


def test_negative_costs_rejected():
    with pytest.raises(ValueError):
        CostMatrix(-1, 1, 1)


def test_decide_and_review_mask():
    p = np.array([0.9, 0.1, 0.5, 0.2])
    rev = review_mask(np.array([False, False, True, False]), np.array([0.0, 0.0, 0.0, 0.9]), review_uncertainty=0.5)
    d = decide(p, 0.5, rev)
    assert list(d) == ["fraud", "legitimate", "review", "review"]


def test_ranking_metrics_report_both_pr_areas():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    m = ranking_metrics(y, p)
    assert 0 < m["auroc"] <= 1 and m["average_precision"] != m["pr_auc_trapezoidal"]
    assert ranking_metrics(np.zeros(3), p[:3])["auroc"] is None


def test_decision_metrics_exclude_review_rows():
    y = np.array([1, 1, 0, 0, 1])
    d = np.array(["fraud", "legitimate", "legitimate", "fraud", "review"], dtype=object)
    m = decision_metrics(y, d)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["review_rows"] == 1 and m["precision"] == 0.5 and m["specificity"] == 0.5


def test_route_shares_and_class_weight():
    shares = route_shares(np.array(["fast", "escalation", "escalation", "fast"]), np.array(["none", "pass", "fail", "none"]))
    assert shares["fast_share"] == 0.5 and shares["verifier_fail_share"] == 0.25
    assert class_weight_ratio(np.array([0, 0, 0, 1])) == 3.0
    with pytest.raises(ValueError):
        class_weight_ratio(np.zeros(5))


def test_routing_rules_eq5_and_band():
    p = np.array([0.05, 0.05, 0.5, 0.9])
    u = np.array([0.1, 0.9, 0.1, 0.1])
    eq5 = route(p, u, RoutingConfig(rule="eq5", gamma_low=0.18, t_u=0.25, uncertainty="margin"))
    band = route(p, u, RoutingConfig(rule="band", gamma_low=0.18, gamma_high=0.72, uncertainty="margin"))
    assert list(eq5) == ["fast", "escalation", "escalation", "escalation"]
    assert list(band) == ["fast", "fast", "escalation", "fast"]


def test_rank_disagreement_ignores_calibration_offset():
    ref = ScoreReference.fit(np.linspace(0, 1, 101), np.linspace(0.4, 0.6, 101))
    cfg = RoutingConfig(uncertainty="rank_disagreement")
    u = uncertainty(np.array([0.1, 0.9]), np.array([0.42, 0.58]), cfg, ref)
    assert np.all(u < 0.02)
    raw = uncertainty(np.array([0.1, 0.9]), np.array([0.42, 0.58]), RoutingConfig(uncertainty="disagreement"))
    assert np.all(raw > 0.3)
    with pytest.raises(ValueError):
        uncertainty(np.array([0.1]), np.array([0.5]), cfg, None)


def test_routing_config_validation():
    with pytest.raises(ValueError):
        RoutingConfig(rule="random")
    with pytest.raises(ValueError):
        RoutingConfig(rule="band", gamma_low=0.5, gamma_high=0.6)
