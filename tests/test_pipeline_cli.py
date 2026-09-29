from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from templar_fraud_lab.cli import main
from templar_fraud_lab.config import load_config, load_costs
from templar_fraud_lab.outputs import PREDICTION_COLUMNS


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke_out")
    rc = main(["smoke", "--config", "configs/smoke.yaml", "--output-dir", str(out)])
    assert rc == 0
    return out


def test_full_synthetic_run_writes_every_artifact(smoke_run):
    for name in ("run_manifest.yaml", "predictions.csv", "metrics.json", "reliability.json",
                 "late_slice_overlap.json", "demo_trace.json", "template_bank.json", "threshold_curve.csv",
                 "models/saint.pt", "models/triage_catboost.json", "models/stacking.json", "models/calibrator_ot.json"):
        assert (smoke_run / name).exists(), name


def test_manifest_follows_contract(smoke_run):
    m = yaml.safe_load((smoke_run / "run_manifest.yaml").read_text(encoding="utf-8"))
    assert m["schema_version"] == 1 and m["project_id"] == "templar-fraud-lab"
    assert m["status"] == "completed" and m["mode"] == "smoke" and m["evidence_status"] == "demo"
    assert m["git_commit"] is None and m["seed"] == 42
    assert m["data"]["source_id"] == "synthetic-fixture-v1" and len(m["data"]["split_manifest_hash"]) == 64
    assert set(m["data"]["sample_counts"]) == {"train", "val", "test", "late"}


def test_predictions_have_exact_columns_and_three_way_decisions(smoke_run):
    frame = pd.read_csv(smoke_run / "predictions.csv")
    assert list(frame.columns) == PREDICTION_COLUMNS
    assert set(frame["decision"].unique()) <= {"fraud", "legitimate", "review"}
    assert set(frame["route"].unique()) <= {"fast", "escalation"}
    fast = frame[frame["route"] == "fast"]
    assert fast["p_escalation"].isna().all()  # fast rows never carry an escalation probability
    failed = frame[frame["verification_status"] == "fail"]
    assert (failed["decision"] == "review").all()  # verifier failure never yields a verdict


def test_metrics_carry_demo_label_and_both_pr_areas(smoke_run):
    m = json.loads((smoke_run / "metrics.json").read_text(encoding="utf-8"))
    assert m["evidence_status"] == "demo" and "not a benchmark" in m["note"].lower()
    test = m["slices"]["test"]
    assert {"average_precision", "pr_auc_trapezoidal", "auroc"} <= set(test["ranking_calibrated"])
    assert sum(b["count"] for b in test["calibration_after_ot"]["bins"]) == test["rows"]
    assert 0.001 <= m["threshold"] <= 0.5
    assert set(m["routing_comparison_test_escalation_share"]) == {"eq5", "band"}


def test_trace_has_required_shape(smoke_run):
    trace = json.loads((smoke_run / "demo_trace.json").read_text(encoding="utf-8"))
    assert trace["schemaVersion"] == 1 and trace["mode"] == "synthetic-demo" and trace["scenarioId"]
    assert trace["steps"]
    for step in trace["steps"]:
        assert {"id", "stage", "inputRefs", "outputRefs", "status", "reasonCodes"} <= set(step)
    stages = {s["stage"] for s in trace["steps"]}
    assert {"ingest", "triage", "routing", "stacking", "calibration", "decision"} <= stages


def test_late_slice_overlap_report(smoke_run):
    rep = json.loads((smoke_run / "late_slice_overlap.json").read_text(encoding="utf-8"))
    assert rep["late_in_test"] == rep["late_rows"] and rep["late_in_train"] == 0


def test_evaluate_reproduces_metrics_and_exits_two_on_missing(smoke_run, tmp_path):
    out = tmp_path / "eval.json"
    rc = main(["evaluate", "--predictions", str(smoke_run / "predictions.csv"), "--costs", "configs/demo_costs.yaml", "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    metrics = json.loads((smoke_run / "metrics.json").read_text(encoding="utf-8"))
    assert report["slices"]["test"]["cost"] == metrics["slices"]["test"]["cost"]
    assert report["slices"]["test"]["decision"]["tp"] == metrics["slices"]["test"]["decision"]["tp"]
    assert main(["evaluate", "--predictions", str(tmp_path / "nope.csv")]) == 2


def test_audit_command_writes_measured_ambiguities(smoke_run):
    assert main(["audit", "--output-dir", str(smoke_run)]) == 0
    audit = json.loads((smoke_run / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["ambiguities"]) == 11
    a01 = next(a for a in audit["ambiguities"] if a["id"] == "A01")
    assert a01["evidence_status"] == "demo" and a01["measured"]["late_rows"] > 0


def test_figures_exit_two_without_demo_outputs(tmp_path):
    assert main(["figures", "--output-dir", str(tmp_path / "empty"), "--figures-dir", str(tmp_path / "figs")]) == 2


def test_make_fixture_command(tmp_path):
    assert main(["make-fixture", "--out", str(tmp_path / "fx"), "--rows", "200", "--seed", "9"]) == 0
    meta = json.loads((tmp_path / "fx" / "fixture_meta.json").read_text(encoding="utf-8"))
    assert meta["rows"] == 200 and meta["seed"] == 9


def test_config_and_costs_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mode: production\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mode"):
        load_config(bad)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
    costs = tmp_path / "costs.yaml"
    costs.write_text("c_fp: 1\nc_fn: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="c_review"):
        load_costs(costs)
    cfg = load_config("configs/demo.yaml")
    assert cfg.routing.rule == "eq5" and len(cfg.configuration_hash()) == 64
