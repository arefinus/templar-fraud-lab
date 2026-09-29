"""End-to-end pipeline on the synthetic fixture.

Stages: load -> split -> preprocess (train-only fit, forward OOF target encoding)
-> SAINT encoder -> CatBoost triage -> routing -> escalation (templates, verifier)
-> stacking (fit on validation) -> OT calibration (fit on validation) -> cost
threshold (searched on validation) -> decisions and metrics on test and late slice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from templar_fraud_lab import escalation as esc
from templar_fraud_lab import stacking as stk
from templar_fraud_lab import triage as trg
from templar_fraud_lab.calibration import fit_isotonic, fit_ot_calibrator
from templar_fraud_lab.config import RunConfig, load_costs
from templar_fraud_lab.data.synthetic import load_fixture
from templar_fraud_lab.decision import decide, review_mask, search_threshold
from templar_fraud_lab.models.saint import SaintConfig
from templar_fraud_lab.models.train_saint import encode_in_context_windows, save_checkpoint, train_saint
from templar_fraud_lab.preprocess.tabular import TabularPreprocessor
from templar_fraud_lab.preprocess.target_encoding import forward_oof_target_encode, target_encode_apply
from templar_fraud_lab.schema import FIXTURE_SCHEMA, LABEL, ROW_ID, TIMESTAMP
from templar_fraud_lab.splits.chrono import (
    ChronoSplit,
    assert_strict_chronology,
    chronological_split,
    split_manifest_hash,
    write_overlap_report,
)
from templar_fraud_lab.templates.bank import demo_bank, save_bank
from templar_fraud_lab.templates.mcts import discover_templates
from templar_fraud_lab.verifier import context_from_preprocessor

TE_COLUMN = "email_domain__te"


@dataclass
class PipelineState:
    cfg: RunConfig
    out_dir: Path
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    split: ChronoSplit | None = None
    pre: TabularPreprocessor | None = None
    x: dict[str, np.ndarray] = field(default_factory=dict)
    y: dict[str, np.ndarray] = field(default_factory=dict)
    ts: dict[str, np.ndarray] = field(default_factory=dict)
    z: dict[str, np.ndarray] = field(default_factory=dict)
    p_z: dict[str, np.ndarray] = field(default_factory=dict)
    p_triage: dict[str, np.ndarray] = field(default_factory=dict)
    u_route: dict[str, np.ndarray] = field(default_factory=dict)
    route: dict[str, np.ndarray] = field(default_factory=dict)
    ver_status: dict[str, list[str]] = field(default_factory=dict)
    ver_codes: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)
    rationales: dict[str, list] = field(default_factory=dict)
    p_esc: dict[str, np.ndarray] = field(default_factory=dict)
    p_stacked: dict[str, np.ndarray] = field(default_factory=dict)
    disagreement: dict[str, np.ndarray] = field(default_factory=dict)
    p_cal: dict[str, np.ndarray] = field(default_factory=dict)
    p_iso: dict[str, np.ndarray] = field(default_factory=dict)
    review: dict[str, np.ndarray] = field(default_factory=dict)
    decision: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    bank: Any = None
    stacking_fit: Any = None
    calibrator: Any = None
    threshold: float | None = None
    costs: Any = None
    saint_history: Any = None
    score_reference: Any = None


def stage_load_and_split(state: PipelineState) -> None:
    cfg = state.cfg
    frame = load_fixture(cfg.fixture_dir)
    if cfg.max_rows:
        frame = frame.iloc[: int(cfg.max_rows)].reset_index(drop=True)
    split = chronological_split(frame, cfg.split_fractions, cfg.late_fraction)
    assert_strict_chronology(frame, split)
    state.split = split
    indexed = frame.set_index(ROW_ID, drop=False)
    for name, ids in (("train", split.train_ids), ("val", split.val_ids), ("test", split.test_ids)):
        state.frames[name] = indexed.loc[list(ids)].reset_index(drop=True)
    state.frames["late"] = indexed.loc[list(split.late_ids)].reset_index(drop=True)
    write_overlap_report(split, state.out_dir / "late_slice_overlap.json")
    state.info["split_manifest_hash"] = split_manifest_hash(split)
    state.info["sample_counts"] = {k: int(len(v)) for k, v in state.frames.items()}
    state.info["fixture_rows"] = int(len(frame))


def stage_preprocess(state: PipelineState) -> None:
    cfg, frames = state.cfg, state.frames
    train = frames["train"]
    te_train, stats, prior = forward_oof_target_encode(train["email_domain"], train[LABEL], cfg.te_smoothing)
    frames["train"] = train.assign(**{TE_COLUMN: te_train})
    for name in ("val", "test", "late"):
        frames[name] = frames[name].assign(**{TE_COLUMN: target_encode_apply(frames[name]["email_domain"], stats, prior, cfg.te_smoothing)})
    pre = TabularPreprocessor(FIXTURE_SCHEMA, extra_numeric=(TE_COLUMN,)).fit(frames["train"])
    state.pre = pre
    for name in ("train", "val", "test", "late"):
        state.x[name] = pre.transform(frames[name])
        state.y[name] = frames[name][LABEL].to_numpy(dtype=np.int64)
        state.ts[name] = frames[name][TIMESTAMP].to_numpy(dtype=np.float64)
    state.info["feature_names"] = pre.feature_names
    state.info["preprocessor_state_hash"] = pre.state_hash()
    state.info["unknown_categories_in_test"] = pre.unknown_categories(frames["test"])


def stage_encoder(state: PipelineState) -> None:
    s, cfg = state, state.cfg.saint
    scfg = SaintConfig(n_features=s.x["train"].shape[1], token_dim=cfg.token_dim, out_dim=cfg.out_dim,
                       n_blocks=cfg.n_blocks, n_heads=cfg.n_heads, ff_mult=cfg.ff_mult,
                       dropout=cfg.dropout, context=cfg.context)
    t0 = time.perf_counter()
    model, result = train_saint(scfg, s.x["train"], s.y["train"], s.ts["train"], s.x["val"], s.y["val"], s.ts["val"],
                                epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay, gamma=cfg.gamma,
                                patience=cfg.patience, seed=state.cfg.seed, loss=cfg.loss)
    s.info["saint_train_seconds"] = round(time.perf_counter() - t0, 3)
    s.saint_history = result
    for name in ("train", "val", "test", "late"):
        s.z[name], s.p_z[name] = encode_in_context_windows(model, s.x[name], s.ts[name])
    models_dir = state.out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, models_dir / "saint.pt")
    s.info["saint_parameters"] = int(sum(p.numel() for p in model.parameters()))


def stage_triage(state: PipelineState) -> None:
    s = state
    model = trg.train_triage(s.z["train"], s.x["train"], s.y["train"], s.cfg.triage)
    for name in ("train", "val", "test", "late"):
        t0 = time.perf_counter()
        s.p_triage[name] = trg.predict_triage(model, s.z[name], s.x[name])
        if name == "test":
            s.info["latency_fast_ms_per_row"] = 1000 * (time.perf_counter() - t0) / max(len(s.x[name]), 1)
    reference = trg.ScoreReference.fit(s.p_triage["train"], s.p_z["train"])
    s.score_reference = reference
    for name in ("train", "val", "test", "late"):
        s.u_route[name] = trg.uncertainty(s.p_triage[name], s.p_z[name], s.cfg.routing, reference)
        s.route[name] = trg.route(s.p_triage[name], s.u_route[name], s.cfg.routing)
    trg.save_triage(model, state.out_dir / "models" / "triage_catboost.json")
    s.info["class_weight_ratio_train"] = trg.class_weight_ratio(s.y["train"])
    alt = trg.RoutingConfig(rule="band" if s.cfg.routing.rule == "eq5" else "eq5",
                            gamma_low=s.cfg.routing.gamma_low, gamma_high=s.cfg.routing.gamma_high,
                            t_u=s.cfg.routing.t_u, uncertainty=s.cfg.routing.uncertainty)
    alt_route = trg.route(s.p_triage["test"], s.u_route["test"], alt)
    s.info["routing_comparison_test"] = {
        s.cfg.routing.rule: float((s.route["test"] == trg.ROUTE_ESCALATION).mean()),
        alt.rule: float((alt_route == trg.ROUTE_ESCALATION).mean()),
    }


def stage_escalation(state: PipelineState) -> None:
    s, cfg = state, state.cfg
    bank = demo_bank()
    if cfg.templates.mcts_enabled:
        bank, log = discover_templates(s.frames["train"], s.y["train"], bank, ("amount", "account_age_days", "velocity_1h", "ip_risk_score"),
                                       ("device_type", "merchant_category"), cfg.templates.mcts)
        s.info["mcts_log"] = log
    save_bank(bank, state.out_dir / "template_bank.json")
    s.bank = bank
    ctx = context_from_preprocessor(s.pre)
    model = esc.fit_escalation_model(bank, s.frames["train"], s.y["train"])
    for name in ("val", "test", "late"):
        escalated = s.route[name] == trg.ROUTE_ESCALATION
        t0 = time.perf_counter()
        statuses, codes, p_esc, rationales = esc.run_escalation(s.frames[name], escalated, bank, ctx, model)
        if name == "test" and escalated.sum():
            s.info["latency_escalation_ms_per_row"] = s.info["latency_fast_ms_per_row"] + 1000 * (time.perf_counter() - t0) / int(escalated.sum())
        s.ver_status[name], s.ver_codes[name], s.p_esc[name], s.rationales[name] = statuses, codes, p_esc, rationales


def _sources(s: PipelineState, name: str) -> np.ndarray:
    return np.column_stack([s.p_triage[name], s.p_esc[name], s.p_z[name]])


def stage_stack_calibrate_decide(state: PipelineState) -> None:
    s, cfg = state, state.cfg
    fit = stk.fit_stacking(_sources(s, "val"), s.y["val"], lr=cfg.stacking.lr, iterations=cfg.stacking.iterations,
                           mc_samples=cfg.stacking.mc_samples, patience=cfg.stacking.patience, seed=cfg.seed)
    s.stacking_fit = fit
    fit.save(state.out_dir / "models" / "stacking.json")
    w = np.array(fit.weights)
    for name in ("val", "test", "late"):
        s.p_stacked[name] = stk.combine(_sources(s, name), w)
        s.disagreement[name] = stk.disagreement(_sources(s, name), w)
    cal = fit_ot_calibrator(s.p_stacked["val"], s.y["val"], n_knots=cfg.calibration.n_knots, min_support=cfg.calibration.min_support)
    iso = fit_isotonic(s.p_stacked["val"], s.y["val"])
    s.calibrator = cal
    cal.save(state.out_dir / "models" / "calibrator_ot.json")
    for name in ("val", "test", "late"):
        s.p_cal[name] = cal.transform(s.p_stacked[name])
        s.p_iso[name] = iso.transform(s.p_stacked[name])
        failed = np.array([st == "fail" for st in s.ver_status[name]])
        s.review[name] = review_mask(failed, s.disagreement[name], cfg.decision.review_uncertainty)
    costs = load_costs(cfg.decision.costs_file)
    s.costs = costs
    search = search_threshold(s.p_cal["val"], s.y["val"], s.review["val"], costs)
    s.threshold = search.threshold
    s.info["threshold_search"] = {"threshold": search.threshold, "val_cost": search.cost, "n_ties": search.n_ties,
                                  "grid_costs": search.grid_costs}
    for name in ("val", "test", "late"):
        s.decision[name] = decide(s.p_cal[name], search.threshold, s.review[name])


def run_training_stages(state: PipelineState) -> None:
    state.out_dir.mkdir(parents=True, exist_ok=True)
    stage_load_and_split(state)
    stage_preprocess(state)
    stage_encoder(state)
    stage_triage(state)


def run_all_stages(state: PipelineState) -> None:
    run_training_stages(state)
    stage_escalation(state)
    stage_stack_calibrate_decide(state)
