"""Run configuration loaded from YAML into frozen dataclasses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from templar_fraud_lab.decision import CostMatrix
from templar_fraud_lab.templates.mcts import MctsConfig
from templar_fraud_lab.triage import RoutingConfig, TriageConfig


@dataclass(frozen=True)
class SaintTrainConfig:
    token_dim: int = 16
    out_dim: int = 16
    n_blocks: int = 2
    n_heads: int = 2
    ff_mult: int = 2
    dropout: float = 0.1
    context: int = 128
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-5
    gamma: float = 2.0
    patience: int = 3
    loss: str = "focal"


@dataclass(frozen=True)
class StackingConfig:
    lr: float = 0.01
    iterations: int = 600
    mc_samples: int = 32
    patience: int = 100


@dataclass(frozen=True)
class CalibrationConfig:
    n_knots: int = 32
    min_support: int = 20
    ece_bins: int = 10


@dataclass(frozen=True)
class DecisionConfig:
    costs_file: str = "configs/demo_costs.yaml"
    review_uncertainty: float = 0.35


@dataclass(frozen=True)
class TemplatesConfig:
    bank: str = "demo"
    mcts_enabled: bool = True
    mcts: MctsConfig = field(default_factory=MctsConfig)


@dataclass(frozen=True)
class RunConfig:
    mode: str
    seed: int
    fixture_dir: str
    output_dir: str
    max_rows: int | None
    split_fractions: tuple[float, float, float]
    late_fraction: float
    te_smoothing: float
    saint: SaintTrainConfig
    triage: TriageConfig
    routing: RoutingConfig
    templates: TemplatesConfig
    stacking: StackingConfig
    calibration: CalibrationConfig
    decision: DecisionConfig
    raw: dict[str, Any]

    def configuration_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _sub(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return value


def load_config(path: str | Path) -> RunConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if raw.get("mode") not in {"smoke", "demo", "experiment"}:
        raise ValueError("config 'mode' must be one of smoke, demo, experiment")
    split = _sub(raw, "split")
    tpl = _sub(raw, "templates")
    mcts_raw = _sub(tpl, "mcts")
    mcts_enabled = bool(mcts_raw.pop("enabled", True))
    seed = int(raw.get("seed", 42))
    mcts_raw.setdefault("seed", seed)
    triage_raw = _sub(raw, "triage")
    triage_raw.setdefault("seed", seed)
    fractions = tuple(float(x) for x in split.get("fractions", (0.70, 0.15, 0.15)))
    if len(fractions) != 3:
        raise ValueError("split.fractions must have three entries")
    return RunConfig(
        mode=raw["mode"], seed=seed,
        fixture_dir=str(raw.get("fixture_dir", "examples/fixtures/v1")),
        output_dir=str(raw.get("output_dir", "examples/output")),
        max_rows=raw.get("max_rows"),
        split_fractions=fractions, late_fraction=float(split.get("late_fraction", 0.10)),
        te_smoothing=float(_sub(raw, "preprocess").get("te_smoothing", 20.0)),
        saint=SaintTrainConfig(**_sub(raw, "saint")),
        triage=TriageConfig(**triage_raw),
        routing=RoutingConfig(**_sub(raw, "routing")),
        templates=TemplatesConfig(bank=str(tpl.get("bank", "demo")), mcts_enabled=mcts_enabled,
                                  mcts=MctsConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                                                     for k, v in mcts_raw.items()})),
        stacking=StackingConfig(**_sub(raw, "stacking")),
        calibration=CalibrationConfig(**_sub(raw, "calibration")),
        decision=DecisionConfig(**_sub(raw, "decision")),
        raw=raw,
    )


def load_costs(path: str | Path) -> CostMatrix:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"cost matrix not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for key in ("c_fp", "c_fn", "c_review"):
        if key not in raw:
            raise ValueError(f"cost matrix missing {key!r} in {p}")
    return CostMatrix(float(raw["c_fp"]), float(raw["c_fn"]), float(raw["c_review"]),
                      label=str(raw.get("label", "demo")))


def config_summary(cfg: RunConfig) -> dict[str, Any]:
    return {"mode": cfg.mode, "seed": cfg.seed, "saint": asdict(cfg.saint), "triage": asdict(cfg.triage),
            "routing": asdict(cfg.routing), "stacking": asdict(cfg.stacking),
            "calibration": asdict(cfg.calibration), "decision": asdict(cfg.decision)}
