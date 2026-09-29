"""Run manifest (contract section 5). `git_commit` is null; it is filled at push
time and never fabricated here."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from templar_fraud_lab import PROJECT_ID


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class RunManifest:
    run_id: str
    mode: str
    seed: int
    configuration_hash: str
    started_at: str = field(default_factory=utc_now)
    status: str = "running"
    evidence_status: str = "demo"
    data: dict[str, Any] = field(default_factory=dict)
    metrics_file: str | None = None
    predictions_file: str | None = None
    finished_at: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "project_id": PROJECT_ID,
            "run_id": self.run_id,
            "status": self.status,
            "mode": self.mode,
            "evidence_status": self.evidence_status,
            "git_commit": None,
            "data": self.data,
            "configuration_hash": self.configuration_hash,
            "seed": self.seed,
            "environment": {"python": platform.python_version(), "device": "cpu"},
            "metrics_file": self.metrics_file,
            "predictions_file": self.predictions_file,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": self.notes,
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.as_dict(), sort_keys=False), encoding="utf-8")


def make_run_id(mode: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{mode}"
