"""Charts drawn from ACTUAL demo predictions. No curve is ever synthesised from a
reported summary number."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from templar_fraud_lab.schema import LABEL  # noqa: E402

CAPTION = "Synthetic fixture v1, seed 42, test slice. Computed by the demo run; not a benchmark result."
COLOR_A, COLOR_B, COLOR_GRID = "#2E5A88", "#C0602B", "#BBBBBB"


def _finish(fig: plt.Figure, ax: plt.Axes, title: str, path: Path) -> None:
    ax.set_title(title)
    ax.grid(color=COLOR_GRID, linewidth=0.5, alpha=0.6)
    fig.text(0.01, 0.005, CAPTION, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, format="svg")
    plt.close(fig)


def roc_chart(test: pd.DataFrame, path: Path) -> None:
    fpr, tpr, _ = roc_curve(test[LABEL], test["p_calibrated"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=COLOR_A, label="calibrated stacked score")
    ax.plot([0, 1], [0, 1], "--", color=COLOR_GRID, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right")
    _finish(fig, ax, "ROC (demo)", path)


def pr_chart(test: pd.DataFrame, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(test[LABEL], test["p_calibrated"])
    prevalence = float(test[LABEL].mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.step(recall, precision, where="post", color=COLOR_A, label="calibrated stacked score")
    ax.axhline(prevalence, linestyle="--", color=COLOR_B, label=f"no-skill baseline (prevalence {prevalence:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    _finish(fig, ax, "Precision-recall (demo)", path)


def reliability_chart(reliability: dict, path: Path) -> None:
    after = reliability["after_ot"]["bins"]
    before = reliability["before_ot"]["bins"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color=COLOR_GRID, label="perfect calibration")
    for bins, color, label in ((before, COLOR_B, "before OT"), (after, COLOR_A, "after OT")):
        xs = [b["mean_predicted"] for b in bins if b["count"] > 0]
        ys = [b["observed_rate"] for b in bins if b["count"] > 0]
        ax.plot(xs, ys, "o-", color=color, label=f"{label} (ECE {reliability['after_ot' if label == 'after OT' else 'before_ot']['ece']:.3f})")
    for b in after:
        if b["count"] > 0:
            ax.annotate(f"n={b['count']}", (b["mean_predicted"], b["observed_rate"]), textcoords="offset points",
                        xytext=(0, 6), fontsize=6, ha="center", color="#333333")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed fraud rate (bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    _finish(fig, ax, "Reliability diagram with bin counts (demo)", path)


def threshold_cost_chart(curve: pd.DataFrame, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(curve["threshold"], curve["val_cost"], color=COLOR_A, label="validation cost (demo cost matrix)")
    ax.axvline(threshold, color=COLOR_B, linestyle="--", label=f"selected t* = {threshold:.3f}")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("cost (demo units)")
    ax.legend(loc="upper right")
    _finish(fig, ax, "Threshold versus cost on validation (demo)", path)


def route_latency_chart(metrics: dict, path: Path) -> None:
    lat = metrics["latency_ms_per_row"]
    labels = ["fast path", "escalation path"]
    values = [lat.get("fast") or 0.0, lat.get("escalation") or 0.0]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(labels, values, color=[COLOR_A, COLOR_B])
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2f} ms", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_ylabel("wall-clock ms per row (this machine)")
    _finish(fig, ax, "Route latency, measured during the demo run", path)


def render_charts(output_dir: Path, figures_dir: Path) -> list[Path]:
    preds = output_dir / "predictions.csv"
    metrics_path, rel_path, curve_path = output_dir / "metrics.json", output_dir / "reliability.json", output_dir / "threshold_curve.csv"
    for p in (preds, metrics_path, rel_path, curve_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run `python -m templar_fraud_lab demo` first")
    frame = pd.read_csv(preds)
    test = frame[frame["split"] == "test"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reliability = json.loads(rel_path.read_text(encoding="utf-8"))
    curve = pd.read_csv(curve_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "chart_roc.svg": lambda p: roc_chart(test, p),
        "chart_pr.svg": lambda p: pr_chart(test, p),
        "chart_reliability.svg": lambda p: reliability_chart(reliability, p),
        "chart_threshold_cost.svg": lambda p: threshold_cost_chart(curve, float(metrics["threshold"]), p),
        "chart_route_latency.svg": lambda p: route_latency_chart(metrics, p),
    }
    written = []
    for name, fn in out.items():
        path = figures_dir / name
        fn(path)
        written.append(path)
    return written
