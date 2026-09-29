"""Command line entry: python -m templar_fraud_lab {smoke,demo,train,evaluate,audit,make-fixture,figures}."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from templar_fraud_lab import __version__
from templar_fraud_lab.config import load_config
from templar_fraud_lab.manifest import RunManifest, make_run_id, utc_now
from templar_fraud_lab.pipeline import PipelineState, run_all_stages, run_training_stages

EXIT_OK, EXIT_FAIL, EXIT_MISSING_INPUT = 0, 1, 2
REQUIRED_DEMO_OUTPUTS = ("run_manifest.yaml", "predictions.csv", "metrics.json", "reliability.json",
                         "late_slice_overlap.json", "demo_trace.json", "template_bank.json")


def _manifest(state: PipelineState, cfg_hash: str) -> RunManifest:
    return RunManifest(run_id=make_run_id(state.cfg.mode), mode=state.cfg.mode, seed=state.cfg.seed,
                       configuration_hash=cfg_hash)


def _finish_manifest(m: RunManifest, state: PipelineState, status: str) -> None:
    m.status = status
    m.finished_at = utc_now()
    m.data = {"source_id": "synthetic-fixture-v1", "version": "v1",
              "split_manifest_hash": state.info.get("split_manifest_hash"),
              "sample_counts": state.info.get("sample_counts", {})}
    m.write(state.out_dir / "run_manifest.yaml")


def run_pipeline_command(config_path: str, output_dir: str | None, training_only: bool = False) -> int:
    from templar_fraud_lab.outputs import write_metrics, write_predictions, write_reliability, write_threshold_curve
    from templar_fraud_lab.trace import write_trace

    cfg = load_config(config_path)
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = PipelineState(cfg=cfg, out_dir=out_dir)
    manifest = _manifest(state, cfg.configuration_hash())
    t0 = time.perf_counter()
    try:
        if training_only:
            run_training_stages(state)
            manifest.notes.append("training stages only: encoder + triage checkpoints written to models/")
        else:
            run_all_stages(state)
            write_predictions(state, out_dir / "predictions.csv")
            metrics = write_metrics(state, out_dir / "metrics.json")
            write_reliability(metrics, out_dir / "reliability.json")
            write_threshold_curve(state, out_dir / "threshold_curve.csv")
            write_trace(state, out_dir / "demo_trace.json")
            manifest.metrics_file = (out_dir / "metrics.json").as_posix()
            manifest.predictions_file = (out_dir / "predictions.csv").as_posix()
    except Exception as exc:  # noqa: BLE001 - we record failure then re-raise as exit code
        _finish_manifest(manifest, state, "failed")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAIL
    _finish_manifest(manifest, state, "completed")
    elapsed = time.perf_counter() - t0
    if not training_only:
        test = metrics["slices"]["test"]
        print(f"{cfg.mode}: completed in {elapsed:.1f}s; outputs in {out_dir}")
        print(f"  test rows {test['rows']} (fraud {test['fraud_rows']}); threshold {metrics['threshold']}; "
              f"routes fast {test['routes']['fast_share']:.3f} / escalation {test['routes']['escalation_share']:.3f}; "
              f"verifier fail {test['routes']['verifier_fail_share']:.3f}")
        print("  synthetic fixture only; metrics are demo evidence, not a benchmark")
    else:
        print(f"train: completed in {elapsed:.1f}s; checkpoints in {out_dir / 'models'}")
    return EXIT_OK


def cmd_smoke(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    rc = run_pipeline_command(args.config, str(out))
    if rc != EXIT_OK:
        return rc
    missing = [n for n in REQUIRED_DEMO_OUTPUTS if not (out / n).exists()]
    if missing:
        print(f"smoke: missing outputs {missing}", file=sys.stderr)
        return EXIT_FAIL
    trace = json.loads((out / "demo_trace.json").read_text(encoding="utf-8"))
    for key in ("schemaVersion", "mode", "scenarioId", "steps"):
        if key not in trace:
            print(f"smoke: demo_trace.json missing key {key}", file=sys.stderr)
            return EXIT_FAIL
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("evidence_status") != "demo" or "slices" not in metrics:
        print("smoke: metrics.json schema mismatch", file=sys.stderr)
        return EXIT_FAIL
    print("smoke: outputs present and schema checks passed")
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    rc = run_pipeline_command(args.config, args.output_dir)
    if rc == EXIT_OK and not args.no_figures:
        return cmd_figures(args)
    return rc


def cmd_train(args: argparse.Namespace) -> int:
    return run_pipeline_command(args.config, args.output_dir, training_only=True)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from templar_fraud_lab.evaluate import evaluate_predictions

    preds = Path(args.predictions)
    if not preds.exists():
        print(f"evaluate: predictions file not found: {preds} (run demo first)", file=sys.stderr)
        return EXIT_MISSING_INPUT
    costs_path = Path(args.costs)
    if not costs_path.exists():
        print(f"evaluate: cost matrix not found: {costs_path}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    report = evaluate_predictions(preds, costs_path, Path(args.out))
    print(json.dumps({k: report[k] for k in ("evidence_status", "rows", "protocol")}, indent=2))
    print(f"evaluate: wrote {args.out}")
    return EXIT_OK


def cmd_audit(args: argparse.Namespace) -> int:
    from templar_fraud_lab.audit import build_audit, format_audit

    audit = build_audit(Path(args.output_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    print(format_audit(audit))
    return EXIT_OK


def cmd_make_fixture(args: argparse.Namespace) -> int:
    from templar_fraud_lab.data.synthetic import write_fixture

    path = write_fixture(args.out, n_rows=args.rows, seed=args.seed)
    print(f"make-fixture: wrote {path} (rows {args.rows}, seed {args.seed})")
    return EXIT_OK


def cmd_figures(args: argparse.Namespace) -> int:
    from templar_fraud_lab.charts import render_charts

    out_dir = Path(getattr(args, "output_dir", None) or "examples/output")
    try:
        written = render_charts(out_dir, Path(args.figures_dir))
    except FileNotFoundError as exc:
        print(f"figures: {exc}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    print("figures: wrote " + ", ".join(p.name for p in written))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="templar_fraud_lab", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke", help="fast offline run on the fixture with schema checks (< 30 s)")
    s.add_argument("--config", default="configs/smoke.yaml")
    s.add_argument("--output-dir", default=".smoke_output")
    s.set_defaults(func=cmd_smoke)

    d = sub.add_parser("demo", help="documented demonstration on the fixture; writes examples/output/")
    d.add_argument("--config", default="configs/demo.yaml")
    d.add_argument("--output-dir", default=None)
    d.add_argument("--figures-dir", default="docs/figures")
    d.add_argument("--no-figures", action="store_true")
    d.set_defaults(func=cmd_demo)

    t = sub.add_parser("train", help="train encoder and triage on the fixture; writes checkpoints")
    t.add_argument("--config", default="configs/demo.yaml")
    t.add_argument("--output-dir", default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate saved predictions with the stated protocol")
    e.add_argument("--predictions", default="examples/output/predictions.csv")
    e.add_argument("--costs", default="configs/demo_costs.yaml")
    e.add_argument("--out", default="examples/output/metrics_evaluate.json")
    e.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("audit", help="print the paper-implementation audit with measured quantities")
    a.add_argument("--output-dir", default="examples/output")
    a.set_defaults(func=cmd_audit)

    m = sub.add_parser("make-fixture", help="regenerate the synthetic fixture")
    m.add_argument("--out", default="examples/fixtures/v1")
    m.add_argument("--rows", type=int, default=6000)
    m.add_argument("--seed", type=int, default=42)
    m.set_defaults(func=cmd_make_fixture)

    f = sub.add_parser("figures", help="render charts from actual demo predictions")
    f.add_argument("--output-dir", default="examples/output")
    f.add_argument("--figures-dir", default="docs/figures")
    f.set_defaults(func=cmd_figures)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
