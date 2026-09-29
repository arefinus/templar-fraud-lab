# Evaluation

Every number on this page was computed by `make demo` on synthetic fixture v1, seed 42, and
is not a benchmark result. The source file is `examples/output/metrics.json`; the run manifest
is `examples/output/run_manifest.yaml`. Paper-reported values live only in
`evidence/paper_reported/templar.json` and are not repeated here as results.

## Protocol

Chronological 70/15/15 split of 6,000 rows (train 4,200, validation 900, test 900); late slice
= final 600 rows, entirely inside the test split. Everything that fits uses train; everything
that selects uses validation; test and late are scored once. Ranking metrics are reported on
the calibrated stacked probability and, for reference, on the stacked, triage-only and
encoder-only scores. Both average precision and trapezoidal PR-AUC are given because they
differ and the paper does not say which it used. ECE uses 10 equal-width bins with counts.

## Demo results (synthetic fixture v1, seed 42, computed by `make demo`, not a benchmark)

| Slice | Rows | Fraud | AUROC (calibrated) | Average precision | PR-AUC trapezoidal | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|---|---|---|
| validation | 900 | 23 | 0.712 | 0.075 | 0.115 | 0.077 | 0.565 | 0.823 | 0.136 |
| test | 900 | 25 | 0.719 | 0.104 | 0.170 | 0.078 | 0.583 | 0.809 | 0.137 |
| late (final 10%) | 600 | 13 | 0.749 | 0.131 | 0.204 | 0.062 | 0.583 | 0.818 | 0.112 |

Component scores on the test slice (synthetic, demo): triage-only AUROC 0.705, encoder-only
AUROC 0.661, stacked (before calibration) AUROC 0.712. A plain logistic regression on the same
preprocessed features reached AUROC 0.78 on the test slice in a side check; the fixture's 74
training positives are too few for the tree ensemble to beat a linear model, which is a
property of the fixture, not a finding about the method.

Routing on the test slice (demo, synthetic): 34.1% fast, 65.9% escalation under `eq5`; the
`band` rule would escalate 50.1% of the same rows. Verifier-fail share 0.6% (5 of 900 rows,
all on the late-appearing `crypto_exchange` category or on amounts above the training
maximum), all routed to review. Decision shares: fraud 20.0%, legitimate 79.4%, review 0.6%.

Calibration on the test slice (demo, synthetic): ECE 0.266 before OT calibration, 0.0007 after,
0.0048 for the isotonic comparator. Read this with care: at 2.8% prevalence nearly every row
falls in the first equal-width bin, where mean predicted and observed rate agree almost by
construction, so a small ECE here is weak evidence. The per-bin counts are in
`examples/output/reliability.json` and drawn in `docs/figures/chart_reliability.svg`.

Cost on the test slice with the demo cost matrix (FP 1, FN 20, review 2; demo units): 166 false
positives, 10 false negatives, 5 reviews, total 376. Selected threshold t* = 0.058 (validation
cost 357, 2 grid points tied, lowest chosen). Stacking weights: p_triage 0.715, p_escalation
0.270, p_z 0.015.

Encoder: 135,313 parameters, best epoch 1 of 5 run (early stopped), validation average
precision at the best epoch 0.156, training time 31.6 s on the machine that produced the
outputs. Route latency measured during the same run: 0.005 ms per row on the fast path and
0.99 ms per row on the escalation path (wall-clock, this machine, synthetic; not a benchmark).

## How to read these numbers

They demonstrate that each stage runs, that the leakage controls hold, and that the outputs
have the documented shape. They do not estimate performance on BAF or IEEE-CIS. The small
number of positives makes every ranking metric noisy (a handful of rows moves AUROC by several
points), and the fixture's label model is deliberately not the kind of structure a real fraud
stream has.

## Charts

`docs/figures/chart_roc.svg`, `chart_pr.svg` (with the no-skill baseline at the test
prevalence), `chart_reliability.svg` (bin counts annotated), `chart_threshold_cost.svg`
(validation cost over the whole grid with t* marked), `chart_route_latency.svg`. All are drawn
from `examples/output/predictions.csv` and `metrics.json`.

## Local test run (2026-09-27)

```
python -m pytest -q
88 passed in 50.14s
```

Run on Python 3.10.11, Windows 11, CPU. Test list by module: data and splits (15),
preprocessing (7), encoder (7), templates and AST (18, of which 9 are parametrised rejection
cases), verifier (9), stacking and calibration (10), decision, metrics and routing (12),
pipeline and CLI (11, including one full synthetic train-to-export run).
