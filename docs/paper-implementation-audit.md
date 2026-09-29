# Paper implementation audit

Source: "TEMPLAR fraud a verifier grounded calibrated and cost sensitive framework for
transaction fraud detection", Discover Data 4:20 (2026), DOI
https://doi.org/10.1007/s44248-026-00117-y. The article cites
https://github.com/raihandinaribn/templar-fraud as its code; that repository was not audited or
reused here. This page lists every point where the article's text underdetermines an
implementation, what this repository does, the evidence status of anything measured, and the
assumption made. `python -m templar_fraud_lab audit` prints a short form with the measured
values from the latest demo run (`examples/output/audit.json`).

Evidence labels: `paper-reported` (transcribed from the article), `demo` (computed here on the
synthetic fixture, not a benchmark), `documented-only` (no measurement possible here).

## A01. Temporal overlap of the final-10% slice with the 15% test split

- Paper says: results are reported on a chronological 70/15/15 split and additionally on the
  final 10% chronological slice of the same dataset, described as a robustness check. Whether
  the 10% slice is carved out of the 15% test split or lies elsewhere is not stated.
- This repository does: defines the late slice as the last ceil(0.10 n) rows, computes its
  intersection with every split and writes `late_slice_overlap.json`.
- Evidence: `demo`. On the fixture 600 of 600 late rows are inside the test split and they
  form 66.7% of it, so late-slice metrics are not independent of test metrics.
- Assumption: same construction on real data; the late slice is a subset of test. The
  paper's late-slice numbers should be read as a within-test drift check, not as an external
  validation.

## A02. Calendar and entity availability in BAF

- Paper says: temporal features for D1 are month index and day-of-week; graphs are
  card-device-IP (D1) and card-merchant-device (D2).
- This repository does: the BAF adapter exposes `month` as the only chronological key and uses
  the row id as the entity id, because the published Base table has no day, no card, no
  device and no IP column. No graph features are built for either dataset.
- Evidence: `documented-only` (the data was not downloaded; the column list is from the BAF
  documentation and is validated by the adapter).
- Assumption: the paper either used a BAF variant or derived fields not in the Base table, or
  the day-of-week and graph features were D2-only. Any BAF result that depends on them cannot
  be reproduced from Base as published.

## A03. Calibration sample count

- Paper says: the calibrator was "fitted using 100,000 validation-fold predictions". The 15%
  validation split of IEEE-CIS is 88,581 rows and BAF's is 150,000.
- This repository does: fits the calibrator on the validation split, whatever its size (900
  rows on the fixture).
- Evidence: `demo` for the fit size; the discrepancy itself is `paper-reported`.
- Assumption: the paper pooled out-of-fold predictions from forward-chaining folds, or
  rounded. We do not pool; pooling would mix rows that were scored by different models.

## A04. Routing rule

- Paper says: Eq. 5 routes to the fast path when `p_triage < Gamma_low` and `u_i < T_u`;
  Section 2.7 also gives low/high thresholds 0.18/0.72 (D1) and 0.16/0.74 (D2) with a
  minimum separation of 0.20, which describes a band rule.
- This repository does: implements both behind `routing.rule` (`eq5` default, `band`
  alternative) and exports the escalation share under each on the test slice.
- Evidence: `demo`. Escalation share 65.9% under `eq5` versus 50.1% under `band` on the same
  900 rows, so the choice changes route shares materially.
- Assumption: the paper's reported verifier-routed shares (7.8% and 6.3%, paper-reported)
  correspond to whichever rule produced its tables; we cannot tell which. The Eq. 9
  disagreement cannot be the routing uncertainty because it needs `p_escalation`, which does
  not exist before routing; we use rank disagreement between `p_triage` and `p_z`.

## A05. Fit scope of stacking and calibration

- Paper says: stacking weights are fitted by variational inference on validation NLL; the
  calibrator on validation predictions. Whether "validation" means the 15% split or pooled
  forward-chaining folds is not stated.
- This repository does: fits both on the 15% validation split and applies them once to test.
- Evidence: `demo`. Weights p_triage 0.715, p_escalation 0.270, p_z 0.015 on the fixture.
- Assumption: the 15% split. Pooled OOF would give more rows but mixes model versions.

## A06. Intersample attention leakage

- Paper says: SAINT-style intersample attention with a "history context" of 128; it does not
  say whether a row may attend to later rows in its batch.
- This repository does: masks attention so that row i attends only to rows with `ts <= ts_i`,
  and forms windows chronologically. A test perturbs later rows and asserts earlier embeddings
  are unchanged.
- Evidence: `demo` (behavioural test passes; no measurement of the effect size on real data).
- Assumption: this is what "history context" is meant to mean. Without the mask, a batch that
  contains a later fraud would leak into earlier rows' embeddings.

## A07. Out-of-fold target encoding

- Paper says: smoothed target encoding for D2 high-cardinality columns, out-of-fold, forward
  order.
- This repository does: expanding-window encoding in which training row i uses only strictly
  earlier rows of its category; validation and test rows use full-train counts; smoothing 20.
- Evidence: `demo` (unit test proves a row's encoding does not depend on its own or later
  labels).
- Assumption: expanding window rather than block-wise folds. Block-wise OOF would let a row
  see later rows of the same block, which is a weaker guarantee.

## A08. What "1-D optimal transport calibration" is

- Paper says: a one-dimensional optimal transport map on validation, 256 quantile knots,
  monotone piecewise-linear, at least 200 observations per interval.
- This repository does: quantile matching between the validation score distribution and the
  distribution of binned label rates, realised at the knots by assigning the k-th smallest
  rate to the k-th score bin, piecewise-linear, clamped to [0, 1]; isotonic regression as a
  comparator. The objective, and how it differs from isotonic, is written in
  `calibration.py`.
- Evidence: `demo`. Test-slice ECE 0.266 before, 0.0007 after OT, 0.0048 isotonic; see
  docs/evaluation.md for why a small ECE at 2.8% prevalence is weak evidence.
- Assumption: this is the construction the paper means. Other readings (for example
  transporting to a Beta target, or a Sinkhorn-regularised map) would give different maps.

## A09. Template search

- Paper says: bank of K = 64 templates discovered by MCTS over a 50,000-row corpus with 100
  simulations per expansion, predicate depth at most 3, at most 3 conjunctive terms.
- This repository does: a hand-written 16-template bank labelled `demo`, plus a small MCTS
  (24 simulations, 3 appended templates, forward-chaining reward) that demonstrates the search
  mechanics on the fixture. The bank and the discovered templates are persisted as JSON.
- Evidence: `demo`. Final bank size 19.
- Assumption: none about the paper's bank; ours is not it and should not be compared to it.

## A10. Cost matrix

- Paper says: `t* = argmin C_FP FP + C_FN FN + C_R R`; selected thresholds 0.034 (D1) and
  0.033 (D2); net savings of 56,485 and 221,283 in unstated units under an author-defined
  cost model. The values of C_FP, C_FN and C_R are not given.
- This repository does: a user-editable demo cost matrix (FP 1, FN 20, review 2, demo units),
  an exhaustive grid search with lowest-threshold tie-break, and the full cost curve exported.
- Evidence: `demo`. t* = 0.058 on the fixture with 2 tied grid points.
- Assumption: none about the paper's costs. The savings figures cannot be interpreted or
  reproduced without units.

## A11. Upstream substitutes

- Paper says: the strongest baseline is FraudGT; card-level entity keys and graph features are
  used; the code is at the cited upstream repository.
- This repository does: uses the community card key for IEEE-CIS (`card1|card2|card3|card5|
  addr1`), builds no graph features, implements no FraudGT, and did not read the upstream
  code.
- Evidence: `documented-only`.
- Assumption: differences in entity key or graph construction would change results on real
  data; this repository makes no claim about them.

## Additional departures made here

- Encoder bottleneck: the flattened row is projected to 64 dimensions before intersample
  attention (the original SAINT attends on the full flattened width). Chosen so the demo
  trains in seconds on CPU.
- Routing uncertainty: rank disagreement between `p_triage` and `p_z` (see A04).
- Rows that escalate but fire no template receive no `p_escalation` and are not counted as
  verifier failures.
- The demo triage uses depth-2 trees because the fixture has 74 training positives; the
  paper's depth 6 is recorded in `configs/paper_pipeline.yaml`.
