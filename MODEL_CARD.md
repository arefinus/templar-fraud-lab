# Model card

This repository exposes several fitted components. All of them are trained only on the
synthetic fixture by `make demo`; the checkpoints under `examples/output/models/` are
regenerated on every run and are not committed.

## Components

| Component | Implementation | Fitted on | Used for |
|---|---|---|---|
| Preprocessor | `preprocess/tabular.py` | training split | median imputation, missing token, one-hot, frequency encoding, RobustScaler |
| Target encoder | `preprocess/target_encoding.py` | training split, forward out-of-fold | smoothed encoding of `email_domain` |
| SAINT-style encoder | `models/saint.py`, torch, 135,313 parameters in the demo configuration (computed by `make demo` on synthetic fixture v1, seed 42) | training split, early stopping on validation average precision | row embedding `z` and probability `p_z` |
| Triage | CatBoost, ordered boosting, `triage.py` | training split | `p_triage` on [z, x], routing |
| Escalation model | logistic regression over template firings, `escalation.py` | training split | `p_escalation` for verifier-approved rows |
| Stacking | logistic-normal variational fit, Dirichlet(1) prior, `stacking.py` | validation split | combined probability and disagreement proxy |
| Calibrator | 1-D optimal transport by quantile matching, `calibration.py` | validation split | calibrated probability |
| Threshold | grid search on expected cost, `decision.py` | validation split | fraud / legitimate cut-off |

## Intended use

Demonstration and inspection of the pipeline mechanics on synthetic data; a starting point
for a real-data study by someone who has obtained BAF or IEEE-CIS under their terms.

## Out-of-scope use

Any operational fraud decision. No component here has seen real transactions. The demo
metrics in `examples/output/metrics.json` describe a 6,000-row synthetic fixture and are
labelled `demo`; they say nothing about performance on real payments.

## Training details (demo configuration, `configs/demo.yaml`)

- Encoder: token dimension 16, output 16, 2 blocks, 2 heads, feed-forward multiplier 2,
  dropout 0.10, intersample bottleneck 64, context window 256 rows, class-weighted BCE, AdamW
  learning rate 0.001, weight decay 1e-5, gradient clipping 1.0, one warmup epoch then cosine
  decay, up to 8 epochs, patience 3.
- Triage: 400 ordered trees, depth 2, learning rate 0.03, l2 5, random strength 0.30, bagging
  temperature 0.50, `scale_pos_weight` = legit/fraud ratio of the training fold.
- Stacking: Adam 0.01, up to 600 iterations, 32 Monte Carlo samples, patience 100 on
  validation NLL.
- Calibration: up to 32 quantile knots, at least 20 validation rows per knot.
- Decision: cost matrix `configs/demo_costs.yaml` (demo units), grid [0.001, 0.500] step 0.001.

The paper's own settings are recorded separately in `configs/paper_saint.yaml` and
`configs/paper_pipeline.yaml` with `evidence_status: paper-reported`; the demo does not run
them.

## Evaluation

See `docs/evaluation.md`. Metrics are reported on the validation, test and late slices with
both average precision and trapezoidal PR-AUC, precision, recall, specificity, F1, ECE with
bin counts, route shares, verifier-fail share and total cost.

## Known failure modes

- With 74 fraud rows in the training split the encoder and the tree ensemble overfit within a
  few epochs or a few hundred trees; early stopping and shallow trees are load-bearing.
- Quantile-matching calibration caps the calibrated probability at the highest binned label
  rate seen on validation; with few positives that cap is low.
- The verifier's range check uses the training [min, max]; legitimate extreme values on new
  data will be routed to review. That is intended, and it costs `c_review` per row.

## Safety and integrity

Predicates are evaluated by an allowlisted interpreter, never `eval`. Checkpoints are
`torch.save` state_dicts loaded with `weights_only=True`, CatBoost JSON, and plain JSON for
the stacking weights and calibrator knots. No pickle of untrusted input.
