# Method

The pipeline has nine stages. Each is a module under `src/templar_fraud_lab/`, and
`pipeline.py` wires them in order. Figures: `docs/figures/fig1_pipeline.svg`,
`fig2_escalation.svg`, `fig3_stacking_calibration_decision.svg` (original drawings adapted
from the structure of the article's figures).

## 1. Preprocessing (`preprocess/`)

Train-only median imputation, an explicit missing token, one-hot for low-cardinality
categoricals, frequency encoding for high-cardinality ones, RobustScaler. Smoothed target
encoding of `email_domain` is out-of-fold in forward order: the encoding of training row i is
`(s + m * prior) / (n + m)` where `n` and `s` count earlier rows of the same category and
their positives; validation and test rows use the full training counts. Smoothing `m` = 20.

## 2. Encoder (`models/saint.py`)

Each scalar feature becomes a token through a per-feature affine map; a CLS token is
prepended. A block applies (a) multi-head self-attention across the tokens of one row with a
feed-forward layer, then (b) intersample attention across the rows of the context window.
Intersample attention in the original SAINT runs on the flattened row; here the flattened row
is projected to `inter_dim` (64) before attention and back afterwards, which keeps the demo
model at 135 k parameters. The CLS token after the last block gives `z` (tanh, dimension 16)
and a logit head gives `p_z`.

Historical-context policy: with `forbid[i, j] = ts_j > ts_i` passed as the attention mask, a
row attends to earlier-or-equal rows only. Windows are contiguous chronological blocks. The
test `test_no_future_context_leaks_into_earlier_rows` perturbs later rows and asserts earlier
embeddings are unchanged to 1e-6.

Training: class-weighted BCE (or focal loss, configurable), AdamW, gradient clipping at 1.0,
one warmup epoch then cosine decay, early stopping on validation average precision.

## 3. Triage and routing (`triage.py`)

CatBoost with ordered boosting on the concatenation [z, x], `scale_pos_weight` equal to the
legit/fraud ratio of the training fold. Two routing rules:

- `eq5` (default): fast if `p_triage < gamma_low` and `u < t_u`, otherwise escalation.
- `band`: fast if `p_triage < low` or `p_triage > high`, escalation inside the band; the two
  thresholds must be at least 0.20 apart.

The uncertainty `u` for `eq5` defaults to rank disagreement, `|F_triage(p_triage) -
F_z(p_z)|`, with `F` the empirical CDF of each score on training rows. Raw
`|p_triage - p_z|` and a margin `1 - 2|p_triage - 0.5|` are available as alternatives; the
docstring in `triage.py` explains why the margin variant collapses to a single threshold.

## 4. Templates and rationales (`templates/`)

A predicate is a JSON tree of `{"const"}`, `{"feat"}` and `{"op", "args"}` nodes. Allowed
operators: comparisons (`> >= < <= == != in`), arithmetic (`+ - * /`), logic (`and or not`)
and aggregates (`sum mean max min`) over feature references. `validate` rejects anything else
and bounds depth; `evaluate` is a small recursive interpreter. A `Rationale` is a frozen
dataclass `(template_id, feature_ids, observed_values, predicates, verification_status,
reason_codes)`.

The hand-written bank has 16 templates and is labelled `demo`. `templates/mcts.py` is a small
Monte Carlo tree search over conjunctions of up to three atoms (quantile thresholds,
categorical equalities, one ratio atom), UCB1 selection, random rollout, reward = mean
forward-chaining validation average precision of a logistic model over the firings of the
current bank plus the candidate. The demo runs 24 simulations per template and appends three
templates (`M01` to `M03`), persisted to `examples/output/template_bank.json`. It is not the
paper's search (see audit A09).

## 5. Verifier (`verifier.py`)

Four deterministic checks, no parameters: the template exists and its predicate matches the
bank; every referenced feature is in the schema, listed in `feature_ids` and present in
`observed_values`; numeric observations lie within the training [min, max] and categorical
observations are training levels; the predicate re-evaluates to true on the observed values.
Any failure gives status `fail` with reason codes and the row is routed to review. A failed
verification never produces a fraud or legitimate verdict.

## 6. Escalation probability (`escalation.py`)

A logistic regression over the template firing vector, fitted on training rows, gives
`p_escalation` for rows whose rationales all passed. Rows with no firing template have no
`p_escalation` and are not counted as failures.

## 7. Stacking (`stacking.py`)

Sources `p_triage`, `p_escalation` (optional), `p_z`. Missing sources are NaN and are masked;
weights are renormalised per row over the available sources. A Dirichlet(1) prior is uniform on
the simplex, so the variational objective reduces to the expected log-likelihood under a
logistic-normal `q` plus its entropy and the softmax log-Jacobian, estimated with
reparameterised Monte Carlo samples and optimised with Adam, early stopping on validation NLL.
The disagreement proxy is the availability-weighted standard deviation of the sources around
the stacked probability.

## 8. Calibration (`calibration.py`)

Validation rows are cut into K quantile bins with at least `min_support` rows each; bin k has
median score x_k and label rate r_k. The 1-D optimal transport map from the score distribution
to the distribution of bin rates is the monotone quantile-matching map, realised by assigning
the k-th smallest rate to the k-th bin and interpolating piecewise-linearly, clamped to [0, 1].
It pushes the validation score distribution onto the label-rate distribution exactly (mean
calibrated probability = prevalence). Isotonic regression is fitted alongside as a comparator:
it minimises squared error under monotonicity and pools adjacent violators, whereas quantile
matching rearranges rates and preserves their marginal distribution.

## 9. Decision (`decision.py`)

`t*` minimises `C_FP FP + C_FN FN + C_R R` over the grid [0.001, 0.500] step 0.001 on the
validation split; ties go to the lowest threshold. Review rows are those with a failed
verification or a disagreement above `review_uncertainty`; they pay `C_R` at every threshold.
Remaining rows are `fraud` if `p >= t*` and `legitimate` otherwise. The cost matrix is a demo
file, `configs/demo_costs.yaml`.

## Settings taken from the source paper

Recorded with `evidence_status: paper-reported` in `configs/paper_saint.yaml` and
`configs/paper_pipeline.yaml`, and NOT run by the demo: encoder with 6 blocks, 8 heads, token
dimension 32, output 64, history context 128, feed-forward x4, dropout 0.10, focal gamma 2,
weight decay 1e-5, AdamW, mixed precision, gradient clip 1.0, 5 warmup epochs, cosine decay,
up to 100 epochs, patience 12, learning rate 3e-4 with batch 2048 (D1) and 2e-4 with batch 1024
(D2); TPE search of 50 trials (10 random) on mean forward-chaining validation AUC-PR with ties
under 0.002 broken by lower validation expected cost; CatBoost 4000 ordered trees, depth 6,
learning rate 0.03, l2 5, random strength 0.30, bagging temperature 0.50; template bank K = 64
from a 50,000-row discovery corpus with 100 MCTS simulations per expansion, depth at most 3
and at most 3 conjuncts; stacking with Dirichlet(1), learning rate 0.01, 3000 iterations, 32 MC
samples, patience 200; calibration with 256 quantile knots and at least 200 observations per
interval; decision grid [0.001, 0.500] step 0.001 with selected thresholds 0.034 (D1) and
0.033 (D2); routing thresholds 0.18/0.72 (D1) and 0.16/0.74 (D2) with minimum separation 0.20;
seeds 11, 23, 37, 53, 79.
