# Data handling

See DATA_CARD.md for the three data sources. This page describes what the code does with
whatever frame an adapter returns.

## Canonical schema

Every adapter maps into the columns in `templar_fraud_lab/schema.py`: `row_id`, `ts`,
`card_id`, `is_fraud`, plus a `FeatureSchema` naming numeric, low-cardinality and
high-cardinality columns in a fixed order. The preprocessor preserves that order and the
tests assert it (`tests/test_preprocess.py::test_exact_feature_order`).

## Chronology

Frames are sorted by `(ts, row_id)` with a stable sort, so ties are broken deterministically.
`derive_temporal_features` refuses an unsorted frame. `time_since_last_card_event` uses only
the previous event of the same card, so computing it once over the whole frame leaks nothing
across splits.

## Splits

`chronological_split` cuts the sorted frame at floor(0.70 n) and floor(0.85 n). The late
slice is the last ceil(0.10 n) rows. `assert_strict_chronology` checks that no training row
is later than any validation row and no validation row later than any test row.
`late_slice_overlap` counts the intersection with each split and writes it to
`late_slice_overlap.json`; on the fixture the late slice is entirely inside the test split
(600 of 600 rows) and makes up two thirds of it.

`forward_chaining_folds(n, 5)` cuts n rows into six contiguous blocks; fold j trains on blocks
0..j and validates on block j+1. It drives the MCTS reward.

## Fitting scope

| Stage | Fitted on | Applied to |
|---|---|---|
| medians, category levels, frequencies, RobustScaler | train | train, val, test, late |
| smoothed target encoding | train (expanding window, each row sees only earlier labels) | val, test, late use full-train counts |
| encoder | train; early stopping on val | all |
| triage, escalation model, MCTS | train | all |
| stacking weights, OT calibrator, isotonic comparator, threshold | val | test, late |

The test-perturbation test (`test_fitted_statistics_do_not_change_when_test_rows_are_perturbed`)
hashes the whole fitted state, transforms a deliberately corrupted test frame, and asserts the
hash is unchanged.

## Unknown categories and out-of-range values

At transform time an unseen category becomes an all-zero one-hot block and a frequency of 0.
The preprocessor also reports which categories were unseen (`unknown_categories`), and the
verifier independently rejects rationales whose observed categorical value is not a training
level or whose numeric value lies outside the training [min, max]. On the fixture the test
slice contains one unseen level, `crypto_exchange`, and a few amounts far above the training
maximum; both reach the verifier and are routed to review.

## Missing values

Numeric missing values are imputed with training medians for the model inputs. Templates see
the raw row, where a missing value is `None`; the predicate interpreter evaluates any
comparison against `None` as false, and a template can test for missingness explicitly
(template T16 does).
