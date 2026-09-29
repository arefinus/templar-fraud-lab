from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from templar_fraud_lab.preprocess.tabular import TabularPreprocessor
from templar_fraud_lab.preprocess.target_encoding import forward_oof_target_encode, target_encode_apply
from templar_fraud_lab.schema import FIXTURE_SCHEMA, LABEL


@pytest.fixture
def fitted(small_frame):
    train = small_frame.iloc[:1000]
    return TabularPreprocessor(FIXTURE_SCHEMA).fit(train), train, small_frame.iloc[1000:].reset_index(drop=True)


def test_exact_feature_order(fitted):
    pre, _, _ = fitted
    names = pre.feature_names
    assert names[: len(FIXTURE_SCHEMA.numeric)] == list(FIXTURE_SCHEMA.numeric)
    onehot = [n for n in names if n.startswith("merchant_category=")]
    assert onehot == sorted(onehot)
    assert names[-1] == "email_domain__freq"
    assert names.index("merchant_category=apparel") < names.index("device_type=emulator")


def test_fitted_statistics_do_not_change_when_test_rows_are_perturbed(fitted):
    pre, train, test = fitted
    before = pre.state_hash()
    perturbed = test.copy()
    perturbed["amount"] = perturbed["amount"] * 1000
    perturbed["merchant_category"] = "totally_new_category"
    perturbed.loc[:, "ip_risk_score"] = np.nan
    _ = pre.transform(perturbed)
    assert pre.state_hash() == before
    refit = TabularPreprocessor(FIXTURE_SCHEMA).fit(train)
    assert refit.state_hash() == before


def test_unknown_categories_map_to_zero_columns(fitted):
    pre, _, test = fitted
    row = test.iloc[[0]].copy()
    row["merchant_category"] = "unseen_level"
    x = pre.transform(row)
    cols = [i for i, n in enumerate(pre.feature_names) if n.startswith("merchant_category=")]
    center = np.array(pre.scaler.center_)[cols]
    scale = np.array(pre.scaler.scale_)[cols]
    assert np.allclose(x[0, cols] * scale + center, 0.0, atol=1e-6)
    assert pre.unknown_categories(row) == {"merchant_category": ["unseen_level"]}


def test_missing_values_are_imputed_with_train_medians(fitted):
    pre, train, test = fitted
    row = test.iloc[[0]].copy()
    row["ip_risk_score"] = np.nan
    x = pre.transform(row)
    j = pre.feature_names.index("ip_risk_score")
    imputed = x[0, j] * pre.scaler.scale_[j] + pre.scaler.center_[j]
    assert imputed == pytest.approx(pre.medians["ip_risk_score"], abs=1e-5)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        TabularPreprocessor(FIXTURE_SCHEMA).transform(pd.DataFrame())


def test_forward_oof_encoding_never_sees_own_or_future_labels():
    cats = pd.Series(["a", "a", "b", "a", "b", "a"])
    y = pd.Series([1, 0, 1, 1, 0, 0])
    enc, stats, prior = forward_oof_target_encode(cats, y, smoothing=1.0)
    # first row of each category sees only the prior
    assert enc[0] == pytest.approx(prior)
    assert enc[2] == pytest.approx(prior)
    # swap labels of rows 3 and 5 (same category, same total so the global prior is unchanged):
    # rows 0..4 must be unchanged (row 3 never sees its own label), row 5 must change.
    y2 = y.copy()
    y2[3], y2[5] = 0, 1
    enc2, _, prior2 = forward_oof_target_encode(cats, y2, smoothing=1.0)
    assert prior2 == pytest.approx(prior)
    assert np.allclose(enc2[:5], enc[:5])
    assert enc2[5] < enc[5]
    assert stats["a"] == (4, 2)


def test_target_encode_apply_unknown_category_returns_prior():
    _, stats, prior = forward_oof_target_encode(pd.Series(["a", "b"]), pd.Series([1, 0]), smoothing=5.0)
    out = target_encode_apply(pd.Series(["zzz"]), stats, prior, smoothing=5.0)
    assert out[0] == pytest.approx(prior)
