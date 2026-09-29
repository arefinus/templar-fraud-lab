from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from templar_fraud_lab.data.baf import load_baf
from templar_fraud_lab.data.errors import DatasetNotAvailable, SchemaMismatch
from templar_fraud_lab.data.ieee_cis import load_ieee_cis
from templar_fraud_lab.data.synthetic import generate_fixture, load_fixture, write_fixture
from templar_fraud_lab.data.temporal import NO_HISTORY_SENTINEL, derive_temporal_features
from templar_fraud_lab.schema import ENTITY_ID, FIXTURE_SCHEMA, LABEL, ROW_ID, TIMESTAMP
from templar_fraud_lab.splits.chrono import (
    assert_strict_chronology,
    chronological_split,
    late_slice_overlap,
    split_manifest_hash,
)
from templar_fraud_lab.splits.forward_chain import assert_forward_only, forward_chaining_folds


def test_fixture_is_deterministic_and_synthetic_looking():
    a, b = generate_fixture(500, seed=3), generate_fixture(500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a[ROW_ID].str.match(r"^SYN-TX-\d{6}$").all()
    assert a[ENTITY_ID].str.match(r"^SYN-CARD-\d{4}$").all()


def test_fixture_prevalence_near_two_percent():
    frame = generate_fixture(6000, seed=42)
    assert 0.012 < frame[LABEL].mean() < 0.03


def test_write_and_load_fixture_roundtrip(tmp_path):
    write_fixture(tmp_path, n_rows=300, seed=5)
    assert (tmp_path / "FIXTURE.md").read_text(encoding="utf-8").startswith("# Fixture v1")
    loaded = load_fixture(tmp_path)
    assert loaded[TIMESTAMP].is_monotonic_increasing
    assert {"hour", "dow", "time_since_last_card_event"} <= set(loaded.columns)
    FIXTURE_SCHEMA.validate_frame(list(loaded.columns))


def test_load_fixture_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="make-fixture"):
        load_fixture(tmp_path)


def test_temporal_features_use_only_past_events():
    frame = pd.DataFrame({ROW_ID: ["a", "b", "c", "d"], TIMESTAMP: [0, 100, 250, 400],
                          ENTITY_ID: ["c1", "c2", "c1", "c1"]})
    out = derive_temporal_features(frame)
    assert list(out["time_since_last_card_event"]) == [NO_HISTORY_SENTINEL, NO_HISTORY_SENTINEL, 250.0, 150.0]


def test_temporal_features_require_sorted_frame():
    frame = pd.DataFrame({ROW_ID: ["a", "b"], TIMESTAMP: [10, 5], ENTITY_ID: ["c", "c"]})
    with pytest.raises(ValueError, match="sorted"):
        derive_temporal_features(frame)


def test_chronological_split_is_strict(small_frame):
    split = chronological_split(small_frame)
    assert_strict_chronology(small_frame, split)
    n = len(small_frame)
    assert len(split.train_ids) == int(0.7 * n)
    assert len(split.val_ids) == int(0.15 * n)
    assert len(split.train_ids) + len(split.val_ids) + len(split.test_ids) == n
    assert not set(split.train_ids) & set(split.val_ids)
    assert not set(split.val_ids) & set(split.test_ids)


def test_late_slice_overlap_is_measured_not_assumed(small_frame):
    split = chronological_split(small_frame)
    report = late_slice_overlap(split)
    assert report["late_rows"] == len(split.late_ids)
    assert report["late_in_test"] == len(set(split.late_ids) & set(split.test_ids))
    assert report["late_in_train"] == 0
    assert report["share_of_late_inside_test"] == pytest.approx(1.0)
    assert 0 < report["share_of_test_that_is_late"] < 1


def test_split_rejects_bad_fractions(small_frame):
    with pytest.raises(ValueError):
        chronological_split(small_frame, fractions=(0.8, 0.15, 0.15))
    with pytest.raises(ValueError):
        chronological_split(small_frame, late_fraction=1.5)


def test_split_manifest_hash_is_stable(small_frame):
    assert split_manifest_hash(chronological_split(small_frame)) == split_manifest_hash(chronological_split(small_frame))


def test_forward_chaining_folds_never_look_ahead():
    ts = np.sort(np.random.default_rng(0).integers(0, 10_000, size=600))
    folds = forward_chaining_folds(600, n_splits=5)
    assert len(folds) == 5
    assert_forward_only(folds, ts)
    for a, b in zip(folds, folds[1:]):
        assert len(b.train_positions) > len(a.train_positions)
        assert a.val_positions.max() < b.val_positions.min()


def test_forward_chaining_rejects_tiny_inputs():
    with pytest.raises(ValueError):
        forward_chaining_folds(3, n_splits=5)


def test_baf_adapter_names_source_when_missing(tmp_path):
    with pytest.raises(DatasetNotAvailable) as exc:
        load_baf(tmp_path)
    assert "feedzai" in str(exc.value) and "data-root" in str(exc.value)


def test_baf_adapter_reports_schema_mismatch(tmp_path):
    pd.DataFrame({"month": [0, 1], "fraud_bool": [0, 1], "income": [0.1, 0.2]}).to_csv(tmp_path / "Base.csv", index=False)
    with pytest.raises(SchemaMismatch) as exc:
        load_baf(tmp_path)
    assert "payment_type" in exc.value.missing


def test_ieee_adapter_names_kaggle_when_missing(tmp_path):
    with pytest.raises(DatasetNotAvailable) as exc:
        load_ieee_cis(tmp_path)
    assert "kaggle" in str(exc.value).lower()
