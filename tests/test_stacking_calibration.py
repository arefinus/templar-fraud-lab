from __future__ import annotations

import numpy as np
import pytest

from templar_fraud_lab.calibration import (
    expected_calibration_error,
    fit_isotonic,
    fit_ot_calibrator,
)
from templar_fraud_lab.stacking import combine, disagreement, fit_stacking


def test_missing_source_weights_renormalise_over_available_sources():
    w = np.array([0.5, 0.3, 0.2])
    p = np.array([[0.2, np.nan, 0.8], [0.1, 0.9, 0.5]])
    out = combine(p, w)
    assert out[0] == pytest.approx((0.5 * 0.2 + 0.2 * 0.8) / 0.7)
    assert out[1] == pytest.approx(0.5 * 0.1 + 0.3 * 0.9 + 0.2 * 0.5)


def test_combine_never_zero_fills_missing_source():
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    p = np.array([[0.9, np.nan, 0.9]])
    assert combine(p, w)[0] == pytest.approx(0.9)  # zero-filling would give 0.6


def test_combine_requires_at_least_one_source():
    with pytest.raises(ValueError):
        combine(np.array([[np.nan, np.nan, np.nan]]), np.array([0.3, 0.3, 0.4]))


def test_disagreement_is_zero_when_sources_agree_and_positive_otherwise():
    w = np.array([0.4, 0.3, 0.3])
    p = np.array([[0.3, 0.3, 0.3], [0.1, np.nan, 0.9]])
    d = disagreement(p, w)
    assert d[0] == pytest.approx(0.0)
    assert d[1] > 0.3


def test_fit_stacking_prefers_the_informative_source():
    rng = np.random.default_rng(0)
    y = rng.random(600) < 0.2
    good = np.clip(y * 0.7 + rng.normal(0.15, 0.1, 600), 0.01, 0.99)
    noise = rng.random(600)
    p = np.column_stack([good, np.where(rng.random(600) < 0.5, np.nan, noise), noise])
    fit = fit_stacking(p, y.astype(int), iterations=300, patience=50, mc_samples=8)
    assert abs(sum(fit.weights) - 1.0) < 1e-6 and min(fit.weights) >= 0
    assert fit.weights[0] > 0.6
    assert fit.iterations <= 300 and fit.best_val_nll > 0


def test_ot_calibrator_bounds_monotone_and_mass_preserving():
    rng = np.random.default_rng(1)
    s = rng.random(2000)
    y = (rng.random(2000) < s ** 2).astype(int)
    cal = fit_ot_calibrator(s, y, n_knots=20, min_support=50)
    out = cal.transform(np.sort(rng.random(500)))
    assert np.all(out >= 0) and np.all(out <= 1)
    assert np.all(np.diff(out) >= -1e-12)
    assert cal.transform(s).mean() == pytest.approx(y.mean(), abs=0.01)
    assert cal.n_bins <= 20


def test_ot_calibrator_respects_min_support():
    s = np.linspace(0, 1, 100)
    y = (s > 0.7).astype(int)
    cal = fit_ot_calibrator(s, y, n_knots=256, min_support=25)
    assert cal.n_bins <= 4


def test_ot_calibrator_rejects_bad_inputs():
    with pytest.raises(ValueError):
        fit_ot_calibrator(np.array([0.1]), np.array([0, 1]))
    with pytest.raises(ValueError):
        fit_ot_calibrator(np.array([0.1, 0.2]), np.array([0, 1]), min_support=0)


def test_isotonic_comparator_is_monotone():
    rng = np.random.default_rng(2)
    s = rng.random(500)
    y = (rng.random(500) < s).astype(int)
    iso = fit_isotonic(s, y)
    out = iso.transform(np.sort(rng.random(200)))
    assert np.all(np.diff(out) >= -1e-12) and out.min() >= 0 and out.max() <= 1


def test_ece_bins_report_counts_and_zero_for_perfect_calibration():
    p = np.array([0.05] * 100 + [0.95] * 100)
    y = np.array([0] * 95 + [1] * 5 + [1] * 95 + [0] * 5)
    ece, bins = expected_calibration_error(p, y, n_bins=10)
    assert sum(b["count"] for b in bins) == 200
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert bins[0]["count"] == 100 and bins[-1]["count"] == 100 and bins[5]["count"] == 0
