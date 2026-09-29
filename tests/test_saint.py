from __future__ import annotations

import numpy as np
import pytest
import torch

from templar_fraud_lab.models.saint import SaintConfig, SaintEncoder, historical_context_mask
from templar_fraud_lab.models.train_saint import (
    encode_in_context_windows,
    load_checkpoint,
    save_checkpoint,
    train_saint,
)

CFG = SaintConfig(n_features=6, token_dim=8, out_dim=4, n_blocks=1, n_heads=2, ff_mult=2, dropout=0.0, context=16, inter_dim=16)


def test_historical_mask_forbids_only_later_rows():
    ts = torch.tensor([0.0, 1.0, 1.0, 2.0])
    forbid = historical_context_mask(ts)
    assert forbid.tolist() == [
        [False, True, True, True],
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, False],
    ]


def test_no_future_context_leaks_into_earlier_rows():
    torch.manual_seed(0)
    model = SaintEncoder(CFG).eval()
    x = torch.randn(12, 6)
    ts = torch.arange(12, dtype=torch.float64)
    with torch.no_grad():
        z_ref, _ = model(x, ts)
        x2 = x.clone()
        x2[8:] += 5.0  # perturb later rows only
        z_pert, _ = model(x2, ts)
    assert torch.allclose(z_ref[:8], z_pert[:8], atol=1e-6)
    assert not torch.allclose(z_ref[8:], z_pert[8:], atol=1e-3)


def test_earlier_rows_do_influence_later_rows():
    torch.manual_seed(0)
    model = SaintEncoder(CFG).eval()
    x = torch.randn(10, 6)
    ts = torch.arange(10, dtype=torch.float64)
    with torch.no_grad():
        z_ref, _ = model(x, ts)
        x2 = x.clone()
        x2[0] += 5.0
        z_pert, _ = model(x2, ts)
    assert not torch.allclose(z_ref[1:], z_pert[1:], atol=1e-4)


def test_wrong_feature_count_raises():
    model = SaintEncoder(CFG)
    with pytest.raises(ValueError, match="expected 6 features"):
        model(torch.randn(4, 5), torch.arange(4, dtype=torch.float64))


def test_checkpoint_save_load_parity(tmp_path):
    torch.manual_seed(1)
    model = SaintEncoder(CFG).eval()
    save_checkpoint(model, tmp_path / "m.pt")
    loaded = load_checkpoint(tmp_path / "m.pt")
    x = np.random.default_rng(0).normal(size=(20, 6)).astype(np.float32)
    ts = np.arange(20, dtype=np.float64)
    z1, p1 = encode_in_context_windows(model, x, ts)
    z2, p2 = encode_in_context_windows(loaded, x, ts)
    assert np.allclose(z1, z2, atol=1e-6) and np.allclose(p1, p2, atol=1e-6)
    assert loaded.cfg == CFG


def test_train_saint_runs_and_tracks_history():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 6)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=64) > 1.0).astype(int)
    ts = np.arange(64, dtype=np.float64)
    model, result = train_saint(CFG, x, y, ts, x, y, ts, epochs=3, lr=1e-3, patience=5, seed=0)
    assert result.epochs_run == 3
    assert len(result.history) == 3
    assert 0.0 <= result.best_val_ap <= 1.0
    z, p = encode_in_context_windows(model, x, ts)
    assert z.shape == (64, 4) and p.shape == (64,)
    assert np.all((p >= 0) & (p <= 1))


def test_train_saint_rejects_unknown_loss():
    x = np.zeros((8, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        train_saint(CFG, x, np.zeros(8), np.arange(8.0), x, np.zeros(8), np.arange(8.0), loss="hinge")
