"""Training loop for the SAINT-style encoder on CPU.

Loss: focal loss (gamma configurable) or class-weighted BCE, AdamW, gradient
clipping at 1.0, linear warmup then cosine decay, early stopping on validation
average precision. Context windows are contiguous chronological blocks so the
historical-context mask has the intended meaning. Checkpoints are state_dicts
loaded with `weights_only=True` (contract rule 11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor

from templar_fraud_lab.models.saint import SaintConfig, SaintEncoder


@dataclass(frozen=True)
class TrainResult:
    epochs_run: int
    best_epoch: int
    best_val_ap: float
    history: tuple[dict[str, float], ...]


def focal_loss(logits: Tensor, y: Tensor, gamma: float, pos_weight: float) -> Tensor:
    p = torch.sigmoid(logits)
    pt = torch.where(y > 0.5, p, 1 - p)
    w = torch.where(y > 0.5, torch.full_like(p, pos_weight), torch.ones_like(p))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (w * (1 - pt).clamp_min(1e-6) ** gamma * bce).mean()


def _windows(n: int, size: int) -> list[slice]:
    return [slice(s, min(s + size, n)) for s in range(0, n, size)]


@torch.no_grad()
def encode_in_context_windows(model: SaintEncoder, x: np.ndarray, ts: np.ndarray,
                              context: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, p) for rows already sorted chronologically, window by window."""
    model.eval()
    ctx = context or model.cfg.context
    xt, tt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(ts, dtype=torch.float64)
    zs, ps = [], []
    for w in _windows(len(x), ctx):
        z, logit = model(xt[w], tt[w])
        zs.append(z.numpy())
        ps.append(torch.sigmoid(logit).numpy())
    return np.vstack(zs), np.concatenate(ps)


def train_saint(
    cfg: SaintConfig, x_train: np.ndarray, y_train: np.ndarray, ts_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray, ts_val: np.ndarray, *,
    epochs: int = 8, lr: float = 3e-4, weight_decay: float = 1e-5, gamma: float = 2.0,
    patience: int = 3, warmup_epochs: int = 1, seed: int = 42, loss: str = "focal",
) -> tuple[SaintEncoder, TrainResult]:
    if loss not in {"focal", "weighted_bce"}:
        raise ValueError("loss must be 'focal' or 'weighted_bce'")
    torch.manual_seed(seed)
    model = SaintEncoder(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = max(float(y_train.sum()), 1.0)
    pos_weight = float((len(y_train) - pos) / pos)
    xt, yt = torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)
    tt = torch.as_tensor(ts_train, dtype=torch.float64)
    windows = _windows(len(x_train), cfg.context)
    total_steps = max(epochs * len(windows), 1)
    warm = max(warmup_epochs * len(windows), 1)
    best_ap, best_epoch, best_state, bad, history, step = -1.0, -1, None, 0, [], 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for w in windows:
            step += 1
            scale = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total_steps - warm, 1)))
            for g in opt.param_groups:
                g["lr"] = lr * scale
            opt.zero_grad()
            _, logit = model(xt[w], tt[w])
            if loss == "focal":
                value = focal_loss(logit, yt[w], gamma, pos_weight)
            else:
                value = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, yt[w], pos_weight=torch.tensor(pos_weight))
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(value.detach())
        _, p_val = encode_in_context_windows(model, x_val, ts_val)
        ap = float(average_precision_score(y_val, p_val)) if y_val.sum() > 0 else 0.0
        history.append({"epoch": epoch, "train_loss": running / len(windows), "val_ap": ap})
        if ap > best_ap:
            best_ap, best_epoch, bad = ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, TrainResult(len(history), best_epoch, best_ap, tuple(history))


def save_checkpoint(model: SaintEncoder, path: str | Path) -> None:
    torch.save({"config": model.cfg.as_dict(), "state_dict": model.state_dict()}, Path(path))


def load_checkpoint(path: str | Path) -> SaintEncoder:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = SaintEncoder(SaintConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
