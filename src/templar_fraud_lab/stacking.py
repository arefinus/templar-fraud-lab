"""Source-mask-aware Bayesian stacking.

Sources: p_triage, p_escalation (optional; verifier-approved rows only), p_z.
Per row, the stacked probability is the weighted mean of the AVAILABLE sources,
with weights renormalised over that row's availability mask (Eq. 8 as we read
it). A missing source is a NaN in the input matrix and is never zero-filled.

Prior: Dirichlet(1, ..., 1) over the simplex, which is uniform, so its density
is constant on the simplex. Variational family: mean-field Gaussian over a
softmax-parameterised weight vector (logistic-normal). The ELBO is estimated
with reparameterised Monte Carlo samples; because the prior is flat, the ELBO
reduces to the expected log-likelihood plus the entropy of q (up to the constant
prior density and the softmax change-of-variables term, which we include via
the log-Jacobian of the softmax). Fitted by Adam on validation NLL with early
stopping. The posterior mean weight vector is used at prediction time.

Disagreement proxy (our reading of Eq. 9): the availability-weighted standard
deviation of the source probabilities around the stacked probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SOURCE_NAMES = ("p_triage", "p_escalation", "p_z")
EPS = 1e-6


@dataclass(frozen=True)
class StackingFit:
    weights: tuple[float, ...]
    sources: tuple[str, ...]
    iterations: int
    best_val_nll: float
    history: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {"weights": dict(zip(self.sources, self.weights)), "iterations": self.iterations,
                "best_val_nll": self.best_val_nll}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def _mask(p: np.ndarray) -> np.ndarray:
    return ~np.isnan(p)


def combine(p: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Renormalise weights over available sources per row and combine."""
    if p.ndim != 2 or p.shape[1] != len(weights):
        raise ValueError("p must be (n, n_sources) matching weights")
    mask = _mask(p).astype(np.float64)
    if (mask.sum(axis=1) == 0).any():
        raise ValueError("every row needs at least one available source")
    w = mask * weights[None, :]
    w = w / w.sum(axis=1, keepdims=True)
    return (w * np.nan_to_num(p, nan=0.0)).sum(axis=1)


def disagreement(p: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mask = _mask(p).astype(np.float64)
    w = mask * weights[None, :]
    w = w / w.sum(axis=1, keepdims=True)
    mean = (w * np.nan_to_num(p, nan=0.0)).sum(axis=1, keepdims=True)
    var = (w * (np.nan_to_num(p, nan=0.0) - mean) ** 2).sum(axis=1)
    return np.sqrt(np.clip(var, 0, None))


def _torch_combine(p: torch.Tensor, mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # p: (n, S) with NaN -> 0; mask (n, S); w (M, S)
    ww = mask[None, :, :] * w[:, None, :]
    ww = ww / ww.sum(dim=2, keepdim=True)
    return (ww * p[None, :, :]).sum(dim=2)  # (M, n)


def fit_stacking(p_val: np.ndarray, y_val: np.ndarray, *, lr: float = 0.01, iterations: int = 3000,
                 mc_samples: int = 32, patience: int = 200, seed: int = 42,
                 sources: tuple[str, ...] = SOURCE_NAMES) -> StackingFit:
    if p_val.shape[1] != len(sources):
        raise ValueError("source count mismatch")
    torch.manual_seed(seed)
    n_s = p_val.shape[1]
    mask = torch.as_tensor(_mask(p_val), dtype=torch.float64)
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("every validation row needs at least one available source")
    p = torch.as_tensor(np.nan_to_num(p_val, nan=0.0), dtype=torch.float64).clamp(EPS, 1 - EPS)
    y = torch.as_tensor(y_val, dtype=torch.float64)
    mu = torch.zeros(n_s, dtype=torch.float64, requires_grad=True)
    log_sigma = torch.full((n_s,), -1.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)
    best, best_it, best_mu, bad, history = float("inf"), 0, mu.detach().clone(), 0, []
    for it in range(iterations):
        opt.zero_grad()
        eps = torch.randn(mc_samples, n_s, dtype=torch.float64)
        v = mu + eps * log_sigma.exp()
        w = torch.softmax(v, dim=1)
        pm = _torch_combine(p, mask, w).clamp(EPS, 1 - EPS)
        loglik = (y[None] * pm.log() + (1 - y[None]) * (1 - pm).log()).sum(dim=1).mean()
        entropy = log_sigma.sum()  # Gaussian entropy up to constants
        log_jacobian = (torch.log(w + EPS).sum(dim=1)).mean()  # softmax change of variables
        elbo = loglik + entropy + log_jacobian
        (-elbo).backward()
        opt.step()
        with torch.no_grad():
            w_mean = torch.softmax(mu, dim=0)
            pv = _torch_combine(p, mask, w_mean[None]).clamp(EPS, 1 - EPS)[0]
            val_nll = float(-(y * pv.log() + (1 - y) * (1 - pv).log()).mean())
        history.append(val_nll)
        if val_nll < best - 1e-7:
            best, best_it, best_mu, bad = val_nll, it, mu.detach().clone(), 0
        else:
            bad += 1
            if bad >= patience:
                break
    weights = torch.softmax(best_mu, dim=0).numpy()
    return StackingFit(tuple(float(x) for x in weights), tuple(sources), len(history), best, tuple(history))
