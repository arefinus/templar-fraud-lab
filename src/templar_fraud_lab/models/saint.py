"""Compact SAINT-style tabular encoder.

Per block: (1) feature-wise self-attention over the tokens of one row followed by
a per-token feed-forward, and (2) intersample attention, where each row attends
to other rows in its context window. In the original SAINT the intersample step
runs on the flattened row (all tokens concatenated). Here the flattened row is
projected to a small `inter_dim` before attention and projected back afterwards,
so the model stays small enough to train in seconds on CPU. This bottleneck is a
deliberate departure from the paper's encoder; see docs/method.md.

HISTORICAL CONTEXT POLICY: row i may attend to row j only if ts_j <= ts_i. At
inference, windows are formed from chronologically contiguous rows, so a row
never sees a later transaction. This is our resolution of the paper's silence on
leakage through intersample attention (docs/paper-implementation-audit.md, A06).

The model exposes a row embedding `z` (dimension `out_dim`) and a logit head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SaintConfig:
    n_features: int
    token_dim: int = 16
    out_dim: int = 16
    n_blocks: int = 2
    n_heads: int = 2
    ff_mult: int = 2
    dropout: float = 0.1
    context: int = 128
    inter_dim: int = 64

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def historical_context_mask(ts: Tensor) -> Tensor:
    """Boolean mask of shape (n, n): True where attention is FORBIDDEN.

    Entry [i, j] is True when ts[j] > ts[i]; rows attend to earlier-or-equal
    rows including themselves."""
    return ts[None, :] > ts[:, None]


class _Block(nn.Module):
    def __init__(self, cfg: SaintConfig) -> None:
        super().__init__()
        d = cfg.token_dim
        self.feature_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff1 = nn.Sequential(nn.Linear(d, d * cfg.ff_mult), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(d * cfg.ff_mult, d))
        self.norm2 = nn.LayerNorm(d)
        width = d * (cfg.n_features + 1)
        self.norm3 = nn.LayerNorm(width)
        self.down = nn.Linear(width, cfg.inter_dim)
        self.sample_attn = nn.MultiheadAttention(cfg.inter_dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.up = nn.Linear(cfg.inter_dim, width)

    def forward(self, tokens: Tensor, forbid: Tensor) -> Tensor:
        # tokens: (n_rows, n_tokens, d)
        h = self.norm1(tokens)
        a, _ = self.feature_attn(h, h, h, need_weights=False)
        tokens = tokens + a
        tokens = tokens + self.ff1(self.norm2(tokens))
        n, t, d = tokens.shape
        flat = tokens.reshape(1, n, t * d)
        h2 = self.down(self.norm3(flat))
        a2, _ = self.sample_attn(h2, h2, h2, attn_mask=forbid, need_weights=False)
        flat = flat + self.up(a2)
        return flat.reshape(n, t, d)


class SaintEncoder(nn.Module):
    def __init__(self, cfg: SaintConfig) -> None:
        super().__init__()
        self.cfg = cfg
        f, d = cfg.n_features, cfg.token_dim
        self.weight = nn.Parameter(torch.randn(f, d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(f, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_blocks))
        self.final_norm = nn.LayerNorm(d)
        self.embed = nn.Linear(d, cfg.out_dim)
        self.head = nn.Linear(cfg.out_dim, 1)

    def tokenize(self, x: Tensor) -> Tensor:
        tokens = x[:, :, None] * self.weight[None] + self.bias[None]
        cls = self.cls.expand(x.shape[0], 1, -1)
        return torch.cat([cls, tokens], dim=1)

    def forward(self, x: Tensor, ts: Tensor) -> tuple[Tensor, Tensor]:
        """x: (n, n_features) float; ts: (n,) numeric timestamps.

        Returns (z, logit) with z of shape (n, out_dim) and logit of shape (n,)."""
        if x.shape[1] != self.cfg.n_features:
            raise ValueError(f"expected {self.cfg.n_features} features, got {x.shape[1]}")
        forbid = historical_context_mask(ts)
        tokens = self.tokenize(x)
        for block in self.blocks:
            tokens = block(tokens, forbid)
        z = torch.tanh(self.embed(self.final_norm(tokens[:, 0])))
        return z, self.head(z).squeeze(-1)
