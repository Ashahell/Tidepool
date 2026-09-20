# src/tmt/transformer_baseline.py
"""Tiny causal transformer reference for the copy task.

Exists to answer one question: is exact recall learnable at all in a
batched offline regime? Probe-compatible via an explicit context buffer
(reset clears, ingest appends with cap, call runs the full context).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 2, heads: int = 4,
                 lr: float = 3e-4, max_ctx: int = 512):
        super().__init__()
        self.dim = dim
        self.max_ctx = max_ctx
        self.embed = nn.Embedding(256, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                           dim_feedforward=4 * dim,
                                           batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(dim, 256)
        self.register_buffer("ctx", torch.zeros(0, dtype=torch.long))
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def reset(self) -> None:
        self.ctx = torch.zeros(0, dtype=torch.long)

    def _logits(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq).unsqueeze(0)
        n = x.shape[1]
        mask = torch.triu(torch.ones(n, n) * float("-inf"), diagonal=1)
        return self.head(self.tr(x, mask=mask)).squeeze(0)

    def ingest(self, curr: int):
        with torch.no_grad():
            self.ctx = torch.cat([self.ctx, torch.tensor([curr])])[-self.max_ctx:]
            return self._logits(self.ctx)[-1:].detach(), torch.tensor([[0.0]])

    def __call__(self, x: torch.LongTensor):
        return self._logits(torch.cat([self.ctx, x]))[-1:], None

    def train_episode(self, ep: bytes) -> float:
        self.train()
        seq = torch.tensor(list(ep), dtype=torch.long)
        logits = self._logits(seq[:-1])
        import torch.nn.functional as F
        loss = F.cross_entropy(logits, seq[1:])
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
