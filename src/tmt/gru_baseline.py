# src/tmt/gru_baseline.py
"""Minimal GRU byte model with the TMT probe interface.

Exists for one purpose: a same-scale recurrent baseline on the copy
task (consultant: scale probes, not just steps). Not a product.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 2, lr: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.nlayers = layers
        self.embed = nn.Embedding(256, dim)
        self.gru = nn.GRU(dim, dim, layers, batch_first=True)
        self.head = nn.Linear(dim, 256)
        self.register_buffer("hidden", torch.zeros(layers, 1, dim))
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def reset(self) -> None:
        with torch.no_grad():
            self.hidden.zero_()

    def forward(self, x: torch.LongTensor):
        h, _ = self.gru(self.embed(x).unsqueeze(0), self.hidden)
        return self.head(h.squeeze(0)), None

    @torch.no_grad()
    def ingest(self, curr: int):
        enc = self.embed(torch.tensor([curr], dtype=torch.long))
        h, hn = self.gru(enc.unsqueeze(0), self.hidden)
        self.hidden.copy_(hn)
        return self.head(h.squeeze(0)), torch.tensor([[0.0]])

    def train_step(self, curr: int, next_: int | None, end: bool):
        self.train()
        x = torch.tensor([curr], dtype=torch.long)
        h, hn = self.gru(self.embed(x).unsqueeze(0), self.hidden)
        logits = self.head(h.squeeze(0))
        self.hidden.data.copy_(hn.detach())
        if next_ is None:
            return torch.tensor(0.0), curr, 0.0
        loss = F.cross_entropy(logits.view(-1, 256), torch.tensor([next_]))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.detach(), curr, 0.0

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
