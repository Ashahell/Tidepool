# src/tmt/recall.py
"""Separate recall readout, trained only on copy-episode query segments.

Online distillation of the linear-probe finding: the trunk already holds
payload information the main decoder cannot use. This head learns to
read it, with the trunk frozen (head-only updates).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecallHead(nn.Module):
    def __init__(self, dim: int, lr: float = 1e-3, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 256))
        else:
            self.proj = nn.Linear(dim, 256)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def train_step(self, state, target: int) -> float:
        self.train()
        logits = self.proj(state.detach())
        loss = F.cross_entropy(logits.view(-1, 256),
                               torch.tensor([target]))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def predict(self, state) -> int:
        self.eval()
        with torch.no_grad():
            out = self.proj(state)
            return int(out.argmax(dim=-1).view(-1)[0].item())
