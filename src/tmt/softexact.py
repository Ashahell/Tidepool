# src/tmt/softexact.py
"""Hybrid store: verbatim keys/values, soft similarity retrieval.

Exact storage (no blending — content preserved) with softmax retrieval
over cosine similarity (generalizes to similar, not just identical,
contexts). Answers whether the v1/v2 failure was blending (destroyed
content) or addressing (wrong shape).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class SoftExact:
    def __init__(self, dim: int, capacity: int = 512, temp: float = 1.0):
        self.dim = dim
        self.capacity = capacity
        self.temp = temp
        self.keys: list = []
        self.vals: list = []

    def reset(self) -> None:
        self.keys = []
        self.vals = []

    def write(self, state: torch.Tensor, byte: int) -> None:
        self.keys.append(state.detach().reshape(-1).clone())
        self.vals.append(int(byte))
        if len(self.keys) > self.capacity:
            self.keys.pop(0)
            self.vals.pop(0)

    def read_dist(self, query: torch.Tensor) -> torch.Tensor:
        q = query.detach().reshape(-1)
        if not self.keys:
            return torch.full((256,), float("-inf"))
        K = torch.stack(self.keys)
        sims = F.cosine_similarity(K, q.unsqueeze(0), dim=-1) / self.temp
        w = F.softmax(sims, dim=0)
        out = torch.zeros(256)
        for i, v in enumerate(self.vals):
            out[v] += float(w[i])
        return torch.log(out.clamp_min(1e-12))

    def predict(self, query: torch.Tensor) -> int:
        return int(self.read_dist(query).argmax().item())
