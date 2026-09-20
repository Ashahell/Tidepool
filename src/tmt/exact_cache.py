# src/tmt/exact_cache.py
"""Bounded exact-match K-gram cache (kNN-LM lineage at byte scale).

Nonparametric memory: records context -> next-byte pairs on ingest
(free, no training), retrieves on exact context match, interpolates
with the parametric model. Answers whether explicit storage fixes
recall where every parametric mechanism failed.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F


class ExactCache:
    def __init__(self, k: int = 16, capacity: int = 4096, lam: float = 0.5):
        self.k = k
        self.capacity = capacity
        self.lam = lam
        self.table: OrderedDict = OrderedDict()
        self.hist: list = []

    def reset(self) -> None:
        self.table.clear()
        self.hist = []

    def observe(self, context: tuple, nxt: int) -> None:
        if context in self.table:
            self.table.move_to_end(context)
        else:
            if len(self.table) >= self.capacity:
                self.table.popitem(last=False)
            self.table[context] = int(nxt)

    def ingest(self, b: int) -> None:
        self.hist.append(int(b))
        if len(self.hist) > self.k:
            self.observe(tuple(self.hist[-self.k - 1:-1]), int(b))

    def lookup(self, context: tuple) -> int | None:
        if context in self.table:
            self.table.move_to_end(context)
            return self.table[context]
        return None

    def current_context(self) -> tuple:
        return tuple(self.hist[-self.k:])

    def interpolate(self, model_logits: torch.Tensor, context: tuple) -> torch.Tensor:
        hit = self.lookup(context)
        if hit is None:
            return model_logits
        hot = torch.full_like(model_logits, float("-inf"))
        hot[..., hit] = 0.0
        return torch.logaddexp(
            torch.log(torch.tensor(self.lam)) + F.log_softmax(hot, dim=-1),
            torch.log(torch.tensor(1.0 - self.lam)) + F.log_softmax(model_logits, dim=-1),
        )
