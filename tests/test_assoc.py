# tests/test_assoc.py
import numpy as np
import torch
from tmt.assoc_probe import BRACKETS, assoc_memory_ingesting


class PerfectAssoc:
    """Pair slots sit at known positions (2 + inter_len apart); pair keys
    are unique per trial, so the query key identifies exactly one slot."""

    def __init__(self, n_pairs=8, inter_len=64):
        self.n_pairs = n_pairs
        self.inter_len = inter_len
        self.hist = []

    def reset(self):
        self.hist = []

    def ingest(self, b):
        self.hist.append(int(b))
        logits = torch.full((1, 256), -1e9)
        logits[0, int(b)] = 0.0
        return logits, torch.tensor([[0.0]])

    def __call__(self, x):
        cur = int(x[0].item())
        step = 2 + self.inter_len
        for j in range(self.n_pairs):
            if self.hist[j * step] == cur:
                logits = torch.full((1, 256), -1e9)
                logits[0, self.hist[j * step + 1]] = 0.0
                return logits, None
        logits = torch.full((1, 256), -1e9)
        logits[0, 0] = 0.0
        return logits, None


class AmnesiacAssoc:
    def reset(self):
        pass

    def ingest(self, b):
        logits = torch.full((1, 256), -1e9)
        logits[0, 0] = 0.0
        return logits, torch.tensor([[0.0]])

    def __call__(self, x):
        logits = torch.full((1, 256), -1e9)
        logits[0, 0] = 0.0
        return logits, None


def test_assoc_orders_memory():
    assert assoc_memory_ingesting(PerfectAssoc(), seed=0).score == 1.0
    assert assoc_memory_ingesting(AmnesiacAssoc(), seed=0).score == 0.0


def test_assoc_brackets():
    kw = dict(pairs=BRACKETS * 2, n_pairs=8, inter_len=64)
    r = assoc_memory_ingesting(PerfectAssoc(n_pairs=8, inter_len=64),
                               seed=1, **kw)
    assert r.score == 1.0
    r2 = assoc_memory_ingesting(AmnesiacAssoc(), seed=1, **kw)
    assert r2.score == 0.0
