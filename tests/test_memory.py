# tests/test_memory.py
import numpy as np
import torch
from tmt.evaluation_suite import EvalConfig, copy_memory_probe

class _CountingOracle:
    """Knows the single-cell protocol AND the probe's read+feedback shape:
    generation does one dummy-0 read plus one feedback call per byte, and
    only the read advances the emission counter. Sound because probe
    targets are randint(32, 127) — never 0 — so in the gen phase
    cur == 0 ⟺ dummy read. (Feed-phase returns are ignored by the probe.)
    Suite stays frozen; the oracle adapts to its documented protocol."""
    def __init__(self, copy_len, inter_len):
        self.copy_len = copy_len
        self.inter_len = inter_len
        self.fed = []
        self.emitted = 0
    def reset(self):
        self.fed = []
        self.emitted = 0
    def _emit(self, g):
        raise NotImplementedError
    def __call__(self, x):
        cur = int(x[0].item())
        if len(self.fed) < self.copy_len + self.inter_len:
            self.fed.append(cur)
            logits = torch.full((1, 256), -1e9)
            logits[0, cur] = 0.0
            return logits, None
        if cur == 0:
            g = self.emitted
            self.emitted += 1
            logits = torch.full((1, 256), -1e9)
            logits[0, self._emit(g)] = 0.0
            return logits, None
        return torch.full((1, 256), -1e9), None

class PerfectMemory(_CountingOracle):
    def _emit(self, g):
        return self.fed[g]

class OneByteMemory(_CountingOracle):
    def _emit(self, g):
        return self.fed[0]

class Amnesiac:
    def reset(self): pass
    def __call__(self, x):
        logits = torch.full((1, 256), -1e9)
        logits[0, 0] = 0.0
        return logits, None

def _cfg():
    return EvalConfig(copy_lengths=[4], intervening_lengths=[8],
                      num_copy_trials=4)

def _score(model):
    np.random.seed(0)
    return copy_memory_probe(model, _cfg()).score

def test_oracle_ordering():
    assert _score(PerfectMemory(4, 8)) == 1.0
    assert _score(OneByteMemory(4, 8)) == 0.0
    assert _score(Amnesiac()) == 0.0
