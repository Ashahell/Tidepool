# tests/test_ingesting.py
import numpy as np
import torch
from tmt.evaluation_suite import EvalConfig
from tmt.ingesting_probe import copy_memory_ingesting


class _IngestOracle:
    """Tracks ingested bytes itself; generation reads emit stored bytes.
    Feed phase = ingest calls; gen phase = forward reads (input 0)."""

    def __init__(self, copy_len, inter_len):
        self.copy_len = copy_len
        self.inter_len = inter_len
        self.fed = []
        self.emitted = 0

    def reset(self):
        self.fed = []
        self.emitted = 0

    def ingest(self, b):
        self.fed.append(int(b))
        logits = torch.full((1, 256), -1e9)
        logits[0, int(b)] = 0.0
        return logits, torch.tensor([[0.0]])

    def _emit(self, g):
        raise NotImplementedError

    def __call__(self, x):
        if len(self.fed) < self.copy_len + self.inter_len:
            raise AssertionError("oracle read during feed phase")
        g = self.emitted
        self.emitted += 1
        logits = torch.full((1, 256), -1e9)
        logits[0, self._emit(g)] = 0.0
        return logits, None


class PerfectIngest(_IngestOracle):
    def _emit(self, g):
        return self.fed[g]


class OneByteIngest(_IngestOracle):
    def _emit(self, g):
        return self.fed[0]


class AmnesiacIngest:
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


def _cfg():
    return EvalConfig(copy_lengths=[4], intervening_lengths=[8],
                      num_copy_trials=4)


def test_ingesting_probe_orders_memory():
    np.random.seed(0)
    assert copy_memory_ingesting(PerfectIngest(4, 8), _cfg()).score == 1.0
    np.random.seed(0)
    assert copy_memory_ingesting(OneByteIngest(4, 8), _cfg()).score == 0.0
    np.random.seed(0)
    assert copy_memory_ingesting(AmnesiacIngest(), _cfg()).score == 0.0
