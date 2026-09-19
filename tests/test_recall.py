# tests/test_recall.py
import torch
from tmt.recall import RecallHead


def test_recall_head_overfits_fixed_mapping():
    torch.manual_seed(0)
    head = RecallHead(dim=8, lr=1e-2)
    states = torch.randn(4, 8)
    targets = [10, 20, 30, 40]
    l0 = sum(head.train_step(s, t) for s, t in zip(states, targets)) / 4
    for _ in range(50):
        for s, t in zip(states, targets):
            head.train_step(s, t)
    l1 = sum(head.train_step(s, t) for s, t in zip(states, targets)) / 4
    assert l1 < l0
    for s, t in zip(states, targets):
        assert head.predict(s) == t
