# tests/test_canonical.py
"""Canonical-path identity: every experimental flag at neutral must
reproduce the default model bit-exactly. Guards the prove-and-freeze
claim that the canonical RTU core is clean."""
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

FLAGS = dict(selective=False, sparse_k=0, write_k=0, replay_size=0,
             replay_k=1, replay_noise=False, ema_decay=0.0,
             recall_hidden=0, slots=0)


def _models():
    torch.manual_seed(0)
    a = TMTModel(TMTConfig(dim=16, layers=2))
    torch.manual_seed(0)
    b = TMTModel(TMTConfig(dim=16, layers=2, **FLAGS))
    return a, b


def test_explicit_neutral_equals_default():
    a, b = _models()
    x = torch.tensor([65], dtype=torch.long)
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert torch.equal(la, lb)
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]


def test_neutral_training_trajectory_identical():
    a, b = _models()
    for i in range(5):
        torch.manual_seed(100 + i)
        la, _, _ = a.training_step(65 + i, 66, False)
        torch.manual_seed(100 + i)
        lb, _, _ = b.training_step(65 + i, 66, False)
        assert float(la.item()) == float(lb.item())
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
