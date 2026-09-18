# tests/test_fixes.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def test_decay_groups_spread():
    m = TMTModel(TMTConfig(dim=16, layers=1, decay_groups=4))
    m.init_decay_groups([8.0, 64.0, 512.0, 4000.0])
    d = torch.sigmoid(m.layers[0].decay_bias).detach()
    assert float(d.max() - d.min()) > 0.2

def test_reset_clears_state():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    m(torch.tensor([65], dtype=torch.long))
    m.reset()
    for layer in m.layers:
        assert float(layer.states.abs().max()) == 0.0
        assert float(layer.embedtrace.abs().max()) == 0.0

def test_update_every_accumulates():
    m = TMTModel(TMTConfig(dim=16, layers=1, update_every=4))
    before = [p.detach().clone() for p in m.parameters()]
    for _ in range(3):
        m.training_step(65, 66, False)
    mid = [p.detach().clone() for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before, mid))
    m.training_step(65, 66, False)
    after = [p.detach().clone() for p in m.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(mid, after))
