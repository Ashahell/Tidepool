# tests/test_ema.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def test_ema_off_by_default():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    assert m.ema_state is None
    with m.using_ema():
        pass

def test_ema_tracks_and_restores():
    m = TMTModel(TMTConfig(dim=16, layers=1, ema_decay=0.9))
    before = [p.detach().clone() for p in m.parameters()]
    m.training_step(65, 66, False)
    m.training_step(67, 68, False)
    assert m.ema_state is not None
    snapped = [p.detach().clone() for p in m.parameters()]
    with m.using_ema():
        ema_now = [p.detach().clone() for p in m.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(snapped, ema_now))
    after = [p.detach().clone() for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(snapped, after))
