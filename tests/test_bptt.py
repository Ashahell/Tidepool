# tests/test_bptt.py — episode-level BPTT (deferred backward).
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel


def test_defer_finish_runs_and_steps():
    torch.manual_seed(0)
    m = TMTModel(TMTConfig(dim=16, layers=1, fw_dk=4))
    m.reset()
    before = [p.detach().clone() for p in m.parameters()]
    total = None
    for b in (65, 66, 67, 68):
        loss, _, _ = m.training_step(b, b + 1, False, defer=True)
        total = loss if total is None else total + loss
    m.finish_episode(total)
    after = [p.detach().clone() for p in m.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_deferred_s_carries_graph():
    torch.manual_seed(0)
    m = TMTModel(TMTConfig(dim=16, layers=1, fw_dk=4))
    m.reset()
    total = None
    for b in (65, 66):
        loss, _, _ = m.training_step(b, b + 1, False, defer=True)
        total = loss if total is None else total + loss
    # S stored attached mid-episode: cross-step credit path exists.
    assert m.layers[0].fw.S.grad_fn is not None
    m.finish_episode(total)
    assert m.layers[0].fw.S.grad_fn is None  # detached at boundary


def test_no_defer_unchanged():
    torch.manual_seed(0)
    m = TMTModel(TMTConfig(dim=16, layers=1))
    m.reset()
    loss, sampled, stop = m.training_step(65, 66, False)
    assert loss.isfinite()
