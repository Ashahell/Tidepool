# tests/test_anchors.py — minimal MARCH-style anchor bank (EXPERIMENTAL).
import torch
from tmt.anchors import AnchorBank


def test_write_read_roundtrip():
    b = AnchorBank(dim=4, dk=4, max_anchors=8)
    b.reset()
    s = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        b.key_proj.copy_(torch.eye(4))
        b.query_proj.copy_(torch.eye(4))
        b.null_logit.fill_(-20.0)  # null off: all mass routable
    b.checkpoint(s)  # key = e1 (normalized), value = s
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    r, w = b.retrieve(q)
    assert torch.allclose(r, s, atol=1e-5)
    assert float(w[0]) > 0.99  # all mass on the single anchor


def test_null_option():
    b = AnchorBank(dim=4, dk=4, max_anchors=8)
    b.reset()
    with torch.no_grad():
        b.key_proj.copy_(torch.eye(4))
        b.query_proj.copy_(torch.eye(4))
        b.null_logit.fill_(20.0)  # null dominates (10.0 leaves 7e-5 leak)
    b.checkpoint(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    r, w = b.retrieve(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(r, torch.zeros(1, 4), atol=1e-5)
    assert float(w[-1]) > 0.99  # last weight slot is null


def test_bank_cap_evicts_oldest():
    b = AnchorBank(dim=2, dk=2, max_anchors=3)
    b.reset()
    for i in range(5):
        b.checkpoint(torch.tensor([[float(i), 0.0]]))
    assert len(b.values) == 3
    with torch.no_grad():
        assert b.values[0][0, 0].item() == 2.0  # 0,1 evicted


def test_reset_empties():
    b = AnchorBank(dim=2, dk=2, max_anchors=3)
    b.checkpoint(torch.ones(1, 2))
    b.reset()
    r, w = b.retrieve(torch.ones(1, 2))
    assert torch.allclose(r, torch.zeros(1, 2))
