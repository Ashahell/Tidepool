# tests/test_fastweight.py — delta-rule fast-weight memory (EXPERIMENTAL).
import torch
from tmt.fastweight import FastWeightMemory


def _identity_mem():
    m = FastWeightMemory(4, 4)
    with torch.no_grad():
        m.Wk.copy_(torch.eye(4))
        m.Wq.copy_(torch.eye(4))
        m.Wv.copy_(torch.eye(4))
        m.wb.zero_()
        m.bb.fill_(20.0)  # beta = 1 - 2e-9 ~ 1 (sigmoid(10) is only 0.99995)
    m.reset()
    return m


def test_write_read_roundtrip():
    m = _identity_mem()
    h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = m.step(h)
    # write v=h at k=e1, then read q=e1: r = h, out = h + r = 2h.
    assert torch.allclose(out, 2 * h, atol=1e-5)


def test_delta_overwrite_same_key():
    m = _identity_mem()
    k = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    m.step(k)  # store k -> k
    out = m.step(k)  # delta residual is 0; read must still equal k exactly
    assert torch.allclose(out, 2 * k, atol=1e-5)


def test_repeated_write_fixed_point():
    # Writing the same association twice must be a fixed point:
    # the delta residual is 0 the second time, so S is unchanged.
    m = _identity_mem()
    k = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    m.step(k)
    with torch.no_grad():
        s1 = m.S.clone()
    m.step(k)
    with torch.no_grad():
        assert torch.allclose(m.S, s1, atol=1e-6)
        q = k / (k.norm() + 1e-8)
        assert torch.allclose((m.S @ q.T).T, k, atol=1e-5)


def test_reset_zeroes():
    m = _identity_mem()
    m.step(torch.ones(1, 4))
    m.reset()
    assert bool((m.S == 0).all())
