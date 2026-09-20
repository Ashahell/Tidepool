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


def test_aux_fw_pressures_keys():
    from tmt.config import TMTConfig
    from tmt.model import TMTModel
    torch.manual_seed(0)
    # update_every=2: first step backprops without stepping, so grads
    # are still readable after training_step returns.
    m = TMTModel(TMTConfig(dim=16, layers=1, fw_dk=4, aux_fw_w=1.0,
                           update_every=2))
    m.reset()
    m.opt.zero_grad()
    loss, _, _ = m.training_step(65, 66, False)
    assert m.last_components["l_fw"] > 0.0
    assert m.layers[0].fw.Wk.grad is not None
    assert bool((m.layers[0].fw.Wk.grad != 0).any())


def test_wiring_end_to_end():
    from tmt.config import TMTConfig
    from tmt.model import TMTModel
    m = TMTModel(TMTConfig(dim=16, layers=1, fw_dk=4))
    assert m.layers[0].fw is not None  # retrieval must actually be wired
    m.reset()
    logits, states = m(torch.tensor([65], dtype=torch.long))
    assert logits.shape == (1, 256)
    loss, sampled, stop = m.training_step(65, 66, False)
    assert loss.isfinite()
    with torch.no_grad():
        assert bool((m.layers[0].fw.S != 0).any())  # a step must write S
