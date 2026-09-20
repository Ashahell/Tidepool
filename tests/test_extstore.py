# tests/test_extstore.py — true external KV store (new hypothesis class).
import torch
from tmt.extstore import ExternalStore


def _fixed_store():
    st = ExternalStore(dim=4, dk=4, nslots=4)
    with torch.no_grad():
        st.Wk.copy_(torch.eye(4))
        st.Wq.copy_(torch.eye(4))
        st.Wv.copy_(torch.eye(4))
        st.ww.zero_(); st.wb.fill_(20.0)  # write gate ~1
        st.we.zero_(); st.eb.fill_(-20.0)  # erase gate ~0
        st.wp.zero_(); st.pb.fill_(-20.0)  # protect value ~0
        st.null_logit.fill_(-20.0)
    st.reset()
    return st


def test_write_read_roundtrip():
    st = _fixed_store()
    s = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    st.write(s)
    r, w = st.retrieve(s)  # query == written content
    assert torch.allclose(r, s, atol=1e-5)
    assert int(w[:-1].argmax()) == 0  # slot 0 (first free)


def test_protect_blocks_overwrite():
    st = _fixed_store()
    s = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        st.wp.zero_(); st.pb.fill_(20.0)  # this content stored protected
    st.write(s)
    other = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        st.pb.fill_(-20.0)  # subsequent writes unprotected
    for _ in range(6):  # more writes than slots
        st.write(other)
    r, _ = st.retrieve(s)
    # slot 0 protected: stored content untouched AND still best match.
    # (Readout mass is diffuse by construction with 4 live slots —
    # sharpness is the learned temperature's job, measured in runs.)
    with torch.no_grad():
        assert torch.allclose(st.V[0], s.squeeze(0), atol=1e-5)
    q = st._norm(s @ st.Wq.T)
    K = st.K.detach()
    assert int(((K @ q.T).squeeze(0)).argmax()) == 0


def test_erase_clears_similar():
    st = _fixed_store()
    s = torch.tensor([[1.0, 1.0, 0.0, 0.0]]) / 1.41421356
    st.write(s)
    with torch.no_grad():
        st.we.zero_(); st.eb.fill_(20.0)  # erase gate ~1
        st.ww.zero_(); st.wb.fill_(-20.0)  # write gate ~0 (erase only)
    st.write(s)  # same content: erases slot holding similar content
    with torch.no_grad():
        assert st.V[0].abs().max().item() < 0.5


def test_evicts_unprotected_oldest():
    st = _fixed_store()
    for i in range(5):
        v = torch.zeros(1, 4)
        v[0, i % 4] = float(i + 1)
        st.write(v)
    # 5 writes, 4 slots, nothing protected → slot 0 reused by write 4
    assert st.write_tick[0] == 4
    with torch.no_grad():
        assert float(st.V[0][0]) == 5.0


def test_reset_clears():
    st = _fixed_store()
    st.write(torch.ones(1, 4))
    st.reset()
    r, _ = st.retrieve(torch.ones(1, 4))
    assert torch.allclose(r, torch.zeros(1, 4))


def test_wiring_end_to_end():
    from tmt.config import TMTConfig
    from tmt.model import TMTModel
    m = TMTModel(TMTConfig(dim=16, layers=1, ext_slots=8, ext_dk=4))
    assert m.extstore is not None
    m.reset()
    logits, _ = m(torch.tensor([65], dtype=torch.long))
    assert logits.shape == (1, 256)
    loss, _, _ = m.training_step(65, 66, False)
    assert loss.isfinite()
    with torch.no_grad():
        assert (m.extstore.V != 0).any()  # a step must write


def test_store_params_owned_by_optimizer():
    # Anchor-bug class: store created after opt leaves addressing untrained.
    from tmt.config import TMTConfig
    from tmt.model import TMTModel
    m = TMTModel(TMTConfig(dim=16, layers=1, ext_slots=8, ext_dk=4))
    owned = {id(p) for g in m.opt.param_groups for p in g["params"]}
    for p in m.extstore.parameters():
        assert id(p) in owned
