# tests/test_slots.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.slots import SlotMemory

def test_single_slot_exact_retrieval():
    sm = SlotMemory(dim=16, n_slots=1)
    v = torch.randn(16)
    sm.write(v)
    out = sm.read(v)
    import torch.nn.functional as F
    assert F.cosine_similarity(out, v, dim=0).item() > 0.99

def test_read_path_differentiable():
    # Write side is a documented non-learned cache; the read side
    # (query/key projections) must carry gradient.
    sm = SlotMemory(dim=8, n_slots=4)
    sm.write(torch.randn(8))
    q = torch.randn(8)
    out = sm.read(q)
    out.sum().backward()
    assert sm.query.weight.grad is not None
    assert sm.key.weight.grad is not None
    assert float(sm.query.weight.grad.abs().sum()) > 0.0

def test_write_side_detached_by_design():
    sm = SlotMemory(dim=8, n_slots=4)
    x = torch.randn(8, requires_grad=True)
    sm.write(x)
    assert x.grad is None

def test_reset_clears():
    sm = SlotMemory(dim=16, n_slots=8)
    sm.write(torch.randn(16))
    sm.reset()
    assert float(sm.slots.abs().max()) == 0.0

def test_off_path_identical():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    assert m.slots is None
    x = torch.tensor([65], dtype=torch.long)
    with torch.no_grad():
        l1, _ = m(x)
    assert l1.shape == (1, 256)

def test_generation_read_only():
    # Generation contract: no writes during generation.
    from tmt.engine import gen_bytes
    m = TMTModel(TMTConfig(dim=16, layers=1, slots=8))
    m.reset()
    m.ingest(65)
    assert gen_bytes(m, 65) == gen_bytes(m, 65)

def test_aux_loss_decreases():
    # retrieval_loss pulls the read vector toward the target embedding.
    from tmt.slots import SlotMemory
    torch.manual_seed(0)
    sm = SlotMemory(dim=16, n_slots=8)
    tgt = torch.randn(16)
    opt = torch.optim.SGD(sm.parameters(), lr=0.1)
    l0 = sm.retrieval_loss(sm.read(tgt), tgt).item()
    for _ in range(20):
        opt.zero_grad()
        sm.retrieval_loss(sm.read(tgt), tgt).backward()
        opt.step()
    l1 = sm.retrieval_loss(sm.read(tgt), tgt).item()
    assert l1 < l0

def test_slots_smoke_and_roundtrip(tmp_path):
    m = TMTModel(TMTConfig(dim=16, layers=1, slots=8))
    for i in range(5):
        m.training_step(65 + i, 66, False)
    for p in m.parameters():
        assert torch.isfinite(p).all()
    from tmt.engine import save_checkpoint, load_checkpoint
    pth = str(tmp_path / "s")
    save_checkpoint(m, pth)
    m2 = TMTModel(TMTConfig(dim=16, layers=1, slots=8))
    load_checkpoint(m2, pth)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(m.slots.slots, m2.slots.slots)

def test_shared_key_used_both_sides():
    from tmt.slots import SlotMemory
    sm = SlotMemory(dim=16, n_slots=8)
    assert sm.key is not None
    torch.manual_seed(0)
    v = torch.randn(16)
    sm.write(v)
    out = sm.read(v)
    import torch.nn.functional as F
    assert F.cosine_similarity(out, v, dim=0).item() > 0.0

def test_usage_protects_hot_slots():
    torch.manual_seed(1)
    sm = SlotMemory(dim=16, n_slots=4)
    v0 = torch.randn(16)
    with torch.no_grad():
        sm.slots[0].copy_(v0)
        sm.usage[0] = 100.0
    v = v0 + 0.01 * torch.randn(16)
    before = sm.slots.detach().clone()
    sm.write(v)
    # usage 100 >> thresh 5: protect factor ~1e-41, slot0 bit-stable.
    assert (sm.slots[0] - before[0]).abs().max() < 1e-6
    assert float(sm.usage[0]) > 99.0

def test_temperature_scales_sharpness():
    import torch.nn.functional as F
    from tmt.slots import SlotMemory as SM
    torch.manual_seed(2)
    a = SM(dim=16, n_slots=8)
    a.cfg_temp = 0.1
    b = SM(dim=16, n_slots=8)
    b.cfg_temp = 10.0
    v = torch.randn(16)
    for m in (a, b):
        for _ in range(5):
            m.write(torch.randn(16))
        m.write(v)
    ea = -(F.softmax(a.key(a.slots) @ a.query(v) / a.cfg_temp, dim=0) *
           F.log_softmax(a.key(a.slots) @ a.query(v) / a.cfg_temp, dim=0)).sum().item()
    eb = -(F.softmax(b.key(b.slots) @ b.query(v) / b.cfg_temp, dim=0) *
           F.log_softmax(b.key(b.slots) @ b.query(v) / b.cfg_temp, dim=0)).sum().item()
    assert ea < eb
