# tests/test_model.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def test_eval_call_shape_and_reset():
    m = TMTModel(TMTConfig(dim=32, layers=2))
    m.reset()
    logits, state = m(torch.tensor([65], dtype=torch.long))
    assert logits.shape == (1, 256)
    assert isinstance(state, list) and len(state) == 2

def test_single_layer_math():
    # Hand-computed: decay=sigmoid(0)=0.5, state=0.5*0+enc+0=enc.
    torch.manual_seed(0)
    m = TMTModel(TMTConfig(dim=4, layers=1))
    with torch.no_grad():
        m.encoder.embed.weight.zero_()
        m.encoder.embed.weight[65] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m.layers[0].decay_bias.zero_()
    m.reset()
    logits, state = m(torch.tensor([65], dtype=torch.long))
    assert torch.allclose(state[0], torch.tensor([[1.0, 2.0, 3.0, 4.0]]), atol=1e-5)

def test_training_step_runs():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    loss, sampled, stop = m.training_step(65, 66, False)
    assert loss.numel() == 1 and 0 <= sampled <= 255 and 0.0 <= stop <= 1.0

def test_training_step_reports_components():
    import math
    m = TMTModel(TMTConfig(dim=16, layers=1))
    loss, _, _ = m.training_step(65, 66, False)
    comp = m.last_components
    assert set(comp) == {"l_var", "l_pred", "l_ce", "l_stop", "l_fw", "state_norm"}
    assert all(math.isfinite(v) for v in comp.values())
    assert comp["state_norm"] >= 0.0
    total = comp["l_var"] + comp["l_pred"] + comp["l_ce"] + comp["l_stop"] + comp["l_fw"]
    assert abs(total - float(loss.item())) < 1e-4

def test_ingest_advances_state_without_learning():
    import copy
    m = TMTModel(TMTConfig(dim=16, layers=1))
    before_params = [p.detach().clone() for p in m.parameters()]
    before_state = m.layers[0].states.detach().clone()
    logits, _ = m.ingest(65)
    assert logits.shape == (1, 256)
    assert not torch.equal(m.layers[0].states, before_state)
    for a, b in zip(before_params, m.parameters()):
        assert torch.equal(a, b)
    assert m.opt.state == {}

def test_sparse_readout_level_and_off_path():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    x = torch.tensor([65], dtype=torch.long)
    with torch.no_grad():
        l_off, _ = m(x)
    m2 = TMTModel(TMTConfig(dim=16, layers=1, sparse_k=4))
    m2.load_state_dict(m.state_dict(), strict=False)
    assert set(m2.state_dict()) >= set(m.state_dict())
    with torch.no_grad():
        enc = m2.encoder(x)
        h = enc
        for layer in m2.layers:
            h, _, _, _ = layer(enc, h)
        assert int((h != 0).sum()) == 16

def test_sparsify_exact_level():
    from tmt.model import sparsify
    torch.manual_seed(0)
    h = torch.randn(1, 16)
    assert sparsify(h, 0) is h
    assert sparsify(h, 16) is h
    s = sparsify(h, 4)
    assert int((s != 0).sum()) == 4
    assert torch.equal(s[s != 0].abs().sort().values,
                       h.abs().flatten().topk(4).values.sort().values)
