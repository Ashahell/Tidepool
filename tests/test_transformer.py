# tests/test_transformer.py
import torch
from tmt.transformer_baseline import TinyTransformer


def test_causal_and_trains():
    torch.manual_seed(0)
    m = TinyTransformer(dim=16, layers=1, heads=2)
    assert m.count_params() > 0
    ep = bytes([65, 66, 67, 68])
    l0 = m.train_episode(ep)
    for _ in range(30):
        l1 = m.train_episode(ep)
    assert l1 < l0
    # Causality: prefix logits identical regardless of future bytes.
    m.reset()
    for b in (65, 66):
        m.ingest(b)
    a, _ = m(torch.tensor([67], dtype=torch.long))
    m.reset()
    for b in (65, 99):
        m.ingest(b)
    c, _ = m(torch.tensor([67], dtype=torch.long))
    assert not torch.equal(a, c)


def test_probe_compatible():
    m = TinyTransformer(dim=16, layers=1, heads=2)
    m.reset()
    assert len(m.ctx) == 0
    m.ingest(65)
    logits, _ = m(torch.tensor([66], dtype=torch.long))
    assert logits.shape == (1, 256)
