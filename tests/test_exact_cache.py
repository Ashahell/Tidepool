# tests/test_exact_cache.py
import torch
from tmt.exact_cache import ExactCache


def test_store_retrieve_exact():
    c = ExactCache(k=4, capacity=16)
    seq = [10, 20, 30, 40, 50, 60]
    for b in seq:
        c.ingest(b)
    assert c.lookup((10, 20, 30, 40)) == 50
    assert c.lookup((1, 2, 3, 4)) is None


def test_capacity_bounded_lru():
    c = ExactCache(k=2, capacity=4)
    for i in range(20):
        c.ingest(i % 256)
        c.ingest((i * 7) % 256)
    assert len(c.table) <= 4


def test_interpolate_endpoints():
    c = ExactCache(k=2, capacity=16, lam=1.0)
    c.ingest(5)
    c.ingest(6)
    c.ingest(7)
    logits = torch.zeros(1, 256)
    out = c.interpolate(logits, (5, 6))
    assert int(out.argmax()) == 7
    c0 = ExactCache(k=2, capacity=16, lam=0.0)
    out0 = c0.interpolate(logits + 1.0, (5, 6))
    assert torch.allclose(out0, logits + 1.0)


def test_reset_clears():
    c = ExactCache(k=2, capacity=16)
    c.ingest(5)
    c.reset()
    assert len(c.table) == 0 and c.hist == []
