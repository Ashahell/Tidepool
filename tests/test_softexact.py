# tests/test_softexact.py
import torch
from tmt.softexact import SoftExact


def test_store_retrieve_exact():
    s = SoftExact(dim=8, capacity=16)
    torch.manual_seed(0)
    k = torch.randn(8)
    s.write(k, 42)
    assert s.predict(k) == 42


def test_similar_query_retrieves():
    torch.manual_seed(1)
    s = SoftExact(dim=16, capacity=16)
    k1 = torch.randn(16)
    k2 = torch.randn(16)
    s.write(k1, 10)
    s.write(k2, 20)
    assert s.predict(k1 + 0.01 * torch.randn(16)) == 10


def test_capacity_bounded():
    s = SoftExact(dim=8, capacity=4)
    for i in range(10):
        s.write(torch.randn(8), i % 256)
    assert len(s.keys) == 4 and len(s.vals) == 4


def test_reset_clears():
    s = SoftExact(dim=8, capacity=16)
    s.write(torch.randn(8), 5)
    s.reset()
    assert s.keys == [] and s.vals == []
    assert float(s.read_dist(torch.randn(8)).max()) == float("-inf")
