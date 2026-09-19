# tests/test_gru.py
import torch
from tmt.gru_baseline import GRUModel


def test_gru_trains_and_interfaces():
    torch.manual_seed(0)
    m = GRUModel(dim=16, layers=1)
    assert m.count_params() > 0
    l0, _, _ = m.train_step(65, 66, False)
    for _ in range(20):
        l1, _, _ = m.train_step(65, 66, False)
    assert float(l1.item()) < float(l0.item())
    logits, _ = m(torch.tensor([65], dtype=torch.long))
    assert logits.shape == (1, 256)
    m.reset()
    assert float(m.hidden.abs().max()) == 0.0
    lg, _ = m.ingest(65)
    assert lg.shape == (1, 256)
