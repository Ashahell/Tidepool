# tests/test_replay.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def _model(**kw):
    torch.manual_seed(0)
    cfg = TMTConfig(dim=16, layers=1, **kw)
    m = TMTModel(cfg)
    return m

def test_buffer_appends_and_caps():
    m = _model(replay_size=8, replay_k=0)
    for i in range(10):
        m.training_step(65 + (i % 10), 66, False)
    assert m.replay_buf is not None and len(m.replay_buf) == 8
    assert m.replay_buf[0] == (67, 66, False)

def test_replay_off_is_none_and_matches_no_replay_weights():
    a = _model(replay_size=0)
    b = _model(replay_size=16, replay_k=0)
    assert a.replay_buf is None
    for i in range(3):
        a.training_step(65, 66, False)
        torch.manual_seed(0)
        b.training_step(65, 66, False)
        torch.manual_seed(0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)

def test_replay_on_diverges():
    a = _model(replay_size=0)
    b = _model(replay_size=16, replay_k=1)
    for i in range(5):
        torch.manual_seed(1)
        a.training_step(65 + i, 66, False)
        torch.manual_seed(1)
        b.training_step(65 + i, 66, False)
    assert any(not torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    assert all(torch.isfinite(p).all() for p in b.parameters())
