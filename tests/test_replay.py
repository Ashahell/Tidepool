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
    assert m.replay_buf[0][:3] == (67, 66, False)
    assert isinstance(m.replay_buf[0][3], float)

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

def test_tags_stored_as_floats():
    m = _model(replay_size=4, replay_k=0)
    m.training_step(65, 66, False)
    assert len(m.replay_buf[0]) == 4
    assert m.replay_buf[0][:3] == (65, 66, False)
    assert isinstance(m.replay_buf[0][3], float)

def test_weighted_skew():
    from collections import Counter
    torch.manual_seed(0)
    m = _model(replay_size=4, replay_k=0, replay_alpha=2.0)
    m.replay_buf.extend([(65, 66, False, 0.1), (65, 66, False, 0.2),
                         (65, 66, False, 10.0), (65, 66, False, 0.1)])
    idx = [m._replay_indices(1)[0] for _ in range(200)]
    assert Counter(idx)[2] > 100

def test_alpha_zero_uniform():
    torch.manual_seed(3)
    m = _model(replay_size=4, replay_k=0, replay_alpha=0.0)
    m.replay_buf.extend([(65, 66, False, 0.1), (65, 66, False, 0.2),
                         (65, 66, False, 10.0), (65, 66, False, 0.1)])
    idx = [m._replay_indices(1)[0] for _ in range(200)]
    assert sorted(__import__("collections").Counter(idx).keys()) == [0, 1, 2, 3]

def test_refresh_updates_tag():
    torch.manual_seed(5)
    m = _model(replay_size=8, replay_k=1)
    for _ in range(5):
        m.training_step(65, 66, False)
    for i in range(5):
        c, n, e, _ = m.replay_buf[i]
        m.replay_buf[i] = (c, n, e, 123.0 if i == 0 else 0.0)
    m.training_step(65, 66, False)
    for _ in range(2):
        if m.replay_buf[0][3] != 123.0:
            break
        m.training_step(65, 66, False)
    assert m.replay_buf[0][3] != 123.0

def test_random_replay_runs_and_differs():
    a = _model(replay_size=0)
    b = _model(replay_size=16, replay_k=1, replay_noise=True)
    assert b.replay_buf is not None
    for i in range(5):
        torch.manual_seed(2)
        a.training_step(65 + i, 66, False)
        torch.manual_seed(2)
        b.training_step(65 + i, 66, False)
    assert any(not torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    assert all(torch.isfinite(p).all() for p in b.parameters())
