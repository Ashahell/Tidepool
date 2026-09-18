# tests/test_wiring.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.engine import save_checkpoint, load_checkpoint, node_json

def test_checkpoint_roundtrip(tmp_path):
    m = TMTModel(TMTConfig(dim=16, layers=1))
    p = str(tmp_path / "m.safetensors")
    save_checkpoint(m, p)
    m2 = TMTModel(TMTConfig(dim=16, layers=1))
    load_checkpoint(m2, p)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(a, b)

def test_node_json_shape():
    from tmt.engine import node_json
    n = node_json({"dim": 16}, {"val_bpb": 5.0}, 0.1, None)
    assert {"config", "final_metrics", "composite_score", "failure_mode", "cost"} <= set(n)
    assert {"run_id", "timestamp", "git_commit", "steps", "bytes_seen"} <= set(n)
    assert n["steps"] == 0 and n["bytes_seen"] == 0

def test_check_finite():
    from tmt.engine import check_finite
    assert check_finite({"loss": 1.0, "l_ce": 0.5}) is True
    assert check_finite({"loss": float("nan")}) is False
    assert check_finite({"loss": float("inf")}) is False

def test_ema_checkpoint_roundtrip(tmp_path):
    import torch
    from tmt.config import TMTConfig
    from tmt.model import TMTModel
    m = TMTModel(TMTConfig(dim=16, layers=1, ema_decay=0.9))
    m.training_step(65, 66, False)
    m.training_step(67, 68, False)
    assert m.ema_state is not None
    p = str(tmp_path / "e.safetensors")
    save_checkpoint(m, p)
    m2 = TMTModel(TMTConfig(dim=16, layers=1, ema_decay=0.9))
    assert m2.ema_state is None
    load_checkpoint(m2, p)
    assert set(m2.ema_state) == set(m.ema_state)
    for k in m.ema_state:
        assert torch.equal(m.ema_state[k], m2.ema_state[k])
