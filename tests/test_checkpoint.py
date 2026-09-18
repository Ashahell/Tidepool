# tests/test_checkpoint.py
import random
import numpy as np
import pytest
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import skip_bytes
from tmt.engine import save_checkpoint, load_checkpoint

def _seed_all(s=5):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def _trained(tmp_path, steps=3):
    _seed_all()
    m = TMTModel(TMTConfig(dim=8, layers=1))
    for i in range(steps):
        m.training_step(65 + i, 66, False)
    p = str(tmp_path / "c")
    save_checkpoint(m, p, meta={"step": steps, "bytes_seen": steps})
    return m, p

def test_model_state_resume_equals_uninterrupted(tmp_path):
    m, p = _trained(tmp_path, 3)
    m2 = TMTModel(TMTConfig(dim=8, layers=1))
    load_checkpoint(m2, p)
    for i in range(3, 6):
        m.training_step(65 + i, 66, False)
        m2.training_step(65 + i, 66, False)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)

def test_dataset_cursor_resume(tmp_path):
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "wiki_a").write_text("".join(f"line{i}\n" for i in range(8)))
    first = b"".join(list(skip_bytes(str(src), 0))[:4])
    rest = b"".join(list(skip_bytes(str(src), 4))[:4])
    assert first + rest == b"".join(list(skip_bytes(str(src), 0))[:8])
    assert len(rest) > 0

def test_missing_checkpoint_raises(tmp_path):
    m = TMTModel(TMTConfig(dim=8, layers=1))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(m, str(tmp_path / "nope"))

def test_corrupt_checkpoint_raises(tmp_path):
    p = tmp_path / "bad"
    (tmp_path / "bad.safetensors").write_bytes(b"not safetensors" * 10)
    m = TMTModel(TMTConfig(dim=8, layers=1))
    with pytest.raises(Exception):
        load_checkpoint(m, str(p))

def test_incompatible_arch_raises(tmp_path):
    m, p = _trained(tmp_path, 1)
    m2 = TMTModel(TMTConfig(dim=16, layers=1))
    with pytest.raises(Exception):
        load_checkpoint(m2, p)

def test_bad_byte_rejected():
    m = TMTModel(TMTConfig(dim=8, layers=1))
    with pytest.raises(ValueError):
        m.training_step(300, 66, False)
