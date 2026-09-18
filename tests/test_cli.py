# tests/test_cli.py
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.engine import gen_bytes

def test_gen_terminates():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    m.eval()
    for seed in (0, 65):
        out = gen_bytes(m, seed)
        assert isinstance(out, bytes)
        assert len(out) <= 256
