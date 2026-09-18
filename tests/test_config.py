# tests/test_config.py
from tmt.config import TMTConfig

def test_defaults_match_ref():
    c = TMTConfig()
    assert (c.dim, c.layers, c.temp, c.lr) == (512, 16, 0.75, 5e-4)
    assert c.update_every == 1

def test_yaml_roundtrip(tmp_path):
    from tmt.config import TMTConfig
    c = TMTConfig(dim=64, layers=2)
    p = tmp_path / "c.yaml"
    p.write_text("dim: 64\nlayers: 2\n")
    c2 = TMTConfig.from_yaml(str(p))
    assert (c2.dim, c2.layers) == (64, 2)
    assert c2.to_dict()["dim"] == 64
