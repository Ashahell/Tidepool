from __future__ import annotations
from dataclasses import asdict, dataclass
import yaml

@dataclass
class TMTConfig:
    dim: int = 512
    layers: int = 16
    temp: float = 0.75
    lr: float = 5e-4
    update_every: int = 1
    w_var: float = 1.0
    w_pred: float = 1.0
    w_ce: float = 1.0
    w_stop: float = 1.0
    decay_groups: int = 4
    grad_clip: float = 1.0
    seed: int = 42
    replay_size: int = 0
    replay_k: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "TMTConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)
