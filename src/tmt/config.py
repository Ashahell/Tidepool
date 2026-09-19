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
    replay_noise: bool = False
    ema_decay: float = 0.0
    recall_hidden: int = 0
    replay_alpha: float = 1.0

    def __post_init__(self):
        for name in ("dim", "layers", "lr", "temp", "update_every",
                     "grad_clip", "decay_groups"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("replay_size", "replay_k", "ema_decay"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    @classmethod
    def from_yaml(cls, path: str) -> "TMTConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = [k for k in data if k not in cls.__dataclass_fields__]
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
