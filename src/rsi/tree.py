from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Node:
    id: str = ""
    parent_id: Optional[str] = None
    config: dict = field(default_factory=dict)
    status: str = "ok"
    metrics: dict = field(default_factory=dict)
    composite: Optional[float] = None
    cost: dict = field(default_factory=dict)
    failure_mode: Optional[str] = None
    checkpoint_path: Optional[str] = None
    policy_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(**{k: d.get(k, v) for k, v in {
            "id": "", "parent_id": None, "config": {}, "status": "ok",
            "metrics": {}, "composite": None, "cost": {},
            "failure_mode": None, "checkpoint_path": None,
            "policy_id": "", "created_at": ""}.items()})


class DiscoveryTree:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, node: Node) -> Node:
        if not node.id:
            node.id = f"{node.policy_id}:{config_hash(node.config)}"
        if not node.created_at:
            node.created_at = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(node.to_dict()) + "\n")
        return node

    def nodes(self) -> list[Node]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(Node.from_dict(json.loads(line)))
        return out

    def best(self) -> Optional[Node]:
        scored = [n for n in self.nodes() if n.composite is not None]
        return max(scored, key=lambda n: n.composite) if scored else None
