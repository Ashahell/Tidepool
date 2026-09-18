from __future__ import annotations
from .tree import config_hash


def replay_score(candidate, history: list[dict], budget: float) -> dict:
    by_hash: dict[str, dict] = {}
    for h in history:
        if h.get("status") == "ok" and h.get("composite") is not None:
            by_hash.setdefault(config_hash(h["config"]), h)
    proposals = candidate.select_next_configs(history, len(history) + 1)
    used: list[dict] = []
    cost = 0.0
    for cfg in proposals:
        node = by_hash.get(config_hash(cfg))
        if node is None or node["id"] in {u["id"] for u in used}:
            continue
        c = float(node.get("cost", {}).get("gpu_hours", 0.0))
        if cost + c > budget:
            break
        used.append(node)
        cost += c
        if candidate.should_stop(used, cost):
            break
    best = max([u["composite"] for u in used], default=None)
    return {"score": best, "cost": cost, "n": len(used)}
