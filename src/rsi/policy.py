from __future__ import annotations
import random


class ExplorationPolicy:
    policy_id = "base"

    def select_next_configs(self, history: list[dict], budget: int) -> list[dict]:
        raise NotImplementedError

    def should_expand(self, node: dict) -> bool:
        return node.get("status") == "ok" and node.get("composite") is not None

    def parallel_degree(self, remaining: float) -> int:
        return min(4, max(1, int(remaining)))

    def should_stop(self, nodes: list[dict], used: float) -> bool:
        return used > 0 and len(nodes) == 0


class InitialParallelRefine(ExplorationPolicy):
    policy_id = "init-parallel-refine"

    def __init__(self, grid: dict[str, list], seed: int = 42,
                 top_k: int = 2, random_frac: float = 0.25):
        self.grid = grid
        self.seed = seed
        self.rng = random.Random(seed)
        self.top_k = top_k
        self.random_frac = random_frac

    def _sample(self) -> dict:
        return {k: self.rng.choice(v) for k, v in self.grid.items()}

    def _refine(self, best: dict) -> dict:
        out = {}
        for k, vals in self.grid.items():
            nums = [v for v in vals if isinstance(v, (int, float))]
            if k in best and nums:
                lo, hi = min(nums), max(nums)
                v = min(max(best[k], lo), hi)
                if all(isinstance(v, int) for v in nums):
                    v = int(round(v))
                out[k] = v
            else:
                out[k] = self.rng.choice(vals)
        return out

    def select_next_configs(self, history: list[dict], budget: int) -> list[dict]:
        scored = [h for h in history
                  if h.get("status") == "ok" and h.get("composite") is not None]
        if not scored:
            rng = random.Random(self.seed)
            return [{k: rng.choice(v) for k, v in self.grid.items()}
                    for _ in range(max(0, budget))]
        scored.sort(key=lambda h: h["composite"], reverse=True)
        tops = [h["config"] for h in scored[: self.top_k]]
        n_random = int(round(budget * self.random_frac))
        cfgs = [self._sample() for _ in range(n_random)]
        while len(cfgs) < budget:
            cfgs.append(self._refine(self.rng.choice(tops)))
        return cfgs
