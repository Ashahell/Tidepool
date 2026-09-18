from rsi.policy import InitialParallelRefine
from rsi.replay import replay_score
from rsi.tree import config_hash


def _hist():
    return [
        {"id": "p:a", "config": {"dim": 32}, "status": "ok",
         "composite": 2.0, "cost": {"gpu_hours": 0.5}},
        {"id": "p:b", "config": {"dim": 64}, "status": "ok",
         "composite": 7.0, "cost": {"gpu_hours": 0.5}},
        {"id": "p:c", "config": {"dim": 128}, "status": "failed",
         "composite": None, "cost": {"gpu_hours": 0.5}},
    ]


class FixedOrder:
    def select_next_configs(self, history, budget):
        return [{"dim": 64}, {"dim": 999}, {"dim": 32}]

    def should_stop(self, nodes, used):
        return False


def test_replay_uses_only_recorded_ok_within_budget():
    r = replay_score(FixedOrder(), _hist(), 10.0)
    assert r == {"score": 7.0, "cost": 1.0, "n": 2}


def test_replay_respects_budget():
    r = replay_score(FixedOrder(), _hist(), 0.5)
    assert r == {"score": 7.0, "cost": 0.5, "n": 1}


def test_replay_empty_when_nothing_matches():
    r = replay_score(FixedOrder(), [], 10.0)
    assert r == {"score": None, "cost": 0.0, "n": 0}


def test_replay_equals_online_for_own_policy():
    p = InitialParallelRefine({"dim": [32, 64]}, seed=4)
    online = p.select_next_configs([], 2)
    hist = [{"id": f"p:{i}", "config": c, "status": "ok",
             "composite": float(i), "cost": {"gpu_hours": 0.5}}
            for i, c in enumerate(online)]
    r = replay_score(p, hist, 10.0)
    assert r["n"] == 2 and r["score"] == 1.0 and r["cost"] == 1.0
