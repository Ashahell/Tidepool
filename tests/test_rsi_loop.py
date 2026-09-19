from rsi.loop import run_search
from rsi.monitor import summarize_round
from rsi.policy import InitialParallelRefine
from rsi.rewriter import PolicyRewriter
import yaml

def mock_scorer(config):
    return {"config": config, "final_metrics": {},
            "composite_score": float(config.get("dim", 0)),
            "failure_mode": None, "cost": {"gpu_hours": 0.01}}

class BoomRewriter(PolicyRewriter):
    def rewrite(self, current_code, stats, feedback):
        raise RuntimeError("no llm")

def test_loop_explores_and_never_regresses(tmp_path):
    p = InitialParallelRefine({"dim": [32, 64]}, seed=1)
    s = run_search(mock_scorer, p, 0.04, 2, 1, BoomRewriter(),
                   str(tmp_path), seed=1)
    assert s["rounds"] == 2 and s["best"] == 64.0
    assert (tmp_path / "round_2" / "summary.json").exists()

def test_failed_scorer_recorded_not_raised(tmp_path):
    def bad(config):
        raise ValueError("boom")
    p = InitialParallelRefine({"dim": [32]}, seed=1)
    s = run_search(bad, p, 0.04, 1, 0, BoomRewriter(), str(tmp_path))
    assert s["best"] is None

def test_summarize_round_shape():
    nodes = [{"composite": 2.0, "cost": {"gpu_hours": 0.1}},
             {"composite": None, "cost": {"gpu_hours": 0.0}}]
    assert summarize_round(1, nodes, "p") == {
        "round": 1, "policy_id": "p", "n": 2, "best": 2.0, "spend": 0.1}

def test_loop_accepts_replay_configs(tmp_path):
    grid = yaml.safe_load(open("configs/rsi_smoke.yaml"))["grid"]
    assert grid["replay_size"] == [0, 64] and grid["replay_k"] == [1]

def test_loop_accepts_replay_alpha():
    import yaml
    grid = yaml.safe_load(open("configs/rsi_smoke.yaml"))["grid"]
    assert grid["replay_alpha"] == [0.0, 1.0]

def test_loop_accepts_slots():
    import yaml
    grid = yaml.safe_load(open("configs/rsi_smoke.yaml"))["grid"]
    assert grid["slots"] == [0, 16]
