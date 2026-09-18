# tests/test_rsi_sandbox.py
import pytest
from rsi.policy import ExplorationPolicy
from rsi.sandbox import PolicyRejected, check_policy_source, load_policy, run_with_timeout

GOOD = '''
class CandidatePolicy(ExplorationPolicy):
    policy_id = "cand"
    def select_next_configs(self, history, budget):
        return [{"dim": 32} for _ in range(budget)]
'''

def test_accepts_clean_policy():
    p = load_policy(GOOD, ExplorationPolicy)
    assert isinstance(p, ExplorationPolicy)
    assert p.select_next_configs([], 2) == [{"dim": 32}, {"dim": 32}]

def test_rejects_bad_imports():
    with pytest.raises(PolicyRejected):
        check_policy_source("import os\nx = 1")

def test_rejects_while_loops():
    with pytest.raises(PolicyRejected):
        check_policy_source("while True:\n    x = 1")

def test_rejects_missing_subclass():
    with pytest.raises(PolicyRejected):
        load_policy("x = 1", ExplorationPolicy)

def test_timeout_fires():
    import time
    with pytest.raises(TimeoutError):
        run_with_timeout(lambda: time.sleep(2), 0.1)
