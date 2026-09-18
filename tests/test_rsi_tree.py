from rsi.tree import config_hash, DiscoveryTree, Node


def test_hash_stable_and_short():
    assert config_hash({"b": 1, "a": 2}) == config_hash({"a": 2, "b": 1})
    assert len(config_hash({"a": 1})) == 12


def test_append_best_roundtrip(tmp_path):
    t = DiscoveryTree(str(tmp_path / "tree.jsonl"))
    t.append(Node(id="", parent_id=None, config={"dim": 32}, status="ok",
                  metrics={}, composite=1.5, cost={"gpu_hours": 0.1},
                  failure_mode=None, checkpoint_path=None,
                  policy_id="p0", created_at=""))
    t.append(Node(id="", parent_id=None, config={"dim": 64}, status="failed",
                  metrics={}, composite=None, cost={"gpu_hours": 0.1},
                  failure_mode="oom", checkpoint_path=None,
                  policy_id="p0", created_at=""))
    nodes = t.nodes()
    assert len(nodes) == 2 and nodes[0].id == "p0:" + config_hash({"dim": 32})
    assert t.best().config == {"dim": 32}
    t2 = DiscoveryTree(str(tmp_path / "tree.jsonl"))
    assert t2.best().composite == 1.5
