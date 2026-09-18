from __future__ import annotations
import inspect
import json
from pathlib import Path
from .monitor import summarize_round
from .replay import replay_score
from .sandbox import load_policy
from .tree import DiscoveryTree, Node

def _policy_source(policy) -> str:
    try:
        return inspect.getsource(policy.__class__)
    except (OSError, TypeError):
        return "# builtin"

def run_search(scorer, policy, budget_per_round: float, rounds: int,
               n_revisions: int, rewriter, workdir: str,
               seed: int = 42) -> dict:
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    all_nodes: list[dict] = []
    incumbent = policy
    incumbent_code = _policy_source(policy)
    summaries = []
    for r in range(1, rounds + 1):
        rdir = root / f"round_{r}"
        rdir.mkdir(parents=True, exist_ok=True)
        tree = DiscoveryTree(str(rdir / "tree.jsonl"))
        spent = 0.0
        cfgs = incumbent.select_next_configs(
            all_nodes, max(1, int(budget_per_round / 0.01)))
        for cfg in cfgs:
            if spent >= budget_per_round:
                break
            try:
                res = scorer(cfg)
                node = Node(parent_id=None, config=res["config"],
                            status="ok", metrics=res.get("final_metrics", {}),
                            composite=res.get("composite_score"),
                            cost=res.get("cost", {"gpu_hours": 0.0}),
                            failure_mode=res.get("failure_mode"),
                            checkpoint_path=res.get("checkpoint_path"),
                            policy_id=incumbent.policy_id)
                spent += float(node.cost.get("gpu_hours", 0.0))
            except Exception as e:
                node = Node(parent_id=None, config=cfg, status="failed",
                            metrics={}, composite=None,
                            cost={"gpu_hours": 0.0},
                            failure_mode=f"{type(e).__name__}: {e}",
                            checkpoint_path=None,
                            policy_id=incumbent.policy_id)
            tree.append(node)
            all_nodes.append(node.to_dict())
        (rdir / "policy.py").write_text(incumbent_code)
        base = replay_score(incumbent, all_nodes, budget_per_round)
        feedback: list[str] = []
        for _ in range(n_revisions):
            try:
                code = rewriter.rewrite(
                    incumbent_code,
                    {"best": base["score"], "spent": spent,
                     "n": len(all_nodes)},
                    feedback)
                cand = load_policy(code, type(incumbent))
                rep = replay_score(cand, all_nodes, budget_per_round)
                if rep["score"] is not None and (
                        base["score"] is None or rep["score"] > base["score"]):
                    incumbent, incumbent_code, base = cand, code, rep
            except Exception as e:
                feedback.append(f"{type(e).__name__}: {e}")
        summary = summarize_round(r, tree.nodes() and
                                  [n.to_dict() for n in tree.nodes()],
                                  incumbent.policy_id)
        summary["spend"] = spent
        (rdir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
    best = max([s["best"] for s in summaries
                if s["best"] is not None], default=None)
    return {"rounds": rounds, "best": best, "summaries": summaries}
