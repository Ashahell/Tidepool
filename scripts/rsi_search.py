from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
import yaml
from rsi.loop import run_search
from rsi.monitor import print_summary
from rsi.policy import InitialParallelRefine
from rsi.rewriter import AgentPolicyRewriter
from tmt.evaluation_suite import train_and_evaluate

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rsi_smoke.yaml")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--workdir", default="runs/rsi")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rounds = args.rounds or cfg.get("rounds", 2)
    budget = args.budget or cfg.get("budget_per_round", 0.08)
    policy = InitialParallelRefine(cfg["grid"], seed=cfg.get("seed", 42))
    val = bytes([(65 + i % 26) for i in range(512)])
    def scorer(config):
        from tmt.model import TMTModel
        from tmt.config import TMTConfig
        base = TMTConfig.from_yaml("configs/base.yaml")
        for k, v in config.items():
            if hasattr(base, k):
                setattr(base, k, v)
        model = TMTModel(base)
        model.init_decay_groups()
        return train_and_evaluate(
            config, val, [val],
            build_fn=lambda c: model,
            train_fn=lambda m, c: 0.0,
            eval_cfg=None) | {"cost": {"gpu_hours": 0.01}}
    summary = run_search(scorer, policy, budget, rounds,
                         cfg.get("n_revisions", 1),
                         AgentPolicyRewriter(f"{args.workdir}/round_1"),
                         args.workdir, seed=cfg.get("seed", 42))
    print_summary(summary)

if __name__ == "__main__":
    main()
