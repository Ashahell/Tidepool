from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
import yaml
from rsi.loop import run_search
from rsi.monitor import print_summary
from rsi.policy import InitialParallelRefine
from rsi.rewriter import AgentPolicyRewriter
from tmt.data import iter_wikipedia_bytes
from tmt.evaluation_suite import EvalConfig, train_and_evaluate

EVAL_PRESETS = {
    "smoke": EvalConfig(max_bytes=2000, lm_eval_tokens=256,
                        copy_lengths=[4], intervening_lengths=[8, 64],
                        num_copy_trials=2, retention_eval_bytes=128),
    "full": EvalConfig(),
}

def load_bytes(path: str, limit: int) -> bytes:
    with open(path, "rb") as f:
        return f.read(limit)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rsi_smoke.yaml")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--workdir", default="runs/rsi")
    ap.add_argument("--val", action="append", default=[],
                    help="file(s) to read validation bytes from (repeatable)")
    ap.add_argument("--continual", action="append", default=[],
                    help="file(s) for continual-retention domains (repeatable)")
    ap.add_argument("--eval-preset", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--train", default="data/vk4a_train")
    ap.add_argument("--train-steps", type=int, default=2000)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rounds = args.rounds or cfg.get("rounds", 2)
    budget = args.budget or cfg.get("budget_per_round", 0.08)
    policy = InitialParallelRefine(cfg["grid"], seed=cfg.get("seed", 42))
    if args.val:
        val = b"".join(load_bytes(p, 4096) for p in args.val)[:8192]
    else:
        val = bytes([(65 + i % 26) for i in range(512)])
    cont = ([b"".join(load_bytes(p, 2048) for p in args.continual)[:4096]]
            if args.continual else [val])
    eval_cfg = EVAL_PRESETS[args.eval_preset]
    train_blobs = []
    for chunk in iter_wikipedia_bytes(args.train):
        train_blobs.append(chunk)
        if sum(len(c) for c in train_blobs) >= args.train_steps + 1:
            break
    train_bytes = b"".join(train_blobs)[: args.train_steps + 1]

    def train_fn(model, config):
        import time
        t0 = time.time()
        for i in range(min(args.train_steps, len(train_bytes) - 1)):
            model.training_step(train_bytes[i], train_bytes[i + 1], False)
        return (time.time() - t0) / 3600.0
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
            config, val, cont,
            build_fn=lambda c: model,
            train_fn=train_fn,
            eval_cfg=eval_cfg)
    summary = run_search(scorer, policy, budget, rounds,
                         cfg.get("n_revisions", 1),
                         AgentPolicyRewriter(f"{args.workdir}/round_1"),
                         args.workdir, seed=cfg.get("seed", 42))
    print_summary(summary)

if __name__ == "__main__":
    main()
