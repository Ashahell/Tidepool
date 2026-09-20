# scripts/memory_causality.py
"""Memory Causality Experiment (consultant prescription).

At the query point of a copy episode, intervene on persistent state and
measure P(correct first payload byte):
  A normal state
  B zeroed state
  C shuffled state (fixed permutation)
  D replaced state (from a different episode)
Want A >> B/C/D. A ~= B ~= C ~= D means information without causal use.
Usage: memory_causality.py --ckpt PATH --config CFG [--trials N]
"""
from __future__ import annotations
import argparse
import random
import sys
sys.path.insert(0, "src")
import torch
import torch.nn.functional as F
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.engine import load_checkpoint
from tmt.data import copy_episode, QUERY_MARKER


def snap(model):
    return [l.states.detach().clone() for l in model.layers]


def restore(model, saved):
    with torch.no_grad():
        for l, s in zip(model.layers, saved):
            l.states.copy_(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/small_lr1e4.yaml")
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    model.eval()
    load_checkpoint(model, args.ckpt)
    rng = random.Random(0)
    perm = torch.randperm(cfg.dim, generator=torch.Generator().manual_seed(0))
    hits = {"A": 0, "B": 0, "C": 0, "D": 0}
    with torch.no_grad():
        # Donor state for condition D, built once from another episode.
        dep = copy_episode(random.Random(999))
        dpre, _ = dep.split(QUERY_MARKER)
        model.reset()
        for b in dpre + QUERY_MARKER:
            model.ingest(b)
        donor = snap(model)
        for _ in range(args.trials):
            ep = copy_episode(rng)
            pre, post = ep.split(QUERY_MARKER)
            model.reset()
            for b in pre + QUERY_MARKER:
                model.ingest(b)
            base = snap(model)
            q = torch.tensor([QUERY_MARKER[-1]], dtype=torch.long)
            for name, fn in [
                    ("A", lambda: None),
                    ("B", lambda: [l.states.zero_() for l in model.layers]),
                    ("C", lambda: [l.states.copy_(l.states[:, perm]) for l in model.layers]),
                    ("D", lambda: restore(model, donor))]:
                restore(model, base)
                fn()
                logits, _ = model(q)
                if int(logits[0].argmax()) == post[0]:
                    hits[name] += 1
    n = args.trials
    print({k: f"{v}/{n}" for k, v in hits.items()})


if __name__ == "__main__":
    main()
