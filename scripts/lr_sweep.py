# scripts/lr_sweep.py
"""Fine learning-rate sweep on pure-copy training (Okpekpe & Orvieto 2025:
recurrent recall lives in a narrow LR window; our 3-value sweep may have
straddled it). Reports payload CE (primary, lower better) + copy probe.
Usage: lr_sweep.py [--steps N] [--trials CSV]; appends JSON lines to
runs/lr_sweep.jsonl. Retention gate: probe movement decides, CE second.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
sys.path.insert(0, "src")
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import copy_episode, QUERY_MARKER
from tmt.evaluation_suite import EvalConfig
from tmt.ingesting_probe import copy_memory_ingesting

LRS = [3e-5, 5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]


def run_lr(lr: float, steps: int, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    cfg = TMTConfig(dim=128, layers=4, lr=lr)
    m = TMTModel(cfg)
    m.reset()
    rng = random.Random(seed)
    n = 0
    while n < steps:
        ep = copy_episode(rng)
        m.reset()
        for i in range(len(ep) - 1):
            m.training_step(ep[i], ep[i + 1], False)
            n += 1
            if n >= steps:
                break
    m.eval()
    rng2 = random.Random(11)
    ce, cnt = 0.0, 0
    with torch.no_grad():
        for _ in range(20):
            ep = copy_episode(rng2)
            pre, post = ep.split(QUERY_MARKER)
            m.reset()
            for b in pre + QUERY_MARKER:
                m.ingest(b)
            logits, _ = m(torch.tensor([QUERY_MARKER[-1]], dtype=torch.long))
            ce += float(-F.log_softmax(logits, dim=-1)[0, post[0]].item())
            cnt += 1
    np.random.seed(7)
    pcfg = EvalConfig(copy_lengths=[4], intervening_lengths=[8, 32],
                      num_copy_trials=10)
    probe = copy_memory_ingesting(m, pcfg).score
    return {"lr": lr, "steps": steps, "payload_ce": round(ce / cnt, 4),
            "probe": round(probe, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lrs", default=",".join(map(str, LRS)))
    ap.add_argument("--out", default="runs/lr_sweep.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for lr in [float(x) for x in args.lrs.split(",")]:
        res = run_lr(lr, args.steps)
        print(res, flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
