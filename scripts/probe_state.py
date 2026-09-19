# scripts/probe_state.py
"""Linear probe: can payload bytes be decoded from frozen state?

Usage: probe_state.py --ckpt PATH --config CFG [--episodes N]
Trains a linear classifier state->first-payload-byte on frozen-model
states. Above-chance accuracy = information present but readout fails;
chance = information absent.
"""
from __future__ import annotations
import argparse
import random
import sys
sys.path.insert(0, "src")
import torch
import torch.nn as nn
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.engine import load_checkpoint
from tmt.data import copy_episode, QUERY_MARKER


def collect(model, n_episodes, seed):
    rng = random.Random(seed)
    states, labels = [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = copy_episode(rng, (4,), (8,))
            pre, post = ep.split(QUERY_MARKER)
            model.reset()
            for b in pre + QUERY_MARKER:
                model.ingest(b)
            states.append(model.layers[-1].states.detach().clone().squeeze(0))
            labels.append(post[0])
    return torch.stack(states), torch.tensor(labels, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/small_lr1e4.yaml")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--train-steps", type=int, default=500)
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    model.eval()
    load_checkpoint(model, args.ckpt)
    Xtr, ytr = collect(model, args.episodes, seed=0)
    Xte, yte = collect(model, args.episodes // 4, seed=1)
    dim = Xtr.shape[1]
    probe = nn.Linear(dim, 256)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for _ in range(args.train_steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(probe(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(Xte).argmax(dim=-1)
        acc = float((pred == yte).float().mean())
        maj = float((yte == yte.mode().values).float().mean())
    print(f"probe_acc={acc:.4f} majority={maj:.4f} chance~{1/95:.4f}")


if __name__ == "__main__":
    main()
