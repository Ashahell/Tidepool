# scripts/compare_baselines.py
"""Uniform / unigram / reset-every-byte TMT / persistent TMT bpb baselines.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compare_baselines.py --val FILE
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from tmt.config import TMTConfig
from tmt.data import load_val_bytes
from tmt.evaluation_suite import compute_bpb
from tmt.model import TMTModel


class ResetWrapper:
    """Hold a model and reset it before each call, then reuse compute_bpb."""
    def __init__(self, model):
        self.model = model

    def reset(self):
        self.model.reset()

    def __call__(self, x):
        self.model.reset()
        return self.model(x)


def unigram_bpb(data: bytes) -> float:
    counts = np.bincount(list(data), minlength=256).astype(np.float64) + 1.0
    probs = counts / counts.sum()
    return float(np.mean([-math.log2(probs[b]) for b in data]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokens", type=int, default=2000)
    args = ap.parse_args()

    data = load_val_bytes(args.val, limit=args.tokens)

    torch.manual_seed(0)
    model = TMTModel(TMTConfig.from_yaml(args.config))
    if args.ckpt is not None:
        from tmt.engine import load_checkpoint
        load_checkpoint(model, args.ckpt)
    model.eval()

    out = {
        "uniform": math.log2(256),
        "unigram": unigram_bpb(data),
        "reset_every_byte": compute_bpb(ResetWrapper(model), data),
        "persistent": compute_bpb(model, data),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
