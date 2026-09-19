# src/tmt/assoc_probe.py
"""Associative retrieval probe (key -> value after interference).

Protocol per trial: feed K (key, value) pairs with filler between them,
then present one query key; the model must emit its value. Feed via
model.ingest(), query via forward read + ingest feedback (mirrors the
ingesting copy probe). Frozen evaluation_suite.py is imported for
EvalConfig/ProbeResult only.
"""
from __future__ import annotations

import numpy as np
import torch
from tmt.evaluation_suite import EvalConfig, ProbeResult

BRACKETS = [(40, 41), (91, 93), (123, 125), (60, 62)]


def assoc_memory_ingesting(model, n_pairs=8, inter_len=64, n_trials=10,
                           pairs=None, seed=0) -> ProbeResult:
    rng = np.random.RandomState(seed)
    correct = 0
    total = 0
    for _ in range(n_trials):
        model.reset()
        if pairs is None:
            keys = rng.choice(256, size=n_pairs, replace=False)
            vals = rng.randint(32, 127, size=n_pairs)
            trial_pairs = list(zip(keys.tolist(), vals.tolist()))
        else:
            trial_pairs = pairs
        for k, v in trial_pairs:
            model.ingest(int(k))
            model.ingest(int(v))
            for b in rng.randint(0, 256, size=inter_len).tolist():
                model.ingest(int(b))
        qk, qv = trial_pairs[rng.randint(len(trial_pairs))]
        logits, _ = model(torch.tensor([int(qk)], dtype=torch.long))
        pred = int(torch.argmax(logits[0]).item())
        model.ingest(pred)
        total += 1
        if pred == int(qv):
            correct += 1
    return ProbeResult(name="assoc_memory",
                       score=correct / total if total else 0.0,
                       details={"n_pairs": n_pairs, "inter_len": inter_len,
                                "n_trials": n_trials})
