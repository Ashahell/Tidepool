# src/tmt/ingesting_probe.py
"""Copy-memory probe that actually ingests.

The frozen suite's copy probe drives forward(), which never writes the
persistent buffers — no real model can score on it. This variant feeds
target + filler through model.ingest() (state motion without learning)
and generates autoregressively (forward read + ingest feedback).
Frozen evaluation_suite.py is imported, never modified.
"""
from __future__ import annotations

import numpy as np
import torch
from tmt.evaluation_suite import EvalConfig, ProbeResult


def copy_memory_ingesting(model, cfg: EvalConfig) -> ProbeResult:
    results = {}
    successes = []
    for copy_len in cfg.copy_lengths:
        for inter_len in cfg.intervening_lengths:
            key = f"copy{copy_len}_after_{inter_len}"
            correct = 0
            for _ in range(cfg.num_copy_trials):
                model.reset()
                target = bytes(np.random.randint(32, 127, size=copy_len).tolist())
                filler = bytes(np.random.randint(0, 256, size=inter_len).tolist())
                for b in target + filler:
                    model.ingest(int(b))
                generated = []
                for _ in range(copy_len):
                    logits, _ = model(torch.tensor([0], dtype=torch.long))
                    pred = int(torch.argmax(logits[0]).item())
                    generated.append(pred)
                    model.ingest(pred)
                if bytes(generated) == target:
                    correct += 1
            acc = correct / cfg.num_copy_trials
            results[key] = acc
            successes.append(acc)
    return ProbeResult(name="copy_memory_ingesting",
                       score=float(np.mean(successes)) if successes else 0.0,
                       details=results)
