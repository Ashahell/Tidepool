# scripts/remeasure.py — Task 5 re-measurement matrix (8 cells).
#
# {rtrl, single-step} x {persistent, reset} x {groups-4, groups-1}.
# Tiny models (dim=32, layers=2), fixed seed, fixed data slice, JSONL out.
# Single-step cell: monkeypatch model._rtrl_enabled = False (gate from Task 1,
# default True). evaluation_suite.py is used frozen — never edited here.
from __future__ import annotations

import itertools
import json
import sys
import time

sys.path.insert(0, "src")

from pathlib import Path

import torch

from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import load_val_bytes
from tmt.evaluation_suite import EvalConfig, compute_bpb, evaluate_model

SEED = 42
STEPS = 2000
TRAIN_ROOT = "data/vk4a_train"
VAL_PATH = "/tmp/rsi_data/val_heldout.bin"
OUT = Path("runs/remeasure.jsonl")

# Smoke preset (mirrors scripts/train.py EVAL_PRESET) but with copy targets
# at 8/64 bytes so question (C) is answered directly.
EVAL_CFG = EvalConfig(
    max_bytes=2000,
    lm_eval_tokens=256,
    copy_lengths=[8, 64],
    intervening_lengths=[8, 64],
    num_copy_trials=2,
    retention_eval_bytes=128,
)


def load_train_bytes(n: int) -> bytes:
    buf = bytearray()
    for p in sorted(Path(TRAIN_ROOT).iterdir()):
        if p.is_file():
            buf += p.read_bytes()
        if len(buf) >= n + 1:
            break
    if len(buf) < n + 1:
        raise RuntimeError(f"short train slice: {len(buf)} < {n + 1}")
    return bytes(buf[: n + 1])


def run_cell(mode: str, state: str, groups: int) -> dict:
    torch.manual_seed(SEED)
    try:
        import numpy as np

        np.random.seed(SEED)
    except ImportError:
        pass
    cfg = TMTConfig(dim=32, layers=2, lr=1e-4, update_every=1,
                    decay_groups=groups, seed=SEED)
    model = TMTModel(cfg)
    model.init_decay_groups()
    if mode == "single-step":
        model._rtrl_enabled = False  # test-only monkeypatch; default True
    data = load_train_bytes(STEPS)
    val = load_val_bytes(VAL_PATH, limit=4096)
    bpb_init = compute_bpb(model, val[: EVAL_CFG.lm_eval_tokens])
    model.reset()
    first_loss, last_loss = None, None
    t0 = time.time()
    for i in range(STEPS):
        if state == "reset":
            model.reset()
        loss, _, _ = model.training_step(data[i], data[i + 1], False)
        lv = float(loss.item())
        if first_loss is None:
            first_loss = lv
        last_loss = lv
    train_s = time.time() - t0
    bpb_final = compute_bpb(model, val[: EVAL_CFG.lm_eval_tokens])
    r = evaluate_model(model, val, [val], cfg=EVAL_CFG,
                       config_dict=cfg.to_dict(), gpu_hours=0.0)
    mem_details = next(p.details for p in r.probes if p.name == "copy_memory")
    return {
        "cell": f"{mode}/{state}/groups-{groups}",
        "mode": mode, "state": state, "groups": groups,
        "seed": SEED, "steps": STEPS, "train_s": round(train_s, 1),
        "loss_first": first_loss, "loss_last": last_loss,
        "bpb_init": bpb_init, "bpb_final": bpb_final,
        "val_bpb": r.val_bpb, "copy_mean": r.long_range_score,
        "copy_details": mem_details, "continual": r.continual_score,
        "stability": r.stability_score, "composite": r.composite_score,
        "failure_mode": r.failure_mode,
    }


def main() -> None:
    assert Path(TRAIN_ROOT).is_dir(), f"missing {TRAIN_ROOT}"
    assert Path(VAL_PATH).is_file(), f"missing {VAL_PATH}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    for mode, state, groups in itertools.product(
            ["rtrl", "single-step"], ["persistent", "reset"], [4, 1]):
        rec = run_cell(mode, state, groups)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{rec['cell']}: loss {rec['loss_first']:.3f}->"
              f"{rec['loss_last']:.3f} bpb {rec['bpb_init']:.3f}->"
              f"{rec['bpb_final']:.3f} copy {rec['copy_mean']:.3f} "
              f"cont {rec['continual']:.3f} ({rec['train_s']}s)",
              flush=True)


if __name__ == "__main__":
    main()
