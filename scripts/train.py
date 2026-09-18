# scripts/train.py
from __future__ import annotations
import argparse
import json
import sys
import time
sys.path.insert(0, "src")
from pathlib import Path
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import iter_wikipedia_bytes, load_val_bytes, skip_bytes
from tmt.engine import (save_checkpoint, load_checkpoint, node_json,
                        check_finite, new_run_id, utc_timestamp,
                        git_commit_short)
from tmt.evaluation_suite import EvalConfig, evaluate_model

EVAL_PRESET = EvalConfig(max_bytes=2000, lm_eval_tokens=256, copy_lengths=[4],
                         intervening_lengths=[8, 64], num_copy_trials=2,
                         retention_eval_bytes=128)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    ap.add_argument("--data", default="wikipedia_clean")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-val", default="")
    ap.add_argument("--epochs", type=int, default=0,
                    help="shuffled passes over --data (0 = legacy single sorted pass)")
    ap.add_argument("--resume", action="store_true",
                    help="load checkpoint meta and seek bytes_seen before training")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    model.init_decay_groups()
    meta = load_checkpoint(model, args.ckpt, allow_missing=True) or {}
    resume_from = int(meta.get("bytes_seen", 0)) if args.resume else 0
    run_id, started_at, commit = new_run_id(), utc_timestamp(), git_commit_short()
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    loss_path = Path("runs/loss.csv")
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not loss_path.exists() or loss_path.stat().st_size == 0
    lf = open(loss_path, "a")
    if need_header:
        lf.write("step,loss,l_var,l_pred,l_ce,l_stop,state_norm\n")
    eval_path = Path("runs/eval.csv")
    if args.eval_every > 0 and args.eval_val:
        eval_bytes = load_val_bytes(args.eval_val)
        if not eval_path.exists() or eval_path.stat().st_size == 0:
            eval_path.write_text("step,bpb,mem,cont,stab,composite\n")
    n = resume_from
    min_loss = None
    failure = None
    t0 = time.time()
    try:
        def _chunks():
            if args.resume and resume_from > 0 and args.epochs <= 0:
                buf = bytearray()
                for b in skip_bytes(args.data, resume_from):
                    buf += b
                    if b == b"\n":
                        yield bytes(buf)
                        buf = bytearray()
                if buf:
                    yield bytes(buf)
                return
            if args.epochs <= 0:
                yield from iter_wikipedia_bytes(args.data)
                return
            from tmt.data import epoch_lines
            for ep in range(args.epochs):
                for line in epoch_lines(args.data, ep, args.seed):
                    b = line.encode("utf-8", errors="ignore")
                    if b:
                        yield b
        for chunk in _chunks():
            if args.epochs > 0:
                model.reset()
            for i in range(len(chunk) - 1):
                loss, _, _ = model.training_step(chunk[i], chunk[i + 1], i == len(chunk) - 2)
                n += 1
                lv = float(loss.item())
                comp = dict(getattr(model, "last_components", {}))
                if min_loss is None or lv < min_loss:
                    min_loss = lv
                lf.write(f"{n},{lv},{comp.get('l_var', 0.0)},{comp.get('l_pred', 0.0)},"
                         f"{comp.get('l_ce', 0.0)},{comp.get('l_stop', 0.0)},"
                         f"{comp.get('state_norm', 0.0)}\n")
                if not check_finite({"loss": lv, **comp}):
                    save_checkpoint(model, "runs/diverged.safetensors")
                    failure = "diverged"
                    print(f"DIVERGED at step {n}; emergency checkpoint saved.",
                          flush=True)
                    return
                if n % 500 == 0:
                    lf.flush()
                    save_checkpoint(model, args.ckpt)
                    el = time.time() - t0
                    Path("runs/state.json").write_text(json.dumps({
                        "step": n, "loss": lv, "components": comp,
                        "bytes_per_sec": round(n / el, 1),
                        "elapsed_s": round(el, 1)}))
                if args.eval_every > 0 and args.eval_val and n % args.eval_every == 0:
                    from contextlib import nullcontext
                    ema_ctx = model.using_ema() if model.ema_state is not None else nullcontext()
                    with ema_ctx:
                        r = evaluate_model(model, eval_bytes, [eval_bytes],
                                           cfg=EVAL_PRESET,
                                           config_dict=cfg.to_dict(), gpu_hours=0.0)
                    with open(eval_path, "a") as ef:
                        ef.write(f"{n},{r.val_bpb:.4f},{r.long_range_score:.4f},"
                                 f"{r.continual_score:.4f},{r.stability_score:.4f},"
                                 f"{r.composite_score:.3f}\n")
                    print(f"[eval] step={n} bpb={r.val_bpb:.3f} "
                          f"composite={r.composite_score:.3f}", flush=True)
                if n >= args.steps:
                    return
    finally:
        save_checkpoint(model, args.ckpt)
        lf.close()
        node = node_json(cfg.to_dict(), {"min_loss": min_loss if min_loss is not None else 0.0}, 0.0, failure, 0.0,
                         run_id=run_id, timestamp=started_at, git_commit=commit,
                         steps=n, bytes_seen=n)
        Path("runs/node.json").parent.mkdir(parents=True, exist_ok=True)
        Path("runs/node.json").write_text(json.dumps(node, indent=2))
        if failure:
            sys.exit(1)

if __name__ == "__main__":
    main()
