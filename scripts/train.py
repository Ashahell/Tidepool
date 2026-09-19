# scripts/train.py
from __future__ import annotations
import argparse
import json
import random
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

def ss_pick(rng, prob: float, prev, curr: int) -> int:
    """Scheduled-sampling input choice: own prediction with prob, else truth."""
    if prev is not None and prob > 0.0 and rng.random() < prob:
        return int(prev)
    return int(curr)


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
    ap.add_argument("--copy-frac", type=float, default=0.0,
                    help="share of lines replaced by copy-task episodes (epoch mode)")
    ap.add_argument("--copy-tiny", action="store_true",
                    help="tiny copy curriculum (payload 2-4, filler 0-4)")
    ap.add_argument("--ss-prob", type=float, default=0.0,
                    help="scheduled sampling: share of steps feeding the model's own prediction")
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
    from tmt.recall import RecallHead
    recall = RecallHead(cfg.dim, hidden=cfg.recall_hidden)
    recall_path = Path(args.ckpt).parent / "recall.pt"
    if recall_path.exists():
        recall.load_state_dict(torch.load(recall_path, weights_only=True))
    meta = load_checkpoint(model, args.ckpt, allow_missing=True) or {}
    resume_from = int(meta.get("bytes_seen", 0)) if args.resume else 0
    run_id, started_at, commit = new_run_id(), utc_timestamp(), git_commit_short()
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.ckpt).parent
    loss_path = run_dir / "loss.csv"
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not loss_path.exists() or loss_path.stat().st_size == 0
    lf = open(loss_path, "a")
    if need_header:
        lf.write("step,loss,l_var,l_pred,l_ce,l_stop,state_norm\n")
    eval_path = run_dir / "eval.csv"
    if args.eval_every > 0 and args.eval_val:
        eval_bytes = load_val_bytes(args.eval_val)
        if not eval_path.exists() or eval_path.stat().st_size == 0:
            eval_path.write_text("step,bpb,mem,cont,stab,composite\n")
    n = resume_from
    min_loss = None
    failure = None
    t0 = time.time()
    srng = random.Random(args.seed + 777)
    prev_sampled = None
    try:
        def _chunks():
            if args.resume and resume_from > 0 and args.epochs <= 0:
                buf = bytearray()
                for b in skip_bytes(args.data, resume_from):
                    buf += b
                    if b == b"\n":
                        yield bytes(buf), False
                        buf = bytearray()
                if buf:
                    yield bytes(buf), False
                return
            if args.epochs <= 0:
                for c in iter_wikipedia_bytes(args.data):
                    yield c, False
                return
            from tmt.data import (copy_episode, epoch_lines, TINY_PAYLOAD_LENS,
                                  TINY_FILLER_LENS)
            erng = random.Random(args.seed + 999)
            for ep in range(args.epochs):
                for line in epoch_lines(args.data, ep, args.seed):
                    if erng.random() < args.copy_frac:
                        if args.copy_tiny:
                            yield copy_episode(erng, TINY_PAYLOAD_LENS,
                                               TINY_FILLER_LENS), True
                        else:
                            yield copy_episode(erng), True
                    else:
                        b = line.encode("utf-8", errors="ignore")
                        if b:
                            yield b, False
        for chunk, is_episode in _chunks():
            if args.epochs > 0 or is_episode:
                model.reset()
                prev_sampled = None
            for i in range(len(chunk) - 1):
                curr = ss_pick(srng, args.ss_prob, prev_sampled, chunk[i])
                loss, sampled, _ = model.training_step(curr, chunk[i + 1], i == len(chunk) - 2)
                prev_sampled = sampled
                n += 1
                lv = float(loss.item())
                comp = dict(getattr(model, "last_components", {}))
                if min_loss is None or lv < min_loss:
                    min_loss = lv
                lf.write(f"{n},{lv},{comp.get('l_var', 0.0)},{comp.get('l_pred', 0.0)},"
                         f"{comp.get('l_ce', 0.0)},{comp.get('l_stop', 0.0)},"
                         f"{comp.get('state_norm', 0.0)}\n")
                if not check_finite({"loss": lv, **comp}):
                    save_checkpoint(model, str(run_dir / "diverged"),
                                    meta={"step": n, "bytes_seen": n, "cursor_bytes": n})
                    failure = "diverged"
                    print(f"DIVERGED at step {n}; emergency checkpoint saved.",
                          flush=True)
                    return
                if n % 500 == 0:
                    lf.flush()
                    save_checkpoint(model, args.ckpt,
                                    meta={"step": n, "bytes_seen": n, "cursor_bytes": n})
                    torch.save(recall.state_dict(), run_dir / "recall.pt")
                    el = time.time() - t0
                    Path(run_dir / "state.json").write_text(json.dumps({
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
            if is_episode:
                # Recall distillation: re-feed pre+marker, train the
                # separate head (trunk frozen) on the payload segment.
                from tmt.data import QUERY_MARKER
                pre, post = bytes(chunk).split(QUERY_MARKER)
                model.reset()
                with torch.no_grad():
                    for b in pre + QUERY_MARKER:
                        model.ingest(b)
                    st = model.layers[-1].states.detach().clone()
                for j, t in enumerate(post):
                    if j > 0:
                        with torch.no_grad():
                            model.ingest(post[j - 1])
                            st = model.layers[-1].states.detach().clone()
                    recall.train_step(st.detach(), int(t))
            if is_episode and model.slots is not None and cfg.aux_mem_w > 0.0:
                from tmt.data import QUERY_MARKER
                pre, post = bytes(chunk).split(QUERY_MARKER)
                model.reset()
                with torch.no_grad():
                    for b in pre + QUERY_MARKER:
                        model.ingest(b)
                    _, _ = model(torch.tensor([QUERY_MARKER[-1]], dtype=torch.long))
                    h_state = model.layers[-1].states.detach().clone()
                read_vec = model.slots.read(h_state)
                tgt = model.encoder(torch.tensor([post[0]], dtype=torch.long)).detach()
                aux = model.slots.retrieval_loss(read_vec, tgt)
                (cfg.aux_mem_w * aux).backward()
            if n >= args.steps:
                return
    finally:
        save_checkpoint(model, args.ckpt,
                        meta={"step": n, "bytes_seen": n, "cursor_bytes": n})
        torch.save(recall.state_dict(), run_dir / "recall.pt")
        lf.close()
        node = node_json(cfg.to_dict(), {"min_loss": min_loss if min_loss is not None else 0.0}, 0.0, failure, 0.0,
                         run_id=run_id, timestamp=started_at, git_commit=commit,
                         steps=n, bytes_seen=n)
        Path(run_dir / "node.json").parent.mkdir(parents=True, exist_ok=True)
        Path(run_dir / "node.json").write_text(json.dumps(node, indent=2))
        if failure:
            sys.exit(1)

if __name__ == "__main__":
    main()
