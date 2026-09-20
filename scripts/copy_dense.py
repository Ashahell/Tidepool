# scripts/copy_dense.py — copy-dense run with query-segment instrumentation.
"""Pure-copy training logging query-segment CE + periodic exact-recall probes.

Every episode is fresh (generalization, not memorization). Logs per step:
query CE (marker[-1] -> payload[0] and trailing payload transitions only)
vs other CE. Probes exact free-generation recall every --probe-every steps.
Usage: copy_dense.py --steps 100000 --dim 64 --layers 3 --out runs/copydense
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import time
sys.path.insert(0, "src")
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import copy_episode, QUERY_MARKER
from tmt.engine import save_checkpoint


def probe_recall(model, rng, trials=20, plen=4, flen=8):
    hits = 0
    model.eval()
    with torch.no_grad():
        for _ in range(trials):
            ep = copy_episode(rng, (plen,), (flen,))
            pre, post = bytes(ep).split(bytes(QUERY_MARKER))
            model.reset()
            for b in pre + bytes(QUERY_MARKER):
                model.ingest(b)
            cur = QUERY_MARKER[-1]
            out = bytearray()
            for _ in range(len(post)):
                logits, _ = model(torch.tensor([cur], dtype=torch.long))
                cur = int(logits[0].argmax())
                out.append(cur)
            hits += bytes(out) == post
    model.train()
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/copydense")
    ap.add_argument("--probe-every", type=int, default=10000)
    ap.add_argument("--fw-dk", type=int, default=0)
    ap.add_argument("--bptt", action="store_true",
                    help="episode BPTT: defer backward to episode end")
    ap.add_argument("--aux-fw-w", type=float, default=0.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 1)
    cfg = TMTConfig(dim=args.dim, layers=args.layers, lr=args.lr,
                    fw_dk=args.fw_dk, aux_fw_w=args.aux_fw_w)
    model = TMTModel(cfg)
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "dense.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "q_ce", "other_ce", "recall"])
    n = 0
    t0 = time.time()
    while n < args.steps:
        ep = copy_episode(rng)
        pre, post = bytes(ep).split(bytes(QUERY_MARKER))
        qpos = set(range(len(pre), len(ep) - 1))
        model.reset()
        total = None
        for i in range(len(ep) - 1):
            loss, _, _ = model.training_step(ep[i], ep[i + 1], i == len(ep) - 2,
                                             defer=args.bptt)
            if args.bptt:
                total = loss if total is None else total + loss
            n += 1
            key = "q" if i in qpos else "o"
            w.writerow([n, f"{float(loss.item()):.4f}" if key == "q" else "",
                        "" if key == "q" else f"{float(loss.item()):.4f}", ""])
            if n >= args.steps:
                break
        if args.bptt and total is not None:
            model.finish_episode(total)
        if n % args.probe_every < len(ep) or n >= args.steps:
            r = probe_recall(model, random.Random(args.seed + 2))
            el = time.time() - t0
            print(f"[probe] step={n} recall={r}/20 elapsed={el:.0f}s", flush=True)
            w.writerow([n, "", "", r])
            log.flush()
    save_checkpoint(model, str(out / "model.safetensors"),
                    meta={"step": n, "seed": args.seed})
    log.close()
    print(f"done step={n} -> {args.out}")


if __name__ == "__main__":
    main()
