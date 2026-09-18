# scripts/train.py
from __future__ import annotations
import argparse
import json
import sys
sys.path.insert(0, "src")
from pathlib import Path
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import iter_wikipedia_bytes
from tmt.engine import save_checkpoint, load_checkpoint, node_json

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    ap.add_argument("--data", default="wikipedia_clean")
    ap.add_argument("--seed", type=int, default=42)
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
    load_checkpoint(model, args.ckpt)
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    loss_path = Path("runs/loss.csv")
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not loss_path.exists() or loss_path.stat().st_size == 0
    lf = open(loss_path, "a")
    if need_header:
        lf.write("step,loss\n")
    n = 0
    min_loss = None
    try:
        for chunk in iter_wikipedia_bytes(args.data):
            for i in range(len(chunk) - 1):
                loss, _, _ = model.training_step(chunk[i], chunk[i + 1], i == len(chunk) - 2)
                n += 1
                lv = float(loss.item())
                if min_loss is None or lv < min_loss:
                    min_loss = lv
                lf.write(f"{n},{lv}\n")
                if n % 500 == 0:
                    lf.flush()
                    save_checkpoint(model, args.ckpt)
                if n >= args.steps:
                    return
    finally:
        save_checkpoint(model, args.ckpt)
        lf.close()
        node = node_json(cfg.to_dict(), {"min_loss": min_loss if min_loss is not None else 0.0}, 0.0, None, 0.0)
        Path("runs/node.json").parent.mkdir(parents=True, exist_ok=True)
        Path("runs/node.json").write_text(json.dumps(node, indent=2))

if __name__ == "__main__":
    main()
