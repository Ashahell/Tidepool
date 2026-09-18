# scripts/train.py
from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import iter_wikipedia_bytes
from tmt.engine import save_checkpoint

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    model.init_decay_groups()
    n = 0
    try:
        for chunk in iter_wikipedia_bytes():
            for i in range(len(chunk) - 1):
                model.training_step(chunk[i], chunk[i + 1], i == len(chunk) - 2)
                n += 1
                if n % 500 == 0:
                    save_checkpoint(model, args.ckpt)
                if n >= args.steps:
                    save_checkpoint(model, args.ckpt)
                    return
    finally:
        save_checkpoint(model, args.ckpt)

if __name__ == "__main__":
    main()
