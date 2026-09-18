# scripts/chat.py
from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.engine import load_checkpoint, save_checkpoint, gen_bytes

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    ap.add_argument("--readonly", action="store_true")
    ap.add_argument("--notrace", action="store_true")
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    load_checkpoint(model, args.ckpt)
    try:
        while True:
            text = input("User >> ") + "\n"
            data = text.encode("utf-8", errors="ignore")
            for i in range(len(data) - 1):
                if args.notrace:
                    with torch.no_grad():
                        logits, _ = model(torch.tensor([data[i]], dtype=torch.long))
                        b = int(torch.argmax(logits[0]).item())
                else:
                    _, b, _ = model.training_step(data[i], data[i + 1], False) if not args.readonly else (None, int(torch.argmax(model(torch.tensor([data[i]], dtype=torch.long))[0][0]).item()), 0.0)
            b = data[-1]
            print("Model >> ", end="", flush=True)
            sys.stdout.buffer.write(gen_bytes(model, b))
            sys.stdout.flush()
            print()
    finally:
        if not args.readonly:
            save_checkpoint(model, args.ckpt)

if __name__ == "__main__":
    main()
