# scripts/eval_memory.py
from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import load_val_bytes
from tmt.engine import load_checkpoint
from tmt.evaluation_suite import EvalConfig, evaluate_model

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    ap.add_argument("--val", default="val.bin")
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    load_checkpoint(model, args.ckpt)
    val = load_val_bytes(args.val)
    eval_cfg = EvalConfig(max_bytes=2000, lm_eval_tokens=512, copy_lengths=[4], intervening_lengths=[8, 64], num_copy_trials=2)
    r = evaluate_model(model, val, [val], cfg=eval_cfg, config_dict=cfg.to_dict(), gpu_hours=0.0)
    print(f"bpb={r.val_bpb:.3f} mem={r.long_range_score:.3f} cont={r.continual_score:.3f} stab={r.stability_score:.3f} composite={r.composite_score:.3f} fail={r.failure_mode}")

if __name__ == "__main__":
    main()
