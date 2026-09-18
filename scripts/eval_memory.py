# scripts/eval_memory.py
from __future__ import annotations
import argparse
import json
import sys
sys.path.insert(0, "src")
from pathlib import Path
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import load_val_bytes
from tmt.engine import load_checkpoint, node_json
from tmt.evaluation_suite import EvalConfig, evaluate_model, save_eval_result

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    ap.add_argument("--val", default="val.bin")
    ap.add_argument("--ema", action="store_true",
                    help="evaluate averaged weights (needs ema in ckpt)")
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    load_checkpoint(model, args.ckpt)
    use_ema = False
    if args.ema:
        if model.ema_state is None:
            print("warning: no ema in checkpoint; evaluating fast weights")
        else:
            use_ema = True
    val = load_val_bytes(args.val)
    eval_cfg = EvalConfig(max_bytes=2000, lm_eval_tokens=512, copy_lengths=[4], intervening_lengths=[8, 64], num_copy_trials=2)
    from contextlib import nullcontext
    ema_ctx = model.using_ema() if use_ema else nullcontext()
    with ema_ctx:
        r = evaluate_model(model, val, [val], cfg=eval_cfg, config_dict=cfg.to_dict(), gpu_hours=0.0)
    print(f"bpb={r.val_bpb:.3f} mem={r.long_range_score:.3f} cont={r.continual_score:.3f} stab={r.stability_score:.3f} composite={r.composite_score:.3f} fail={r.failure_mode}")
    node = node_json(cfg.to_dict(), r.raw_metrics, 0.0, r.failure_mode, r.composite_score)
    Path("runs/node.json").parent.mkdir(parents=True, exist_ok=True)
    Path("runs/node.json").write_text(json.dumps(node, indent=2))
    save_eval_result(r, Path("runs/eval_result.json"))

if __name__ == "__main__":
    main()
