# src/tmt/engine.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import math
import random
import secrets
import subprocess
import torch
from safetensors.torch import save_file, load_file

def check_finite(vals: dict) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in vals.values())

def _strip_suffix(path: str) -> str:
    if path.endswith(".state.pt"):
        return path[: -len(".state.pt")]
    for suf in (".safetensors", ".pt"):
        if path.endswith(suf):
            return path[: -len(suf)]
    return path

def git_commit_short() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"

def new_run_id() -> str:
    return secrets.token_hex(4)

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def save_checkpoint(model, path: str, meta: Optional[dict] = None) -> None:
    base = _strip_suffix(path)
    state = {f"m.{k}": v.cpu().clone() for k, v in model.state_dict().items()}
    for i, layer in enumerate(model.layers):
        state[f"trace_state.{i}"] = layer.states.cpu().clone()
        state[f"trace_decay.{i}"] = layer.decaytrace.cpu().clone()
        state[f"trace_embed.{i}"] = layer.embedtrace.cpu().clone()
    if getattr(model, "ema_state", None):
        for k, v in model.ema_state.items():
            state[f"ema.{k}"] = v.cpu().clone()
    save_file(state, base + ".safetensors")
    m = dict(meta) if meta else {}
    m.setdefault("step", 0)
    m.setdefault("bytes_seen", 0)
    m.setdefault("cursor_bytes", m["bytes_seen"])
    try:
        import numpy as np
        numpy_rng = np.random.get_state()
    except ImportError:
        numpy_rng = None
    payload = {
        "optimizer": model.opt.state_dict(),
        "accum": int(getattr(model, "_accum", 0)),
        "meta": m,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "numpy_rng": numpy_rng,
        "config": model.cfg.to_dict(),
        "git_commit": git_commit_short(),
    }
    torch.save(payload, base + ".state.pt")

def _restore_rng(payload: dict) -> None:
    try:
        torch.set_rng_state(payload["torch_rng"])
    except Exception as e:
        raise ValueError(f"bad torch RNG state: {e}")
    if payload.get("cuda_rng") is not None:
        if torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
            except Exception as e:
                raise ValueError(f"bad CUDA RNG state: {e}")
    try:
        random.setstate(payload["python_rng"])
    except Exception as e:
        raise ValueError(f"bad Python RNG state: {e}")
    if payload.get("numpy_rng") is not None:
        try:
            import numpy as np
            np.random.set_state(payload["numpy_rng"])
        except Exception as e:
            raise ValueError(f"bad NumPy RNG state: {e}")

def load_checkpoint(model, path: str, allow_missing: bool = False) -> dict:
    from pathlib import Path as _P
    base = _strip_suffix(path)
    weights = _P(base + ".safetensors")
    if not weights.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"checkpoint not found: {weights}")
    data = load_file(str(weights))
    sd = {k[2:]: v for k, v in data.items() if k.startswith("m.")}
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        # Compat: checkpoints predating selective gates lack gate keys;
        # gate defaults are mathematical zeros, so loading them as such
        # is exact, not silent. Anything else still raises.
        missing = set(model.state_dict()) - set(sd)
        unexpected = set(sd) - set(model.state_dict())
        if missing and not unexpected and all("gate" in k for k in missing):
            print(f"warning: backfilling {len(missing)} gate keys with zeros")
            model.load_state_dict(sd, strict=False)
        else:
            raise
    for i, layer in enumerate(model.layers):
        if f"trace_state.{i}" in data:
            layer.states.copy_(data[f"trace_state.{i}"])
            layer.decaytrace.copy_(data[f"trace_decay.{i}"])
            layer.embedtrace.copy_(data[f"trace_embed.{i}"])
    ema = {k[4:]: v for k, v in data.items() if k.startswith("ema.")}
    if ema:
        names = {n for n, _ in model.named_parameters()}
        model.ema_state = {k: v.clone() for k, v in ema.items() if k in names}
    state_path = _P(base + ".state.pt")
    if not state_path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"checkpoint state not found: {state_path}")
    payload = torch.load(str(state_path), map_location="cpu", weights_only=False)
    try:
        model.opt.load_state_dict(payload["optimizer"])
    except ValueError:
        # Old optimizer state (e.g. predating gate params): keep the fresh
        # optimizer rather than fail the whole load. Warn loudly.
        print("warning: optimizer state incompatible; using fresh optimizer")
    model._accum = int(payload.get("accum", 0))
    _restore_rng(payload)
    meta = payload.get("meta", {})
    return dict(meta) if isinstance(meta, dict) else {}

def gen_bytes(model, seed_byte: int, max_bytes: int = 256, stop_threshold: float = 0.35) -> bytes:
    out = bytearray()
    b = int(seed_byte) % 256
    with torch.no_grad():
        for _ in range(max_bytes):
            # Same computation as model.forward, but keeps the decoder
            # input so the stop head can be read for termination.
            x = torch.tensor([b], dtype=torch.long)
            enc = model.encoder(x)
            h = enc
            for layer in model.layers:
                h, _, _ = layer(enc, h)
            logits, stop = model.decoder(h)
            nxt = int(torch.argmax(logits[0]).item())
            out.append(nxt & 0xFF)
            if nxt == 10:
                break
            if float(stop.item()) > stop_threshold:
                break
            b = nxt
    return bytes(out)

def node_json(config: dict, metrics: dict, gpu_hours: float, failure: Optional[str], composite: float = 0.0, *,
              run_id: Optional[str] = None, timestamp: Optional[str] = None,
              git_commit: Optional[str] = None, steps: int = 0,
              bytes_seen: int = 0) -> Dict[str, Any]:
    return {
        "config": config,
        "final_metrics": metrics,
        "composite_score": composite,
        "failure_mode": failure,
        "cost": {"gpu_hours": gpu_hours},
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "steps": steps,
        "bytes_seen": bytes_seen,
    }
