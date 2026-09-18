# src/tmt/engine.py
from __future__ import annotations
from typing import Any, Dict, Optional
import math
import torch
from safetensors.torch import save_file, load_file

def check_finite(vals: dict) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in vals.values())

def save_checkpoint(model, path: str) -> None:
    state = {f"m.{k}": v.cpu().clone() for k, v in model.state_dict().items()}
    for i, layer in enumerate(model.layers):
        state[f"trace_state.{i}"] = layer.states.cpu().clone()
        state[f"trace_decay.{i}"] = layer.decaytrace.cpu().clone()
        state[f"trace_embed.{i}"] = layer.embedtrace.cpu().clone()
    if getattr(model, "ema_state", None):
        for k, v in model.ema_state.items():
            state[f"ema.{k}"] = v.cpu().clone()
    save_file(state, path)

def load_checkpoint(model, path: str) -> None:
    from pathlib import Path as _P
    if not _P(path).exists():
        return
    data = load_file(path)
    sd = {k[2:]: v for k, v in data.items() if k.startswith("m.")}
    model.load_state_dict(sd, strict=False)
    for i, layer in enumerate(model.layers):
        if f"trace_state.{i}" in data:
            layer.states.copy_(data[f"trace_state.{i}"])
            layer.decaytrace.copy_(data[f"trace_decay.{i}"])
            layer.embedtrace.copy_(data[f"trace_embed.{i}"])
    ema = {k[4:]: v for k, v in data.items() if k.startswith("ema.")}
    if ema:
        names = {n for n, _ in model.named_parameters()}
        model.ema_state = {k: v.clone() for k, v in ema.items() if k in names}

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

def node_json(config: dict, metrics: dict, gpu_hours: float, failure: Optional[str], composite: float = 0.0) -> Dict[str, Any]:
    return {
        "config": config,
        "final_metrics": metrics,
        "composite_score": composite,
        "failure_mode": failure,
        "cost": {"gpu_hours": gpu_hours},
    }
