# src/tmt/engine.py
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from safetensors.torch import save_file, load_file

def save_checkpoint(model, path: str) -> None:
    state = {f"m.{k}": v.cpu().clone() for k, v in model.state_dict().items()}
    for i, layer in enumerate(model.layers):
        state[f"trace_state.{i}"] = layer.states.cpu().clone()
        state[f"trace_decay.{i}"] = layer.decaytrace.cpu().clone()
        state[f"trace_embed.{i}"] = layer.embedtrace.cpu().clone()
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

def node_json(config: dict, metrics: dict, gpu_hours: float, failure: Optional[str], composite: float = 0.0) -> Dict[str, Any]:
    return {
        "config": config,
        "final_metrics": metrics,
        "composite_score": composite,
        "failure_mode": failure,
        "cost": {"gpu_hours": gpu_hours},
    }
