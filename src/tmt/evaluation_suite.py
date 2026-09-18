"""
TMT Evaluation Suite (PyTorch port)
===================================
Standardized, reproducible metrics for architecture & continual-learning search.
Used as the inner-loop scorer inside Dream-RSI discovery trees.

Ported from the MLX reference version. Model interface contract:

    model.reset() -> None
    logits, state = model(byte_tensor)   # byte_tensor: LongTensor shape (1,)
        logits: FloatTensor shape (1, 256)
        state:  sequence of Tensors/None (for norm tracking), or None

Generation uses the same single-byte call. Online training hooks are
injected (see train_and_evaluate) so the suite stays model-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------
# Configuration & Result Containers
# ---------------------------------------------------------------


@dataclass
class EvalConfig:
    # Core
    max_bytes: int = 50_000
    batch_size: int = 1
    seed: int = 42

    # Language modeling
    lm_eval_tokens: int = 20_000

    # Memory probes
    copy_lengths: List[int] = field(default_factory=lambda: [16, 32, 64])
    intervening_lengths: List[int] = field(
        default_factory=lambda: [128, 512, 2048, 8192, 20000]
    )
    num_copy_trials: int = 20

    # Continual learning
    continual_tasks: int = 4
    bytes_per_task: int = 8_000
    retention_eval_bytes: int = 2_000

    # Stability
    state_norm_threshold: float = 50.0
    garbage_threshold: float = 0.15

    # Scoring weights (tune these)
    weight_bpb: float = -12.0
    weight_long_range: float = 6.0
    weight_continual: float = 9.0
    weight_stability: float = 4.0
    weight_efficiency: float = -0.8


@dataclass
class ProbeResult:
    name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    config_hash: str
    val_bpb: float
    long_range_score: float
    continual_score: float
    stability_score: float
    efficiency_score: float
    composite_score: float
    wall_time: float
    probes: List[ProbeResult] = field(default_factory=list)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    failure_mode: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_valid_utf8(byte_seq: bytes) -> bool:
    try:
        byte_seq.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def bytes_to_tensor(data: bytes) -> torch.LongTensor:
    return torch.tensor(list(data), dtype=torch.long)


@torch.no_grad()
def compute_bpb(model, data: bytes, context_reset: bool = True) -> float:
    """Bits-per-byte on a byte sequence (lower is better)."""
    if context_reset:
        model.reset()

    total_nll = 0.0
    total_bytes = 0
    x = bytes_to_tensor(data)

    for i in range(len(x) - 1):
        logits, _ = model(x[i : i + 1])
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs[0, int(x[i + 1].item())]
        total_nll += float(nll.item())
        total_bytes += 1

    return total_nll / (total_bytes * math.log(2)) if total_bytes > 0 else 99.0


# ---------------------------------------------------------------
# Individual Probes
# ---------------------------------------------------------------


def language_modeling_probe(model, val_data: bytes, cfg: EvalConfig) -> ProbeResult:
    bpb = compute_bpb(model, val_data[: cfg.lm_eval_tokens])
    return ProbeResult(
        name="language_modeling",
        score=bpb,
        details={"bpb": bpb, "tokens": cfg.lm_eval_tokens},
    )


@torch.no_grad()
def copy_memory_probe(model, cfg: EvalConfig) -> ProbeResult:
    """Classic copy task at multiple distances."""
    results = {}
    successes = []

    for copy_len in cfg.copy_lengths:
        for inter_len in cfg.intervening_lengths:
            key = f"copy{copy_len}_after_{inter_len}"
            correct = 0

            for _ in range(cfg.num_copy_trials):
                model.reset()
                target = bytes(np.random.randint(32, 127, size=copy_len).tolist())
                filler = bytes(np.random.randint(0, 256, size=inter_len).tolist())

                seq = target + filler
                for b in seq:
                    model(torch.tensor([b], dtype=torch.long))

                generated = []
                for _ in range(copy_len):
                    # NOTE: replace the dummy-0 input with the model's real
                    # autoregressive step once the torch Model lands.
                    logits, _ = model(torch.tensor([0], dtype=torch.long))
                    pred = int(torch.argmax(logits[0]).item())
                    generated.append(pred)
                    model(torch.tensor([pred], dtype=torch.long))

                if bytes(generated) == target:
                    correct += 1

            acc = correct / cfg.num_copy_trials
            results[key] = acc
            successes.append(acc)

    mean_acc = float(np.mean(successes)) if successes else 0.0
    return ProbeResult(name="copy_memory", score=mean_acc, details=results)


def continual_learning_probe(
    model,
    task_datas: List[bytes],
    cfg: EvalConfig,
    online_update: Optional[Callable[[bytes], None]] = None,
) -> ProbeResult:
    """Sequential training on several tasks, measure retention."""
    retention_scores = []
    model.reset()

    for task_idx, data in enumerate(task_datas):
        if online_update is not None:
            for i in range(0, len(data) - 1, 64):
                online_update(data[i : i + 64])

        task_retentions = []
        for prev_idx in range(task_idx + 1):
            prev_data = task_datas[prev_idx][: cfg.retention_eval_bytes]
            bpb = compute_bpb(model, prev_data, context_reset=True)
            score = max(0.0, 1.0 - (bpb / 10.0))
            task_retentions.append(score)

        retention_scores.append(float(np.mean(task_retentions)))

    final_continual = float(np.mean(retention_scores)) if retention_scores else 0.0
    return ProbeResult(
        name="continual_learning",
        score=final_continual,
        details={"per_task_retention": retention_scores},
    )


@torch.no_grad()
def stability_probe(model, data: bytes, cfg: EvalConfig) -> ProbeResult:
    """Detect state explosion, decay collapse, and garbage output."""
    model.reset()
    state_norms = []
    garbage_count = 0
    total = 0

    for i, b in enumerate(data[: cfg.max_bytes]):
        logits, state = model(torch.tensor([b], dtype=torch.long))
        if state is not None:
            norm = sum(
                float(torch.linalg.norm(s).item())
                for s in state
                if s is not None and torch.is_tensor(s)
            )
            state_norms.append(norm)

        if i % 200 == 0:
            for _ in range(32):
                pred = int(torch.argmax(logits[0]).item())
                total += 1
                if pred < 32 or pred > 126:
                    garbage_count += 1
                logits, _ = model(torch.tensor([pred], dtype=torch.long))

    avg_norm = float(np.mean(state_norms[-100:])) if state_norms else 0.0
    garbage_rate = garbage_count / max(1, total)

    norm_penalty = (
        1.0
        if avg_norm < cfg.state_norm_threshold
        else max(0.0, 1.0 - (avg_norm / 200))
    )
    garbage_penalty = max(0.0, 1.0 - garbage_rate / cfg.garbage_threshold)
    stability = 0.6 * norm_penalty + 0.4 * garbage_penalty

    failure = None
    if avg_norm > cfg.state_norm_threshold * 2:
        failure = "state_explosion"
    elif garbage_rate > 0.4:
        failure = "high_garbage"

    return ProbeResult(
        name="stability",
        score=stability,
        details={
            "avg_state_norm": avg_norm,
            "garbage_rate": garbage_rate,
            "failure_mode": failure,
        },
    )


# ---------------------------------------------------------------
# Main Evaluation Entry Point
# ---------------------------------------------------------------


def evaluate_model(
    model,
    val_data: bytes,
    continual_datas: List[bytes],
    cfg: EvalConfig = EvalConfig(),
    config_dict: Optional[Dict] = None,
    gpu_hours: float = 0.0,
    online_update: Optional[Callable[[bytes], None]] = None,
) -> EvalResult:
    """Full evaluation of a TMT model instance.

    Returns a structured EvalResult ready for a Discovery Tree node.
    """
    set_seed(cfg.seed)
    start_time = time.time()
    probes = []

    lm_probe = language_modeling_probe(model, val_data, cfg)
    probes.append(lm_probe)

    mem_probe = copy_memory_probe(model, cfg)
    probes.append(mem_probe)

    cont_probe = continual_learning_probe(
        model, continual_datas, cfg, online_update=online_update
    )
    probes.append(cont_probe)

    stab_probe = stability_probe(model, val_data, cfg)
    probes.append(stab_probe)
    failure_mode = stab_probe.details.get("failure_mode")

    efficiency = max(0.0, 1.0 - gpu_hours)

    val_bpb = lm_probe.score
    long_range = mem_probe.score
    continual = cont_probe.score
    stability = stab_probe.score

    composite = (
        cfg.weight_bpb * val_bpb
        + cfg.weight_long_range * long_range
        + cfg.weight_continual * continual
        + cfg.weight_stability * stability
        + cfg.weight_efficiency * efficiency
    )

    wall_time = time.time() - start_time
    config_hash = hashlib.sha1(
        json.dumps(config_dict or {}, sort_keys=True).encode()
    ).hexdigest()[:12]

    return EvalResult(
        config_hash=config_hash,
        val_bpb=val_bpb,
        long_range_score=long_range,
        continual_score=continual,
        stability_score=stability,
        efficiency_score=efficiency,
        composite_score=composite,
        wall_time=wall_time,
        probes=probes,
        raw_metrics={
            "val_bpb": val_bpb,
            "long_range": long_range,
            "continual": continual,
            "stability": stability,
            "gpu_hours": gpu_hours,
        },
        failure_mode=failure_mode,
    )


def train_and_evaluate(
    config: dict,
    val_data: bytes,
    continual_datas: List[bytes],
    build_fn: Callable[[dict], Any],
    train_fn: Callable[[Any, dict], float],
    eval_cfg: Optional[EvalConfig] = None,
) -> dict:
    """Dream-RSI inner-loop scorer: build -> train -> evaluate.

    build_fn(config) -> model; train_fn(model, config) -> gpu_hours.
    Returns everything needed for the Discovery Tree node.
    """
    model = build_fn(config)
    gpu_hours = train_fn(model, config)
    result = evaluate_model(
        model=model,
        val_data=val_data,
        continual_datas=continual_datas,
        cfg=eval_cfg or EvalConfig(),
        config_dict=config,
        gpu_hours=gpu_hours,
    )
    return {
        "config": config,
        "final_metrics": result.raw_metrics,
        "composite_score": result.composite_score,
        "failure_mode": result.failure_mode,
        "eval_result": result.to_dict(),
        "cost": {"gpu_hours": gpu_hours},
    }


# ---------------------------------------------------------------
# Convenience: Save / Load results
# ---------------------------------------------------------------


def save_eval_result(result: EvalResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_eval_result(path: Path) -> EvalResult:
    with open(path) as f:
        data = json.load(f)
    data["probes"] = [ProbeResult(**p) for p in data.get("probes", [])]
    return EvalResult(**data)
