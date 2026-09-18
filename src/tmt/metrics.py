# src/tmt/metrics.py
"""Thin re-export of the evaluation-suite public interface."""
from __future__ import annotations
from .evaluation_suite import (
    EvalConfig,
    EvalResult,
    ProbeResult,
    evaluate_model,
    train_and_evaluate,
    save_eval_result,
    load_eval_result,
)

__all__ = [
    "EvalConfig",
    "EvalResult",
    "ProbeResult",
    "evaluate_model",
    "train_and_evaluate",
    "save_eval_result",
    "load_eval_result",
]
