# tests/mlx_reference.py
"""Pure-NumPy transcription of the upstream MLX model equations.

Source-level equivalence reference for the torch port (Dream-RSI RTRL work).

Upstream: https://github.com/jrz97619761/test-model-thing/blob/main/main.py
Fetched: 2026-09-18.

Transcribes Model.step / Model.__call__ (forward + loss + RTRL persistent
trace update) in float64 NumPy. No autograd: only the persistent buffer motion
(states / embedtrace / decaytrace) and the loss scalars are reproduced.
"""
from __future__ import annotations

import numpy as np

LN_EPS = 1e-5
VAR_EPS = 1e-4


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _layernorm(x, w, b, eps=LN_EPS):
    mu = np.mean(x)
    var = np.mean((x - mu) ** 2)  # population variance, matches MLX/torch
    return (x - mu) / np.sqrt(var + eps) * w + b


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def init_params(seed=11, dim=4, layers=2):
    """Random float64 params: embed (256,dim); per-layer decay/weight/ln_w/ln_b;
    dec_w (256,dim), dec_b (256,), stop_w (dim,1), stop_b (1,)."""
    rng = np.random.default_rng(seed)
    P = {"embed": rng.standard_normal((256, dim))}
    P["layers"] = [{
        "decay": rng.standard_normal((dim,)),
        "weight": rng.standard_normal((dim, dim)),
        "ln_w": rng.standard_normal((dim,)),
        "ln_b": rng.standard_normal((dim,)),
    } for _ in range(layers)]
    P["dec_w"] = rng.standard_normal((256, dim))
    P["dec_b"] = rng.standard_normal((256,))
    P["stop_w"] = rng.standard_normal((dim, 1))
    P["stop_b"] = rng.standard_normal((1,))
    return P


def mlx_step(P, traces, curr, next_, end):
    """One transcribed upstream step.

    traces is None (=> zero states/traces) or a dict with per-layer lists
    under "states", "embedtrace", "decaytrace".
    Returns (loss, logits, states, new_traces).
    """
    layers = P["layers"]
    n_layers = len(layers)
    if traces is None:
        dim = P["embed"].shape[1]
        old_states = [np.zeros((dim,)) for _ in range(n_layers)]
        old_embedtrace = [np.zeros((256, dim)) for _ in range(n_layers)]
        old_decaytrace = [np.zeros((dim,)) for _ in range(n_layers)]
    else:
        old_states = traces["states"]
        old_embedtrace = traces["embedtrace"]
        old_decaytrace = traces["decaytrace"]

    enc = P["embed"][curr]
    x = enc.copy()
    states, decays = [], []
    for li, L in enumerate(layers):
        decay = _sigmoid(L["decay"])
        state = decay * old_states[li] + enc  # dummy is zero
        x = x + _silu(_layernorm(state, L["ln_w"], L["ln_b"]) @ L["weight"].T)
        states.append(state)
        decays.append(decay)

    logits = x @ P["dec_w"].T + P["dec_b"]
    stop = _sigmoid(x @ P["stop_w"] + P["stop_b"])

    l_var = max(0.0, 1.0 - np.sqrt(np.mean((x - np.mean(x)) ** 2) + VAR_EPS))
    loss = l_var
    if next_ is not None:
        tgt = P["embed"][next_]
        loss = loss + float(np.mean((x - tgt) ** 2))
        loss = loss + float(-logits[next_] + _logsumexp(logits))
        target_stop = 1.0 if end else 0.0
        loss = loss + float(np.mean((stop - target_stop) ** 2))

    new_states, new_embedtrace, new_decaytrace = [], [], []
    one_hot = np.zeros_like(old_embedtrace[0])
    one_hot[curr] += 1.0
    for li in range(n_layers):
        d = decays[li]
        new_states.append(states[li].copy())
        new_embedtrace.append(old_embedtrace[li] * d + one_hot)
        new_decaytrace.append(d * old_decaytrace[li] + d * (1.0 - d) * old_states[li])
    new_traces = {
        "states": new_states,
        "embedtrace": new_embedtrace,
        "decaytrace": new_decaytrace,
    }
    return float(loss), logits.copy(), [s.copy() for s in states], new_traces
