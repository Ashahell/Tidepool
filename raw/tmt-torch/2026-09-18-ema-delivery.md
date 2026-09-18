# EMA weight averaging — delivery record

> Source: tmt-torch commits d0f01ed + 3e65fe4 (suite output at commit time)
> Collected: 2026-09-18
> Published: 2026-09-18

## Delivered 2026-09-18

TMTConfig gains ema_decay (default 0.0, off). TMTModel tracks ema_state
(param-name shadow dict) on every optimizer step: init-on-first-step,
then ema = decay*ema + (1-decay)*param under no_grad. Context manager
using_ema() swaps averaged params in (parameters only; recurrent state
and trace buffers untouched) and restores after; no-op when ema_state is
None. Checkpoints persist ema_* keys alongside params/traces and restore
them into matching param names. train.py periodic eval uses averaged
weights whenever ema_state exists; eval_memory.py gains --ema (warns and
evaluates fast weights if the checkpoint has no ema). Suite: 50 passed
at commit time (2 ema tests + ema checkpoint roundtrip).
