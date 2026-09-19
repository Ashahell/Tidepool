# Tidepool Architecture and Evidence

> Sources: tmt-torch session records, 2026-09-18
> Raw: [2026-09-18-pytorch-port-record](../../raw/tmt-torch/2026-09-18-pytorch-port-record.md); [2026-09-18-rsi-loop-record](../../raw/tmt-torch/2026-09-18-rsi-loop-record.md); [2026-09-18-vk4a-training](../../raw/tmt-torch/2026-09-18-vk4a-training.md); [2026-09-18-monitoring](../../raw/tmt-torch/2026-09-18-monitoring.md); [2026-09-18-long-run](../../raw/tmt-torch/2026-09-18-long-run.md); [2026-09-18-replay-comparison](../../raw/tmt-torch/2026-09-18-replay-comparison.md); [2026-09-18-priority-comparison](../../raw/tmt-torch/2026-09-18-priority-comparison.md); [2026-09-18-noise-control](../../raw/tmt-torch/2026-09-18-noise-control.md); [2026-09-18-ema-comparison](../../raw/tmt-torch/2026-09-18-ema-comparison.md); [2026-09-18-coverage-confound](../../raw/tmt-torch/2026-09-18-coverage-confound.md); [2026-09-18-rtrl-200k](../../raw/tmt-torch/2026-09-18-rtrl-200k.md); [2026-09-18-memory-zero](../../raw/tmt-torch/2026-09-18-memory-zero.md); [2026-09-18-pure-copy](../../raw/tmt-torch/2026-09-18-pure-copy.md); [2026-09-18-copy-accum](../../raw/tmt-torch/2026-09-18-copy-accum.md); [2026-09-18-reset-completion](../../raw/tmt-torch/2026-09-18-reset-completion.md); [2026-09-18-remediation-record](../../raw/tmt-torch/2026-09-18-remediation-record.md)
> Updated: 2026-09-18

## Overview

Tidepool is a PyTorch port of a byte-level recurrent LM plus a
Dream-RSI outer loop and a full experiment harness. The port's training
math is proven against the reference equations (RTRL gradients match
finite differences to 4.3e-9 at 5 steps); training is stable through
millions of steps; every cure tried for forgetting has been measured
and most failed honestly. Long-range retention is unproven: memory
probes read 0.0 everywhere, including on the best model.

## Architecture

Byte embedding (256 symbols) → stack of RTU layers (per-dim sigmoid
decay, LayerNorm, square linear, SiLU residual, persistent state) →
byte-logit + stop heads. Learned online one byte at a time with exact
RTRL trace corrections (embedding add, decay recurrent-add), true
gradient accumulation, EMA shadow weights, and an optional recency /
loss-weighted / noise replay buffer. Checkpoints are two-file
(safetensors weights+traces, torch.save optimizer/RNG/cursor) with
resume. Train/val splits are disjoint; epoch mode shuffles lines and
resets state per line; copy-task episodes (triple-null marker,
configurable frac) train recall. The RSI loop (tree, exact-match
replay, sandboxed policy rewriting, never-regress selection) searches
configs through the same scorer. Monitoring: per-component loss CSV,
heartbeat, NaN watchdog, periodic held-out eval, curves plot.

## Tests (72 Green)

Equivalence 2 (forward/states/traces to 1e-5, loss 1e-3); RTRL FD proof
3 (1/2/5-step, all 9 parameter classes); accumulation 5 (explicit-sum
equality, reset isolation, half-life invariant, temp, config
validation); checkpoints 6 (weight-equal resume, cursor, loud failures,
bad bytes); memory oracles 1 + ingesting 1 (Perfect 1.0 > OneByte 0.0 =
Amnesiac 0.0); replay 8; RSI loop/policy/replay/sandbox/rewriter/tree
22; model/config/data/fixes/gate/wiring/ema/CLI 24.

## Results

Prediction works (held-out bpb 5.6–7.6 sustained over 6000000 steps with
reset+RTRL; train CE to 0.07734909653663635). Retention does not: copy probes 0.0 at
all distances on all models. Cures measured: lower LR delays (not
cures); uniform replay ends 3.5 bpb worse (12.119 vs 8.616); priority
replay 8.4 worse (17.012 vs 8.616); random-replay control splits the
difference (9.442 — content matters); EMA ties then loses (20.398 vs
18.115 at 200k); accumulated updates move payload CE 7.923 → 5.846 with
zero recall; pure-copy training cannot teach recall at dim128.
Per-line reset fixed the acute stale-state poisoning (state_norm 15k →
70). RTRL ≈ single-step at tiny scale; reset cells bit-identical by
derivation.

## See Also

- [TMT PyTorch Port](pytorch-port.md)
- [Dream-RSI Loop](rsi-loop.md)
- [Vulkan4Aros Training Data](vk4a-training.md)
