# Tidepool Architecture and Evidence

> Sources: tmt-torch session records, 2026-09-18/19
> Raw: [2026-09-18-pytorch-port-record](../../raw/tmt-torch/2026-09-18-pytorch-port-record.md); [2026-09-18-rsi-loop-record](../../raw/tmt-torch/2026-09-18-rsi-loop-record.md); [2026-09-18-vk4a-training](../../raw/tmt-torch/2026-09-18-vk4a-training.md); [2026-09-18-monitoring](../../raw/tmt-torch/2026-09-18-monitoring.md); [2026-09-18-knob-round](../../raw/tmt-torch/2026-09-18-knob-round.md); [2026-09-18-long-run](../../raw/tmt-torch/2026-09-18-long-run.md); [2026-09-18-replay-comparison](../../raw/tmt-torch/2026-09-18-replay-comparison.md); [2026-09-18-priority-comparison](../../raw/tmt-torch/2026-09-18-priority-comparison.md); [2026-09-18-noise-control](../../raw/tmt-torch/2026-09-18-noise-control.md); [2026-09-18-ema-comparison](../../raw/tmt-torch/2026-09-18-ema-comparison.md); [2026-09-18-coverage-confound](../../raw/tmt-torch/2026-09-18-coverage-confound.md); [2026-09-18-rtrl-200k](../../raw/tmt-torch/2026-09-18-rtrl-200k.md); [2026-09-18-memory-zero](../../raw/tmt-torch/2026-09-18-memory-zero.md); [2026-09-18-pure-copy](../../raw/tmt-torch/2026-09-18-pure-copy.md); [2026-09-18-copy-accum](../../raw/tmt-torch/2026-09-18-copy-accum.md); [2026-09-18-reset-completion](../../raw/tmt-torch/2026-09-18-reset-completion.md); [2026-09-18-remediation-record](../../raw/tmt-torch/2026-09-18-remediation-record.md); [2026-09-19-consultant-review](../../raw/tmt-torch/2026-09-19-consultant-review.md); [2026-09-19-gru-contrast](../../raw/tmt-torch/2026-09-19-gru-contrast.md); [2026-09-19-scale-copy](../../raw/tmt-torch/2026-09-19-scale-copy.md); [2026-09-19-rank-flat](../../raw/tmt-torch/2026-09-19-rank-flat.md); [2026-09-19-decay-audit](../../raw/tmt-torch/2026-09-19-decay-audit.md); [2026-09-19-transformer-anchor](../../raw/tmt-torch/2026-09-19-transformer-anchor.md); [2026-09-19-single-episode](../../raw/tmt-torch/2026-09-19-scheduled-sampling.md); [2026-09-19-decode-ceiling](../../raw/tmt-torch/2026-09-19-decode-ceiling.md); [2026-09-19-hybrid-fails](../../raw/tmt-torch/2026-09-19-hybrid-fails.md); [2026-09-19-lr-sweep](../../raw/tmt-torch/2026-09-19-lr-sweep.md); [2026-09-19-lr-fine-sweep](../../raw/tmt-torch/2026-09-19-lr-fine-sweep.md); [2026-09-19-low-lr-long](../../raw/tmt-torch/2026-09-19-low-lr-long.md); [2026-09-19-rank-analysis](../../raw/tmt-torch/2026-09-19-rank-analysis.md); [2026-09-19-linear-probe](../../raw/tmt-torch/2026-09-19-linear-probe.md); [2026-09-19-exact-cache](../../raw/tmt-torch/2026-09-19-exact-cache.md); [2026-09-19-ingate-copy](../../raw/tmt-torch/2026-09-19-ingate-copy.md)
> Updated: 2026-09-19

## Overview

Tidepool is a PyTorch port of a byte-level recurrent LM plus a
Dream-RSI outer loop and a full experiment harness. The port's training
math is proven against the reference equations (RTRL gradients match
finite differences to 4.3e-9 at 5 steps, all 9 parameter classes plus
gate extensions); training is stable through millions of steps; every
cure tried for forgetting and recall has been measured and nearly all
failed honestly. Long-range retention is unproven: exact-match probes
read 0.0 everywhere, though rank/linear-probe diagnostics show a weak
diffuse trace (payload rank ~50 vs 128 chance, 7–10x chance decoding).

## Architecture

Byte embedding (256 symbols) → stack of RTU layers (per-dim sigmoid
decay, LayerNorm, square linear, SiLU residual, persistent state) →
byte-logit + stop heads, with optional selective decay/input gates,
top-k write masks, top-k readout sparsity, EMA shadows, replay buffers,
and sidecar slot stores. Learned online one byte at a time with exact
RTRL trace corrections (embedding add, decay/in-gate recurrent adds),
true gradient accumulation, scheduled sampling, per-line reset epoch
training, copy-task episodes (triple-null marker, tiny curriculum),
and auxiliary recall/retrieval heads. Checkpoints are two-file with
resume; the RSI loop searches configs through the same scorer.
Baselines: GRU, tiny transformer, unigram/uniform oracles. Monitoring:
per-component loss CSV, heartbeat, NaN watchdog, periodic held-out
eval, curves plot.

## Tests (110 Green)

Equivalence 2; RTRL FD proof 9 (staged 1/2/5, selective, write-mask);
accumulation 5; checkpoints 6; memory oracles + ingesting + assoc +
exact-cache + hybrid (1 + 1 + 2 + 4 + 4); replay 8; recall/EMA/train
utils 5; RSI loop/policy/replay/sandbox/rewriter/tree 22;
model/config/data/fixes/gate/wiring/CLI/GRU/transformer 24; slots 11.

## Results

Prediction works (held-out bpb 5.6–7.6 sustained over 6000000 steps;
train CE to 0.07734909653663635; payload CE below uniform at 1M copy
steps). Retention does not: exact probes 0.0 at all distances on all
models including dim256, GRU, and transformer baselines. Cures
measured: lower LR delays (not cures; 11 values exhausted);
uniform/priority/random replay hurt or tie; EMA ties then loses;
accumulation helps payload CE only; per-line reset fixed stale-state
poisoning (state_norm 15k → 70); selective + input gates + sparsity
move prediction, never recall; scheduled sampling perturbs attractor
collapse without curing; decoding constraints reveal fragments, never
exact bytes. Trace diagnostics: rank stationary ~50, linear decode
7–10x chance, per-position 2–4x chance flat, timescales intact
(~3–1000 steps). RTRL ≈ single-step at tiny scale.

## External Review (Adopted)

An external consultant confirmed the machinery and named the central
failure: retention zero everywhere — prediction without memory.
Adopted prescription stands: retention-only gate, hard inductive-bias
changes one at a time, RSI loop retargeted at retention, scale only
after capability exists. Current RTU treated as local feature
extractor until retention moves.

## See Also

- [TMT PyTorch Port](pytorch-port.md)
- [Dream-RSI Loop](rsi-loop.md)
- [Vulkan4Aros Training Data](vk4a-training.md)
