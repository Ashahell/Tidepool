# Tidepool Architecture and Evidence

> Sources: tmt-torch session records, 2026-09-18/19
> Raw: [2026-09-18-pytorch-port-record](../../raw/tmt-torch/2026-09-18-pytorch-port-record.md); [2026-09-18-rsi-loop-record](../../raw/tmt-torch/2026-09-18-rsi-loop-record.md); [2026-09-18-vk4a-training](../../raw/tmt-torch/2026-09-18-vk4a-training.md); [2026-09-18-monitoring](../../raw/tmt-torch/2026-09-18-monitoring.md); [2026-09-18-knob-round](../../raw/tmt-torch/2026-09-18-knob-round.md); [2026-09-18-long-run](../../raw/tmt-torch/2026-09-18-long-run.md); [2026-09-18-replay-comparison](../../raw/tmt-torch/2026-09-18-replay-comparison.md); [2026-09-18-priority-comparison](../../raw/tmt-torch/2026-09-18-priority-comparison.md); [2026-09-18-noise-control](../../raw/tmt-torch/2026-09-18-noise-control.md); [2026-09-18-ema-comparison](../../raw/tmt-torch/2026-09-18-ema-comparison.md); [2026-09-18-coverage-confound](../../raw/tmt-torch/2026-09-18-coverage-confound.md); [2026-09-18-rtrl-200k](../../raw/tmt-torch/2026-09-18-rtrl-200k.md); [2026-09-18-memory-zero](../../raw/tmt-torch/2026-09-18-memory-zero.md); [2026-09-18-pure-copy](../../raw/tmt-torch/2026-09-18-pure-copy.md); [2026-09-18-copy-accum](../../raw/tmt-torch/2026-09-18-copy-accum.md); [2026-09-18-reset-completion](../../raw/tmt-torch/2026-09-18-reset-completion.md); [2026-09-18-remediation-record](../../raw/tmt-torch/2026-09-18-remediation-record.md); [2026-09-19-consultant-review](../../raw/tmt-torch/2026-09-19-consultant-review.md); [2026-09-19-gru-contrast](../../raw/tmt-torch/2026-09-19-gru-contrast.md); [2026-09-19-scale-copy](../../raw/tmt-torch/2026-09-19-scale-copy.md); [2026-09-19-rank-flat](../../raw/tmt-torch/2026-09-19-rank-flat.md); [2026-09-19-decay-audit](../../raw/tmt-torch/2026-09-19-decay-audit.md); [2026-09-19-transformer-anchor](../../raw/tmt-torch/2026-09-19-transformer-anchor.md); [2026-09-19-single-episode](../../raw/tmt-torch/2026-09-19-scheduled-sampling.md); [2026-09-19-decode-ceiling](../../raw/tmt-torch/2026-09-19-decode-ceiling.md); [2026-09-19-hybrid-fails](../../raw/tmt-torch/2026-09-19-hybrid-fails.md); [2026-09-19-lr-sweep](../../raw/tmt-torch/2026-09-19-lr-sweep.md); [2026-09-19-lr-fine-sweep](../../raw/tmt-torch/2026-09-19-lr-fine-sweep.md); [2026-09-19-low-lr-long](../../raw/tmt-torch/2026-09-19-low-lr-long.md); [2026-09-19-rank-analysis](../../raw/tmt-torch/2026-09-19-rank-analysis.md); [2026-09-19-linear-probe](../../raw/tmt-torch/2026-09-19-linear-probe.md); [2026-09-19-exact-cache](../../raw/tmt-torch/2026-09-19-exact-cache.md); [2026-09-19-ingate-copy](../../raw/tmt-torch/2026-09-19-ingate-copy.md); [2026-09-19-review-3](../../raw/tmt-torch/2026-09-19-review-3.md)
> Updated: 2026-09-19

## Executive State

Tidepool is a PyTorch CUDA implementation of a byte-level recurrent
language model with exact RTRL gradient corrections, persistent
recurrent state, a Dream-RSI experiment loop, and a reproducible
experiment harness.

The implementation has passed numerical and engineering validation:
RTRL gradients match finite differences to approximately 4.3e-9 at five
timesteps across the tested parameter classes and gate extensions;
gradient accumulation has dedicated equivalence tests;
checkpoint/resume has dedicated tests; model, data, training, baseline,
memory, and RSI components have automated coverage (110 tests);
training remains numerically stable over multi-million-step runs.

The central unresolved result is persistent memory. The model learns
prediction, but exact long-range recall remains 0.0 on the current
memory probes — also true for the tested GRU and transformer baselines.
Linear probes recover weak payload information (rank ~50, 7–10x chance
decoding), so the state carries signal it cannot causally use.

Therefore: state persistence demonstrated; information survival weakly
demonstrated; information decoding weakly demonstrated; causal memory
use unproven; exact task recall failed. Tidepool is currently a
recurrent feature extractor with persistent state, not a demonstrated
long-term memory system. The primary objective is determining whether
the recurrent state can be made causally useful — not improving the
language model.

## Evidence Rule

Every architectural claim needs a hypothesis, a controlled experiment,
a measurable pass/fail criterion, a baseline, and a reproducible
record. A diagnostic signal is not a capability. A correlation is not
causation. A successful training run is not evidence of memory. A
finite-difference match validates an implementation, not an
architecture.

## Research Hierarchy

correctness → gradient correctness → state persistence → information
survival → causal state influence → exact recall → language-model
benefit → automated search. Failure at a lower level blocks
optimization of higher levels.

## Know vs Think

Measured: RTRL FD error ≈ 4.3e-9; training stable; state holds
recoverable payload information; state persists. Not demonstrated:
state as useful memory. Hypotheses: attractor causes recall failure;
better gates fix recall; RTRL should beat single-step; RSI can
discover memory mechanisms.

## Architecture

Core model: byte embedding → RTU layers (persistent state, per-dim
sigmoid decay, LayerNorm, square linear, SiLU residual, exact RTRL
traces) → byte logits + stop head. Everything else lives as isolated
experimental variants, never in the canonical path: selective
decay/input gates, write masks, readout sparsity, EMA, replay buffers,
slot stores, scheduled sampling, auxiliary heads, copy curricula. No
variant joins the canonical architecture without a reproducible
retention improvement.

Canonical baselines for every memory experiment: uniform, unigram,
persistent TMT, reset-every-step TMT, GRU, tiny transformer; RTRL vs
single-step variants where practical.

## Memory Benchmark (Primary Instrument)

Write payload → intervening distractors → query. Measure exact byte
accuracy, bit accuracy, sequence accuracy, linear/nonlinear probe
accuracy, and causal intervention effect, across a distance sweep.
Causal protocol: compare normal state vs zeroed vs shuffled vs
replaced memory at the query point — memory must change output, or it
is information without use.

## Retention Gate

An architectural change counts as a retention improvement only if it
improves the retention metric against the unchanged control under
comparable training. Better bpb, norms, rank, or probe accuracy alone
do not qualify.

## Search Gate

RSI searches broadly for retention only after a manual intervention
shows nonzero causal/exact recall. Until then it may optimize
engineering/diagnostic tasks, never as evidence the memory works.

## Research Stop Condition

If exact recall stays zero with ~zero causal effect while RTRL ≈
single-step, stop modifying the RTU: freeze the architecture and
analyze the state update equations instead.

## Rejected Hypotheses

Lower LR enables retention → delays failure. Rejected. Replay improves
retention → no. Rejected. EMA stabilizes memory → no. Rejected. Input
gates enable recall → no. Rejected. Sparsity enables recall → no.
Rejected. Scheduled sampling enables recall → no. Rejected. Slots
(v1/v2) enable recall → no. Rejected. Exact cache enables recall → no,
wrong shape. Scale (5x) enables recall → no. Rejected. LR window exists
→ no, 11 values exhausted. Rejected.

## Tests (110 Green)

Equivalence 2; RTRL FD 9+; accumulation, checkpoints, oracles,
ingesting/assoc/exact-cache/hybrid probes, replay, recall, EMA, train
utils, RSI loop/policy/replay/sandbox/rewriter/tree, model/config/data,
GRU/transformer baselines. Full per-file inventory via pytest
--collect-only.

## Results (Headlines)

Prediction: bpb 5.6–7.6 sustained; payload CE below uniform at 1M copy
steps. Retention: exact 0.0 everywhere; rank ~50 stationary; linear
decode 7–10x chance; per-position 2–4x flat; timescales intact. No
variant moves recall: replay ×3, EMA, accumulation, selective, input
gate, sparsity, scheduled sampling, slots ×2, exact/soft stores, scale,
LR ×11, time to 1M.

## See Also

- [TMT PyTorch Port](pytorch-port.md)
- [Dream-RSI Loop](rsi-loop.md)
- [Vulkan4Aros Training Data](vk4a-training.md)
