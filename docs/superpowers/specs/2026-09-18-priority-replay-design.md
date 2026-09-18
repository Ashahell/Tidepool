# Prioritized Replay — Design

**Date:** 2026-09-18
**Status:** Sections 1–2 approved, awaiting spec review
**Motivation:** uniform recency replay lost to baseline at all 12
checkpoints (12.119 vs 8.616); it re-trains learned bytes. Sample by
forgetting signal instead.

## 1. Fields + sampling

- `replay_alpha: float = 1.0` — priority exponent; 0 reproduces uniform.
- Entries become `(curr, next, end, loss)`; append stores the live loss,
  replay overwrites the sampled entry's tag after its update (free —
  the loss is already computed).
- Weights `w = loss^alpha`, normalized; all-equal/zero tags fall back to
  uniform (no div-zero). Draws via `torch.multinomial`, without
  replacement when k < buffer length.
- Default 1.0 is safe: no existing test pins exact replay trajectories.

## 2. Integration + tests

- No `_update` signature change (it already returns the detached loss).
- `training_step` shape unchanged: live update, append, k sampled
  updates with tag write-back. Monitoring stays live-only.
- Tests: weighted skew (seeded draw counts favor high-loss entries);
  refresh (tag changes after replay); alpha=0 uniform parity; full suite
  green.
- The priority-vs-off bpb comparison is a follow-up experiment, not a
  gate.

## Out of scope

Reservoir sampling, EWC, separate accumulation, replay-aware monitoring.
