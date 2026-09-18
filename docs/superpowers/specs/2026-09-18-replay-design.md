# In-Model Experience Replay — Design

**Date:** 2026-09-18
**Status:** Sections 1–2 approved, awaiting spec review
**Sequencing:** buffer now, EWC later (Fisher machinery deferred)

## Goal

Give TMT a searchable continual-learning lever against slow forgetting:
an in-model byte replay buffer with mixed extra updates, exposed as
TMTConfig fields so the RSI loop can search it.

## 1. Config + buffer semantics

- `replay_size: int = 0` — deque maxlen; 0 disables replay entirely
  (exact current path, zero overhead).
- `replay_k: int = 1` — extra sampled updates per training step.
- Buffer holds `(curr, next, end)` tuples of recent steps (recency deque;
  reservoir only if recency proves insufficient).
- Uniform sampling via torch RNG (global seed covers determinism).

## 2. Integration + verification

- `training_step` body becomes `_update(curr, next, end)` with identical
  math, returning the live loss.
- `training_step` = live `_update` + buffer append + `replay_k` sampled
  `_update`s (skipped while the buffer is empty).
- Shared grad accumulation and optimizer stepping (no new optimizer
  semantics); returned loss and `last_components` describe the live
  update only.
- Tests: appends land correctly; seeded on/off runs diverge after N
  steps; off path identical to today (existing suite is the gate).
- A replay-on/off bpb comparison run is a follow-up experiment, not a
  gate for this change.

## Out of scope

EWC / Fisher penalties; reservoir or prioritized sampling; separate
replay accumulation; replay-aware monitoring (live-only stays).
