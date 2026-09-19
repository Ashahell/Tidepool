# RTRL re-measurement matrix (Task 5) — 2026-09-18

8 cells: {rtrl, single-step} x {persistent, reset} x {groups-4, groups-1}.
Tiny models (TMTModel dim=32, layers=2, lr=1e-4, update_every=1), 2000 steps on
`data/vk4a_train` bytes (fixed slice: concatenated dir bytes, first 2001),
seed 42, smoke eval on `/tmp/rsi_data/val_heldout.bin` (frozen
`evaluation_suite.py`, copy targets 8/64). Single-step cell = test-only
monkeypatch `model._rtrl_enabled = False` (Task 1 gate, default True).
Source: `runs/remeasure.jsonl` (8 lines). Runtime ~5.6 min total (~42 s/cell).
No outcome gate: RTRL ≈ single-step is a finding, not a failure.

## Per-cell numbers (transcribed from jsonl)

| cell | loss first→last | bpb init→final | copy mean (8/64 keys) | continual | stability |
|---|---|---|---|---|---|
| rtrl/persistent/groups-4 | 7.269→4.986 | 8.324→7.454 | 0.0 (all four 0.0) | 0.231 | 0.958 |
| rtrl/persistent/groups-1 | 7.269→5.282 | 8.324→7.470 | 0.0 | 0.230 | 0.992 |
| rtrl/reset/groups-4 | 7.269→5.017 | 8.324→7.474 | 0.0 | 0.227 | 0.992 |
| rtrl/reset/groups-1 | 7.269→5.017 | 8.324→7.474 | 0.0 | 0.227 | 0.992 |
| single-step/persistent/groups-4 | 7.269→5.003 | 8.324→7.446 | 0.0 | 0.232 | 0.958 |
| single-step/persistent/groups-1 | 7.269→5.348 | 8.324→7.477 | 0.0 | 0.229 | 0.992 |
| single-step/reset/groups-4 | 7.269→5.017 | 8.324→7.474 | 0.0 | 0.227 | 0.992 |
| single-step/reset/groups-1 | 7.269→5.017 | 8.324→7.474 | 0.0 | 0.227 | 0.992 |

No failures (`failure_mode` None everywhere). Composite ≈ −84.3 everywhere
(bpb-dominated; not discriminative at this scale).

## Questions A–F, one line each

- (A) Learning — bpb improves: YES, weakly. 8.324→~7.45 in every cell
  (loss 7.27→~5.0). Note init bpb 8.32 is *worse* than uniform (8.0):
  random-init head starts below chance, crosses it during the 2000 steps.
- (B) State — persistent beats reset: BARELY. Persistent bpb edge is
  ~0.02 (7.454 vs 7.474, groups-4); loss edge ~0.03. Direction right,
  magnitude near noise — 2000 steps is too short for state to matter much.
- (C) Memory — copy at 8/64: FLOOR. 0.0 on all four keys in all 8 cells.
  A dim-32 model after 2000 steps copies nothing; the probe is uninformative
  at this scale, not evidence of no memory.
- (D) Timescales — groups differ: YES, but only when state persists.
  Persistent loss 4.986 (groups-4) vs 5.282 (groups-1); under reset the two
  group settings are *exactly* identical (see below).
- (E) Continual — retention nonzero: YES, ~0.23 in all cells. Weak but
  nonzero and uniform; single-task probe, so this is a floor not a curve.
- (F) Ablation — RTRL vs single-step: ≈ IDENTICAL. Persistent/groups-4 bpb
  7.454 vs 7.446 (single-step marginally better, within noise); reset cells
  bit-identical. At 2000 steps on tiny models the correction changes nothing
  measurable — a finding, not a failure.

## Exact identities (predicted, not bugs)

All four reset cells are bit-identical (loss 5.0173, bpb 7.4740). This is
*derived*, not suspicious: `reset()` zeroes states AND traces each step, so
the RTRL correction terms (`old_embed * old_decay`, `old_decay * decaytrace`)
are exactly zero — RTRL ≡ single-step under reset-every-byte by construction.
Likewise groups cannot matter under reset: with zero state,
`state = decay*0 + enc = enc` (decay cancels), and the decay_bias direct
gradient term scales with the zeroed prior state — so decay never affects
forward, loss, or updates. The identity confirms the plumbing; it carries
no information about (D) or (F).

## Open questions (stayed open)

1. Copy floor: does any config achieve nonzero copy at 8/64, or is a bigger
   model/longer run needed first? Unanswered — all cells 0.0.
2. RTRL separation: needs longer horizons and/or larger models before the
   correction can separate from single-step; 2000 tiny steps cannot resolve it.
3. State advantage (B) is directionally right but tiny; longer runs needed.
4. Init bpb worse-than-uniform (8.32 > 8.0): worth a glance in a later task,
   but out of scope here — recorded, not investigated.
