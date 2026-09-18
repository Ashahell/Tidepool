# Per-line reset — interim confirmation

> Source: tmt-torch run runs/epoch_reset (train.log, live; gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Measured 2026-09-18

train.py resets recurrent state per line in epoch mode (2-line change,
committed bb85be6). Reset run, same shape as the stopped no-reset run
(dim128/lr1e4, vk4a_train, held-out val): evals at 20000 bpb=6.261,
40000 bpb=5.367, 60000 bpb=6.262 — flat, best sustained numbers ever
recorded (previous best single point 6.3535). No-reset run at the same
steps: 7.028, 12.694, 17.419. Throughput unchanged (~625 B/s).

The stopped no-reset run's full curve (recovered from its logs): bpb to
25.637 at 200k and 34.073 at 240k, then silent process death at 256500
steps with finite loss 2.11 (harness reaping, not divergence — no
traceback, memory fine). Its death is a gap: no evals past 240k.

Reading: stale state across unrelated lines was the poison, confirmed
by intervention. Reset is now standard in epoch mode. Full 2-epoch reset
curve pending; it decides whether forgetting is truly gone or merely
slower.
