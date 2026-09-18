# Reset run extension (160k steps) — forgetting still absent

> Source: tmt-torch run runs/epoch_reset (train.log, live; gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Measured 2026-09-18

Per-line-reset run continued past the interim checkpoint (162500 steps
and alive): evals at 80000 bpb=7.601 composite=-86.399, 100000
bpb=6.315 composite=-69.473, 120000 bpb=8.175 composite=-93.445,
140000 bpb=6.786 composite=-75.728, 160000 bpb=6.367
composite=-70.020. Trajectory oscillates 6.3–8.2 with no monotonic
decay; the no-reset run occupied 11–18 across the same span and kept
climbing. Forgetting has not reappeared through 160k steps. Full
2-epoch curve still pending.
