# Fine LR sweep (10 values) — no window with recall

> Source: tmt-torch scripts/lr_sweep.py output runs/lr_sweep.jsonl (10k steps each, dim128, frac 1.0)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Payload CE by LR (probe 0.0 at all ten): 3e-5: 7.2816, 5e-5: 7.7849,
7e-5: 8.0575, 1e-4: 8.1906, 1.5e-4: 8.3867, 2e-4: 8.2990, 3e-4:
8.1569, 5e-4: 8.3463, 7e-4: 8.3571, 1e-3: 9.1566. Best CE at the
lowest LR; monotonic worsening upward; recall zero everywhere.

Reading: no narrow LR window with recall at 10k steps (Okpekpe &
Orvieto's window does not appear at this horizon/scale). Caveat: 10k
steps may be too few for any LR — their evidence comes from far longer
runs. Lowest LR best on CE, consistent with everything: slower is
safer, nothing recalls.
