# Long low-LR run — window hypothesis fails fair test

> Source: tmt-torch run runs/copylowlong (200k steps, lr 3e-5, frac 1.0) + evals (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Best-LR (3e-5) copy run, 200000 steps, exit 0: payload CE 7.036
(similar to its 10k reading 7.282 — glacial learning), ingesting probe
0.0 at copy4 x after8/32, held-out evals flat ~7.0–7.4 throughout.

Reading: the literature narrow window does not appear even given a
fair long run at the best LR. LR is exhausted as a recall lever at
this scale: 11 values, 10k–200k steps, all zero. Remaining: protocol
with teeth, major scale, structural memory — or accepting the small
online regime cannot do exact recall.
