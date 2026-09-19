# Copy-task training, first attempt — no signal yet

> Source: tmt-torch run runs/copyexp (20k steps, frac 0.2) + payload-CE diagnostic (inline output)
> Collected: 2026-09-18
> Published: 2026-09-18

## Measured 2026-09-18

copy_episode() landed (payload 4/8/16, filler 8/32/128, triple-null
marker, trailing payload target); train.py --copy-frac mixes episodes
with per-episode reset; suite 72 green. 20k steps dim128/lr1e-4 frac
0.2: ingesting copy probe 0.0 at copy4/8 x after8/32. Diagnostic: mean
first-payload-byte CE 7.382 vs uniform 5.545 — the model learned nothing
of the marker contingency, not even marker recognition.

Reading: inconclusive on capability. A few hundred episodes cannot teach
a recall mapping diluted 5:1 by normal bytes. Next: longer copy training
(100k+ steps) and/or higher frac before any claim about whether the
architecture can retain when paid to.
