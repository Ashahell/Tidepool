# Accumulated updates on pure copy — recall still zero

> Source: tmt-torch run runs/copyue32 (30k steps, frac 1.0, update_every 32) + payload-CE + probe eval (inline output)
> Collected: 2026-09-18
> Published: 2026-09-18

## Measured 2026-09-18

Same pure-copy conditions with update_every 32 (32x gradient
accumulation, fewer noisier steps): payload CE 5.846 (vs 7.923 at
update_every 1; uniform 5.545), ingesting probe 0.0 at copy4 x
after8/32. Run's own held-out bpb ~8.2 (pure-copy stream vs prose
eval, expected worse).

Reading: the noise hypothesis is weak — 32x accumulation buys 2 points
of payload CE but zero recall. The remaining levers are capacity
(scale up) and protocol (different recall signaling), not update
statistics.
