# Long selective run — CE improves, recall absent

> Source: tmt-torch run runs/sel150k (150k steps, frac 1.0, selective) + evals (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Selective model, 150000 pure-copy steps: payload CE 6.276 (vs 7.614 at
30k), ingesting probe 0.0 at copy4 x after8/32. Plain dim128 at 150k
read 6.494 — selective edge persists but narrows relatively. Held-out
evals flat ~7.0 throughout.

Reading: selectivity improves payload prediction with time (6.3 and
falling) without producing any exact recall. The mechanism helps the
thing that was already working (prediction), not the thing that is
broken (recall). Recall remains at zero across every mechanism, scale,
and budget tried.
