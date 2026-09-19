# Recall head — fits, doesn't generalize

> Source: tmt-torch run runs/recall30k + recall-head evals (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Separate Linear recall head (own Adam), trained on episode query
segments with trunk frozen, 30k steps frac 1.0 dim128. Held-out
generation with the head: copy2 0/20, copy4 0/20. Fixed-set check (8
payloads x 200 reps, same ckpt trunk): 8/8 exact.

Reading: the head fits (state separates payloads; capacity suffices)
but does not generalize to fresh payloads — possibly memorizing
filler signatures rather than extracting payload content. The
representation holds instance traces; the recall mapping does not
transfer. Next: nonlinear readout, far more varied recall training, or
accepting that linear recall of novel content is the wall.
