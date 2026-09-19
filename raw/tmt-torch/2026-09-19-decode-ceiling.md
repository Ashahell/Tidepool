# Decoding constraints plateau

> Source: tmt-torch inline constrained-generation evals on /tmp/ss300.pt (output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Same SS-300 single-episode model, presence penalty sweep + sampling:
pen 0 fragments (oo@~ repeats), pen 2 longer fragments, pen 5
near-miss (o@~Mx~sc#[?&4O.o vs want o@~Mx~sc#[?s&4.O), pen 10 and 20
identical to each other with exotic bytes (b4 97) intruding, sampled
decoding 0/10 exact.

Reading: constraints monotonically reveal more payload, then saturate
into exotic-byte forcing — the true bytes never top the distribution.
Decoding tweaks are exhausted as a path to exact recall; the
distribution itself lacks the peak. Left: training that sharpens the
peak (contrastive, longer), scale, structural memory.
