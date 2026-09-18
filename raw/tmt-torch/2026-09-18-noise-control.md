# Random-replay control (60k steps) — content matters

> Source: tmt-torch run runs/ncmp_noise (train.log, gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Run measured 2026-09-18

Same fair pair plus a third arm: replay_noise samples uniform random
bytes for the k extra updates (same update budget as buffer replay).
60k steps, eval every 5000, exit 0. bpb: 6.840, 6.926, 6.934, 7.889,
7.931, 8.259, 8.283, 9.564, 7.133, 9.727, 7.917, 9.442.

## Four-way ordering at 60k steps

Off 8.616 < random 9.442 < uniform-recency 12.119 < priority 17.012.

Reading: extra updates alone cost ~0.8 bpb (random vs off), but content
adds far more on top (recency +2.7, priority +7.6). The pure
effective-LR story is dead; the recency-bias and dig-into-mistakes
mechanisms survive the control. Replay content matters, and both tried
contents point the wrong way.
