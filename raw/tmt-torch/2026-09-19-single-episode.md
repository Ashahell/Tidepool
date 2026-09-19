# Single-episode memorization fails

> Source: tmt-torch inline experiment (fixed 16-byte episode x 300 reps, dim128) output
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

One fixed copy episode (16-byte payload), 300 repetitions, dim128:
generated b'~\x00~\x00...' vs want b'~Mx~sc#[?s&4.O\\?' — match False.
Output is a degenerate marker/null cycle, not the payload.

Reading: the main decoder path cannot memorize even one fixed episode
in 300 tries. Combined with the recall head's 8/8 fixed-set fit, the
split is precise: a separately trained readout on frozen states can
fit; the jointly trained main path cannot — even with zero
generalization required. The failure is in joint training of
recall through the main head, not in state capacity alone.
