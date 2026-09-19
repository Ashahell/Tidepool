# Decay audit — timescales survive, retention still zero

> Source: tmt-torch decay_bias inspection on saved checkpoints (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Per-dim half-lives from sigmoid(decay_bias), all 4 layers, both runs:
copy150k L0 [3.7, 29.7, 192.3, 911.6], L1 [3.1, 29.5, 194.5, 1033.7],
L2 [7.2, 30.6, 182.4, 922.1], L3 [3.7, 31.1, 197.8, 960.0]; rtrl200k
L0 [4.6, 22.6, 146.3, 739.2] through L3 [4.9, 24.3, 146.4, 762.9]
(std ~0.06, range 0.77–1.0). Multi-timescale structure survived
training in every layer — no collapse.

Reading: timescale diversity is present and persistent, yet recall is
zero even at 8-byte gaps where ~1000-step dims should dominate. The
"decays collapsed" explanation is dead. Timescale diversity is
necessary but not sufficient; information is not written retrievably
regardless of horizon. Left: selective gates, reconstructive losses,
explicit slots.
