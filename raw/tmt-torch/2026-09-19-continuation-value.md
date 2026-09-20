# Continuation value: the threshold dissolves, the gate is memorization

> Source: arithmetic over recorded tmt-torch measurements (no new runs)
> Collected: 2026-09-19
> Published: 2026-09-19

## The consultant's Phase-0 question, answered

"Compute P(recall) required to beat the no-memory optimizer optimum."
On the copy probe the no-memory optimum is chance: (1/256)^4 ≈
2.3e-10 exact-match. Any nonzero reliable recall beats it. The
threshold is trivially low, so the threshold analysis dissolves —
the decision is pure base-rate × cost, not threshold.

## Base rates (measured)

~15 mechanism trials at 130–160k params: 0 with recall > 0. Laplace
successor for the next distinct mechanism: ≈1/17 ≈ 6%, optimistic
(the easy hypotheses failed first). GRU and transformer baselines
also 0.0 — the failure is not architecture-specific.

## The load-bearing fact

Single fixed 16-byte episode × 300 reps: main head cannot memorize
it (null-cycle attractor). Recall head on frozen states: 8/8
fixed-set fit. Split is precise: storage + readout fit; joint
generation through the main head cannot — with zero generalization
required. Every proposed mechanism (gated write, slot-attention) sits
ABOVE this gate: none can help if one-episode memorization is broken.

## Decision

1. Next experiment is not a mechanism but the gate itself: make the
   main path memorize ONE fixed episode (optimizer/loss/decoding
   surgery, hours not days). If that fails, no memory mechanism can
   succeed — pivot or stop.
2. The under-celebrated positive: frozen-state + trained readout
   already memorizes (8/8). The failing component is generation, not
   storage. Highest-EV direction may be constrained decoding at query
   (emit from readout, never free-generate) rather than new memory.
3. RSI stays parked; slot-attention+RTRL waits behind gate (1).
