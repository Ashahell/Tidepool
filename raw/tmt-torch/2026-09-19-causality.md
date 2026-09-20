# Memory causality: A=B=C=D (no causal use)

> Source: tmt-torch scripts/memory_causality.py on runs/copy150k (50 trials, inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Query-point intervention on persistent state, P(first payload byte
correct): A normal 1/50, B zeroed 1/50, C shuffled 1/50, D replaced
1/50. Perfectly flat — manipulating memory changes nothing.

Reading: information without causal use, exactly the outcome the
review specified as decisive. The state does not drive the output at
the query point; the decoder decides without it. This closes the
diagnostic arc: retention absent (probes), trace diffuse (rank),
readout failing (heads), and now causally inert (intervention).
Companion fix: checkpoint backfill now matches exact gate key names
(in_* keys contain neither "gate" nor "ing").
