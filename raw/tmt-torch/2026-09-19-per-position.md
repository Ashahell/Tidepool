# Per-position accuracy — weak but nonzero signal

> Source: tmt-torch recall-head eval on runs/recall30k (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Recall-head per-position accuracy over 50 fresh episodes (chance
0.0105): pos0 0.04, pos1 0.02, pos2 0.02, pos3 0.04. No positional
decay (first byte not better) — flat 2–4x chance everywhere.

Reading: exact-match zeros hide a weak uniform signal, consistent with
rank ~50 and linear-probe 10%. The trace is real, diffuse, and
position-independent — not recency. Still orders of magnitude from
usable recall.
