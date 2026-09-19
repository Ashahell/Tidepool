# GRU baseline contrast — failure is not TMT-specific

> Source: tmt-torch GRU training (30k pure-copy steps) + probe evals (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Minimal GRU baseline (src/tmt/gru_baseline.py, dim96/layers2, 161152
params vs TMT dim128/layers4 132993) with reset/ingest/forward/train
interface, trained 30000 steps on pure copy episodes (frac 1.0, same
protocol as the TMT copy runs). Scores: copy GRU 0.0, copy TMT 0.0,
assoc GRU 0.0, assoc TMT 0.0 (copy4 x after8/32; assoc default 8 pairs).

Reading: classic gated recurrence fails identically at this
scale/training. The zero is not an RTU-specific defect — exact recall
from online single-sample training does not emerge in either
architecture here. Next discriminators: scale (params and steps),
not further architecture swaps at 130–160k.
