# Transformer anchor — task hard at small scale, period

> Source: tmt-torch transformer baseline runs (3 + 20 epochs) + probe (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Tiny causal transformer (dim128/layers2/heads4, 462336 params),
offline whole-sequence training on copy episodes: 3 epochs loss 5.254
to 5.033, probe 0.0; 20 epochs loss to 4.019, probe still 0.0.

Reading: a standard architecture in its best regime learns the
distribution slowly (4.02) with zero exact recall. The 0.0s across all
TMT variants are consistent with task-hardness-at-small-scale, not an
RTU-specific defect. Exact copy needs more scale/training than anything
tried — or the probe bar (exact argmax chains) exceeds what small
models deliver anywhere.
