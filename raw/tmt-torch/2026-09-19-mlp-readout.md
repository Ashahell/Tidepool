# Nonlinear readout — capacity not the gap

> Source: tmt-torch run runs/copymlp (30k steps, frac 1.0, recall_hidden 256) + evals (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

MLP recall head (128-256-256, own Adam) trained on episode queries,
30k pure-copy steps dim128: held-out copy4 0/20, fixed-set 8/8 —
identical split to the linear head at higher capacity.

Reading: readout capacity is not the gap either (linear and MLP both
fit, both fail transfer). The mapping from state to novel payload does
not generalize at this training scale regardless of readout power.
Remaining: far heavier varied training, scale, structural memory — or
the possibility that single-pass online state simply does not factor
into content-addressable form.
