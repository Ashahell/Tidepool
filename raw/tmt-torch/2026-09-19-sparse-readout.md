# Readout sparsity — marginal gain, still zero

> Source: tmt-torch run runs/copysparse (30k steps, frac 1.0, sparse_k 16) + evals (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Top-k (16/128) readout sparsity, recurrence and traces untouched:
payload CE 7.462 (vs 7.923 dense), probe 0.0 at copy4 x after8/32.
Suite green including exact-level sparsify tests.

Reading: disentangling readout mixing helps prediction slightly and
recall not at all. The superposition hypothesis for the READOUT is
weak; the blur lives upstream (in what gets written to state), not in
how it is read.
