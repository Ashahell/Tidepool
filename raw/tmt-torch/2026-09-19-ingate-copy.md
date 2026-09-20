# Input write-gate — helps prediction, not recall

> Source: tmt-torch runs runs/copyingate (10k) + runs/copyingate60k (60k, frac 1.0, selective+ingate) + evals (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Selective model with learned per-dim input gate (FD-proven with the
decay gate to 1e-4-class tolerances), pure copy: 10k payload CE 8.262,
probe 0.0; 60k payload CE 6.973, probe 0.0. Best payload CE at this
scale outside the 1M run, yet exact recall still zero.

Reading: the write gate joins every other mechanism in moving
prediction without touching recall. The split is now total: a dozen
levers move bpb/CE, zero move exact match. Recall at small scale is
not a knob problem.
