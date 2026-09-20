# Retrieval gap proven: retention present, recall absent

> Source: tmt-torch scripts/copy_dense.py runs + probe_state.py on two ckpts
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Copy-dense runs (fresh random episodes, query-segment logging):
100k steps dim64/L3 → recall 0/20 all probes; 400k steps seed 8 →
recall 0/20 all 17 probes. Query composite loss 6.27 → 5.02
(below uniform 5.545), but teacher-forced per-position accuracy on
fresh episodes is 0/50, 0/50, 1/50, 0/50 — the sub-uniform loss is
mass-shifting, not mapping.

Decay autopsy: training builds slow channels by itself — copy150k
(with init_decay_groups) ends min 0.77 / mean 0.96 / 64 units >0.99
per layer; even the no-slow-init 400k run pushes max to 0.86.
Retention is not the scarce resource.

Double dissociation (probe_state.py, 400 train / 100 test, copy4/after8):
copy150k state → linear probe 10% vs majority 5% vs chance ~1%
(information present, 10x chance); dense400k state → probe 0.00%
(absent). Both score exact recall 0.0.

## Reading

Retention present + recall absent = the failure is retrieval/use,
not storage. The state holds the payload (slow channels, linear
decodability) and nothing at query time reads it out — consistent
with the flat causality (A=B=C=D) and the frozen-readout 8/8 fit.
Prescription narrows to retrieval-side mechanisms (query-conditioned
readout, real attentional competition); further retention-side work
(decay, gates, replay, EMA) is now contraindicated by evidence.
scripts/copy_dense.py committed as the dense-run harness.
