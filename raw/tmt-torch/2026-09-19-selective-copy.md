# Selective copy run — marginal CE gain, still zero recall

> Source: tmt-torch run runs/copyselective (30k steps, frac 1.0, selective) + probe eval (inline output)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Selective model (input-dependent per-dim gate, FD-proven) on pure copy,
30k steps: payload CE 7.614 (vs 7.923 plain), ingesting probe 0.0 at
copy4 x after8/32. Gate engaged (weights ~0.03, traces live). Early
train evals coincided with the plain run to 3 decimals (same
seed/data/init + tiny gate — determinism, not duplication; direct
6-decimal bpb differs: 7.790672 vs 7.674337).

Reading: selectivity moves payload CE slightly, recall not at all at
30k. Inconclusive rather than negative — needs longer + wider
conditions before judging the mechanism. Companion fix: checkpoint
loader backfills missing gate keys with zeros (exact) and falls back
to a fresh optimizer on incompatible state (loud warnings); strictness
otherwise intact.
