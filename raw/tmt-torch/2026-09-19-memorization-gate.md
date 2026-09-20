# The gate opens: one-episode memorization at 2000 reps

> Source: tmt-torch tests/test_memorize_one.py (GREEN) + /tmp/memprobe probes
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Fixed copy episode (payload 4 + filler 4 + marker + payload, true
data.copy_episode shape), dim32/layers2, free generation after reset:
300 reps → [0,0,0,0] (query transition rank 2, null wins); 2000 reps
→ all four query positions rank 1, free generation == payload
EXACT (test GREEN, 40s). TF within-episode 12/14 at 2000 (overtraining
shifts two bigrams; generation unaffected).

## Root-cause chain (systematic-debugging, all Phases)

1. First test version was mis-specified (episode lacked the trailing
   payload — the query transition was never trained). Suspect the
   assertion first: fixed, re-ran.
2. Failure isolated to ONE decision: marker→payload[0] (positions 1–3
   always rank 1 given true prefix).
3. forward() never commits persistent state; ingest() does — checked
   both paths, identical predictions, eliminated.
4. Single hypothesis "under-training on the ambiguous transition":
   300 → rank 2, 2000 → rank 1. Confirmed minimally.

## Reading

The gate was training amount, not architecture. The 2026-09-19
continuation-value note's gate ("make the main path memorize ONE
fixed episode") is now GREEN. Prescription for real runs: copy
training failed at 150k distributed steps — the query transition saw
too few effective updates (copy-frac dilution + per-line reset).
Next: a copy-dense run (high copy-frac, small model, steps to
query-transition convergence), not new mechanisms.
