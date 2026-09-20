# Fast-weight retrieval: memorization 7x, generalization zero

> Source: tmt-torch src/tmt/fastweight.py + runs/copydense_fw{,300k}
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Delta-rule fast weights (S += b(v − Sk)kT, unit-norm keys, read h+Sq;
Ba et al. 2016 / DeltaNet design). Unit behavior verified:
write-read roundtrip exact, same-key overwrite exact, repeated write
fixed point (tests/test_fastweight.py, 5 tests).

One-episode gate (fixed episode, dim32/L2): WITHOUT fw 2000 reps to
exact free generation; WITH fw_dk=8, 300 reps — all query ranks 1,
free generation byte-exact. ~7x fewer updates. First mechanism ever
to move a memory number in the right direction.

Generalization (fresh episodes, dim64/L3, fw_dk=16): 100k steps
recall 0/20 all probes, query-CE 6.92 → 5.50 (faster early than
no-fw 6.42 → 5.73); 300k steps recall 0/20 all 13 probes,
query-CE plateaus ~5.13 (HIGHER than no-fw 400k's 5.02).

Implementation note: S must detach across steps (else autograd
re-enters freed graphs); the read uses attached S_new so Wk/Wq/Wv
keep single-step grads. Committed in fastweight.py.

## Reading

fw helps FIT (expressive readout path) but not the copy ALGORITHM.
Mechanistic account: learning store-now-for-later keys/values needs
credit across the filler gap; v1 has single-step grads only (RTRL
traces cover RTU states, not S — disclosed in the plan). The
memorization win + generalization zero is exactly the signature of
missing cross-step credit. Next if continued: RTRL trace extension
to S (days of derivation), not more steps — the plateau says steps
are exhausted.
