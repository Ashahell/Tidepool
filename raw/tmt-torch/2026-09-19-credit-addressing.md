# Credit and addressing: 2x2 memorization, generalization null

> Source: tmt-torch defer/BPTT machinery + runs/copydense_fw_bptt{,700k,_aux,_auxq}
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Episode-BPTT machinery (defer backward to episode end; S stays
attached in-episode; clone-on-read fixes delayed-backward version
errors; RTRL correction refactored per-step): tests/test_bptt.py, 3
tests green. Full suite 112 green.

Memorization 2x2 (episodes to exact one-episode gate, dim32/L2):
no-fw single-step 2000; fw single-step 300 (fw 7x); no-fw BPTT 100
(credit 20x); fw+BPTT 50 (combined 40x). Both effects real,
roughly multiplicative. Credit dominates.

Generalization (fresh episodes, dim64/L3, fw_dk=16): BPTT 100k
tokens recall 0/20 (update-starved: 1k updates); BPTT 700k tokens
(~11.6k updates) recall 0/20, query-CE 7.56 → 5.54 — better
per-update than single-step, no snap.

Addressing autopsy (query-key match on trained ckpts, chance ~0.31):
BPTT700k payload-top4 0.362, keys ~random (|cos| 0.246), beta 0.49 —
no learned addressing. Aux retrieval loss (CE(decoder(r), next))
applied EVERYWHERE: payload-top4 0.188, BELOW chance — at
unpredictable positions it teaches retrieval to be useless
(residual bypass lets h do the work; r gets anti-pressure).
Query-gated aux (3k episodes): payload-top4 0.263 — still chance.

## Reading

Credit massively accelerates FIT (40x combined) but the copy
ALGORITHM still doesn't emerge, and addressing never rises above
chance under single-step, BPTT, everywhere-aux, or query-gated aux.
The slow weights do not learn keys under any regime tried. Remaining
ideas (span pointer supervision, contrastive keys, 8h+ runs) have
poor base rates at rising cost. Retrieval work stops here; the
write-up is now fully armed: retention present, retrieval absent,
credit helps fit not algorithm, addressing unlearnable in-regime.
