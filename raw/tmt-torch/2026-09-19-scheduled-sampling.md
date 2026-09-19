# Scheduled sampling — partial then collapse

> Source: tmt-torch inline single-episode experiments + scripts/train.py --ss-prob (committed c4fc8eb)
> Collected: 2026-09-19
> Published: 2026-09-19

## Measured 2026-09-19

Root-cause chain for joint-training failure: fillerless episodes fail
identically (filler-noise dead); teacher-forced payload CE 0.028
(learned) vs free generation collapse (attractor) — exposure bias
class, not optimization.

Fix attempt: scheduled sampling (--ss-prob, own-prediction inputs,
tested in tests/test_train_utils.py). Single fixed 16-byte episode:
300 reps output payload fragments (oo@~...) not null cycles — shape
improvement, still no match; 1000 reps regressed to the null cycle
(o\x00 repeating, first byte right).

Reading: SS perturbs attractor dynamics but does not remove them;
longer SS training collapses back. Exposure-bias mitigation alone is
insufficient — the attractor (marker/null fixed point) dominates free
generation. Next: generation constraints, attractor analysis (which
states trap), or structural decode changes.
