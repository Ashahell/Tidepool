# First genuine memory measurement — true zero

> Source: tmt-torch ingesting-probe run on runs/rtrl200k checkpoint (inline eval output)
> Collected: 2026-09-18
> Published: 2026-09-18

## Measured 2026-09-18

New ingesting probe (feed via model.ingest, oracles order 1.0/0.0/0.0,
suite 70 passed) scored the trained 200k RTRL checkpoint (dim128,
bpb ~6): copy4_after_8, copy4_after_64, copy4_after_512,
copy16_after_8, copy16_after_64, copy16_after_512 all 0.0, mean 0.0.
(Checkpoint predates EMA code: ema_state False, fast weights scored.)

Reading: with a probe that can actually see state, retention is still
zero — even 4 bytes after 8 intervening. Good prediction (bpb ~6) with
zero measurable retention confirms they are different phenomena. The
memory quest is now properly instrumented with a true zero baseline.
