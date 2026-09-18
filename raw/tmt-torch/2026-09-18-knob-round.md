# Knob round (LR, update_every) — cure record

> Source: tmt-torch runs runs/knob_lr1e4, runs/knob_ue32, runs/knob_tiny20k (train.log files, gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Runs measured 2026-09-18

Three 5-minute runs on data/vk4a_train, eval every 2000 steps on held-out
bytes, all exit 0.

lr 1e-4 (small_lr1e4, dim128/layers4, 10k steps): bpb 6.929, 6.543,
6.734, 6.449, 6.328 — flat-to-improving, all below the 8.0 uniform
baseline. The monotonic climb of serious run 1 is gone at one-fifth the
learning rate.

update_every 32 (small_ue32, dim128/layers4, 10k steps): bpb 7.844,
7.494, 7.845, 7.546, 7.415 — flat, no climb. Chunked updates also
stabilize.

tiny 20k (tiny.yaml dim64/layers2, 20k steps): bpb 7.115, 6.674, 7.403,
7.102, 7.173, 7.144, 7.812, 8.060, 7.884, then 9.423 at step 20000 —
flat for 18k steps with one late spike (single eval; noise vs onset
unclear, needs a repeat before any claim).

Reading: the serious-run-1 pathology was step-size noise, not structural
amnesia. Cure by hand: adopt lr 1e-4 as the baseline. No architecture
search needed for this; the RSI loop stays parked until a structural
residual appears.
