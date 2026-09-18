# Real-training scorer — wiring record

> Source: tmt-torch run runs/rsi_real1 (tree.jsonl, summary; gitignored) + suite output
> Collected: 2026-09-18
> Published: 2026-09-18

## Wired 2026-09-18

rsi_search.py scorer now trains for real: --train root (default
data/vk4a_train) + --train-steps (default 2000) feed train_fn, which
runs that many single-byte steps and returns measured wall-clock
gpu_hours. The 0.01 stub cost override is gone; replay budgets now
account real spend.

## Proof run measured 2026-09-18

One round, smoke grid, 2000 train steps per config:
round 1: n=8 best=-73.14682381448415 spend=0.004
policy=init-parallel-refine. First node: dim 32 / layers 1 /
update_every 1 / lr 0.0005, composite -73.15, gpu_hours
0.0009240440527598063, status ok. Training moves composites ~22 points
above random-init (-95). Suite: 37 passed.
