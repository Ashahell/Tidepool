# Training monitor slice — delivery record

> Source: tmt-torch session record (live mini-run runs/mon_slice + suite output)
> Collected: 2026-09-18
> Published: 2026-09-18

## Delivered 2026-09-18

Held-out split: scripts/make_splits.py over split_corpus (val takes 5%
line tails) produced data/vk4a_train (5.0 MB) and data/vk4a_val (196 KB),
disjoint by construction (test_split_disjoint_and_covering).

Per-component logging: training_step stashes last_components
(l_var, l_pred, l_ce, l_stop, state_norm) with total == loss; train.py
writes step,loss,l_var,l_pred,l_ce,l_stop,state_norm to runs/loss.csv.

Heartbeat + watchdog: runs/state.json every 500 steps (step, loss,
components, bytes_per_sec, elapsed_s); non-finite values save
runs/diverged.safetensors, record failure_mode diverged, exit 1
(engine.check_finite, unit-tested).

Periodic eval: --eval-every N with --eval-val writes runs/eval.csv
(step,bpb,mem,cont,stab,composite) using the smoke preset.
scripts/plot_curves.py renders runs/curves.png (matplotlib, Agg).

## Live proof (runs/mon_slice, tiny config, 300 steps)

loss.csv header confirmed with component columns; first row step 1 loss
9.57589340209961 with l_var -1.239776611328125e-05, l_pred
2.828108549118042, l_ce 6.513428688049316, l_stop 0.23436792194843292,
state_norm 16.256256103515625. The variance term sits at ~0 from the
start (already satisfied); cross-entropy does the work. Periodic evals
on held-out bytes: step 150 bpb=7.862 composite=-90.945, step 300
bpb=7.490 composite=-85.089 — below the 8.0 uniform baseline and
improving. curves.png written (172575 bytes). Full suite: 37 passed.
