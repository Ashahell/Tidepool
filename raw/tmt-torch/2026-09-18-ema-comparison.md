# EMA comparison (60k + 200k) — negative result

> Source: tmt-torch runs runs/ecmp_ema + runs/ecmp_ema200k (train.log files, gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Runs measured 2026-09-18

EMA (ema_decay 0.999) with eval on averaged weights, same fair
conditions (dim128/layers4/lr 1e-4, vk4a_train, held-out val).

60k: 6.579, 6.541, 6.295, 7.341, 7.199, 7.055, 7.467, 8.907, 6.931,
8.379, 8.888, 8.534 — ties baseline (off ended 8.616).

200k (eval every 10000): 6.135, 6.390, 7.076, 7.721, 7.325, 9.303,
8.403, 10.631, 12.249, 12.908, 12.781, 11.760, 14.473, 16.080, 14.579,
18.010, 18.693, 17.169, 19.240, 20.398 — ends worse than off-run 18.115.

Reading: averaging noisy weights does not preserve what forgetting
destroys; the averaged model follows the same decay, slightly worse.
Cures so far: none hold past ~40k. Tried and failed: uniform replay,
priority replay, EMA. Untried: reservoir diversity, EWC, slower
schedules, bigger models.
