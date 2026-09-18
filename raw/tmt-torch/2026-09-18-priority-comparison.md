# Prioritized replay comparison (60k steps) — negative result

> Source: tmt-torch runs runs/pcmp_off + runs/pcmp_on (train.log files, gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Runs measured 2026-09-18

Fair pair, dim128/layers4/lr 1e-4, 60000 steps on data/vk4a_train, eval
every 5000 on held-out bytes, exit 0 both.

Off: 6.576, 6.638, 6.353, 7.397, 7.129, 7.027, 7.410, 8.968, 6.893,
9.016, 8.516, 8.616.

Priority on (size 512, k 1, alpha 1.0): 6.277, 6.657, 6.541, 7.469,
8.504, 9.762, 11.111, 16.045, 11.150, 17.275, 16.296, 17.012.

Priority tracks baseline to ~20k, then diverges upward from ~25k and
ends 8.4 bpb worse (17.012 vs 8.616) — a bigger loss than uniform
recency replay (which ended 3.5 worse).

Reading: replaying high-loss bytes with extra updates digs into
mistakes instead of smoothing them — overconfidence on hard/noisy bytes
(markdown tables, code symbols) compounds. Loss-weighting sharpens
memorization of outliers. Both replay shapes tried so far hurt;
remaining candidates: reservoir/diverse sampling, EWC, or accepting that
single-pass online updates need a different stabilizer (slower
schedules, weight averaging).
