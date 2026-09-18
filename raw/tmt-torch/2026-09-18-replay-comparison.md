# Replay on/off comparison (60k steps) — negative result

> Source: tmt-torch runs runs/cmp_off + runs/cmp_on (train.log files, gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Runs measured 2026-09-18

Fair pair, dim128/layers4/lr 1e-4, 60000 steps on data/vk4a_train, eval
every 5000 on held-out bytes, exit 0 both.

Off (replay_size 0): 6.576, 6.638, 6.353, 7.397, 7.129, 7.027, 7.410,
8.968, 6.893, 9.016, 8.516, 8.616.

On (replay_size 512, replay_k 1): 6.378, 6.775, 6.893, 8.362, 7.971,
7.600, 8.358, 9.230, 7.394, 10.002, 9.685, 12.119.

Replay never beats baseline at any of the 12 checkpoints and ends 3.5
bpb worse (12.119 vs 8.616).

Reading: a uniform recency buffer replays what the model just saw —
already-learned bytes — doubling down on recency bias instead of
countering it. Equal-weighted recency replay is the wrong shape;
candidates: reservoir/diverse buffer, high-loss prioritization, or EWC.
The buffer stays as a searchable mechanism, but its current form is a
negative result, recorded as such.
