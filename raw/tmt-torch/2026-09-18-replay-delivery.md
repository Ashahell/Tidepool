# In-model replay buffer — delivery record

> Source: tmt-torch SDD execution on branch feat/replay (ledger rulings + suite output; workspace removed per process)
> Collected: 2026-09-18
> Published: 2026-09-18

## Delivered 2026-09-18

TMTConfig gains replay_size (default 0, off) and replay_k (default 1).
TMTModel holds a recency deque of (curr, next, end) tuples;
training_step runs one live _update, appends, then replay_k sampled
_update calls under shared accumulation. Returned loss and
last_components stay live-only. Smoke grid exposes replay_size [0, 64]
and replay_k [1]; the RSI scorer picks them up with zero code change.

Two tasks, both review-clean, no fix loops. Final review clean (2
trivial deferred notes). Merged to master fast-forward; suite 41 passed
controller-verified on the merged tree.

Still open (spec-deferred): replay-on/off bpb comparison run; EWC;
reservoir sampling.
