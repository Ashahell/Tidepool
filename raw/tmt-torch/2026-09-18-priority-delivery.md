# Prioritized replay — delivery record

> Source: tmt-torch SDD execution on branch feat/priority-replay (ledger rulings; workspace removed per process)
> Collected: 2026-09-18
> Published: 2026-09-18

## Delivered 2026-09-18

TMTConfig gains replay_alpha (default 1.0; 0 reproduces uniform).
Entries are (curr, next, end, loss) 4-tuples; _replay_indices draws via
torch.multinomial over loss^alpha with uniform fallback on equal/zero
tags; sampled entries get their tags refreshed after update. No _update
change; monitoring stays live-only. Smoke grid exposes replay_alpha
[0.0, 1.0] with zero code change (hasattr overlay).

Two tasks, both review-clean; one fix wave (empty-buf guard, stronger
alpha-zero test, deterministic refresh test) re-reviewed clean; final
review clean. Merged to master fast-forward; suite 46 passed
controller-verified on the merged tree. No plan conflicts surfaced; no
ruling beyond routine approvals.

Still open: priority-vs-off bpb comparison run (the experiment that
decides whether loss-weighting beats the negative uniform result).
