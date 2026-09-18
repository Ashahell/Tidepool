# tmt-torch Dream-RSI loop — session record

> Source: tmt-torch project session record (SDD execution log + test output on branch feat/rsi-loop)
> Collected: 2026-09-18
> Published: 2026-09-18

## Execution measured 2026-09-18

Built the single-machine Dream-RSI loop in src/rsi/ (7 modules: tree,
policy, replay, sandbox, rewriter, loop, monitor) plus
scripts/rsi_search.py, configs/rsi_smoke.yaml, and 6 test files
(test_rsi_tree.py, test_rsi_policy.py, test_rsi_replay.py,
test_rsi_sandbox.py, test_rsi_rewriter.py, test_rsi_loop.py).
Work ran on branch feat/rsi-loop: 14 commits, fast-forward merged to
17680c5. Merged-tree gate: 33 passed (20 RSI + 13 TMT).

Six subagent-driven tasks each passed task review; one fix round changed
loop selection from >= to strict > so ties keep the incumbent. Final
whole-branch review was clean with 6 deferred minors, none blocking.

## Key semantics recorded

Replay is exact-match on config_hash with recorded gpu_hours costs and a
replay==online gate test. Deployment keeps the incumbent unless a
candidate scores strictly higher on full history. Sandbox allows imports
math, json, random, itertools, collections only, bans while loops, and
runs policy calls with a timeout. Rewriter retries ≤3 times; secrets come
from env (OPENAI_API_KEY, OPENAI_BASE_URL) only. Smoke defaults:
budget_per_round 0.08, rounds 2, n_revisions 1, grid dim [32, 64],
layers [1], update_every [1], lr [0.0005], seed 42.

## Rulings recorded

Round-0 fresh-RNG and clamped-best refine (verbatim code failed its own
tests); replay test seed 3 to 4 (seed 3 draws duplicates); sandbox exec
namespace gains __build_class__ and __name__ (classes cannot execute
otherwise); rewriter test asserts best not sum (no such key is sent);
reviewer's ties-keep catch upheld; CUDA pin and tolerant checkpoint loads
parked from the earlier port run.
