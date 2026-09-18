# Dream-RSI Loop

> Sources: tmt-torch session record, 2026-09-18
> Raw: [2026-09-18-rsi-loop-record](../../raw/tmt-torch/2026-09-18-rsi-loop-record.md)
> Updated: 2026-09-18

## Overview

The full Dream-RSI loop now runs on one machine in src/rsi/ (7 modules
plus CLI and smoke config): discovery trees in JSONL, exact-match replay
over recorded costs, LLM-written candidate policies through an
AST-whitelisted sandbox, and never-regress deployment. Built in six
reviewed tasks on branch feat/rsi-loop (14 commits, fast-forward merged
to 17680c5) with a merged-tree gate of 33 passed (20 RSI + 13 TMT).

## How It Works

Online rounds explore TMT configs via an initial parallel-refine policy
and log every result; dreaming evaluates rewritten policies by replaying
them over history (a config counts only on exact config_hash match, costs
charged as recorded); a candidate deploys only on strictly higher replay
score, so ties keep the incumbent. Rewriter backends are an
OpenAI-compatible client (≤3 retries, env keys only) and an agent-session
file flow. Smoke defaults run ≤8 configs over 2 rounds (budget_per_round
0.08, grid dim [32, 64], layers [1]).

## Execution Notes

Three plan defects corrected by measurement (round-0 determinism, refine
tolerance, replay seed), one exec-namespace fix (classes need
__build_class__), one reviewer-caught spec violation (ties-keep now
strict >). Parked: machine-pinned CUDA test, tolerant checkpoint loads.
Next: first real smoke search, then scale-up by config.

## See Also

- [TMT PyTorch Port](pytorch-port.md)
