# Dream-RSI Loop

> Sources: tmt-torch session record, 2026-09-18
> Raw: [2026-09-18-rsi-loop-record](../../raw/tmt-torch/2026-09-18-rsi-loop-record.md); [2026-09-18-rsi-smoke-run](../../raw/tmt-torch/2026-09-18-rsi-smoke-run.md); [2026-09-18-real-scorer](../../raw/tmt-torch/2026-09-18-real-scorer.md); [2026-09-18-rtrl-remeasurement](../../raw/tmt-torch/2026-09-18-rtrl-remeasurement.md)
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

## RTRL Ablation (No Difference at Tiny Scale)

An 8-cell matrix ({RTRL, single-step} × {persistent, reset} ×
{groups-4, groups-1}, 2000 steps each) shows RTRL ≈ single-step on every
metric (bpb 7.45 vs 7.45, copy 0.0 everywhere) and reset cells
bit-identical (zero state implies zero correction — derived, not a bug).
The correction is proven right by finite differences but buys nothing
measurable at this scale and horizon; the question stays open for
longer runs and bigger models.

## First Live Run

On 2026-09-18 the loop ran end-to-end with real weights and real
Vulkan4Aros bytes (wiki prose + C source): round 1 n=8
best=-95.55782821967064 spend=0.080, round 2 n=8 best=-95.78051670751206
spend=0.080, 16 nodes total. Scores are random-init baselines (no
gradient training yet) — machinery validated, not model quality.
Incumbent held both rounds (no rewrite reply existed). Known wart: both
rounds' rewrite requests landed in round_1/rewrite_request.md.

## Real-Training Scorer (Live)

The scorer now trains each config (default 2000 steps on the train
split) and reports measured gpu_hours — no more stub costs, so replay
budgets account real spend. Proof round: n=8 best=-73.14682381448415
spend=0.004, ~22 points above random-init. The loop can now hunt
continual-learning rules for real.

## Execution Notes

Three plan defects corrected by measurement (round-0 determinism, refine
tolerance, replay seed), one exec-namespace fix (classes need
__build_class__), one reviewer-caught spec violation (ties-keep now
strict >). Parked: machine-pinned CUDA test, tolerant checkpoint loads.
Next: first real smoke search, then scale-up by config.

## See Also

- [TMT PyTorch Port](pytorch-port.md)
