# Dream-RSI Loop over TMT — Design

**Date:** 2026-09-18
**Status:** Sections 1–3 approved, awaiting spec review
**Approach:** A (single-machine orchestrator, tiny-smoke default budget)
**Foundation:** tmt-torch on master (`ea3c3ed` + `15186c8` wiki migration)

## Goal

Close the full Dream-RSI loop around TMT: online exploration of TMT
configs, exact replay simulation over logged discovery trees, and
LLM-driven exploration-policy improvement with a never-regress selection
rule — all on one box, defaulting to a tiny smoke budget.

## 1. Layout + interfaces

New package `src/rsi/` in tmt-torch:

- `tree.py` — `DiscoveryTree` / `Node`, append-only JSONL + in-memory index.
- `policy.py` — `ExplorationPolicy` base + `InitialParallelRefine`.
- `replay.py` — `replay_score(policy, history, budget)` pure function.
- `rewriter.py` — `PolicyRewriter` interface + `OpenAIPolicyRewriter` + `AgentPolicyRewriter`.
- `sandbox.py` — whitelisted `exec` + timeouts.
- `loop.py` — outer rounds: explore → dream → select → redeploy.
- `monitor.py` — round summaries + composite curves.
- `scripts/rsi_search.py` — `--config`, `--rounds`, `--budget` CLI.
- `configs/rsi_smoke.yaml` — tiny default (≤8 configs/round, ≤2 rounds, dim ≤64).

Reuses `train_and_evaluate`, `EvalConfig`, `EvalResult` untouched.
Policy-rewriting LLM: `PolicyRewriter` interface; OpenAI-compatible client
(env key, model from config); agent-session flow via request/reply markdown.

## 2. Tree schema + replay semantics

Node: `{id, parent_id, config, status, metrics {raw_metrics, composite},
cost {gpu_hours, steps}, failure_mode, checkpoint_path, policy_id,
created_at}`. Trees append to `runs/rsi/round_N/tree.jsonl`; the history
pool (all prior rounds) is immutable.

`replay_score(candidate, history, budget)` simulates the candidate's
decisions against recorded nodes only: a config counts as expanded iff its
exact `config_hash` exists in history; cost charged from recorded cost;
score = best recorded composite reachable under budget order. Gate:
replaying the initial policy over its own history returns its exact online
score and cost (summary-JSON byte-equality).

## 3. Policy, rewriter, sandbox, budgets, verification

Initial policy: round 0 parallel quasi-random over the tiny grid
(dim × layers × decay_groups × update_every × lr); later rounds refine
top-2 (halve numeric distance) + 25% random anti-collapse sample.
Deploy a candidate only if its replay score ≥ incumbent on full history;
ties keep the incumbent.

`OpenAIPolicyRewriter`: env `OPENAI_API_KEY` / `OPENAI_BASE_URL`, model
from config, prompt carries history stats + failure modes, ≤3 retries.
`AgentPolicyRewriter`: writes `runs/rsi/round_N/rewrite_request.md`, reads
`rewrite_reply.md` — the Opencode-session flow, offline-capable.

Sandbox: trimmed `__builtins__`, AST pre-check (stdlib math/json/random/
itertools only; no os/sys/subprocess/socket; loops need constant bounds),
per-call wall timeout; violations score −inf, loop continues on incumbent.

Verification: replay==online byte-gate; sandbox rejection tests
(import-os, unbounded loop, slow call); never-regress test (bad candidates
keep incumbent); end-to-end smoke (mock scorer + ≥1 real tiny config).

## Out of scope

Parallel GPU workers / persistent server; nearest-neighbor or model-based
replay; real-scale search (a config change, not a rewrite); the ambitious
hybrid (TMT inside the loop as world model).
