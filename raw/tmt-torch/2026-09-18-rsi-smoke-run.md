# First live RSI smoke search — run record

> Source: tmt-torch smoke run runs/rsi_smoke1 (summaries + trees on disk, gitignored)
> Collected: 2026-09-18
> Published: 2026-09-18

## Run measured 2026-09-18

First end-to-end loop run with real model weights and real bytes (no
mocks): `.venv/bin/python scripts/rsi_search.py --rounds 2 --workdir
runs/rsi_smoke1` with val from Vulkan4Aros llm-wiki/log.md (4096 bytes)
and continual domains from one wiki article (2048 bytes) plus
src/vulkan/library/icd_draw.c (2048 bytes), smoke eval preset.

Printed: `round 1: n=8 best=-95.55782821967064 spend=0.080
policy=init-parallel-refine` and `round 2: n=8 best=-95.78051670751206
spend=0.080 policy=init-parallel-refine`. 16 nodes total across
round_1/tree.jsonl and round_2/tree.jsonl (8 lines each). Scores are
random-init baselines (train_fn is a 0.0 stub; no gradient training yet),
so the run validates machinery, not model quality.

Policy stayed incumbent both rounds: AgentPolicyRewriter wrote
round_1/rewrite_request.md, no reply file existed, the FileNotFoundError
was caught into feedback, and selection kept init-parallel-refine.
Known wart: the rewriter was constructed with the round_1 directory, so
both rounds' requests landed in round_1/rewrite_request.md (overwritten).
