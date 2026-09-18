# TMT PyTorch port: plan, execution, and record

📋 **2026-09-18 — port + fixes landed on `feat/tmt-port` (11 commits, 13/13 green, unmerged).**
Torch 2.14.0+cu130 on RTX 5070 Ti. Full SDD record with rulings below.

## Sources

- TMT: [jrz97619761/test-model-thing](https://github.com/jrz97619761/test-model-thing) (152★, MIT) — 230-line MLX `main.py` + `benchmark.py`. 4.5M params (dim=512, layers=16), ~12h on Simple Wikipedia. Byte I/O, JEPA-style latent prediction + byte CE, per-dim sigmoid-decay RTU state, single-byte online updates with RTRL traces, test-time training.
- Dream-RSI: [zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI) (688★) — paper + site out, code pending. Meta-loop (online explore → replay simulator → dreaming policy improvement) over coding-agent discovery trees. Relevance here: the eval suite + Node-JSON logging are built to plug into that outer loop later; the loop itself is out of scope.

## Approach (A: faithful port + surgical fixes)

Keep the single-byte online loop and RTRL trace structure so behavior stays
comparable to MLX; fix during the port rather than after. Modernized core
(complex LRU, selective SSM, FFN, hierarchical latents) deferred to
search-space options, not day-1 defaults.

## Module mapping (MLX → torch)

- `Encoder` → `nn.Embedding(256, dim)`; `Decoder` → byte logits + stop head.
- `Layer` → per-dim `sigmoid(decay_bias)` + state update + LayerNorm + Linear(no bias) + SiLU; traces as buffers (`states`, `decaytrace`, `embedtrace`), not params.
- `Model` → autograd main path + manual RTRL trace adds (same math as MLX `value_and_grad` on dummies), AdamW, safetensors (params + optimizer + traces), `^C`-safe periodic save.
- `Runtime` → `scripts/train.py` / `chat.py` / `eval_memory.py`.
- RSI-ready: every run emits Node-shaped JSON (config, curves, final_metrics, costs, failure_mode, checkpoint).

Two transcription bugs caught by tests, both fixed with plan errata:
`yaml.safe_load()` → `safe_load(f)`; `(1,1)` byte tensors → 1-D (crashed `states.copy_`); no `__call__` alias (`nn.Module` dispatches to `forward`).

## Fix list (ported together)

- Decays: round-robin init across 4 half-life groups (8/64/512/4000 → per-step `0.5**(1/h)` = [0.917, 0.989, 0.999, 1.0]) + `−var` diversity penalty (0.01, gated on `decay_groups>1`). Log mean/std/min/max per group. Note the values cluster near 1 — that is correct; timescales differ 500×.
- Stability: variance loss kept, state-norm guard, configurable grad clip.
- Leakage: explicit `reset()` per eval example; continual carry only when flagged.
- Updates: `update_every: 1|32|128` (accumulate, single step; identical math at 1).
- Losses (config-weighted): `w_var·L_var + w_pred·MSE + w_ce·CE + w_stop·MSE`.
- UTF-8: score bytes, report validity separately; no sampler constraint yet.
- Benchmark: per-example reset, MCC zero-div guard, held-out bpb, copy-after-N + bracket probes, continual retention.

## Eval suite (`src/tmt/evaluation_suite.py`, frozen contract)

Ported from the provided MLX reference: `EvalConfig`, `ProbeResult`,
`EvalResult` (composite = −12·bpb + 6·long-range + 9·continual + 4·stability − 0.8·efficiency),
`compute_bpb`, copy-memory / continual / stability probes, `evaluate_model`,
`train_and_evaluate(config, val, cont, build_fn, train_fn)` as the Dream-RSI
inner-loop scorer, JSON save/load. Model contract: `reset()`, `model(LongTensor(1,)) → (logits (1,256), state list)`.
Deltas vs reference: injected `online_update` callable; copy probe still uses
the dummy-0 input until the real autoregressive step lands.

## Execution (subagent-driven, 5 tasks + fix wave)

- T1 config — clean.
- T2 model core — clean.
- T3 decay groups — fix round 1: implementer proved the plan's `>0.2` spread test mathematically impossible for the spec's formula (measured 0.083) and shipped an alternative mapping; ruled against it (spec binds half-lives), restored `0.5**(1/h)`, test now asserts exact per-group values + ordering + `>0.05`.
- T4 engine/CLIs — clean (`.clone()` required: safetensors rejects shared-storage tensors).
- T5 gate — clean, 12/12 first run.
- Final review → one fix wave: bounded `gen_bytes` (chat loop hung on short inputs), `metrics.py` re-export, train `--data/--seed`/resume/loss.csv/node.json, eval node.json + eval_result.json writes → re-review all-addressed.
- Controller-verified `pytest tests/`: **13/13 green**.

Parked with rulings: CUDA `(12,0)` pin (machine-specific gate by design);
`strict=False` checkpoint loads (cross-config tolerance); `one_hot` full-row
bump (correct RTRL math, MLX-faithful). Deferred minors: mutable-default
`half_lives`, import-in-loop + input-len break in chat.py (brief-verbatim),
`torch.var` unbiased at dim=1 (non-issue).

## What's next

Merge `feat/tmt-port` → master (pending decision); first real training run
(small model, short corpus, `eval_memory.py`); copy-probe real autoregressive
step; then the Dream-RSI outer loop (discovery tree + replay + policy
rewriting) on top of `train_and_evaluate`.
