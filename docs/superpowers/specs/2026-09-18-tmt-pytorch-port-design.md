# TMT PyTorch/CUDA Port + Fixes — Design

**Date:** 2026-09-18
**Status:** Approved sections 1–3, awaiting spec review
**Sources:** jrz97619761/test-model-thing (152★, MIT, MLX, 4.5M dim=512/layers=16) + zhengkid/Dream-RSI (688★, paper+site, code pending)
**Machine:** RTX 5070 Ti 16GB, no torch installed yet

## Goal

Port TMT off MLX to CUDA/PyTorch as the foundation for everything else
(Dream-RSI-style meta-search over TMT configs later). Approach A:
faithful port + surgical fixes in one go. No modernized core (complex
LRU/selective SSM, FFN, hierarchical latents) on day 1 — those become
search-space options later.

## 1. Directory + module mapping

New repo: `/home/miller/Work/projects/tmt-torch` (fresh git).

Layout:
- `src/tmt/{model.py, engine.py, config.py, data.py, metrics.py}`
- `scripts/{train.py, chat.py, eval_memory.py}`
- `configs/base.yaml`
- `tests/`
- `docs/superpowers/specs/`

Torch mapping (from 230-line MLX `main.py`):
- `Encoder`: `nn.Embedding(256, dim)`.
- `Decoder`: byte logits `Linear(dim, 256)` + stop head `Linear(dim, 1)` + sigmoid.
- `Layer`: per-dim sigmoid decay + state update
  `state = decay*states + enc + dummy`; `LayerNorm(dim)` +
  `Linear(dim, dim, bias=False)` + SiLU residual. Traces kept as
  buffers (not params): `decaytrace (dim)`, `embedtrace (256, dim)`.
- `Model.step/call`: torch autograd for main path + manual RTRL trace
  adds (same math as MLX `value_and_grad` on dummies), AdamW,
  safetensors save of params + optimizer state + traces. `^C`-safe
  periodic save.
- `Runtime`: train/chat/dataset loops with `readonly` / `notrace` modes.
- RSI-ready: every run emits `node.json` in the DiscoveryTree Node shape
  (config, loss curves, final_metrics, costs, failure_mode, checkpoint).

## 2. Fix list + memory probes

- Decays: per-dim learnable bias init spread across 4 groups
  (half-lives ~8 / 64 / 512 / 4k bytes) + clamp + diversity penalty.
  Log mean/std/min/max per group to detect collapse.
- Stability: keep variance loss, add state-norm guard + configurable
  grad clip, AdamW unchanged.
- Leakage: explicit `reset()` per example in eval; continual-mode state
  carry only when flagged + logged.
- Updates: `update_every: 1 | 32 | 128` (accumulate, single step).
- Losses (config-weighted):
  `L = w_var*L_var + w_pred*MSE(x, stopgrad(enc(next)))
       + w_ce*CE(logits, next) + w_stop*MSE(stop, end)`.
- UTF-8: score bytes, report valid-UTF8 rate separately; no sampler
  constraint yet.
- Probes: 32B copy after 128/1k/5k/20k gap, bracket/quote-close,
  val bits-per-byte on held-out split, continual sequential retention,
  fixed MCC (zero-div guard).

## 3. Eval, logging, verification, env

- Scripts: `train.py` (wikipedia_clean glob + `--data` override),
  `chat.py` (`--readonly`, `--notrace`), `eval_memory.py` (all probes).
- Logging: per-step CSV + per-run `node.json`; seeds; resume.
- Tests: `test_parity_math`, `test_trace_update`,
  `test_reset_no_leak`, `test_mcc`, `test_eval_smoke` (CPU, <60s),
  `test_cuda_smoke` (one GPU step).
- Env: venv + CUDA torch build supporting Blackwell sm_120
  (recent torch + CUDA 12.8+; verify `torch.cuda.is_available()`),
  pytest.

## Risks

- 5070 Ti needs a new-enough torch/CUDA combo; install may need
  nightly. Mitigation: verify smoke test before port work.
- Single-byte online updates are noisy; chunked mode is the hedge.
- Dream-RSI code is unreleased; we only scaffold the Node/replay-ready
  logging now, no meta-loop in this spec.

## Out of scope

Modernized recurrent core, full Dream-RSI loop, large-scale training,
multimodality adapters, production serving.
