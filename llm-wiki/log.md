# tmt-torch log

Chronological activity log. One line per event, newest at bottom.
Detail lives in `raw/articles/`; code lives in `src/`, `scripts/`, `tests/`.

## 2026-09-18 — repo scaffold + design spec (branch feat/tmt-port base)
- Created `/home/miller/Work/projects/tmt-torch` (fresh git) with `src/tmt`, `configs`, `scripts`, `tests`, `docs/superpowers/{specs,plans}`.
- Brainstormed (architectural path): goal = PyTorch/CUDA port first, port+fix together, approach A (faithful + surgical fixes, Dream-RSI-ready scaffolding).
- Committed design spec `docs/superpowers/specs/2026-09-18-tmt-pytorch-port-design.md` (`851d3c8`).

## 2026-09-18 — torch install (RTX 5070 Ti)
- Arch-based box had no pip: installed `uv` + `python-pip` via pacman.
- `uv venv` + torch 2.14.0+cu130 (CUDA 13.0), numpy, safetensors, pyyaml, pytest.
- Verified: `cuda_available True`, RTX 5070 Ti, sm (12, 0).

## 2026-09-18 — evaluation suite MLX→PyTorch (`7f56154`)
- Ported the provided MLX eval suite to `src/tmt/evaluation_suite.py` (torch seeds/tensors/log_softmax/norms, `no_grad` probes, injected `online_update`, new `train_and_evaluate` for the Dream-RSI inner loop).
- Smoke-verified with dummy model (save/load roundtrip exact).

## 2026-09-18 — implementation plan (`0723543`)
- Wrote 5-task TDD plan `docs/superpowers/plans/2026-09-18-tmt-pytorch-port.md` (config → model → decay fixes → engine/CLIs → gate).

## 2026-09-18 — SDD execution, 5 tasks + fix wave (branch feat/tmt-port)
- Task 1 config clean (plan fix: `safe_load(f)`).
- Task 2 model core clean (plan fix: 1-D byte tensors, no `__call__` alias).
- Task 3 decay groups: fix round 1 — restored spec's `0.5**(1/h)` formula, relaxed impossible `>0.2` test to exact-values + ordering + `>0.05` (measured spread 0.083).
- Task 4 engine/CLIs clean (`.clone()` for safetensors shared-storage).
- Task 5 gate clean (12/12).
- Final review: 1 blocking (chat gen-loop) + spec gap (metrics/CLI logging) → one fix wave (`e7be8a5`: bounded `gen_bytes`, `metrics.py`, `--data/--seed`, loss.csv, node.json) → re-review clean.
- Controller-verified: 13/13 green. Full record: `raw/articles/2026-09-18-tmt-pytorch-port.md`.
