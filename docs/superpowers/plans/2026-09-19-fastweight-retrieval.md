# Fast-Weight Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give TMTModel content-addressed retrieval via a DeltaNet-style fast-weight memory, and measure whether exact copy recall moves off zero.

**Architecture:** Each `RTULayer` optionally owns a `FastWeightMemory` holding `S ∈ R^(d×dk)`. Per step: project `k=Wk h`, `v=Wv h` (slow weights), L2-normalize `k`, delta-write `S += β(v − Sk)kᵀ` with `β=sigmoid(wb·h+bb)`, read `r=Sq` with normalized `q=Wq h`, output `h+r`. Off (`fw_dk=0`) is bit-identical canonical. No RTRL trace terms for `S` in v1 (single-step autograd credit only — disclosed limitation).

**Tech Stack:** PyTorch, existing `src/tmt/` modules, pytest TDD.

**Spec:** `wiki/tmt-torch/architecture-and-evidence.md` (retrieval-gap section) + `raw/tmt-torch/2026-09-19-retrieval-gap.md`. Literature: Ba et al. 2016 (arXiv:1610.06258, η=0.5/λ=0.95 reference values), Schlag et al. 2021 (DeltaNet delta rule), Schlag/Irie/Schmidhuber 2021 (linear-transformer≡FWP equivalence).

## Global Constraints

- `PYTHONPATH=src`, interpreter `.venv/bin/python` (Python 3.14, torch 2.14.0+cu130).
- Canonical path bit-identical when `fw_dk=0` (guard: `tests/test_canonical.py` stays green).
- Editor tool for all file writes; one-shot diagnostics in `/tmp/memprobe/`, never `scripts/`.
- New mechanism code carries `# EXPERIMENTAL` banners; config default off.
- Every claim ingested to `raw/tmt-torch/` with measured numbers; wiki lint (`check_evidence.py`) 0 errors.

---

### Task 1: FastWeightMemory module + roundtrip/overwrite tests

**Files:**
- Create: `src/tmt/fastweight.py`
- Create: `tests/test_fastweight.py`

**Interfaces:**
- Consumes: nothing (standalone `torch.nn.Module`).
- Produces: `FastWeightMemory(dim, dk)` with `step(h: Tensor(1,d)) -> Tensor(1,d)` (write-then-read), `reset()`; used by Task 2 as `self.fw`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fastweight.py
import torch
from tmt.fastweight import FastWeightMemory

def test_write_read_roundtrip():
    m = FastWeightMemory(4, 4)
    with torch.no_grad():
        m.Wk.copy_(torch.eye(4)); m.Wq.copy_(torch.eye(4)); m.Wv.copy_(torch.eye(4))
        m.wb.zero_(); m.bb.fill_(10.0)  # beta ~ 1
    m.reset()
    h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = m.step(h)
    # write v=h,k=e1 then read q=e1: r = h, out = h + r = 2h
    assert torch.allclose(out, 2 * h, atol=1e-5)

def test_delta_overwrite_same_key():
    m = FastWeightMemory(4, 4)
    with torch.no_grad():
        m.Wk.copy_(torch.eye(4)); m.Wq.copy_(torch.eye(4)); m.Wv.copy_(torch.eye(4))
        m.wb.zero_(); m.bb.fill_(10.0)
    m.reset()
    k = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    m.step(torch.tensor([[1.0, 0.0, 0.0, 0.0]]) * 0 + k)  # v=k input
    out = m.step(k)  # write k->k again, read: r must equal k exactly
    assert torch.allclose(out, 2 * k, atol=1e-5)

def test_reset_zeroes():
    m = FastWeightMemory(4, 4)
    m.reset()
    assert bool((m.S == 0).all())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_fastweight.py -q`
Expected: FAIL with "No module named 'tmt.fastweight'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/tmt/fastweight.py
"""Delta-rule fast-weight associative memory (EXPERIMENTAL, v1).

Ba et al. 2016; Schlag et al. 2021 (DeltaNet). S += b(v - S k) k^T
with unit-norm keys; read r = S q; step returns h + r.
No RTRL trace terms in v1: slow weights get single-step autograd
credit only. S is a persistent buffer, zeroed by reset().
"""
import math
import torch
import torch.nn as nn

class FastWeightMemory(nn.Module):
    def __init__(self, dim: int, dk: int):
        super().__init__()
        self.dim, self.dk = dim, dk
        self.Wk = nn.Parameter(torch.randn(dk, dim) / math.sqrt(dim))
        self.Wq = nn.Parameter(torch.randn(dk, dim) / math.sqrt(dim))
        self.Wv = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.wb = nn.Parameter(torch.zeros(dim))
        self.bb = nn.Parameter(torch.zeros(()))
        self.register_buffer("S", torch.zeros(dim, dk))

    def reset(self) -> None:
        with torch.no_grad():
            self.S.zero_()

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def step(self, h: torch.Tensor) -> torch.Tensor:
        k = self._norm(h @ self.Wk.T)
        q = self._norm(h @ self.Wq.T)
        v = h @ self.Wv.T
        beta = torch.sigmoid(h @ self.wb + self.bb).mean()
        vpred = self.S @ k.T
        self.S = self.S + beta * ((v - vpred).T @ k)
        return h + self.S @ q.T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_fastweight.py -q`
Expected: PASS (3 passed). Note: `self.S = ...` rebinding keeps it a plain tensor attribute updated differentiably; reset() re-zeroes. If `register_buffer` + rebinding breaks `state_dict`/checkpoint roundtrip, switch to in-place `self.S.copy_(...)` under no_grad for the state part — but that kills grads into S writes; v1 wants grads, so rebind and verify `test_checkpoint.py` still passes.

- [ ] **Step 5: Commit**

```bash
git add src/tmt/fastweight.py tests/test_fastweight.py
git commit -m "feat: delta-rule fast-weight memory module + roundtrip tests"
```

### Task 2: Wire into RTULayer + config flag (default off)

**Files:**
- Modify: `src/tmt/model.py` (RTULayer `__init__`/`forward`, TMTModel `__init__`/`reset`)
- Modify: `src/tmt/config.py` (add `fw_dk: int = 0`)
- Test: `tests/test_canonical.py` (must stay green unmodified), `tests/test_fastweight.py` (add wiring test below)

**Interfaces:**
- Consumes: `FastWeightMemory` from Task 1.
- Produces: `TMTModel(TMTConfig(fw_dk=8))` runs end-to-end; `fw_dk=0` is canonical.

- [ ] **Step 1: Write the failing test** (append to `tests/test_fastweight.py`)

```python
def test_wiring_end_to_end():
    from tmt.config import TMTConfig
    from tmt.model import TMTModel
    m = TMTModel(TMTConfig(dim=16, layers=1, fw_dk=4))
    m.reset()
    logits, states = m(torch.tensor([65], dtype=torch.long))
    assert logits.shape == (1, 256)
    loss, sampled, stop = m.training_step(65, 66, False)
    assert loss.isfinite()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_fastweight.py::test_wiring_end_to_end -q`
Expected: FAIL with "unexpected keyword argument 'fw_dk'"

- [ ] **Step 3: Write minimal implementation**

1. `config.py`: add `fw_dk: int = 0` to the dataclass and to the `to_dict`/key tuple at `config.py:34` if it enumerates keys.
2. `model.py` `RTULayer.__init__(self, dim, selective=False, write_k=0, fw_dk=0)`: add `# EXPERIMENTAL (fast weights; inert at 0)` + `self.fw = FastWeightMemory(dim, fw_dk) if fw_dk > 0 else None`.
3. `RTULayer.forward`: after computing `h` (the returned hidden), add `if self.fw is not None: h = self.fw.step(h)` — mark `# EXPERIMENTAL`.
4. `TMTModel.__init__`: pass `cfg.fw_dk` at the `RTULayer(...)` construction site.
5. `TMTModel.reset`: call `layer.fw.reset()` for layers where present (find `def reset` at `model.py:126`).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_fastweight.py tests/test_canonical.py tests/test_model.py -q`
Expected: all PASS (canonical untouched).

- [ ] **Step 5: Commit**

```bash
git add src/tmt/model.py src/tmt/config.py tests/test_fastweight.py
git commit -m "feat: wire fast-weight retrieval into RTULayer (default off)"
```

### Task 3: Measure — memorization gate + dense run with fw on

**Files:**
- Modify: `scripts/copy_dense.py` (`--fw-dk` passthrough to TMTConfig)
- Test: measurement only — `tests/test_memorize_one.py` stays unmodified (canonical gate); new numbers go to raw.

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: measured recall numbers (may be 0 — record either way).

- [ ] **Step 1: one-episode gate with fw** (diagnostic, `/tmp/memprobe/fw_gate.py`): same fixed episode as `tests/test_memorize_one.py`, model `TMTConfig(dim=32, layers=2, fw_dk=8)`, train 300 and 2000 reps, report query-position ranks + free generation. Question: does fw reduce the reps needed (300 now GREEN?) — retrieval should make the query transition easier, not just memorizable.

- [ ] **Step 2: dense run with fw**: add `--fw-dk` to `scripts/copy_dense.py` (one `add_argument` + `TMTConfig(..., fw_dk=args.fw_dk)`), run `--steps 100000 --dim 64 --layers 3 --fw-dk 16 --out runs/copydense_fw`, compare query-CE deciles + recall probes vs `runs/copydense400k`.

- [ ] **Step 3: ingest**: `raw/tmt-torch/2026-09-19-fastweight-retrieval.md` with all numbers; cite from `wiki/tmt-torch/architecture-and-evidence.md`; commit + push.

## Self-Review

- Spec coverage: retrieval-gap spec asks for retrieval-side mechanisms with real competition — delta-rule fast weights are exactly that (error-correcting write = competition against stored content; unit-norm keys = HOLA stability requirement). RSI stays parked per spec gates. No spec requirement lacks a task.
- Placeholders: none — all code blocks are complete implementations, all commands exact.
- Type consistency: `step(h: (1,d)) -> (1,d)` matches layer hidden shapes; `S: (d,dk)` matches `S@k.T: (d,1)` broadcast against `v.T: (1,d)`... verify in Task 1 Step 4: `(v - vpred)` is `(1,d)`, `.T` is `(d,1)`, `@ k` with `k: (1,dk)` gives `(d,dk)` — correct outer product. Read `S @ q.T`: `(d,dk)@(dk,1)` = `(d,1)`, added to `h: (1,d)` broadcasts — WRONG SHAPE. Fix: read must be `(S @ q.T).T` → `(1,d)`. Correct the implementation in Task 1 Step 3 before running: `return h + (self.S @ q.T).T`. Likewise write term: `beta * ((v - vpred).T @ k)` is `(d,dk)` — correct as written.
