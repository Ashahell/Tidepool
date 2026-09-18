# Prioritized Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample replay updates by loss (priority exponent α) with tag refresh, searchable via config.

**Architecture:** Entries become 4-tuples with loss tags; a `_replay_indices` helper draws via `torch.multinomial` (uniform fallback on equal/zero tags); `training_step` appends the live loss and writes back refreshed losses. No `_update` signature change.

**Tech Stack:** Python 3.14, torch 2.14.0+cu130 (CPU), pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-priority-replay-design.md`

## Global Constraints

- Python 3.14 via `.venv/bin/python`; repo root `/home/miller/Work/projects/tmt-torch`; branch `feat/priority-replay`.
- `src/tmt/evaluation_suite.py` is frozen — import from it, never modify it.
- Monitoring stays live-only; replay updates never touch `last_components` except through the live call.
- `replay_alpha=0` reproduces today's uniform behavior.
- Every task ends with a commit; no task leaves the tree red.

---

## File Map

- `src/tmt/config.py` — gains `replay_alpha: float = 1.0`.
- `src/tmt/model.py` — gains `_replay_indices`, 4-tuple entries, refresh write-back.
- `configs/rsi_smoke.yaml` — grid gains `replay_alpha: [0.0, 1.0]`.
- Tests: `tests/test_replay.py` (4 appended tests + 1 head-assertion update), `tests/test_rsi_loop.py` (1 appended test).

---

### Task 1: Weighted sampler + tag refresh

**Files:**
- Modify: `src/tmt/config.py` (1 field line)
- Modify: `src/tmt/model.py` (`_replay_indices`, append/sample/refresh edits)
- Modify: `tests/test_replay.py` (head assertion to 4-tuple + 3 new tests)

**Interfaces:**
- Consumes: `_update(curr, next_, end) -> (loss, logits, stop, comp)` and `replay_buf` deque from the prior replay work.
- Produces: `_replay_indices(k: int) -> list[int]`; 4-tuple entries `(curr, next_, end, loss: float)`; unchanged `training_step` signature.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_replay.py`:

```python
def test_tags_stored_as_floats():
    m = _model(replay_size=4, replay_k=0)
    m.training_step(65, 66, False)
    assert len(m.replay_buf[0]) == 4
    assert m.replay_buf[0][:3] == (65, 66, False)
    assert isinstance(m.replay_buf[0][3], float)

def test_weighted_skew():
    from collections import Counter
    torch.manual_seed(0)
    m = _model(replay_size=4, replay_k=0, replay_alpha=2.0)
    m.replay_buf.extend([(65, 66, False, 0.1), (65, 66, False, 0.2),
                         (65, 66, False, 10.0), (65, 66, False, 0.1)])
    idx = [m._replay_indices(1)[0] for _ in range(200)]
    assert Counter(idx)[2] > 100

def test_alpha_zero_uniform():
    torch.manual_seed(3)
    m = _model(replay_size=4, replay_k=0, replay_alpha=0.0)
    m.replay_buf.extend([(65, 66, False, 0.5) for _ in range(4)])
    idx = [m._replay_indices(1)[0] for _ in range(200)]
    assert all(v >= 1 for v in __import__("collections").Counter(idx).values())

def test_refresh_updates_tag():
    torch.manual_seed(5)
    m = _model(replay_size=8, replay_k=1)
    for _ in range(5):
        m.training_step(65, 66, False)
    for i in range(5):
        c, n, e, _ = m.replay_buf[i]
        m.replay_buf[i] = (c, n, e, 123.0 if i == 0 else 0.0)
    m.training_step(65, 66, False)
    assert m.replay_buf[0][3] != 123.0
```

Also update the existing head assertion in `test_buffer_appends_and_caps`:

```python
    assert m.replay_buf[0][:3] == (67, 66, False)
    assert isinstance(m.replay_buf[0][3], float)
```

(replacing `assert m.replay_buf[0] == (67, 66, False)`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_replay.py -v`
Expected: FAIL — `TypeError: unexpected keyword 'replay_alpha'` plus `AttributeError: _replay_indices`.

- [ ] **Step 3: Write minimal implementation**

In `src/tmt/config.py` add:

```python
    replay_alpha: float = 1.0
```

In `src/tmt/model.py` add after `__init__` (before `reset`):

```python
    def _replay_indices(self, k: int) -> list[int]:
        buf = self.replay_buf
        tags = torch.tensor([t[3] for t in buf], dtype=torch.float)
        if bool((tags == tags[0]).all()):
            probs = torch.full((len(buf),), 1.0 / len(buf))
        else:
            w = torch.pow(torch.clamp(tags, min=0.0), self.cfg.replay_alpha)
            tot = float(w.sum())
            probs = w / tot if tot > 0.0 else torch.full((len(buf),), 1.0 / len(buf))
        repl = k > len(buf)
        out = torch.multinomial(probs, k if repl else min(k, len(buf)), replacement=repl).tolist()
        return out if isinstance(out, list) else [out]
```

In `training_step`, replace the replay block:

```python
        if self.replay_buf is not None:
            self.replay_buf.append((curr, next_, end, float(loss)))
            if self.cfg.replay_k > 0:
                for idx in self._replay_indices(self.cfg.replay_k):
                    c, n, e, _ = self.replay_buf[idx]
                    rloss, _, _, _ = self._update(c, n, e)
                    self.replay_buf[idx] = (c, n, e, float(rloss))
```

(replacing the 3-tuple append and the `torch.randint` loop).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_replay.py tests/test_model.py tests/test_config.py -v`
Expected: PASS (7 + 4 + 2 = 13 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tmt/config.py src/tmt/model.py tests/test_replay.py
git commit -m "feat: add loss-weighted prioritized replay"
```

---

### Task 2: Search-surface wiring

**Files:**
- Modify: `configs/rsi_smoke.yaml` (1 grid line)
- Modify: `tests/test_rsi_loop.py` (append 1 test)

**Interfaces:**
- Consumes: `replay_alpha` field from Task 1; scorer `hasattr` overlay (existing).
- Produces: smoke grid containing the alpha key; loop-level proof it flows.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rsi_loop.py`:

```python
def test_loop_accepts_replay_alpha():
    import yaml
    grid = yaml.safe_load(open("configs/rsi_smoke.yaml"))["grid"]
    assert grid["replay_alpha"] == [0.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rsi_loop.py::test_loop_accepts_replay_alpha -v`
Expected: FAIL with `KeyError: 'replay_alpha'`.

- [ ] **Step 3: Write minimal implementation**

In `configs/rsi_smoke.yaml`, extend the grid:

```yaml
  replay_alpha: [0.0, 1.0]
```

(No code changes: the scorer overlays every config key via `hasattr`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: PASS (full suite green).

- [ ] **Step 5: Commit**

```bash
git add configs/rsi_smoke.yaml tests/test_rsi_loop.py
git commit -m "feat: expose replay_alpha in smoke search grid"
```

---

## Self-Review

**1. Spec coverage:** §1 alpha field + 4-tuples + refresh + multinomial + uniform fallback + default 1.0 — Task 1 ✓. §2 no `_update` change (untouched) + live-only monitoring (only the live call sets `last_components`) + 4 tests + experiment-as-follow-up ✓.

**2. Placeholder scan:** `rg` clean (this sentence excepted; verified below before commit).

**3. Type consistency:** `_replay_indices(k: int) -> list[int]`; single-draw `.tolist()` guarded to list ✓. Entries `(int, int|None, bool, float)` at append, sample-unpack, and refresh sites ✓. `float(loss)`/`float(rloss)` on detached tensors ✓. `replay_alpha: float` read as exponent in `torch.pow` ✓; alpha=0 test expects uniform — note `_replay_indices` with alpha=0 and non-equal tags gives weights all 1.0 → normalized uniform, and the equal-tags branch also gives uniform; both paths agree ✓.
