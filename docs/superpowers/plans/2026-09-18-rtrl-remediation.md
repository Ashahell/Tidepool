# RTRL Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the torch port algorithmically faithful: prove MLX equivalence, port the missing RTRL gradient correction, fix accumulation/clipping/checkpoints, and gate memory honestly.

**Architecture:** A NumPy transcription of the MLX equations anchors equivalence; the torch `_update` gains `retain_grad` state gradients plus the exact `dlds * trace` corrections (embed add, decay overwrite); accumulation/clip/step collapse into the correct order; checkpoints become resumable; provenance and baselines close the methodology gaps.

**Tech Stack:** Python 3.14, torch 2.14.0+cu130 (CPU, float64 in gradient tests), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-replay-design.md` is superseded for training math by this plan plus the consultant review (pasted 2026-09-18, recorded in `raw/` by the controller at execution time). Prior training findings are relabeled single-step-autograd dynamics until re-measured (Task 5).

## Global Constraints

- Python 3.14 via `.venv/bin/python`; repo root `/home/miller/Work/projects/tmt-torch`; branch `feat/rtrl-fix` (controller creates it).
- `src/tmt/evaluation_suite.py` stays frozen.
- No MLX dependency (unavailable on this host): the reference is NumPy transcribed from the upstream source.
- Every task ends with a commit; no task leaves the tree red.

---

## File Map

- `tests/mlx_reference.py` — NumPy transcription of the MLX equations (test helper, not shipped).
- `src/tmt/model.py` — RTRL correction, accumulation fix, temp rename, variance doc, groups-driven init.
- `src/tmt/engine.py` — resumable checkpoints, loud missing-file errors, strict loads.
- `src/tmt/config.py` — unknown-field rejection.
- `scripts/train.py` — resume fields, strict checkpoint errors.
- `scripts/eval_memory.py` — strict checkpoint errors.
- `scripts/compare_baselines.py` — unigram / reset-every-byte / persistent bpb.
- Tests: `tests/test_equivalence.py`, `tests/test_rtrl.py`, `tests/test_accum.py`, `tests/test_checkpoint.py`, `tests/test_memory.py`, plus appended negative/config tests.

---

### Task 0: MLX equivalence reference

**Files:**
- Create: `tests/mlx_reference.py`
- Test: `tests/test_equivalence.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `mlx_step(params, traces, curr, next_, end) -> (loss, logits, states, new_traces)` in pure NumPy, transcribed from upstream `main.py` (URL + fetch date in the file header).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_equivalence.py
import numpy as np
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tests.mlx_reference import init_params, mlx_step

def test_forward_equivalence():
    torch.manual_seed(11)
    cfg = TMTConfig(dim=4, layers=2)
    m = TMTModel(cfg)
    P = init_params(seed=11, dim=4, layers=2)
    m.load_numpy_params(P)
    traces = None
    for curr, nxt in [(65, 66), (66, 67), (67, 68)]:
        with torch.no_grad():
            logits, states = m(torch.tensor([curr], dtype=torch.long))
        (nloss, nlogits, nstates, traces, saved) = mlx_step(
            P, traces, curr, nxt, False, saved_init=(traces is None))
        assert np.max(np.abs(nlogits - logits.detach().numpy())) < 1e-5
        for a, b in zip(nstates, states):
            assert np.max(np.abs(a - b.detach().numpy())) < 1e-5
        assert abs(saved["embed_maxdiff"]) < 1e-6
```

Settling the interface contract precisely (implementer must honor it):
- `init_params(seed, dim, layers)` returns dict with keys `embed (256,dim)`, per-layer `decay (dim,)`, `weight (dim,dim)`, `ln_w (dim,)`, `ln_b (dim,)`, `dec_w (256,dim)`, `dec_b (256,)`, `stop_w (dim,1)`, `stop_b (1,)` — all float64 NumPy.
- `TMTModel.load_numpy_params(P)` (new, Task 0 adds it to model.py): copies each array into the matching torch param/buffer (`encoder.embed.weight`, `layers[i].decay_bias`, `layers[i].weights.weight`, `layers[i].norm.weight/bias`, `decoder.decode.weight/bias`, `decoder.stop.weight/bias`), casts to model dtype, resets traces. Raises `ValueError` on shape mismatch.
- `mlx_step(P, traces, curr, next_, end, saved_init)` transcribes: `enc = embed[curr]`; per layer `decay = sigmoid(decay_bias)`; `state = decay*states + enc` (dummy zero); `x = x + silu((norm(state))*weight)` with LayerNorm eps 1e-5; decoder logits + sigmoid stop head; losses variance (`max(0,1-sqrt(var+1e-4))`, population var), pred MSE vs `embed[next_]`, CE `-logits[next_]+logsumexp`, stop MSE; traces `embedtrace/decaytrace` exactly as upstream; returns updated traces plus `saved` dict carrying `embed_maxdiff` (max abs diff between the embed row read and `P["embed"][curr]`, must be ~0 — guards the embedding lookup itself).
- Loss cross-check tolerance 1e-3 (MLX `var` ddof convention hedge); states/logits 1e-5; embedding 1e-6.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_equivalence.py -v`
Expected: FAIL with "No module named 'tests.mlx_reference'" (then `load_numpy_params` missing after the helper lands — two-stage RED is fine, record both).

- [ ] **Step 3: Write minimal implementation**

`tests/mlx_reference.py` (~90 lines NumPy, header cites `https://github.com/jrz97619761/test-model-thing/blob/main/main.py` fetched 2026-09-18). `load_numpy_params` on `TMTModel` (~20 lines, `torch.no_grad`, `ValueError` on shape mismatch).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_equivalence.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/mlx_reference.py tests/test_equivalence.py src/tmt/model.py
git commit -m "test: add MLX-equation NumPy reference with forward equivalence"
```

---

### Task 1: RTRL correction + finite-difference proof

**Files:**
- Modify: `src/tmt/model.py` (`retain_grad`, `dlds` corrections, trace order)
- Test: `tests/test_rtrl.py`

**Interfaces:**
- Consumes: `_update` from current code; `load_numpy_params` from Task 0.
- Produces: corrected `_update` where `decay_bias.grad` is overwritten by `dlds * new_decaytrace` and `encoder.embed.grad` gains `Σ_layers dlds * (old_embedtrace * decay)`; all other grads pure single-step autograd.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rtrl.py
import numpy as np
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def _seq_model():
    cfg = TMTConfig(dim=3, layers=1, update_every=1000, decay_groups=1)
    m = TMTModel(cfg).double()
    torch.manual_seed(4)
    for p in m.parameters():
        torch.nn.init.uniform_(p, -0.5, 0.5)
    m.reset()
    return m

def _total_loss(m, seq):
    m.reset()
    tot = 0.0
    for i in range(len(seq) - 1):
        loss, _, _ = m.training_step(seq[i], seq[i + 1], i == len(seq) - 2)
        tot += float(loss.item())
    return tot

def _fd_grad(m, seq, param, eps=1e-5):
    base = _total_loss(m, seq)
    g = torch.zeros_like(param.detach(), dtype=torch.float64)
    flat_p, flat_g = param.detach().reshape(-1), g.reshape(-1)
    for j in range(flat_p.numel()):
        orig = flat_p[j].item()
        flat_p[j] = orig + eps
        with torch.no_grad():
            lp = _total_loss(m, seq)
        flat_p[j] = orig - eps
        with torch.no_grad():
            lm = _total_loss(m, seq)
        flat_p[j] = orig
        flat_g[j] = (lp - lm) / (2 * eps)
    return g

def test_rtrl_matches_finite_differences():
    m = _seq_model()
    seq = [65, 66, 67, 68, 69, 70]
    _total_loss(m, seq)  # accumulates RTRL grads, update_every=1000 so no step
    for name, p in [("embed", m.encoder.embed.weight),
                    ("decay", m.layers[0].decay_bias),
                    ("w", m.layers[0].weights.weight)]:
        got = p.grad.detach().clone().double()
        ref = _fd_grad(m, seq, p)
        rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-12)
        assert rel.item() < 1e-4, (name, rel.item())
```

Notes the implementer must honor: `update_every=1000` guarantees no optimizer step during the 5-byte accumulation window; `_total_loss` resets traces each call so every FD evaluation starts identically; `decay_groups=1` disables the diversity penalty (keeps the FD target exact); double precision throughout.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rtrl.py -v`
Expected: FAIL — relative error ~1.0 on decay (autograd single-step vs trace truth).

- [ ] **Step 3: Write minimal implementation**

In `_update`, after the layer loop add `state.retain_grad()` per state (implement as: `states = [s.retain_grad() or s for ...]` — no; write it plainly: loop `for s in states: s.retain_grad()` right after the forward section, before loss use). After `loss.backward()` and before the trace-update block, insert:

```python
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                dlds = states[i].grad.detach().squeeze(0)
                old_embed = layer.embedtrace.detach().clone()
                old_decay = torch.sigmoid(layer.decay_bias).detach()
                self.encoder.embed.weight.grad += dlds * (old_embed * old_decay)
                new_trace = old_decay * layer.decaytrace.detach() + old_decay * (1.0 - old_decay) * layer.states.detach().squeeze(0)
                layer.decay_bias.grad = (dlds * new_trace).clone()
```

Then the existing trace-update block runs unchanged (it recomputes the same `new_trace` into `layer.decaytrace` — keep both computations; do not merge them, the grad correction must use pre-update traces).

Edge: `encoder.embed.weight.grad` may be None if the embedding saw no use — it always sees use (`enc` feeds the graph), but guard with `if ... is not None` for the `next_=None` generation path where `_update` still runs backward (variance-only loss keeps the graph alive; keep the guard anyway). Same guard for `decay_bias.grad` before overwrite: if None, assign the trace term directly (identical code path — assignment covers both).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rtrl.py -v`
Expected: PASS (rel err < 1e-4 on all three param classes; runtime ~1–2 min, tiny model).

- [ ] **Step 5: Commit**

```bash
git add src/tmt/model.py tests/test_rtrl.py
git commit -m "feat: port MLX RTRL gradient correction with FD proof"
```

---

### Task 2: Accumulation, clipping, temp, variance doc, groups-driven init

**Files:**
- Modify: `src/tmt/model.py`, `src/tmt/config.py`
- Test: `tests/test_accum.py`

**Interfaces:**
- Consumes: corrected `_update` from Task 1.
- Produces: true gradient accumulation; clip-after-accumulate-before-step; `adaptive_temp` internal name; documented variance term; `init_decay_groups()` honoring `cfg.decay_groups` (canonical `[8, 64, 512, 4000]` when groups == 4, else log-spaced half-lives between 8 and 4000).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_accum.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def _fresh(**kw):
    cfg = TMTConfig(dim=8, layers=1, lr=0.01, decay_groups=1, **kw)
    m = TMTModel(cfg)
    torch.manual_seed(9)
    for p in m.parameters():
        torch.nn.init.uniform_(p, -0.5, 0.5)
    m.reset()
    return m

def test_accumulates_four_gradients():
    torch.manual_seed(9)
    m = _fresh(update_every=4)
    for i in range(4):
        m.training_step(65 + i, 66, False)
    stepped = [p.detach().clone() for p in m.parameters()]
    torch.manual_seed(9)
    r = _fresh(update_every=1)
    manual = None
    for i in range(4):
        r.opt.zero_grad()
        loss, _, _, _ = r._update_no_step(65 + i, 66, False)
        if manual is None:
            manual = [g.detach().clone() for g in
                      [p.grad for p in r.parameters()]]
        else:
            for acc, p in zip(manual, r.parameters()):
                acc += p.grad.detach().clone()
    torch.manual_seed(9)
    s = _fresh(update_every=4)
    s.opt.zero_grad()
    for p, g in zip(s.parameters(), manual):
        p.grad = g.clone()
    torch.nn.utils.clip_grad_norm_(s.parameters(), s.cfg.grad_clip)
    s.opt.step()
    for a, b in zip(stepped, s.parameters()):
        assert torch.allclose(a, b, atol=1e-6)

def test_adaptive_temp_monotonic():
    from tmt.model import TMTModel as M
    cfg = TMTConfig(dim=8, layers=1)
    m = M(cfg)
    assert abs(m._adaptive_temp(0.0) - 0.75) < 1e-6
    assert abs(m._adaptive_temp(1.0) - 0.1875) < 1e-6
    assert m._adaptive_temp(1.0) < m._adaptive_temp(0.5) < m._adaptive_temp(0.0)

def test_groups_drive_init():
    m = TMTModel(TMTConfig(dim=16, layers=1, decay_groups=2))
    m.init_decay_groups()
    import torch as t
    d = t.sigmoid(m.layers[0].decay_bias).detach()
    assert len(set(round(float(v), 4) for v in d.tolist())) == 2
```

This test needs two small helpers the implementer must add: `_update_no_step(curr, next_, end)` — identical to `_update` minus accumulation/step/clip (forward + backward + RTRL corrections only, returns `(loss_detached, ...)` same shape); and `_adaptive_temp(entropy: float) -> float` housing the formula currently inline in `_sample` (rename internal variable, keep `cfg.temp`).

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_accum.py -v`
Expected: FAIL — `AttributeError: _update_no_step` (then accumulation mismatch).

- [ ] **Step 3: Write minimal implementation**

Restructure the tail of `_update` exactly as:

```python
        if self._accum == 0:
            self.opt.zero_grad()
        loss.backward()
        <RTRL correction block from Task 1>
        <trace update block, unchanged>
        self._accum += 1
        if self._accum >= self.cfg.update_every:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad()
            self._accum = 0
```

`_update_no_step` = same body with the accum/step/clip tail removed (duplicate ~15 lines deliberately — the test helper must not share the stepping path). Document the variance term at its definition site:

```python
        # Activation-scale regularizer: forces per-token feature variance
        # toward >= 1 (population var across dim of the single-token state).
```

`init_decay_groups(self, half_lives=None)`: `None` → `[8.0, 64.0, 512.0, 4000.0]` if `cfg.decay_groups == 4` else log-spaced `decay_groups` values between 8 and 4000 (`math.exp` interpolation); explicit list still honored and must match `decay_groups` in length else `ValueError`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_accum.py tests/test_rtrl.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tmt/model.py src/tmt/config.py tests/test_accum.py
git commit -m "fix: true gradient accumulation with proof; temp/init hygiene"
```

---

### Task 3: Checkpoints, provenance, negative tests

**Files:**
- Modify: `src/tmt/engine.py`, `src/tmt/config.py`, `scripts/train.py`, `scripts/eval_memory.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: corrected model from Tasks 1–2.
- Produces: `save_checkpoint` persisting params + traces + optimizer + `_accum` + step + RNG states (`torch`, `random`) + config hash; `load_checkpoint(path, allow_missing=False)` raising `FileNotFoundError`, `strict=True` loads; `node_json` gaining `run_id`, `timestamp`, `git_commit`, `steps`, `bytes_seen` (all defaulted, old call sites keep working); `TMTConfig.from_yaml` raising `ValueError` on unknown keys; `training_step` raising `ValueError` on bytes outside `[0, 255]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_checkpoint.py
import pytest
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.engine import save_checkpoint, load_checkpoint

def _trained(tmp_path, steps=3):
    m = TMTModel(TMTConfig(dim=8, layers=1))
    torch.manual_seed(5)
    for i in range(steps):
        m.training_step(65 + i, 66, False)
    p = str(tmp_path / "c.safetensors")
    meta = {"step": steps, "bytes_seen": steps}
    save_checkpoint(m, p, meta=meta)
    return m, p

def test_resume_equals_uninterrupted(tmp_path):
    m, p = _trained(tmp_path, 3)
    m2 = TMTModel(TMTConfig(dim=8, layers=1))
    load_checkpoint(m2, p)
    for i in range(3, 6):
        m.training_step(65 + i, 66, False)
        m2.training_step(65 + i, 66, False)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)

def test_missing_checkpoint_raises(tmp_path):
    m = TMTModel(TMTConfig(dim=8, layers=1))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(m, str(tmp_path / "nope.safetensors"))

def test_corrupt_checkpoint_raises(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(b"not a safetensors file at all" * 10)
    m = TMTModel(TMTConfig(dim=8, layers=1))
    with pytest.raises(Exception):
        load_checkpoint(m, str(p))

def test_incompatible_arch_raises(tmp_path):
    m, p = _trained(tmp_path, 1)
    m2 = TMTModel(TMTConfig(dim=16, layers=1))
    with pytest.raises(Exception):
        load_checkpoint(m2, p)

def test_bad_byte_rejected():
    m = TMTModel(TMTConfig(dim=8, layers=1))
    with pytest.raises(ValueError):
        m.training_step(300, 66, False)

def test_unknown_config_field_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dim: 16\nnonsense_field: 1\n")
    with pytest.raises(ValueError):
        TMTConfig.from_yaml(str(p))
```

`save_checkpoint(model, path, meta: dict|None = None)` — meta keys `step`, `bytes_seen` stored under `meta.*` prefix; optimizer state under `opt.*` (via `torch.save`-compatible tensors — implementer: serialize `model.opt.state_dict()` tensors with `opt.{k}` keys plus non-tensor step counters as float tensors; RNG via `torch.get_rng_state()` + `random.getstate()` (store `random` state as int64 tensor of the MT array); load restores all and returns the meta dict. `load_checkpoint` signature gains `allow_missing=False`; `scripts/train.py` passes `allow_missing=True` (fresh runs); `eval_memory.py` keeps strict default. `node_json` gains keyword-only `run_id=None` (generates 8-hex), `timestamp=None` (UTC ISO), `git_commit=None` (best-effort `git rev-parse --short HEAD`, `"unknown"` on failure), `steps=0`, `bytes_seen=0`; train.py passes real values.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL — `TypeError: unexpected keyword 'meta'`, no `FileNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Per the interface paragraph above. RNG restore: `torch.set_rng_state`, `random.setstate` (guard malformed with `ValueError`). Optimizer restore: `model.opt.load_state_dict` rebuilt from `opt.*` keys (tensors) — moments are per-param float tensors, restorable exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_checkpoint.py tests/test_rtrl.py tests/test_accum.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tmt/engine.py src/tmt/config.py src/tmt/model.py scripts/train.py scripts/eval_memory.py tests/test_checkpoint.py
git commit -m "fix: resumable checkpoints, provenance, loud failures"
```

---

### Task 4: Baselines + memory sensitivity gate

**Files:**
- Create: `scripts/compare_baselines.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `evaluate_model`, `EvalConfig` (frozen suite).
- Produces: `scripts/compare_baselines.py --val FILE --out FILE` printing bpb for unigram / reset-every-byte TMT / persistent TMT on the same bytes; `test_memory.py` proving the copy probe separates perfect-memory from amnesiac stub models.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory.py
import torch
from tmt.evaluation_suite import EvalConfig, evaluate_model

class Amnesiac:
    def reset(self): pass
    def __call__(self, x):
        torch.manual_seed(0)
        return torch.randn(1, 256), [torch.randn(1, 8)]

class PerfectEcho:
    """Returns the byte it was shown, one step delayed (perfect 1-byte memory)."""
    def __init__(self):
        self.prev = 0
    def reset(self): self.prev = 0
    def __call__(self, x):
        cur = int(x[0].item())
        logits = torch.full((1, 256), -1e9)
        logits[0, self.prev] = 0.0
        self.prev = cur
        return logits, None

def _cfg():
    return EvalConfig(max_bytes=600, lm_eval_tokens=64, copy_lengths=[4],
                      intervening_lengths=[8], num_copy_trials=2,
                      retention_eval_bytes=64)

def test_probe_separates_memory_from_amnesia():
    val = bytes((65 + i % 26) for i in range(256))
    r_bad = evaluate_model(Amnesiac(), val, [val], cfg=_cfg(),
                           config_dict={}, gpu_hours=0.0)
    r_good = evaluate_model(PerfectEcho(), val, [val], cfg=_cfg(),
                            config_dict={}, gpu_hours=0.0)
    assert r_good.long_range_score > r_bad.long_range_score
    assert r_good.val_bpb < r_bad.val_bpb
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_memory.py -v`
Expected: FAIL with "No module named ..." — no; the suite exists, so this test PASSES immediately (it only exercises the frozen suite). Record that honestly in the report (no RED possible; the value is the sensitivity proof itself) and proceed — do not invent failure.

- [ ] **Step 3: Write minimal implementation**

`scripts/compare_baselines.py` (~60 lines): `--val` file in, JSON out; unigram = NumPy byte counts with +1 smoothing, bpb on the file; reset-every-byte TMT = `TMTModel` with `model.reset()` before every `model(byte)` call inside `compute_bpb`-equivalent loop (import and reuse `compute_bpb` by wrapping: subclass or closure resetting per byte — implement as a `ResetWrapper` class in the script holding a model and resetting before each call); persistent TMT = plain calls. Print all three bpb values.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_memory.py -v` plus `PYTHONPATH=src .venv/bin/python scripts/compare_baselines.py --val /tmp/rsi_data/val_heldout.bin` (any readable bytes file; prints three bpb numbers).
Expected: PASS + three printed numbers.

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_baselines.py tests/test_memory.py
git commit -m "feat: baseline comparisons and memory sensitivity gate"
```

---

## Self-Review

**1. Spec coverage:** equivalence Task 0 ✓; RTRL correction + FD proof (review §1–4,7) Task 1 ✓; accumulation + clip + temp + variance doc + groups-init (review §2,3,18,19,21) Task 2 ✓; checkpoint/resume + provenance + negative tests (review §11–15,22-partial) Task 3 ✓; baselines + memory gate + state-isolation test (review §5-gates, §8, §9, §22-partial, §23) Task 4 ✓. State-reset isolation: covered by Task 4's sensitivity framing? Add explicitly — the `PerfectEcho.reset` + `Amnesiac.reset` exercise reset paths; plus Task 3's resume test covers state persistence. Acceptable. Chat-loop plan staleness (§16) and snippet indentation (§17): plan-doc hygiene, controller fixes directly at execution (not agent tasks).

**2. Placeholder scan:** `rg` clean required before commit (run it).

**3. Type consistency:** `load_numpy_params(P: dict)` keys fixed in Task 0 ✓ used by nothing later (FD test builds torch models directly) — no drift surface. `_update_no_step` returns same 4-tuple as `_update` ✓. `save_checkpoint(model, path, meta=None)` — old 2-arg calls keep working ✓. `load_checkpoint(model, path, allow_missing=False)` — old 2-arg calls keep working, now loud ✓. `node_json` new kwargs all defaulted ✓.
