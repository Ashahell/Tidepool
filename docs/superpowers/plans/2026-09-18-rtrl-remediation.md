# RTRL Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the torch port algorithmically faithful: prove MLX equivalence, port the missing RTRL gradient correction, fix accumulation/clipping/checkpoints, and gate memory honestly.

**Architecture:** A NumPy transcription of the MLX equations anchors equivalence; the torch `_update` gains `retain_grad` state gradients plus the exact `dlds * trace` corrections (embed add, decay overwrite); accumulation/clip/step collapse into the correct order; checkpoints become resumable; provenance and baselines close the methodology gaps.

**Tech Stack:** Python 3.14, torch 2.14.0+cu130 (CPU, float64 in gradient tests), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-replay-design.md` is superseded for training math by this plan plus the consultant review (pasted 2026-09-18, recorded in `raw/` by the controller at execution time). Prior training findings are relabeled single-step-autograd dynamics until re-measured (Task 5).

## Global Constraints

- Python 3.14 via `.venv/bin/python`; repo root `/home/miller/Work/projects/tmt-torch`; branch `feat/rtrl-fix` (controller creates it at execution).
- `src/tmt/evaluation_suite.py` stays frozen.
- No MLX dependency (unavailable on this host): the reference is NumPy transcribed from the upstream source.
- Every task ends with a commit; no task leaves the tree red.

---

## Non-Negotiable Invariants

These are the real specification. Each has a test that can actually fail.

- Invariant 1 — Source-level forward equivalence: NumPy transcription == PyTorch (embeddings, decay, states, logits, loss within stated tolerances). Called source-level, not MLX, because both derive from one human transcription.
- Invariant 2 — Persistent state equivalence: `layer.states`, `layer.embedtrace`, `layer.decaytrace` match the reference after every timestep.
- Invariant 3 — Gradient equivalence: RTRL gradients == finite differences at 1, 2, and 5 steps, for every parameter class.
- Invariant 4 — Accumulation equivalence: N accumulated steps == explicit gradient sum followed by one clip+step.
- Invariant 5 — Reset equivalence: `reset(); sequence` == `fresh_model(); sequence` (outputs and states).
- Invariant 6 — Resume equivalence: `train(N)` == `train(K) + checkpoint + train(N-K)` (weights bit-equal) AND dataset cursor continues (no byte reprocessed).
- Invariant 7 — Memory oracle separation: perfect-memory model scores high, one-byte model partial, amnesiac/random score ~0 — on exact-bytes-recovered, never on bpb alone.

STOP THE IMPLEMENTATION IF: state mismatch > tolerance; FD mismatch > tolerance; resume != uninterrupted; reset leaks state. Never adjust a test to fit the implementation — make the implementation satisfy the test.

---

## File Map

- `tests/mlx_reference.py` — NumPy transcription of the MLX equations (test helper, not shipped).
- `src/tmt/model.py` — RTRL correction, accumulation fix, temp rename, variance doc, groups-driven init.
- `src/tmt/engine.py` — resumable checkpoints, loud missing-file errors, strict loads.
- `src/tmt/config.py` — unknown-field rejection.
- `scripts/train.py` — resume fields, strict checkpoint errors.
- `scripts/eval_memory.py` — strict checkpoint errors.
- `scripts/compare_baselines.py` — uniform / unigram / reset-every-byte / persistent bpb.
- `scripts/remeasure.py` — RTRL ablation matrix runner (Task 5).
- Tests: `tests/test_equivalence.py`, `tests/test_rtrl.py`, `tests/test_accum.py`, `tests/test_checkpoint.py`, `tests/test_memory.py`, plus appended negative/config tests.

---

### Task 0: Source-level equivalence reference

**Files:**
- Create: `tests/mlx_reference.py`
- Test: `tests/test_equivalence.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `mlx_step(params, traces, curr, next_, end) -> (loss, logits, states, new_traces)` in pure NumPy, transcribed from upstream `main.py` (URL + fetch date in the file header). Called source-level equivalence: both implementations derive from one human transcription, so this proves PyTorch ≈ transcription, not PyTorch ≈ MLX runtime.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_equivalence.py
import numpy as np
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tests.mlx_reference import init_params, mlx_step

def test_reference_embedding_lookup():
    P = init_params(seed=11, dim=4, layers=2)
    assert np.max(np.abs(P["embed"][65] - P["embed"][65])) == 0.0
    assert P["embed"].shape == (256, 4)

def test_forward_equivalence():
    torch.manual_seed(11)
    cfg = TMTConfig(dim=4, layers=2, decay_groups=1)
    m = TMTModel(cfg)
    P = init_params(seed=11, dim=4, layers=2)
    m.load_numpy_params(P)
    traces = None
    for curr, nxt in [(65, 66), (66, 67), (67, 68)]:
        with torch.no_grad():
            logits, states = m(torch.tensor([curr], dtype=torch.long))
        (nloss, nlogits, nstates, traces) = mlx_step(P, traces, curr, nxt, False)
        assert np.max(np.abs(nlogits - logits.detach().numpy())) < 1e-5
        for a, b in zip(nstates, states):
            assert np.max(np.abs(a - b.detach().numpy())) < 1e-5
        for li, layer in enumerate(m.layers):
            assert np.max(np.abs(traces["states"][li] - layer.states.detach().numpy())) < 1e-5
            assert np.max(np.abs(traces["embedtrace"][li] - layer.embedtrace.detach().numpy())) < 1e-5
            assert np.max(np.abs(traces["decaytrace"][li] - layer.decaytrace.detach().numpy())) < 1e-5
```

Settling the interface contract precisely (implementer must honor it):
- `init_params(seed, dim, layers)` returns dict with keys `embed (256,dim)`, per-layer `decay (dim,)`, `weight (dim,dim)`, `ln_w (dim,)`, `ln_b (dim,)`, `dec_w (256,dim)`, `dec_b (256,)`, `stop_w (dim,1)`, `stop_b (1,)` — all float64 NumPy.
- `TMTModel.load_numpy_params(P)` (new, Task 0 adds it to model.py): copies each array into the matching torch param/buffer (`encoder.embed.weight`, `layers[i].decay_bias`, `layers[i].weights.weight`, `layers[i].norm.weight/bias`, `decoder.decode.weight/bias`, `decoder.stop.weight/bias`), casts to model dtype, resets traces. Raises `ValueError` on shape mismatch.
- `mlx_step(P, traces, curr, next_, end)` transcribes: `enc = embed[curr]`; per layer `decay = sigmoid(decay_bias)`; `state = decay*states + enc` (dummy zero, `traces=None` means zero states/traces); `x = x + silu((norm(state))*weight)` with LayerNorm eps 1e-5; decoder logits + sigmoid stop head; losses variance (`max(0,1-sqrt(var+1e-4))`, population var), pred MSE vs `embed[next_]`, CE `-logits[next_]+logsumexp`, stop MSE; traces `embedtrace/decaytrace` exactly as upstream; returns `(loss, logits, states, new_traces)` where `new_traces` is a dict with `states`, `embedtrace`, `decaytrace` per-layer lists.
- Loss cross-check tolerance 1e-3 (MLX `var` ddof convention hedge); states/logits/persistent traces 1e-5.
- NOTE: order matters and the first draft of this step had it backwards (training_step first compares torch-from-new-buffers against NumPy-from-old-traces — a guaranteed sequencing artifact). Canonical order per step: `mlx_step` + pure `m(...)` from the SAME old state (compare logits/states), then `training_step` under `update_every=1000000` (both advance; compare buffers + loss). The landed test in `tests/test_equivalence.py` is authoritative.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_equivalence.py -v`
Expected: FAIL with "No module named 'tests.mlx_reference'" (then `load_numpy_params` missing after the helper lands — two-stage RED is fine, record both).

- [ ] **Step 3: Write minimal implementation**

`tests/mlx_reference.py` (~90 lines NumPy, header cites `https://github.com/jrz97619761/test-model-thing/blob/main/main.py` fetched 2026-09-18). `load_numpy_params` on `TMTModel` (~20 lines, `torch.no_grad`, `ValueError` on shape mismatch).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_equivalence.py -v`
Expected: PASS (2 passed).

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

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rtrl.py
import numpy as np
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

# Gradient classification (also documented in model.py above the
# correction block): embedding + decay_bias carry temporal influence
# through persistent state and get RTRL corrections; recurrent weight,
# LayerNorm, decoder, and stop-head affect only the current step, so
# plain autograd is exact for them. Every class below must match FD.

def _seq_model():
    cfg = TMTConfig(dim=3, layers=1, update_every=1000, decay_groups=1,
                    grad_clip=float("inf"))
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
    # NO no_grad wrapper here: _total_loss runs training_step, whose
    # backward() needs an enabled grad graph (a no_grad wrapper raises
    # RuntimeError). No optimizer step occurs (update_every=1000).
    g = torch.zeros_like(param.detach(), dtype=torch.float64)
    flat_p, flat_g = param.detach().reshape(-1), g.reshape(-1)
    for j in range(flat_p.numel()):
        orig = flat_p[j].item()
        flat_p[j] = orig + eps
        lp = _total_loss(m, seq)
        flat_p[j] = orig - eps
        lm = _total_loss(m, seq)
        flat_p[j] = orig
        flat_g[j] = (lp - lm) / (2 * eps)
    return g

def _all_params(m):
    L = m.layers[0]
    return {"embed": m.encoder.embed.weight, "decay": L.decay_bias,
            "w": L.weights.weight, "ln_w": L.norm.weight,
            "ln_b": L.norm.bias, "dec_w": m.decoder.decode.weight,
            "dec_b": m.decoder.decode.bias, "stop_w": m.decoder.stop.weight,
            "stop_b": m.decoder.stop.bias}

def _check(seq, names):
    m = _seq_model()
    _total_loss(m, seq)  # accumulates grads, update_every=1000 so no step
    skip_rows = set(seq[1:])  # pred-target rows: upstream stop_gradient
    for name in names:  # excludes them, so FD truth contains a target-role
        p = _all_params(m)[name]  # term no faithful implementation reports
        got = p.grad.detach().clone().double()
        ref = _fd_grad(m, seq, p)
        if name == "embed":
            mask = torch.ones(ref.shape[0], dtype=torch.bool)
            mask[list(skip_rows)] = False
            got, ref = got[mask], ref[mask]
        rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-12)
        assert rel.item() < 1e-4, (name, len(seq) - 1, rel.item())

def test_rtrl_one_step_all_classes():
    _check([65, 66], list(_all_params(_seq_model()).keys()))

def test_rtrl_two_step_all_classes():
    _check([65, 66, 67], list(_all_params(_seq_model()).keys()))

def test_rtrl_five_step_temporal_classes():
    _check([65, 66, 67, 68, 69, 70], ["embed", "decay", "w"])
```

Notes the implementer must honor: `update_every=1000` guarantees no optimizer step during accumulation; `_total_loss` resets traces each call so every FD evaluation starts identically; `decay_groups=1` disables the diversity penalty (keeps the FD target exact); double precision throughout; runtime a few minutes (tiny model).

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rtrl.py -v`
Expected: FAIL — relative error ~1.0 on decay and embed at 2+ steps (autograd single-step vs trace truth); one-step may partially pass.

- [ ] **Step 3: Write minimal implementation**

In `_update`, after the layer loop add `state.retain_grad()` per state (write it plainly: loop `for s in states: s.retain_grad()` right after the forward section, before loss use). After `loss.backward()` and before the trace-update block, insert the classification comment plus corrections:

```python
        # Gradient classification: embedding + decay_bias influence future
        # steps through persistent state -> RTRL trace corrections below.
        # weights/LayerNorm/decoder/stop-head affect only the current step
        # -> plain autograd is exact, no correction.
        # _rtrl_enabled (default True) is the Task 5 ablation gate.
        with torch.no_grad():
            if getattr(self, "_rtrl_enabled", True):
                for i, layer in enumerate(self.layers):
                    dlds = states[i].grad.detach().squeeze(0)
                    old_embed = layer.embedtrace.detach().clone()
                    old_decay = torch.sigmoid(layer.decay_bias).detach()
                    self.encoder.embed.weight.grad += dlds * (old_embed * old_decay)
                    new_trace = old_decay * layer.decaytrace.detach() + old_decay * (1.0 - old_decay) * layer.states.detach().squeeze(0)
                    layer.decay_bias.grad = (dlds * new_trace).clone()
```

Then the existing trace-update block runs unchanged (it recomputes the same `new_trace` into `layer.decaytrace` — keep both computations; do not merge them, the grad correction must use pre-update traces).

Finally, restructure the accumulation tail in the same edit (required: per-step `zero_grad` would wipe the accumulation the FD proof measures):

```python
        if self._accum == 0:
            self.opt.zero_grad()
        loss.backward()
```

moves BEFORE the RTRL correction block (i.e. zero-once at the top instead of zero-every-call), and the tail becomes:

```python
        self._accum += 1
        if self._accum >= self.cfg.update_every:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad()
            self._accum = 0
```

(clip-after-accumulate-before-step; update_every is sequential accumulation across evolving timesteps, NOT a minibatch — state keeps evolving, never detach/reset at accumulation boundaries).

Edge: `encoder.embed.weight.grad` may be None if the embedding saw no use — it always sees use (`enc` feeds the graph), but guard with `if ... is not None` for the `next_=None` generation path where `_update` still runs backward (variance-only loss keeps the graph alive; keep the guard anyway). Same guard for `decay_bias.grad` before overwrite: if None, assign the trace term directly (identical code path — assignment covers both).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rtrl.py -v`
Expected: PASS (3 tests, rel err < 1e-4 every class; runtime a few minutes, tiny model).

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
    # update_every means sequential accumulation across timesteps while
    # state evolves (NOT a minibatch reset): gradients from 4 successive
    # bytes sum, then one clip+step. Proved by comparing against an
    # explicit out-of-model accumulation (no second model path).
    torch.manual_seed(9)
    m = _fresh(update_every=4)
    for i in range(4):
        m.training_step(65 + i, 66, False)
    stepped = [p.detach().clone() for p in m.parameters()]
    torch.manual_seed(9)
    r = _fresh(update_every=1000)
    r.opt.zero_grad()
    for i in range(4):
        r.training_step(65 + i, 66, False)
    manual = [p.grad.detach().clone() for p in r.parameters()]
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
    import math
    for groups, want in [(2, [8.0, 4000.0]),
                         (3, [8.0, math.sqrt(8.0 * 4000.0), 4000.0]),
                         (4, [8.0, 64.0, 512.0, 4000.0])]:
        m = TMTModel(TMTConfig(dim=16, layers=1, decay_groups=groups))
        m.init_decay_groups()
        import torch as t
        d = t.sigmoid(m.layers[0].decay_bias).detach()
        got = sorted(set(round(float(v), 4) for v in d.tolist()))
        assert len(got) == groups
        for g, h in zip(got, sorted(want)):
            # sigmoid(decay_bias) must equal the per-step decay 0.5**(1/h)
            assert abs(g - round(0.5 ** (1.0 / h), 4)) < 1e-3, (groups, g, h)

def test_reset_isolates_sequences():
    # Losses (not samples: sampling consumes RNG) and persistent states
    # must be identical for identical post-reset sequences, regardless of
    # what ran before the reset.
    def run(m, seq):
        losses, states = [], []
        for b in seq:
            loss, _, _ = m.training_step(b, b + 1, False)
            losses.append(float(loss.item()))
            states.append(m.layers[0].states.detach().clone())
        return losses, states
    # update_every=1000: no optimizer steps, so weights stay identical and
    # only state/traces evolve — isolates the reset semantics under test.
    m = _fresh(update_every=1000)
    run(m, (65, 66, 67))
    m.reset()
    losses_a, states_a = run(m, (70, 71, 72))
    m2 = _fresh(update_every=1000)
    losses_b, states_b = run(m2, (70, 71, 72))
    assert losses_a == losses_b
    for a, b in zip(states_a, states_b):
        assert torch.equal(a, b)
    m.reset()
    losses_c, states_c = run(m, (70, 71, 72))
    assert losses_c == losses_a
    for a, c in zip(states_a, states_c):
        assert torch.equal(a, c)
```

This test needs one small helper the implementer must add: `_adaptive_temp(entropy: float) -> float` housing the formula currently inline in `_sample` (rename internal variable, keep `cfg.temp`). No second training path is introduced: the accumulation proof works entirely through public `training_step` plus torch ops in the test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_accum.py -v`
Expected: FAIL — `AttributeError: _adaptive_temp`, accumulation mismatch, half-life mismatch, reset leakage accepted silently.

- [ ] **Step 3: Write minimal implementation**

The accumulation tail was already restructured in Task 1 (zero-on-first, clip-after-accumulate-before-step); verify it is intact, do not rework it. This task adds: `_adaptive_temp`, variance doc comment, `init_decay_groups` groups logic, `__post_init__` validation. Restructure nothing else in `_update`.

Document the variance term at its definition site:

```python
        # Activation-scale regularizer: forces per-token feature variance
        # toward >= 1 (population var across dim of the single-token state).
```

`init_decay_groups(self, half_lives=None)`: `None` → `[8.0, 64.0, 512.0, 4000.0]` if `cfg.decay_groups == 4` else log-spaced `decay_groups` values between 8 and 4000 (`math.exp` interpolation); explicit list still honored and must match `decay_groups` in length else `ValueError`.

`TMTConfig.__post_init__`: reject non-positive `dim`, `layers`, `lr`, `temp`, `update_every`, `grad_clip`, `decay_groups`, and negative `replay_size`/`replay_k`/`ema_decay` with `ValueError`; `from_yaml` rejects unknown keys the same way.

```python
def test_config_validation(tmp_path):
    import pytest
    from tmt.config import TMTConfig
    p = tmp_path / "c.yaml"
    p.write_text("dim: 16\nnonsense_field: 1\n")
    with pytest.raises(ValueError):
        TMTConfig.from_yaml(str(p))
    for bad in [dict(dim=-10), dict(layers=0), dict(lr=-1.0),
                dict(temp=0.0), dict(update_every=0), dict(decay_groups=-4),
                dict(grad_clip=-1.0), dict(replay_size=-1)]:
        with pytest.raises(ValueError):
            TMTConfig(**{"dim": 16, "layers": 1, **bad}})
```

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
- Produces: two-file checkpoints — `<base>.safetensors` (params + traces, strict-loadable) + `<base>.state.pt` (`torch.save`: optimizer `state_dict`, `_accum`, `step`, `bytes_seen`, torch/CUDA/Python/NumPy RNG states, config dict, git commit); `save/load_checkpoint(model, base_path, ...)` with `meta` dict and `allow_missing=False` (train.py passes `allow_missing=True` for fresh runs; eval stays strict); `node_json` gaining `run_id`, `timestamp`, `git_commit`, `steps`, `bytes_seen` (all defaulted); `training_step` raising `ValueError` on bytes outside `[0, 255]`; `data.skip_bytes(root, n)` byte cursor.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checkpoint.py
import random
import numpy as np
import pytest
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import skip_bytes
from tmt.engine import save_checkpoint, load_checkpoint

def _seed_all(s=5):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def _trained(tmp_path, steps=3):
    _seed_all()
    m = TMTModel(TMTConfig(dim=8, layers=1))
    for i in range(steps):
        m.training_step(65 + i, 66, False)
    p = str(tmp_path / "c")
    save_checkpoint(m, p, meta={"step": steps, "bytes_seen": steps})
    return m, p

def test_model_state_resume_equals_uninterrupted(tmp_path):
    m, p = _trained(tmp_path, 3)
    m2 = TMTModel(TMTConfig(dim=8, layers=1))
    load_checkpoint(m2, p)
    for i in range(3, 6):
        m.training_step(65 + i, 66, False)
        m2.training_step(65 + i, 66, False)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)

def test_dataset_cursor_resume(tmp_path):
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "wiki_a").write_text("".join(f"line{i}\n" for i in range(8)))
    first = b"".join(list(skip_bytes(str(src), 0))[:4])
    rest = b"".join(list(skip_bytes(str(src), 4))[:4])
    assert first + rest == b"".join(list(skip_bytes(str(src), 0))[:8])
    assert len(rest) > 0

def test_missing_checkpoint_raises(tmp_path):
    m = TMTModel(TMTConfig(dim=8, layers=1))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(m, str(tmp_path / "nope"))

def test_corrupt_checkpoint_raises(tmp_path):
    p = tmp_path / "bad"
    (tmp_path / "bad.safetensors").write_bytes(b"not safetensors" * 10)
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
```

`save_checkpoint(model, base_path, meta: dict|None = None)`: `base_path` may carry a legacy `.safetensors`/`.pt` suffix — strip it to the base first, so every existing call site (`runs/model.safetensors`, old tests) keeps working unchanged. Writes `<base>.safetensors` (params under `m.` + traces, `.cpu().clone()`, strict-loadable — no aliases) and `<base>.state.pt` via `torch.save` holding `optimizer` state_dict, `_accum`, meta (`step`, `bytes_seen`, `cursor_bytes`), `torch_rng` (`torch.get_rng_state()`), `cuda_rng` (`torch.cuda.get_rng_state_all()` or None), `python_rng` (`random.getstate()`), `numpy_rng` (`np.random.get_state()`), `config` dict, `git_commit`. `load_checkpoint(model, base_path, allow_missing=False)`: missing base → `FileNotFoundError` unless allowed; `load_state_dict(..., strict=True)`; restores optimizer/`_accum`/all RNG states (`torch.set_rng_state`, `torch.cuda.set_rng_state_all` guarded, `random.setstate`, `np.random.set_state`); returns the meta dict. `data.skip_bytes(root, n)`: yields the byte stream of `iter_wikipedia_bytes(root)` with the first `n` bytes dropped (implemented as offset counting across chunks, no materialization). `scripts/train.py`: `--resume` flag loads checkpoint meta and seeks `bytes_seen` via `skip_bytes` before training (thin wiring, ~10 lines; covered by the helper test plus inspection). `node_json` gains keyword-only `run_id=None` (8-hex), `timestamp=None` (UTC ISO), `git_commit=None` (best-effort short HEAD, `"unknown"` on failure), `steps=0`, `bytes_seen=0`; train.py passes real values.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL — `TypeError: unexpected keyword 'meta'`, no `FileNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Per the interface paragraph above (two files, no bespoke tensor-packing format): safetensors for params/traces, `torch.save` for everything else. RNG restore guards malformed states with `ValueError`. Optimizer restore via `model.opt.load_state_dict` on the unpickled dict — moments restorable exactly.

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
- Consumes: `copy_memory_probe`, `EvalConfig` (frozen suite); `compute_bpb` for the script.
- Produces: `scripts/compare_baselines.py --val FILE` printing uniform / unigram / reset-every-byte TMT / persistent TMT bpb as JSON; `test_memory.py` proving the copy probe orders PerfectMemory (1.0) > OneByteMemory (0.0) = Amnesiac (0.0) on exact-bytes-recovered — asserted on `long_range_score` only, never on bpb or composite.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory.py
import numpy as np
import torch
from tmt.evaluation_suite import EvalConfig, copy_memory_probe

class _CountingOracle:
    """Knows the single-cell protocol: feed copy_len target + inter_len
    filler bytes, then emit. Subclasses decide what to emit."""
    def __init__(self, copy_len, inter_len):
        self.copy_len = copy_len
        self.inter_len = inter_len
        self.fed = []
    def reset(self):
        self.fed = []
    def _emit(self, g):
        raise NotImplementedError
    def __call__(self, x):
        cur = int(x[0].item())
        if len(self.fed) < self.copy_len + self.inter_len:
            self.fed.append(cur)
            logits = torch.full((1, 256), -1e9)
            logits[0, cur] = 0.0
            return logits, None
        g = len(self.fed) - (self.copy_len + self.inter_len)
        self.fed.append(cur)
        logits = torch.full((1, 256), -1e9)
        logits[0, self._emit(g)] = 0.0
        return logits, None

class PerfectMemory(_CountingOracle):
    def _emit(self, g):
        return self.fed[g]

class OneByteMemory(_CountingOracle):
    def _emit(self, g):
        return self.fed[0]

class Amnesiac:
    def reset(self): pass
    def __call__(self, x):
        logits = torch.full((1, 256), -1e9)
        logits[0, 0] = 0.0
        return logits, None

def _cfg():
    return EvalConfig(copy_lengths=[4], intervening_lengths=[8],
                      num_copy_trials=4)

def _score(model):
    np.random.seed(0)
    return copy_memory_probe(model, _cfg()).score

def test_oracle_ordering():
    assert _score(PerfectMemory(4, 8)) == 1.0
    assert _score(OneByteMemory(4, 8)) == 0.0
    assert _score(Amnesiac()) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_memory.py -v`
Expected: FAIL with "No module named ..." — no; the suite exists, so this test PASSES immediately (it only exercises the frozen suite). Record that honestly in the report (no RED possible; the value is the sensitivity proof itself) and proceed — do not invent failure.

- [ ] **Step 3: Write minimal implementation**

`scripts/compare_baselines.py` (~70 lines): `--val` file in, JSON out with keys `uniform`, `unigram`, `reset_every_byte`, `persistent`; uniform = 8.0 theoretical (`log2(256)`); unigram = NumPy byte counts with +1 smoothing, bpb on the file; reset-every-byte TMT = `TMTModel` with `model.reset()` before every `model(byte)` call (implement as a `ResetWrapper` class in the script holding a model and resetting before each call, then reuse the frozen `compute_bpb` on the wrapper); persistent TMT = plain calls. Print all four bpb values.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_memory.py -v` plus `PYTHONPATH=src .venv/bin/python scripts/compare_baselines.py --val /tmp/rsi_data/val_heldout.bin` (any readable bytes file; prints four bpb numbers).
Expected: PASS + four printed numbers (uniform exactly 8.0).

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_baselines.py tests/test_memory.py
git commit -m "feat: baseline comparisons and memory sensitivity gate"
```

---

### Task 5: Experimental re-measurement matrix

**Files:**
- Create: `raw/2026-09-18-rtrl-remeasurement.md` (result record; wiki ingest follows project practice)
- Test: none new (this task runs experiments, gated on completion + record, never on outcomes)

**Interfaces:**
- Consumes: corrected implementation from Tasks 0–4.
- Produces: a filled matrix of short-horizon runs (tiny configs, fixed seed, fixed data slice) answering: (A) Learning — bpb improves; (B) State — persistent beats reset-every-byte; (C) Memory — copy accuracy at 8/64 bytes; (D) Timescales — decay-group ablations differ; (E) Continual — retention nonzero; (F) Ablation — RTRL vs single-step-autograd (latter via a `--no-rtrl` temporary flag? No new shipped flag: implement as a test-only monkeypatch in the run script section below — do NOT add a permanent second training path).

- [ ] **Step 1: Write the run script**

```python
# scripts/remeasure.py (thin, ~50 lines): for each cell in
# {rtrl, single-step} x {persistent, reset} x {groups-4, groups-1}:
# build TMTModel(dim=32, layers=2, lr=1e-4, update_every=1), train 2000
# steps on data/vk4a_train bytes, evaluate smoke preset on held-out val,
# append one JSON line per cell to runs/remeasure.jsonl.
# Single-step cell: monkeypatch model._rtrl_enabled = False honored by
# _update (implementer adds this single boolean gate in Task 1's block:
# `if getattr(self, "_rtrl_enabled", True): <corrections>`; default True).
```

- [ ] **Step 2: Run it to completion**

Run: `PYTHONPATH=src .venv/bin/python scripts/remeasure.py`
Expected: `runs/remeasure.jsonl` with 8 lines (2×2×2), exit 0. Runtime ~10 min (tiny models).

- [ ] **Step 3: Write minimal implementation**

The `_rtrl_enabled` gate only (one `if` + attribute, default True). Everything else is the existing corrected code.

- [ ] **Step 4: Record results honestly**

Write `raw/2026-09-18-rtrl-remeasurement.md`: per-cell bpb/copy/retention numbers transcribed from the jsonl, one-line interpretation each, explicit statement of which questions stayed open. No outcome gate: a cell showing RTRL ≈ single-step is a finding, not a failure.

- [ ] **Step 5: Commit**

```bash
git add src/tmt/model.py scripts/remeasure.py raw/2026-09-18-rtrl-remeasurement.md
git commit -m "feat: RTRL ablation gate and re-measurement record"
```

---

## Self-Review

**1. Spec coverage:** equivalence Task 0 ✓; RTRL correction + staged FD + all-class proof (review §1,2,4,7) Task 1 ✓; accumulation + reset isolation + clip + temp + variance doc + groups-init + config validation (review §2,3,8,10,11,12,18,19,21) Task 2 ✓; two-file checkpoint + provenance + CUDA RNG + cursor resume + negative tests (review §11–15,22-partial) Task 3 ✓; oracles + uniform baseline + separate scores (review §4,5-gates,§8-orbited,§9,§15,§16,§17,§22-partial,§23) Task 4 ✓; re-measurement A–F incl. RTRL ablation (review §18) Task 5 ✓; invariants + STOP (review §18) in header ✓; chat-loop staleness (§16) + snippet indentation (§17): controller fixes at execution ✓.

**2. Placeholder scan:** `rg` clean required before commit (run it).

**3. Type consistency:** `load_numpy_params(P: dict)` keys fixed in Task 0 ✓ used by nothing later (FD test builds torch models directly) — no drift surface. `save_checkpoint(model, path, meta=None)` — old 2-arg calls keep working ✓. `load_checkpoint(model, path, allow_missing=False)` — old 2-arg calls keep working, now loud ✓. `node_json` new kwargs all defaulted ✓. `_rtrl_enabled` gate default True — all existing tests run corrected path unless set False ✓.
