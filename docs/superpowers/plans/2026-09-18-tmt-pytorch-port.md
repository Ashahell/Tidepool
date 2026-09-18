# TMT PyTorch Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 230-line MLX TMT model to PyTorch CUDA with the approved surgical fixes, wired to the existing eval suite.

**Architecture:** Keep the single-byte online loop and RTRL trace structure from MLX; traces are torch buffers, main path uses autograd, trace corrections are manual adds. Config-driven, Node-JSON logging, eval-suite-compatible call signature.

**Tech Stack:** Python 3.14, torch 2.14.0+cu130, safetensors, pyyaml, pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-09-18-tmt-pytorch-port-design.md`

## Global Constraints

- Python 3.14 via `.venv/bin/python`; never use system python for runs.
- torch 2.14.0+cu130 with CUDA live on RTX 5070 Ti (sm 12,0); CPU fallback allowed only in tests.
- All runs from repo root `/home/miller/Work/projects/tmt-torch`.
- Eval-suite call contract is frozen: `model.reset() -> None`; `logits, state = model(LongTensor(1,))` with `logits` shape `(1, 256)`.
- No MLX imports anywhere in `src/` or `tests/`.
- Every task ends with a commit; no task leaves the tree red.

---

## File Map

- `src/tmt/config.py` — `TMTConfig` dataclass, `from_yaml`/`to_dict`, defaults match the 4.5M ref (`dim=512, layers=16, temp=0.75, lr=5e-4`).
- `configs/base.yaml` — the same defaults in YAML.
- `src/tmt/model.py` — `Encoder`, `ByteDecoder`, `RTULayer`, `TMTModel` (eval call + `training_step`).
- `src/tmt/engine.py` — `train_loop`, `chat_loop`, `save_checkpoint`, `load_checkpoint`, `node_json`.
- `src/tmt/data.py` — `iter_wikipedia_bytes`, `load_val_bytes`.
- `scripts/train.py`, `scripts/chat.py`, `scripts/eval_memory.py` — thin CLIs.
- `src/tmt/evaluation_suite.py` — EXISTS (frozen contract, do not change its model interface).
- Tests: `tests/test_config.py`, `tests/test_model.py`, `tests/test_fixes.py`, `tests/test_wiring.py`.

---

### Task 1: Config + YAML defaults

**Files:**
- Create: `src/tmt/config.py`
- Create: `configs/base.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TMTConfig(dim, layers, temp, lr, update_every, w_var, w_pred, w_ce, w_stop, decay_groups, grad_clip, seed)` with `TMTConfig.from_yaml(path)`, `TMTConfig.to_dict()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from tmt.config import TMTConfig

def test_defaults_match_ref():
    c = TMTConfig()
    assert (c.dim, c.layers, c.temp, c.lr) == (512, 16, 0.75, 5e-4)
    assert c.update_every == 1

def test_yaml_roundtrip(tmp_path):
    from tmt.config import TMTConfig
    c = TMTConfig(dim=64, layers=2)
    p = tmp_path / "c.yaml"
    p.write_text("dim: 64\nlayers: 2\n")
    c2 = TMTConfig.from_yaml(str(p))
    assert (c2.dim, c2.layers) == (64, 2)
    assert c2.to_dict()["dim"] == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with "No module named 'tmt.config'".

- [ ] **Step 3: Write minimal implementation**

```python
# src/tmt/config.py
from __future__ import annotations
from dataclasses import asdict, dataclass
import yaml

@dataclass
class TMTConfig:
    dim: int = 512
    layers: int = 16
    temp: float = 0.75
    lr: float = 5e-4
    update_every: int = 1
    w_var: float = 1.0
    w_pred: float = 1.0
    w_ce: float = 1.0
    w_stop: float = 1.0
    decay_groups: int = 4
    grad_clip: float = 1.0
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TMTConfig":
        with open(path) as f:
            data = yaml.safe_load() or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)
```

```yaml
# configs/base.yaml
dim: 512
layers: 16
temp: 0.75
lr: 0.0005
update_every: 1
w_var: 1.0
w_pred: 1.0
w_ce: 1.0
w_stop: 1.0
decay_groups: 4
grad_clip: 1.0
seed: 42
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tmt/config.py configs/base.yaml tests/test_config.py
git commit -m "feat: add TMTConfig with YAML defaults"
```

---

### Task 2: Model core (eval call + training step)

**Files:**
- Create: `src/tmt/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `TMTConfig` from Task 1 (exact field names).
- Produces: `Encoder(dim)`, `ByteDecoder(dim)`, `RTULayer(dim)`, `TMTModel(cfg)` with `reset() -> None`, `forward(x: LongTensor(1,)) -> (logits(1,256), state list)`, `__call__` alias of forward, `training_step(curr:int, next_:int|None, end:bool) -> (loss Tensor, sampled int, stop float)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def test_eval_call_shape_and_reset():
    m = TMTModel(TMTConfig(dim=32, layers=2))
    m.reset()
    logits, state = m(torch.tensor([65], dtype=torch.long))
    assert logits.shape == (1, 256)
    assert isinstance(state, list) and len(state) == 2

def test_single_layer_math():
    # Hand-computed: decay=sigmoid(0)=0.5, state=0.5*0+enc+0=enc.
    torch.manual_seed(0)
    m = TMTModel(TMTConfig(dim=4, layers=1))
    with torch.no_grad():
        m.encoder.embed.weight.zero_()
        m.encoder.embed.weight[65] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m.layers[0].decay_bias.zero_()
    m.reset()
    logits, state = m(torch.tensor([65], dtype=torch.long))
    assert torch.allclose(state[0], torch.tensor([[1.0, 2.0, 3.0, 4.0]]), atol=1e-5)

def test_training_step_runs():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    loss, sampled, stop = m.training_step(65, 66, False)
    assert loss.numel() == 1 and 0 <= sampled <= 255 and 0.0 <= stop <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL with "No module named 'tmt.model'".

- [ ] **Step 3: Write minimal implementation**

```python
# src/tmt/model.py
from __future__ import annotations
from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TMTConfig

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.embed(x)

class ByteDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(x), torch.sigmoid(self.stop(x))

class RTULayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.decay_bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.register_buffer("states", torch.zeros(1, dim))
        self.register_buffer("decaytrace", torch.zeros(dim))
        self.register_buffer("embedtrace", torch.zeros(256, dim))

    def forward(self, enc: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.decay_bias)
        state = decay * self.states + enc
        out = x + self.silu(self.weights(self.norm(state)))
        return out, state, decay

class TMTModel(nn.Module):
    def __init__(self, cfg: TMTConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.dim)
        self.decoder = ByteDecoder(cfg.dim)
        self.layers = nn.ModuleList([RTULayer(cfg.dim) for _ in range(cfg.layers)])
        self.opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr)
        self._accum = 0

    def reset(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                layer.states.zero_()
                layer.decaytrace.zero_()
                layer.embedtrace.zero_()
        self._accum = 0

    def forward(self, x: torch.LongTensor):
        enc = self.encoder(x)
        h = enc
        states: List[torch.Tensor] = []
        for layer in self.layers:
            h, state, _ = layer(enc, h)
            states.append(state)
        logits, stop = self.decoder(h)
        return logits, states

    def _sample(self, logits: torch.Tensor) -> int:
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            entropy = float(-(probs * (probs + 1e-8).log()).sum() / math.log(256))
            temp = max(0.1, self.cfg.temp * (1.0 - self.cfg.temp * entropy))
            return int(torch.distributions.Categorical(logits=logits / temp).sample().item())

    def training_step(self, curr: int, next_: Optional[int], end: bool):
        self.train()
        c = torch.tensor([[curr]], dtype=torch.long)
        enc = self.encoder(c)
        h = enc
        states, decays = [], []
        for layer in self.layers:
            h, state, decay = layer(enc, h)
            states.append(state)
            decays.append(decay)
        logits, stop = self.decoder(h)
        x = h
        loss = torch.maximum(
            torch.tensor(0.0),
            1.0 - torch.sqrt(x.var(unbiased=False) + 1e-4),
        ) * self.cfg.w_var
        if next_ is not None:
            with torch.no_grad():
                tgt = self.encoder(torch.tensor([[next_]], dtype=torch.long))
            loss = loss + self.cfg.w_pred * torch.mean((x - tgt) ** 2)
            loss = loss + self.cfg.w_ce * (F.cross_entropy(logits.view(-1, 256), torch.tensor([next_])))
            target_stop = torch.tensor([[1.0 if end else 0.0]])
            loss = loss + self.cfg.w_stop * torch.mean((stop - target_stop) ** 2)
        self.opt.zero_grad()
        loss.backward()
        # RTRL trace update (matches MLX dummy-gradient correction).
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                d = decays[i].detach()
                s = states[i].detach()
                one_hot = torch.zeros_like(layer.embedtrace)
                one_hot[curr] += 1.0
                layer.embedtrace.mul_(d).add_(one_hot)
                layer.decaytrace.mul_(d).add_(d * (1.0 - d) * layer.states.squeeze(0))
                layer.states.copy_(s)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
        self._accum += 1
        if self._accum >= self.cfg.update_every:
            self.opt.step()
            self.opt.zero_grad()
            self._accum = 0
        with torch.no_grad():
            sampled = self._sample(logits.detach())
            stop_v = float(stop.detach().item())
        return loss.detach(), sampled, stop_v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tmt/model.py tests/test_model.py
git commit -m "feat: add torch TMT model core with RTRL traces"
```

---

### Task 3: Decay-group init + diversity + leak guard

**Files:**
- Modify: `src/tmt/model.py` (add `init_decay_groups`, `decay_diversity_penalty`)
- Test: `tests/test_fixes.py`

**Interfaces:**
- Consumes: `TMTModel(cfg)` from Task 2.
- Produces: `TMTModel.init_decay_groups(half_lives=[8,64,512,4000])`, `decay_diversity_penalty() -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def test_decay_groups_spread():
    m = TMTModel(TMTConfig(dim=16, layers=1, decay_groups=4))
    m.init_decay_groups([8.0, 64.0, 512.0, 4000.0])
    d = torch.sigmoid(m.layers[0].decay_bias).detach()
    assert float(d.max() - d.min()) > 0.2

def test_reset_clears_state():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    m(torch.tensor([65], dtype=torch.long))
    m.reset()
    for layer in m.layers:
        assert float(layer.states.abs().max()) == 0.0
        assert float(layer.embedtrace.abs().max()) == 0.0

def test_update_every_accumulates():
    m = TMTModel(TMTConfig(dim=16, layers=1, update_every=4))
    before = [p.detach().clone() for p in m.parameters()]
    for _ in range(3):
        m.training_step(65, 66, False)
    mid = [p.detach().clone() for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before, mid))
    m.training_step(65, 66, False)
    after = [p.detach().clone() for p in m.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(mid, after))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_fixes.py -v`
Expected: FAIL with "has no attribute 'init_decay_groups'".

- [ ] **Step 3: Write minimal implementation**

Add these two methods to `TMTModel` in `src/tmt/model.py` (after `reset`, before `forward`):

```python
    def init_decay_groups(self, half_lives=[8.0, 64.0, 512.0, 4000.0]):
        with torch.no_grad():
            for layer in self.layers:
                dim = layer.dim
                g = len(half_lives)
                idx = torch.arange(dim) % g
                targets = torch.tensor([0.5 ** (1.0 / h) for h in half_lives])
                chosen = targets[idx].clamp(1e-4, 1.0 - 1e-4)
                layer.decay_bias.copy_(torch.log(chosen / (1.0 - chosen)))

    def decay_diversity_penalty(self):
        pens = []
        for layer in self.layers:
            d = torch.sigmoid(layer.decay_bias)
            pens.append(-torch.var(d))
        return torch.stack(pens).mean()
```

Then edit `training_step` in the same file — change this exact block:

```python
        loss = torch.maximum(
            torch.tensor(0.0),
            1.0 - torch.sqrt(x.var(unbiased=False) + 1e-4),
        ) * self.cfg.w_var
```

to:

```python
        loss = torch.maximum(
            torch.tensor(0.0),
            1.0 - torch.sqrt(x.var(unbiased=False) + 1e-4),
        ) * self.cfg.w_var
        if self.cfg.decay_groups > 1:
            loss = loss + 0.01 * self.decay_diversity_penalty()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_fixes.py tests/test_model.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tmt/model.py tests/test_fixes.py
git commit -m "feat: add multi-timescale decay init and leak/update guards"
```

---

### Task 4: Engine + data + CLI wiring

**Files:**
- Create: `src/tmt/engine.py` (save/load/node_json/train_loop/chat_loop)
- Create: `src/tmt/data.py` (`iter_wikipedia_bytes`, `load_val_bytes`)
- Create: `scripts/train.py`, `scripts/chat.py`, `scripts/eval_memory.py`
- Test: `tests/test_wiring.py`

**Interfaces:**
- Consumes: `TMTConfig`, `TMTModel`, `evaluate_model` (frozen).
- Produces: `save_checkpoint(model, path)`, `load_checkpoint(model, path)`, `node_json(...) -> dict`, `iter_wikipedia_bytes(root)`, CLIs with `--config`, `--steps`, `--readonly`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wiring.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.engine import save_checkpoint, load_checkpoint, node_json

def test_checkpoint_roundtrip(tmp_path):
    m = TMTModel(TMTConfig(dim=16, layers=1))
    p = str(tmp_path / "m.safetensors")
    save_checkpoint(m, p)
    m2 = TMTModel(TMTConfig(dim=16, layers=1))
    load_checkpoint(m2, p)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(a, b)

def test_node_json_shape():
    from tmt.engine import node_json
    n = node_json({"dim": 16}, {"val_bpb": 5.0}, 0.1, None)
    assert set(n) == {"config", "final_metrics", "composite_score", "failure_mode", "cost"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_wiring.py -v`
Expected: FAIL with "No module named 'tmt.engine'".

- [ ] **Step 3: Write minimal implementation**

```python
# src/tmt/data.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator

def iter_wikipedia_bytes(root: str = "wikipedia_clean") -> Iterator[bytes]:
    for p in sorted(Path(root).rglob("wiki_*")):
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                b = line.encode("utf-8", errors="ignore")
                if b:
                    yield b

def load_val_bytes(path: str, limit: int = 20000) -> bytes:
    data = Path(path).read_bytes()[:limit]
    return data
```

```python
# src/tmt/engine.py
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from safetensors.torch import save_file, load_file

def save_checkpoint(model, path: str) -> None:
    state = {f"m.{k}": v.cpu() for k, v in model.state_dict().items()}
    for i, layer in enumerate(model.layers):
        state[f"trace_state.{i}"] = layer.states.cpu()
        state[f"trace_decay.{i}"] = layer.decaytrace.cpu()
        state[f"trace_embed.{i}"] = layer.embedtrace.cpu()
    save_file(state, path)

def load_checkpoint(model, path: str) -> None:
    from pathlib import Path as _P
    if not _P(path).exists():
        return
    data = load_file(path)
    sd = {k[2:]: v for k, v in data.items() if k.startswith("m.")}
    model.load_state_dict(sd, strict=False)
    for i, layer in enumerate(model.layers):
        if f"trace_state.{i}" in data:
            layer.states.copy_(data[f"trace_state.{i}"])
            layer.decaytrace.copy_(data[f"trace_decay.{i}"])
            layer.embedtrace.copy_(data[f"trace_embed.{i}"])

def node_json(config: dict, metrics: dict, gpu_hours: float, failure: Optional[str], composite: float = 0.0) -> Dict[str, Any]:
    return {
        "config": config,
        "final_metrics": metrics,
        "composite_score": composite,
        "failure_mode": failure,
        "cost": {"gpu_hours": gpu_hours},
    }
```

```python
# scripts/train.py
from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import iter_wikipedia_bytes
from tmt.engine import save_checkpoint

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    model.init_decay_groups()
    n = 0
    try:
        for chunk in iter_wikipedia_bytes():
            for i in range(len(chunk) - 1):
                model.training_step(chunk[i], chunk[i + 1], i == len(chunk) - 2)
                n += 1
                if n % 500 == 0:
                    save_checkpoint(model, args.ckpt)
                if n >= args.steps:
                    save_checkpoint(model, args.ckpt)
                    return
    finally:
        save_checkpoint(model, args.ckpt)

if __name__ == "__main__":
    main()
```

```python
# scripts/chat.py
from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.engine import load_checkpoint, save_checkpoint

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    ap.add_argument("--readonly", action="store_true")
    ap.add_argument("--notrace", action="store_true")
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    load_checkpoint(model, args.ckpt)
    try:
        while True:
            text = input("User >> ") + "\n"
            data = text.encode("utf-8", errors="ignore")
            for i in range(len(data) - 1):
                if args.notrace:
                    with torch.no_grad():
                        logits, _ = model(torch.tensor([data[i]], dtype=torch.long))
                        b = int(torch.argmax(logits[0]).item())
                else:
                    _, b, _ = model.training_step(data[i], data[i + 1], False) if not args.readonly else (None, int(torch.argmax(model(torch.tensor([data[i]], dtype=torch.long))[0][0]).item()), 0.0)
            b = data[-1]
            print("Model >> ", end="", flush=True)
            while True:
                with torch.no_grad():
                    logits, _ = model(torch.tensor([b], dtype=torch.long))
                import torch.nn.functional as F
                probs = F.softmax(logits, dim=-1)
                b = int(torch.argmax(probs[0]).item())
                sys.stdout.buffer.write(bytes([b]))
                sys.stdout.flush()
                if len(data) > 64:
                    break
            print()
    finally:
        if not args.readonly:
            save_checkpoint(model, args.ckpt)

if __name__ == "__main__":
    main()
```

```python
# scripts/eval_memory.py
from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "src")
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import load_val_bytes
from tmt.engine import load_checkpoint
from tmt.evaluation_suite import EvalConfig, evaluate_model

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="runs/model.safetensors")
    ap.add_argument("--val", default="val.bin")
    args = ap.parse_args()
    cfg = TMTConfig.from_yaml(args.config)
    model = TMTModel(cfg)
    load_checkpoint(model, args.ckpt)
    val = load_val_bytes(args.val)
    eval_cfg = EvalConfig(max_bytes=2000, lm_eval_tokens=512, copy_lengths=[4], intervening_lengths=[8, 64], num_copy_trials=2)
    r = evaluate_model(model, val, [val], cfg=eval_cfg, config_dict=cfg.to_dict(), gpu_hours=0.0)
    print(f"bpb={r.val_bpb:.3f} mem={r.long_range_score:.3f} cont={r.continual_score:.3f} stab={r.stability_score:.3f} composite={r.composite_score:.3f} fail={r.failure_mode}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_wiring.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tmt/engine.py src/tmt/data.py scripts/train.py scripts/chat.py scripts/eval_memory.py tests/test_wiring.py
git commit -m "feat: add engine, data, and CLI wiring"
```

---

### Task 5: Full gate (pytest + CUDA + eval smoke with real model)

**Files:**
- Test: `tests/test_gate.py` (cuda smoke + 30s real-model eval smoke)

**Interfaces:**
- Consumes: all Tasks 1-4.
- Produces: green gate proof.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate.py
import torch

def test_cuda_live():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (12, 0)

def test_real_model_eval_smoke():
    from tmt.config import TMTConfig
    from tmt.model import TMTModel
    from tmt.evaluation_suite import EvalConfig, evaluate_model
    m = TMTModel(TMTConfig(dim=32, layers=2, update_every=2))
    m.init_decay_groups()
    cfg = EvalConfig(max_bytes=400, lm_eval_tokens=64, copy_lengths=[4], intervening_lengths=[8], num_copy_trials=1, retention_eval_bytes=64)
    val = bytes([(65 + i % 26) for i in range(256)])
    r = evaluate_model(m, val, [val], cfg=cfg, config_dict={"dim": 32}, gpu_hours=0.0)
    assert r.val_bpb < 20.0 and r.stability_score >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_gate.py -v`
Expected: FAIL (model/engine from later tasks missing or red).

- [ ] **Step 3: Fix whatever is red (no new features; wire only)**

Run full suite, fix import/device mismatches only.

- [ ] **Step 4: Run full gate**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -v`
Expected: PASS (all files, 10+ tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_gate.py
git commit -m "test: add full CUDA + real-model eval gate"
```
