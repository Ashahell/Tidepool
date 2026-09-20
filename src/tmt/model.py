# src/tmt/model.py
#
# LAYOUT (prove-and-freeze): the CANONICAL RTU core is Encoder,
# ByteDecoder, RTULayer recurrence, TMTModel forward/_update/training
# loop, reset, and load_numpy_params with all experimental flags off.
# Everything marked EXPERIMENTAL below is inert unless its flag is set
# (proven by tests/test_canonical.py) and must stay that way: no
# experimental branch may alter canonical-path numerics.
from __future__ import annotations
from typing import List, Optional, Tuple
from collections import deque
from contextlib import contextmanager
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TMTConfig


def sparsify(h: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k readout mask (k <= 0 = passthrough). Recurrence untouched."""
    if k <= 0 or k >= h.shape[-1]:
        return h
    top, _ = torch.topk(h.abs(), k, dim=-1)
    thresh = top[..., -1].unsqueeze(-1)
    return torch.where(h.abs() >= thresh, h, torch.zeros_like(h))


def write_mask(enc_vec: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """Top-k write mask over dims (k <= 0 = all ones). Input-dependent
    constant for parameter gradients: exact RTRL needs no new terms
    beyond scaling the direct-write inhomogeneous by this mask."""
    if k <= 0 or k >= dim:
        return torch.ones(dim, dtype=enc_vec.dtype)
    top, _ = torch.topk(enc_vec.abs(), k)
    thresh = top[-1]
    return (enc_vec.abs() >= thresh).to(enc_vec.dtype)
from .slots import SlotMemory
from .fastweight import FastWeightMemory

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
    # EXPERIMENTAL params below (selective gates, write mask, fast
    # weights): inert unless flagged.
    def __init__(self, dim: int, selective: bool = False, write_k: int = 0,
                 fw_dk: int = 0):
        super().__init__()
        self.dim = dim
        self.use_selective = selective
        self.write_k = write_k
        self.fw = (FastWeightMemory(dim, fw_dk)  # EXPERIMENTAL (fw_dk > 0)
                   if fw_dk > 0 else None)
        self.decay_bias = nn.Parameter(torch.zeros(dim))
        self.gate_w = nn.Parameter(torch.zeros(dim))
        self.in_bias = nn.Parameter(torch.zeros(dim))
        self.in_w = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.register_buffer("states", torch.zeros(1, dim))
        self.register_buffer("decaytrace", torch.zeros(dim))
        self.register_buffer("embedtrace", torch.zeros(256, dim))
        self.register_buffer("gatetrace", torch.zeros(dim, dim))
        self.register_buffer("ingtrace", torch.zeros(dim))
        self.register_buffer("ingwmat", torch.zeros(dim, dim))

    def forward(self, enc: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_selective:  # EXPERIMENTAL
            decay = torch.sigmoid(self.decay_bias + self.gate_w * enc_v)
            ingate = torch.sigmoid(self.in_bias + self.in_w * enc_v)
        else:
            decay = torch.sigmoid(self.decay_bias)
            ingate = torch.ones_like(decay)
        wmask = write_mask(enc.squeeze(0), self.write_k, self.dim)  # EXPERIMENTAL when write_k > 0
        state = decay * self.states + ingate.unsqueeze(0) * (wmask.unsqueeze(0) * enc)
        out = x + self.silu(self.weights(self.norm(state)))
        if self.fw is not None:  # EXPERIMENTAL (fast-weight retrieval)
            out = self.fw.step(out)
        return out, state, decay, ingate

class TMTModel(nn.Module):
    def __init__(self, cfg: TMTConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.dim)
        self.decoder = ByteDecoder(cfg.dim)
        self.layers = nn.ModuleList([RTULayer(cfg.dim, cfg.selective, cfg.write_k, cfg.fw_dk) for _ in range(cfg.layers)])
        # NOTE (task-1 deviation): slots/mem_head are created BEFORE the
        # optimizer so AdamW owns the query/key/mem_head params; the brief's
        # "after replay_buf lines" placement would leave them untrained.
        # EXPERIMENTAL (all inert unless flagged — see tests/test_canonical.py):
        self.slots = SlotMemory(cfg.dim, cfg.slots, temp=cfg.slot_temp) if cfg.slots > 0 else None
        self.mem_head = nn.Linear(2 * cfg.dim, 256) if cfg.slots > 0 else None
        self.opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr)
        self._accum = 0
        self.last_components: dict = {}
        self.ema_state = None  # EXPERIMENTAL (EMA shadows; inert at ema_decay 0)
        self.replay_buf = (deque(maxlen=cfg.replay_size)  # EXPERIMENTAL (replay; None at size 0)
                            if cfg.replay_size > 0 else None)

    def _replay_indices(self, k: int) -> list[int]:
        buf = self.replay_buf
        if not buf:
            return []
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

    def reset(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                layer.states.zero_()
                layer.decaytrace.zero_()
                layer.embedtrace.zero_()
                layer.gatetrace.zero_()
                layer.ingtrace.zero_()
                layer.ingwmat.zero_()
                if layer.fw is not None:  # EXPERIMENTAL
                    layer.fw.reset()
        self._accum = 0

    def load_numpy_params(self, P: dict) -> None:
        """Copy float64 NumPy reference params into torch params; reset traces."""
        import numpy as np

        def _as(name, arr, shape):
            a = np.asarray(arr, dtype=np.float64)
            if tuple(a.shape) != tuple(shape):
                raise ValueError(f"{name}: expected shape {tuple(shape)}, got {tuple(a.shape)}")
            return torch.from_numpy(a)

        dim = self.cfg.dim
        with torch.no_grad():
            self.encoder.embed.weight.copy_(
                _as("embed", P["embed"], (256, dim)).to(
                    self.encoder.embed.weight.dtype))
            if len(P["layers"]) != len(self.layers):
                raise ValueError(
                    f"layers: expected {len(self.layers)}, got {len(P['layers'])}")
            for i, (L, layer) in enumerate(zip(P["layers"], self.layers)):
                layer.decay_bias.copy_(
                    _as(f"layers[{i}].decay", L["decay"], (dim,)).to(
                        layer.decay_bias.dtype))
                layer.weights.weight.copy_(
                    _as(f"layers[{i}].weight", L["weight"], (dim, dim)).to(
                        layer.weights.weight.dtype))
                layer.norm.weight.copy_(
                    _as(f"layers[{i}].ln_w", L["ln_w"], (dim,)).to(
                        layer.norm.weight.dtype))
                layer.norm.bias.copy_(
                    _as(f"layers[{i}].ln_b", L["ln_b"], (dim,)).to(
                        layer.norm.bias.dtype))
            self.decoder.decode.weight.copy_(
                _as("dec_w", P["dec_w"], (256, dim)).to(
                    self.decoder.decode.weight.dtype))
            self.decoder.decode.bias.copy_(
                _as("dec_b", P["dec_b"], (256,)).to(
                    self.decoder.decode.bias.dtype))
            self.decoder.stop.weight.copy_(
                _as("stop_w", P["stop_w"], (dim, 1)).t().to(
                    self.decoder.stop.weight.dtype))
            self.decoder.stop.bias.copy_(
                _as("stop_b", P["stop_b"], (1,)).to(
                    self.decoder.stop.bias.dtype))
            for layer in self.layers:
                layer.states.zero_()
                layer.decaytrace.zero_()
                layer.embedtrace.zero_()
        self._accum = 0

    def init_decay_groups(self, half_lives=None):
        if half_lives is None:
            if self.cfg.decay_groups == 4:
                half_lives = [8.0, 64.0, 512.0, 4000.0]
            else:
                g = self.cfg.decay_groups
                # log-spaced half-lives between 8 and 4000
                half_lives = [math.exp(math.log(8.0) + (math.log(4000.0) - math.log(8.0)) * i / (g - 1))
                              for i in range(g)] if g > 1 else [8.0]
        elif len(half_lives) != self.cfg.decay_groups:
            raise ValueError(
                f"half_lives length {len(half_lives)} != decay_groups {self.cfg.decay_groups}")
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

    def forward(self, x: torch.LongTensor):
        enc = self.encoder(x)
        h = enc
        states: List[torch.Tensor] = []
        for layer in self.layers:
            h, state, _, _ = layer(enc, h)
            states.append(state)
        logits, stop = self._decode(h)
        return logits, states

    def _decode(self, h: torch.Tensor):
        # EXPERIMENTAL branches (slots/mem_head/sparsity): plain decoder
        # path is the canonical behavior, identical when flags are off.
        if self.slots is None:
            return self.decoder(sparsify(h, self.cfg.sparse_k))
        read_vec = self.slots.read(h)
        logits = self.mem_head(torch.cat([sparsify(h, self.cfg.sparse_k).squeeze(0), read_vec], dim=-1).unsqueeze(0))
        _, stop = self.decoder(h)
        return logits, stop

    def _slot_write(self, h: torch.Tensor) -> None:
        # EXPERIMENTAL (slots): no-op unless slots are on.
        if self.slots is not None:
            self.slots.write(h)

    def _adaptive_temp(self, entropy: float) -> float:
        return max(0.1, self.cfg.temp * (1.0 - self.cfg.temp * entropy))

    def _sample(self, logits: torch.Tensor) -> int:
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            entropy = float(-(probs * (probs + 1e-8).log()).sum() / math.log(256))
            temp = self._adaptive_temp(entropy)
            return int(torch.distributions.Categorical(logits=logits / temp).sample().item())

    def _update(self, curr: int, next_: Optional[int], end: bool):
        self.train()
        c = torch.tensor([curr], dtype=torch.long)
        enc = self.encoder(c)
        h = enc
        states, decays, ingates = [], [], []
        for layer in self.layers:
            h, state, decay, ingate = layer(enc, h)
            states.append(state)
            decays.append(decay)
            ingates.append(ingate)
        for s in states:
            s.retain_grad()
        logits, stop = self._decode(h)
        x = h
        # Slot write AFTER decode: read-before-write per step.
        self._slot_write(h)
        # Activation-scale regularizer: forces per-token feature variance
        # toward >= 1 (population var across dim of the single-token state).
        loss = torch.maximum(
            torch.tensor(0.0),
            1.0 - torch.sqrt(x.var(unbiased=False) + 1e-4),
        ) * self.cfg.w_var
        if self.cfg.decay_groups > 1:
            loss = loss + 0.01 * self.decay_diversity_penalty()
        t_pred = t_ce = t_stop = None
        if next_ is not None:
            with torch.no_grad():
                tgt = self.encoder(torch.tensor([next_], dtype=torch.long))
            t_pred = self.cfg.w_pred * torch.mean((x - tgt) ** 2)
            t_ce = self.cfg.w_ce * (F.cross_entropy(logits.view(-1, 256), torch.tensor([next_])))
            target_stop = torch.tensor([[1.0 if end else 0.0]])
            t_stop = self.cfg.w_stop * torch.mean((stop - target_stop) ** 2)
            loss = loss + t_pred + t_ce + t_stop
        with torch.no_grad():
            parts = [t for t in (t_pred, t_ce, t_stop) if t is not None]
            l_var = float((loss - sum(parts)).detach()) if parts else float(loss.detach())
            comp = {
                "l_var": l_var,
                "l_pred": float(t_pred.detach()) if t_pred is not None else 0.0,
                "l_ce": float(t_ce.detach()) if t_ce is not None else 0.0,
                "l_stop": float(t_stop.detach()) if t_stop is not None else 0.0,
                "state_norm": float(sum(torch.linalg.norm(s).detach() for s in states)),
            }
        # update_every is sequential accumulation across evolving timesteps,
        # NOT a minibatch: never detach/reset state at accumulation boundaries.
        if self._accum == 0:
            self.opt.zero_grad()
        loss.backward()
        # Gradient classification: embedding + decay_bias influence future
        # steps through persistent state -> RTRL trace corrections below.
        # weights/LayerNorm/decoder/stop-head affect only the current step
        # -> plain autograd is exact, no correction.
        # Selective gate (vector w_g): same treatment as decay_bias —
        # autograd holds the direct term, the correction adds only the
        # recurrent part dL_t/ds_t·d·G_{t-1}.
        # _rtrl_enabled (default True) is the Task 5 ablation gate.
        # Decay accumulation rule: autograd already holds each step's DIRECT
        # term dL_t/ds_t·d(1-d)·s_{t-1} (decay's only in-graph use is the
        # state computation). The correction adds ONLY the recurrent part
        # dL_t/ds_t·d·T_{t-1}; their sum is dL_t/ds_t·T_t, exactly the
        # upstream overwrite term — but accumulated across steps instead of
        # discarded. (Upstream overwrites because it steps every __call__;
        # under accumulation, overwrite keeps the last step only.)
        with torch.no_grad():
            if getattr(self, "_rtrl_enabled", True):
                for i, layer in enumerate(self.layers):
                    dlds = states[i].grad.detach().squeeze(0)
                    old_embed = layer.embedtrace.detach().clone()
                    old_decay = decays[i].detach()
                    # old_embedtrace already carries past gate influence via
                    # the trace recursion below; the current-step gate path
                    # is autograd's (in-graph), so no extra term here.
                    embed_corr = dlds * (old_embed * old_decay)
                    if self.encoder.embed.weight.grad is not None:
                        self.encoder.embed.weight.grad += embed_corr
                    else:
                        self.encoder.embed.weight.grad = embed_corr.clone()
                    rec = (dlds * old_decay * layer.decaytrace.detach()).clone()
                    if layer.decay_bias.grad is None:
                        layer.decay_bias.grad = rec
                    else:
                        layer.decay_bias.grad += rec
                    if layer.use_selective:
                        dcur = decays[i].detach()
                        grec = ((dlds * dcur).unsqueeze(0) @ layer.gatetrace.detach()).squeeze(0).clone()
                        if layer.gate_w.grad is None:
                            layer.gate_w.grad = grec
                        else:
                            layer.gate_w.grad += grec
                        # Input-gate params: same recurrent-add pattern, and
                        # same gating — inert (grad None) when flag off.
                        # Autograd holds each step's direct term; the
                        # correction adds only dL_t/ds_t·d·T_{t-1}.
                        irec = (dlds * old_decay * layer.ingtrace.detach()).clone()
                        if layer.in_bias.grad is None:
                            layer.in_bias.grad = irec
                        else:
                            layer.in_bias.grad += irec
                        iwrec = ((dlds * old_decay).unsqueeze(0) @ layer.ingwmat.detach()).squeeze(0).clone()
                        if layer.in_w.grad is None:
                            layer.in_w.grad = iwrec
                        else:
                            layer.in_w.grad += iwrec
        # RTRL trace update (matches MLX dummy-gradient correction).
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                d = decays[i].detach()
                s = states[i].detach()
                g = ingates[i].detach()
                one_hot = torch.zeros_like(layer.embedtrace)
                one_hot[curr] += 1.0
                # Direct input Jacobian is g-scaled under selective input
                # gating (g == 1 exactly when the flag is off) and masked
                # by the write mask (ones when write_k <= 0).
                wmask = write_mask(enc.detach().squeeze(0), layer.write_k, layer.dim)
                layer.embedtrace.mul_(d).add_(one_hot * (g * wmask).unsqueeze(0))
                layer.decaytrace.mul_(d).add_(d * (1.0 - d) * layer.states.squeeze(0))
                if layer.use_selective:
                    enc_vec = enc.detach().squeeze(0)
                    s_old = layer.states.detach().squeeze(0)
                    m = d * (1.0 - d) * s_old * enc_vec
                    layer.gatetrace.mul_(d.unsqueeze(1)).add_(torch.diag(m))
                    wg = layer.gate_w.detach()
                    layer.embedtrace.add_(one_hot * (d * (1.0 - d) * s_old * wg).unsqueeze(0))
                    iw = layer.in_w.detach()
                    # In-gate inhomogeneous terms flow through the masked
                    # write (s = d·s_old + g·(m⊙e)): scale by wmask.
                    layer.ingtrace.mul_(d).add_(wmask * g * (1.0 - g) * enc_vec)
                    layer.ingwmat.mul_(d.unsqueeze(1)).add_(
                        torch.diag(wmask * g * (1.0 - g) * enc_vec * enc_vec))
                    # In-gate term flows through the masked write: scale by wmask.
                    layer.embedtrace.add_(one_hot * ((g * (1.0 - g) * enc_vec * iw) * wmask).unsqueeze(0))
                layer.states.copy_(s)
        self._accum += 1
        if self._accum >= self.cfg.update_every:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad()
            self._accum = 0
            self._ema_track()
        return loss.detach(), logits.detach(), stop.detach(), comp

    @torch.no_grad()
    def _ema_track(self):  # EXPERIMENTAL (EMA; no-op at decay 0)
        if self.cfg.ema_decay <= 0.0:
            return
        if self.ema_state is None:
            self.ema_state = {n: p.detach().clone()
                              for n, p in self.named_parameters()}
            return
        d = self.cfg.ema_decay
        for n, p in self.named_parameters():
            self.ema_state[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def using_ema(self):
        if self.ema_state is None:
            yield
            return
        saved = {n: p.detach().clone() for n, p in self.named_parameters()}
        try:
            with torch.no_grad():
                for n, p in self.named_parameters():
                    p.copy_(self.ema_state[n])
            yield
        finally:
            with torch.no_grad():
                for n, p in self.named_parameters():
                    p.copy_(saved[n])

    def training_step(self, curr: int, next_: Optional[int], end: bool):
        for name, v in (("curr", curr), ("next", next_)):
            if v is None and name == "next":
                continue
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255:
                raise ValueError(f"{name} byte out of range [0, 255]: {v!r}")
        self.train()
        loss, logits, stop, comp = self._update(curr, next_, end)
        # EXPERIMENTAL (replay): skipped entirely when replay_buf is None.
        if self.replay_buf is not None:
            self.replay_buf.append((curr, next_, end, float(loss)))
            if self.cfg.replay_k > 0:
                if self.cfg.replay_noise:
                    for _ in range(self.cfg.replay_k):
                        c = int(torch.randint(0, 256, (1,)).item())
                        n = int(torch.randint(0, 256, (1,)).item())
                        self._update(c, n, False)
                else:
                    for idx in self._replay_indices(self.cfg.replay_k):
                        c, n, e, _ = self.replay_buf[idx]
                        rloss, _, _, _ = self._update(c, n, e)
                        self.replay_buf[idx] = (c, n, e, float(rloss))
        self.last_components = comp
        with torch.no_grad():
            sampled = self._sample(logits)
            stop_v = float(stop.item())
        return loss, sampled, stop_v

    @torch.no_grad()
    def ingest(self, curr: int):
        """Advance persistent state without learning: forward + commit.

        The evaluation probes drive forward(), which never touches the
        persistent buffers, so no real model can score on them. Probes
        that must measure retention call ingest() instead: identical
        state motion to training, minus gradients, traces, and steps.
        """
        enc = self.encoder(torch.tensor([curr], dtype=torch.long))
        h = enc
        for layer in self.layers:
            h, state, _, _ = layer(enc, h)
            layer.states.copy_(state)
        logits, stop = self._decode(h)
        self._slot_write(h)
        return logits, stop
